"""
murmur cache layer — 缓存优化中间件
基于 NyraSeithhh/cache 的思路，适配我们的 murmur chat 后端。

核心：把请求切成「不变」和「会变」两堆。
不变的挂 cache_control 标记，变的排到最后面。
三条路径自动适配：anthropic-bp / or-blocks / oai-passthrough

用法：
    from cache_layer import CacheOptimizer
    optimizer = CacheOptimizer(user_id="shenwen-murmur-stable")
    body = optimizer.optimize(model, api_msgs, body, volatile_content="...")
"""

import re
import hashlib
from typing import Optional

# 已知的上游域名 → 模式映射
UPSTREAM_MODES = {
    # Anthropic 原生协议
    "api.anthropic.com": "anthropic-bp",
    "anthropic.msui.uk": "anthropic-bp",
    # OpenRouter
    "openrouter.ai": "or-blocks",
    # 已知不支持 cache_control 的站（会报 4xx）
    "ekan": "blacklisted",
}

# 黑名单文件：记录对 cache_control 报错的 host，之后自动跳过
BLACKLIST_FILE = "/data/murmur/chat/cache_blacklist.txt"


def _load_blacklist():
    try:
        with open(BLACKLIST_FILE) as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()


def _save_blacklist(hosts: set):
    with open(BLACKLIST_FILE, "w") as f:
        f.write("\n".join(sorted(hosts)))


def detect_mode(base_url: str) -> str:
    """根据上游 base_url 自动判定缓存模式。"""
    blacklist = _load_blacklist()
    url_lower = base_url.lower()

    for pattern, mode in UPSTREAM_MODES.items():
        if pattern in url_lower:
            return mode

    # 检查黑名单
    host = re.sub(r"https?://", "", url_lower).split("/")[0]
    if host in blacklist:
        return "blacklisted"

    # 默认：普通 OAI 中转
    return "oai-passthrough"


def mark_blacklist(base_url: str):
    """把某个 host 加入黑名单（收到 4xx 后调用）。"""
    host = re.sub(r"https?://", "", base_url.lower()).split("/")[0]
    bl = _load_blacklist()
    bl.add(host)
    _save_blacklist(bl)


class CacheOptimizer:
    """
    缓存优化器。在 make_stream 发请求前调用 optimize()。

    支持三种模式：
    - anthropic-bp: Anthropic 原生 /v1/messages，system 拆 blocks 挂标
    - or-blocks: OpenRouter OAI 协议，嵌 content block 挂标
    - oai-passthrough: 普通 OAI 中转，挂 cache_control + user_id（中转不认就忽略）
    """

    def __init__(self, user_id: str = "shenwen-murmur-stable"):
        self.user_id = user_id

    def optimize(
        self,
        base_url: str,
        api_msgs: list,
        body: dict,
        volatile_content: str = "",
        is_thinking_model: bool = False,
        think_hint: str = "",
    ) -> tuple:
        """
        优化请求的缓存布局。

        返回 (api_msgs, body) —— 可能被修改过的消息列表和请求体。

        参数：
        - base_url: 上游 API 地址
        - api_msgs: [{"role":"system","content":"..."}, ...] 消息列表
        - body: 请求体 dict（包含 model, messages, stream 等）
        - volatile_content: 易变内容（关键词世界书等），排到缓存边界外
        - is_thinking_model: 是否 thinking 模型
        - think_hint: 非 thinking 模型的 think 指令注入文本
        """
        mode = detect_mode(base_url)

        if mode == "blacklisted":
            # 这个站报过错，不挂任何标，直接透传
            # 但 volatile 和 user_id 还是要处理
            api_msgs = self._inject_volatile(api_msgs, volatile_content)
            body["messages"] = api_msgs
            body["metadata"] = {"user_id": self.user_id}
            return api_msgs, body

        # ===== 1. System 稳定化 =====
        # think_hint 是固定文本，烘焙进 system（不影响缓存稳定性）
        if think_hint and not is_thinking_model:
            if api_msgs and api_msgs[0].get("role") == "system":
                api_msgs = [dict(api_msgs[0], content=api_msgs[0]["content"] + think_hint)] + list(api_msgs[1:])

        # ===== 2. 按模式挂 cache_control =====
        if mode == "anthropic-bp":
            api_msgs = self._apply_anthropic_bp(api_msgs)
        elif mode == "or-blocks":
            api_msgs = self._apply_or_blocks(api_msgs)
        else:  # oai-passthrough
            api_msgs = self._apply_oai_passthrough(api_msgs)

        # ===== 3. BP4 rolling: 倒数第二条 user 挂标 =====
        user_indices = [i for i, m in enumerate(api_msgs) if m.get("role") == "user"]
        if len(user_indices) >= 2:
            bp4_idx = user_indices[-2]
            api_msgs[bp4_idx] = dict(api_msgs[bp4_idx])
            api_msgs[bp4_idx]["cache_control"] = {"type": "ephemeral"}

        # ===== 4. Volatile 内容排到最后（缓存边界之外）=====
        api_msgs = self._inject_volatile(api_msgs, volatile_content)

        # ===== 5. metadata.user_id 粘后端 =====
        body["messages"] = api_msgs
        body["metadata"] = {"user_id": self.user_id}

        return api_msgs, body

    def _apply_anthropic_bp(self, msgs: list) -> list:
        """Anthropic 原生：system 挂 cache_control。"""
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = dict(msgs[0])
            msgs[0]["cache_control"] = {"type": "ephemeral"}
        return msgs

    def _apply_or_blocks(self, msgs: list) -> list:
        """OpenRouter：system 内容转成 content blocks 格式挂标。"""
        if msgs and msgs[0].get("role") == "system":
            sys_msg = dict(msgs[0])
            content = sys_msg["content"]
            if isinstance(content, str):
                sys_msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            msgs = [sys_msg] + list(msgs[1:])
        return msgs

    def _apply_oai_passthrough(self, msgs: list) -> list:
        """普通 OAI 中转：挂 cache_control，中转不认就忽略。"""
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = dict(msgs[0])
            msgs[0]["cache_control"] = {"type": "ephemeral"}
        return msgs

    def _inject_volatile(self, msgs: list, volatile_content: str) -> list:
        """把 volatile 内容注入到缓存边界之外（最后一条 user 之前）。"""
        if not volatile_content or not volatile_content.strip():
            return msgs
        volatile_msg = {
            "role": "user",
            "content": f"<gateway_volatile_context>\n{volatile_content}\n</gateway_volatile_context>",
        }
        ack_msg = {"role": "assistant", "content": "Understood."}
        # 插到最后一条消息之前
        msgs = list(msgs)
        msgs.insert(len(msgs) - 1, volatile_msg)
        msgs.insert(len(msgs) - 1, ack_msg)
        return msgs

    def report_error(self, base_url: str, status_code: int):
        """收到 4xx 时调用，自动把该上游加入黑名单。"""
        if 400 <= status_code < 500:
            mark_blacklist(base_url)


# === 实用函数 ===

def get_cache_stats(cached_tokens: int, input_tokens: int) -> dict:
    """计算缓存命中率，用于前端展示。"""
    if input_tokens <= 0:
        return {"hit_rate": 0, "cached": 0, "total": 0, "status": "no_data"}

    hit_rate = round(cached_tokens / input_tokens * 100, 1) if cached_tokens > 0 else 0

    if cached_tokens == 0:
        status = "miss"
    elif hit_rate > 80:
        status = "excellent"
    elif hit_rate > 50:
        status = "good"
    else:
        status = "partial"

    return {
        "hit_rate": hit_rate,
        "cached": cached_tokens,
        "total": input_tokens,
        "status": status,
    }
