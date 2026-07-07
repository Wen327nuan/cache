# murmur 缓存适配笔记

原始参考：[NyraSeithhh/cache](https://github.com/NyraSeithhh/cache) —— prompt cache 实测 96% 命中率的方块布局。

## 我们的情况

- 后端：`/data/murmur/chat/main.py`（FastAPI + httpx stream）
- 中转：55/玖时（OAI 兼容协议，`oai-passthrough` 模式）
- 协议：`/v1/chat/completions`

## 已做的改动

### 1. `get_model_and_prompt` → 返回三值
```
model, stable_prompt, volatile_content = get_model_and_prompt(c, persona_id, user_message)
```
- `stable_prompt`: 人设 + 常驻世界书 → 缓存安全，逐字节不变
- `volatile_content`: 关键词触发的世界书条目 → 排到缓存边界外

### 2. `make_stream` 缓存优化
- `metadata.user_id = "shenwen-murmur-stable"` → 粘后端
- system message 挂 `cache_control: {type: ephemeral}`
- BP4 rolling：倒数第二条 user 消息挂缓存标
- volatile 内容注入到缓存边界之后的伪 user 消息
- think 指令烘焙进 system（固定文本，不影响缓存前缀）

### 3. `cache_layer.py`（独立缓存中间件）
可以直接 import 到 main.py 替代手写逻辑：
```python
from cache_layer import CacheOptimizer
optimizer = CacheOptimizer(user_id="shenwen-murmur-stable")
api_msgs, body = optimizer.optimize(base_url, api_msgs, body, volatile_content=volatile)
```
自动检测上游类型，三条路径自适应。中转报 4xx 自动降级。

## 下一步
- [ ] 接 Anthropic 直连 key 后，实测 cache_read 数据
- [ ] 前端给 token 统计区域加缓存命中率展示
- [ ] 尝试 OpenRouter 路径验证 or-blocks 模式
- [ ] 实现 BP2 每日内容（日期、日活信息等，一天换一次的层）
- [ ] 实现 BP3 会话压缩摘要（80K token 触发压缩）
