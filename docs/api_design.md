# API 接口设计

## FastAPI 端点 (app/server.py)

### 1. Agent 对话 `/agent`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/agent/stream` | Agent 流式对话 |
| POST | `/agent/invoke` | Agent 同步调用 |

请求体:
```json
{
  "input": {
    "messages": [
      {"role": "user", "content": "查询今天成都双流的航班"}
    ]
  }
}
```

响应 (流式 SSE):
```
event: metadata / event: updates / event: end
```

### 2. 流量预测 `/dl/predict`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/dl/predict` | LSTM 预测未来流量 |

请求体:
```json
{
  "airport": "ZUUU",
  "hours": 24
}
```

响应:
```json
{
  "airport": "ZUUU",
  "airport_name": "成都双流国际机场",
  "predictions": [22.1, 18.3, 14.7, ...],
  "hours": 24
}
```

### 3. 异常检测 `/dl/anomaly`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/dl/anomaly` | Autoencoder 异常检测 |

请求体:
```json
{
  "airport": "ZUUU"
}
```

响应:
```json
{
  "airport": "ZUUU",
  "airport_name": "成都双流国际机场",
  "reconstruction_error": 340.6,
  "z_score": 0.12,
  "is_anomaly": false
}
```

### 4. 健康检查 `/health`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |

响应:
```json
{
  "status": "ok",
  "chroma_ok": true,
  "flights_records": 10000,
  "flow_records": 43800,
  "model_loaded": true
}
```

## Chainlit 前端 (ui/app.py)

Chainlit 通过 `astream_events` 流式消费 Agent 输出：

```python
@cl.on_chat_start  → 创建 Agent, 发欢迎消息
@cl.on_message     → history.append → agent.astream_events
                   → on_chat_model_stream: 流式文本
                   → on_tool_start: 展示工具调用
                   → on_tool_end: 展示工具输出摘要
```

## 错误码

| 状态码 | 说明 |
|--------|------|
| 400 | 参数校验失败 |
| 500 | 内部错误 (LLM超时/模型未加载/数据文件缺失) |
| 503 | 服务不可用 (API Key未配置) |
