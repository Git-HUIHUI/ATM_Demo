# API 接口设计

> 最后更新: 2026-06-29

`app/server.py` 是可选独立模块，将 Agent 和 DL 模型包装为 REST API。Chainlit 不依赖它启动。

## 端点总览

| 方法 | 路径 | 说明 | 模块 |
|------|------|------|------|
| GET | `/health` | 健康检查 | 运维 |
| POST | `/agent/stream` | Agent 流式对话 | Agent |
| POST | `/dl/predict` | LSTM 预测未来流量 | DL |
| POST | `/dl/anomaly` | Autoencoder 异常检测 | DL |

## 1. 健康检查 `GET /health`

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

## 2. Agent 对话 `POST /agent/stream`

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

响应: SSE 流 (event: metadata / updates / end)

## 3. LSTM 流量预测 `POST /dl/predict`

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
  "predictions": [22.1, 18.3, 14.7, "...24个值"],
  "hours": 24
}
```

## 4. Autoencoder 异常检测 `POST /dl/anomaly`

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

判定逻辑: `abs(z_score) > 2` → 异常（双向，偏高/偏低均可检测）

## 错误码

| 状态码 | 说明 |
|--------|------|
| 400 | 参数校验失败（机场不存在/数据不足24小时） |
| 500 | 内部错误 (LLM超时/模型未加载/数据文件缺失) |
| 503 | 服务不可用 (API Key未配置) |
