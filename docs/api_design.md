# API 接口设计

## LangServe 端点

### 1. RAG 问答 `/rag`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/rag/playground/` | LangServe 交互式测试页 |
| POST | `/rag/invoke` | 同步调用 |
| POST | `/rag/stream` | 流式调用 |

请求体:
```json
{
  "input": {
    "question": "成都双流机场的跑道小时容量是多少？"
  }
}
```

响应:
```json
{
  "output": "根据CCAR-93第十二条，成都双流国际机场（ZUUU）双跑道运行小时容量约为 48 架次。",
  "metadata": {
    "sources": ["CCAR-93-空中交通管理规则.md"]
  }
}
```

### 2. Agent 对话 `/agent`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/agent/playground/` | LangServe 交互式测试页 |
| POST | `/agent/invoke` | 同步调用 |
| POST | `/agent/stream` | 流式调用 |

请求体:
```json
{
  "input": {
    "messages": [
      {"role": "user", "content": "查询今天从成都双流出发的航班"}
    ]
  }
}
```

响应 (流式):
```
data: {"type": "agent", "content": "正在查询..."}
data: {"type": "tool", "name": "query_flights", "input": {"dep": "ZUUU"}}
data: {"type": "tool_result", "content": "找到 45 个航班"}
data: {"type": "agent", "content": "今天从成都双流出发共有45个航班..."}
```

### 3. 流量预测 `/dl`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/dl/predict` | 流量预测 |

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
  "predictions": [22, 18, 14, 12, 10, ...],
  "timestamps": ["2026-06-27T22:00", "2026-06-27T23:00", ...]
}
```

### 4. 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |

响应:
```json
{
  "status": "ok",
  "chroma_docs": 350,
  "flights_records": 5000,
  "model_loaded": true
}
```

## Chain 定义

```python
# retriever → prompt → llm → output parser
rag_chain = create_stuff_documents_chain(llm, prompt) | retriever

# Agent
agent = create_react_agent(llm, tools, checkpointer)
```

## 错误码

| 状态码 | 说明 |
|--------|------|
| 400 | 参数校验失败 |
| 500 | 内部错误 (LLM超时/模型未加载/数据文件缺失) |
| 503 | 服务不可用 (API Key未配置) |
