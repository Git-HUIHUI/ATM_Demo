# 系统架构

## 架构图

```
┌──────────────────────────────────────────────────────┐
│                  Chainlit 2.11                        │
│  ┌─────────────────────────────────────────────────┐ │
│  │            Agent 单页对话 (ui/app.py)            │ │
│  │  5 工具: 航班查询 | 法规检索 | 流量分析 |       │ │
│  │          流量预测 | 异常检测                     │ │
│  └──────────────────────┬──────────────────────────┘ │
└─────────────────────────┼────────────────────────────┘
                          │ astream_events
                          ▼
┌──────────────────────────────────────────────────────┐
│              LangGraph ReAct Agent                    │
│              (core/agent_graph.py)                    │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐             │
│  │ flights  │ │regulations│ │  flow    │             │
│  │  tool    │ │  tool     │ │  tools   │             │
│  └────┬─────┘ └────┬──────┘ └────┬─────┘             │
└───────┼────────────┼─────────────┼───────────────────┘
        │            │             │
        ▼            ▼             ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│ Pandas   │ │ Chroma   │ │ PyTorch      │
│ CSV 查询 │ │ 向量检索  │ │ LSTM/AE 推理 │
└──────────┘ └────┬─────┘ └──────────────┘
                  │
                  ▼
┌──────────────────────┐     ┌────────────────┐
│ text2vec-base-chinese│     │  DashScope      │
│ (本地 Embedding)     │     │  通义千问       │
└──────────────────────┘     │  (外部 LLM API) │
                             └────────────────┘
```

## 可选: FastAPI 独立 API

```
FastAPI (app/server.py)
├── GET  /health         健康检查
├── POST /agent/stream   Agent 对话流
├── POST /dl/predict     LSTM 流量预测
└── POST /dl/anomaly     Autoencoder 异常检测
```

## 技术选型理由

| 选择 | 理由 |
|------|------|
| 通义千问 | 国有背景企业倾向国产模型；DashScope 兼容 OpenAI 协议 |
| LangChain | JD 明确要求，生态最完善 |
| Chroma | 轻量级、本地部署、与 LangChain 原生集成 |
| LangGraph | LangChain 官方 Agent 框架，ReAct 模式成熟 |
| Chainlit | 为 LLM 应用设计的现代对话界面，Agent 工具调用可视化 |
| FastAPI | 异步高性能，LangServe 原生支持 |

## 数据流

```
法规文档 (.md) → TextSplitter → Embedding → Chroma
                                                ↓
用户提问 → Agent 判断 → search_regulations 工具 → 相似度检索 → LLM → 回答

航班/流量 (.csv) → Pandas DataFrame → Agent Tool → 统计摘要 → LLM → 结构化回答

流量时序 (.csv) → 滑动窗口 → LSTM → 预测值
                              → Autoencoder → 异常判定 (z-score)
```
