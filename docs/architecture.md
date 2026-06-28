# 系统架构

## 架构图

```
┌──────────────────────────────────────────────────┐
│                  Chainlit UI                       │
│  ┌─────────┐ ┌─────────┐ ┌────────────────────┐  │
│  │ RAG 问答 │ │Agent 对话│ │ 流量预测看板        │  │
│  └────┬─────┘ └────┬─────┘ └────────┬───────────┘  │
└───────┼─────────────┼───────────────┼──────────────┘
        │             │               │
        ▼             ▼               ▼
┌──────────────────────────────────────────────────┐
│              LangServe API (FastAPI)               │
│  /rag/playground  /agent/playground  /dl/predict  │
└──────────────────────────────────────────────────┘
        │             │               │
        ▼             ▼               ▼
┌───────────┐ ┌─────────────┐ ┌──────────────┐
│ RAG Chain │ │Agent Graph   │ │  DL Models    │
│           │ │              │ │               │
│ Retrieval │ │ Tools:       │ │ LSTM          │
│ QA Chain  │ │ - flights    │ │ Autoencoder   │
│           │ │ - regulations│ │               │
└─────┬─────┘ └──────┬───────┘ └──────┬────────┘
      │              │               │
      ▼              ▼               ▼
┌──────────┐ ┌─────────────┐ ┌──────────────┐
│ Chroma   │ │ Pandas       │ │ PyTorch       │
│ (向量库) │ │ (航班数据)   │ │ (.pt 模型)    │
└──────────┘ └─────────────┘ └──────────────┘
      │
      ▼
┌──────────┐
│DashScope  │  ← 外部 LLM API
│通义千问   │
└──────────┘
```

## 技术选型理由

| 选择 | 理由 |
|------|------|
| 通义千问 | 国有背景企业倾向国产模型；DashScope 兼容 OpenAI 协议 |
| LangChain | JD 明确要求，生态最完善 |
| Chroma | 轻量级、本地部署、与 LangChain 原生集成 |
| LangGraph | LangChain 官方 Agent 框架，ReAct 模式成熟 |
| Chainlit | 为 LLM 应用设计的现代对话界面，Agent 步骤可视化 |
| FastAPI | 异步高性能，LangServe 原生支持 |

## 数据流

```
法规文档 (.md) → TextSplitter → Embedding → Chroma
                                                ↓
用户提问 → Embedding → 相似度检索 → Prompt 模板 → LLM → 回答

航班数据 (.csv) → Pandas DataFrame → Agent Tool → LLM → 结构化回答

流量时序 (.csv) → 滑动窗口 → LSTM → 预测值 → 可视化
```
