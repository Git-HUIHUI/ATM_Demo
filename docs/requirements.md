# 需求分析与功能清单

## JD 对照

### 职位要求第4条：大模型应用

| 子项 | 实现方式 |
|------|---------|
| RAG 检索增强生成 | LangChain + Chroma 向量库 + text2vec-base-chinese 本地 Embedding，构建空管法规知识库问答 |
| Agent 智能体 | LangGraph ReAct Agent，支持航班查询/法规检索/流量查询/LSTM预测/异常检测 5 工具 |
| 向量数据库 | Chroma (轻量级，本地持久化) |
| 文档切片与索引 | RecursiveCharacterTextSplitter，chunk_size=500, overlap=50 |

### 职位要求第5条：传统深度学习

| 子项 | 实现方式 |
|------|---------|
| 深度学习框架 | PyTorch 2.x |
| 时序预测 | LSTM (hidden=64, layers=2)，输入24h历史流量，输出24h预测 |
| 异常检测 | Autoencoder (24→16→8→16→24)，z-score 双向异常判定 |

### 职位要求第1-3条：数据治理

Demo 通过合成数据生成器模拟端到端数据链路：
- 数据采集：generate_flights / generate_flow_history
- 数据清洗：缺失值处理、异常值过滤
- 数据标准：统一机场ICAO代码、航班号格式
- CAAC 2024 公报校准：延误率、月度波动、起降分布等真实统计参数

## 功能清单

### P0 (必须实现)
- [x] RAG 法规知识库问答
- [x] Agent 多工具对话 (5 工具)
- [x] LSTM 流量预测
- [x] Chainlit 对话界面

### P1 (加分项)
- [x] Autoencoder 异常检测
- [x] FastAPI 独立 API (app/server.py)
- [x] 多机场数据覆盖 (5 机场流量 + 11 机场航班)
- [ ] 预测结果图表可视化
- [x] DeepSeek 风格主题定制 (public/custom.css)
