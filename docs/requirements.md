# 需求分析与功能清单

## JD 对照

### 职位要求第4条：大模型应用

| 子项 | 实现方式 |
|------|---------|
| RAG 检索增强生成 | LangChain + Chroma 向量库 + text-embedding-v3，构建空管法规知识库问答 |
| Agent 智能体 | LangGraph ReAct Agent，支持航班查询/法规检索/流量查询工具调用 |
| 向量数据库 | Chroma (轻量级，适合本地 Demo) |
| 文档切片与索引 | RecursiveCharacterTextSplitter，chunk_size=500, overlap=50 |

### 职位要求第5条：传统深度学习

| 子项 | 实现方式 |
|------|---------|
| 深度学习框架 | PyTorch 2.x |
| 时序预测 | LSTM 预测机场未来24小时流量 |
| 异常检测 | Autoencoder 检测流量异常模式 |

### 职位要求第1-3条：数据治理

Demo 通过合成数据生成器模拟端到端数据链路：
- 数据采集：generate_flights / generate_flow_history
- 数据清洗：缺失值处理、异常值过滤
- 数据标准：统一机场ICAO代码、航班号格式

## 功能清单

### P0 (必须实现)
- [ ] RAG 法规知识库问答
- [ ] Agent 多工具对话 (航班查询 + 法规检索)
- [ ] LSTM 流量预测
- [ ] Streamlit 可视化

### P1 (加分项)
- [ ] Autoencoder 异常检测
- [ ] LangServe API 部署
- [ ] 多机场对比分析
- [ ] 预测结果图表可视化
