# 空管智能助手 (ATM Intelligence Suite)

基于大模型 + RAG + Agent + 深度学习的空管业务智能化演示系统。

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM | 通义千问 (DashScope, OpenAI 兼容) |
| RAG 框架 | LangChain + Chroma + 本地 Embedding (text2vec-base-chinese) |
| Agent 框架 | LangGraph (ReAct, 工具调用) |
| 深度学习 | PyTorch (LSTM, Autoencoder) |
| 后端框架 | FastAPI + LangServe |
| 前端框架 | Chainlit 2.11 (LLM 原生对话 UI) |
| 环境管理 | Conda (atm_demo) |

## 环境准备

```bash
# 1. 激活 conda 环境
conda activate atm_demo

# 2. 配置 API Key
# 编辑 .env 文件，填入阿里云百炼 API Key:
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxx
```

## 快速启动

```bash
# 1. 生成模拟数据（CAAC 2024 公报校准）
python core/data_generator.py

# 2. 构建 RAG 知识库（向量化 + Chroma 持久化）
python -c "import sys; sys.path.insert(0,'.'); from core.rag_chains import build_vectorstore; build_vectorstore(); print('Done')"

# 3. 训练深度学习模型
python core/train.py

# 4. 启动 Chainlit 应用
chainlit run ui/app.py --port 8000

# 浏览器打开 http://localhost:8000

# 如果从非项目目录启动，需指定项目根:
#   set CHAINLIT_APP_ROOT=D:\ATM_Demo
#   再运行 chainlit run ui/app.py --port 8000
```

## 项目结构

```
ATM_Demo/
├── app/
│   └── server.py              # FastAPI + LangServe API
├── core/
│   ├── config.py               # 全局配置
│   ├── llm.py                  # LLM / Embedding 工厂
│   ├── data_generator.py       # 合成数据生成 (CAAC 校准)
│   ├── rag_chains.py           # RAG 检索问答链
│   ├── agent_graph.py          # LangGraph Agent (5 工具)
│   ├── dl_models.py            # LSTM + Autoencoder 模型定义
│   └── train.py                # 模型训练脚本
├── ui/
│   └── app.py                  # Chainlit 前端
├── public/
│   └── custom.css               # DeepSeek 风格主题 (Chainlit 静态文件)
├── .chainlit/
│   └── config.toml              # Chainlit 配置 (主题/CoT/custom_css)
├── data/
│   ├── regulations/            # 6 篇 CCAR 法规 .md
│   ├── flights.csv             # 航班时刻表 (10K 条, 3 个月)
│   └── flow_history.csv        # 机场流量时序 (5 机场, 365 天)
├── chroma_db/                  # Chroma 向量库
├── models/                     # .pt 模型文件
└── docs/                       # 项目文档
```

## 数据校准参数 (CAAC 2024)

| 参数 | 数值 | 来源 |
|------|------|------|
| 全国航班正常率 | 87.1% | 2024 民航航班运行效率报告 |
| 天气原因占延误 | 58.31% | 同上 |
| 雷雨季(6-8月)正常率 | ~77% | 同上 |
| ZUUU 双跑道容量 | 48 架次/小时 | CCAR-93 第十二条 |
| ZUUU2 三跑道容量 | 70 架次/小时 | 同上 |
| 成渝机场群正常率 | 89.52% | 四大机场群最高 |

## API 端点 (LangServe)

| 路径 | 说明 |
|------|------|
| `/docs` | Swagger UI |
| `/agent/playground` | Agent 对话交互 |
| `/dl/predict` | 流量预测 (POST) |
| `/dl/anomaly` | 异常检测 (POST) |
| `/health` | 健康检查 |

## 注意事项

- API Key 需在阿里云百炼平台申请 DashScope API Key
- Embedding 使用本地模型 `shibing624/text2vec-base-chinese`，首次运行需下载
- 航班数据日期范围: 2025-03-01 ~ 2025-05-29
- Agent 调用工具时 LLM 可能传入错误日期参数（日期幻觉），已在 tool 描述中给出提示
