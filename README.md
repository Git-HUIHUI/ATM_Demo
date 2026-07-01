# 空管智能助手 (ATM Intelligence Suite)

基于大模型 + RAG + Agent + 深度学习的空管业务智能辅助系统。

## 项目定位

**用户**: 空管管制员、流量管理人员。决策窗口几十秒，信息分散在多个系统。

**本系统做什么**: 空管员的"第二双眼睛 + 决策外脑"。

```
检测异常 → 评估影响 → 推荐处置措施 → 生成通知清单 → 事后复盘
```

**减少什么损失**: 延误传导（提前反应，每早一分钟少一环波及）→ 容量浪费（流量恢复立即通知放行）→ 决策合规风险（每条措施附法规依据）→ 通知遗漏（自动生成清单，管制员只做确认）

**怎么听到问题**: 四条通道——① 数据自动监控（Autoencoder 持续推理，标红自触发）② 外部系统对接（雷达、气象、军航、飞行计划汇总）③ 管制员自然语言追问 ④ 机载子系统（飞机突发故障自动打包上下文发给地面，详见"空地协同"）

**空地协同**: 机载轻量子系统实时监测飞机状态——传感器数据异常（引擎/舵面/舱压）自动触发、气象雷达检测雷暴、机组隐蔽触发安全告警、或触发医疗紧急模式。突发时自动打包（位置+高度+故障类型+剩余油量+机上人数）通过卫星数据链发给地面。地面端 3 秒内激活评估引擎：硬件故障→算备降方案+通知消防维修；雷暴→算绕飞航路+协调空域；医疗急救→筛选含医疗条件的机场；安全威胁→静默模式，通知公安/军方。机组只做最终确认，全程不需要打一个电话。详见 `docs/V2_plan.md`。

当前 V1 实现了检测层（5 工具 Agent + DL 异常检测）。V2 补齐完整闭环，详见 `docs/V2_plan.md`。

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM | 通义千问 (DashScope, OpenAI 兼容) |
| RAG 框架 | LangChain + Chroma + 本地 Embedding (text2vec-base-chinese) |
| Agent 框架 | LangGraph (ReAct, 工具调用) |
| 深度学习 | PyTorch (LSTM, Autoencoder) |
| 后端 | Chainlit + LangGraph (可选: FastAPI API 模块) |
| 前端 | Chainlit 2.11 (shadcn/ui + Tailwind CSS) |
| 环境管理 | Conda (atm_demo) |

## 环境准备

```bash
# 1. 激活 conda 环境
conda activate atm_demo

# 2. 配置 API Key（编辑 .env 文件）
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxx
```

## 快速启动

> 数据生成、RAG 构建、模型训练只需执行一次，之后每次只需第 4 步启动。

```bash
# 1. 生成模拟数据（CAAC 2024 公报校准）
python core/data_generator.py

# 2. 构建 RAG 知识库（向量化 + Chroma 持久化）
python -c "from core.rag_chains import build_vectorstore; build_vectorstore()"

# 3. 训练深度学习模型
python core/train.py

# 4. 启动 Chainlit 应用
chainlit run ui/app.py --port 8000

# 浏览器打开 http://localhost:8000
```

## 项目结构

```
ATM_Demo/
├── app/
│   └── server.py              # FastAPI 独立 API（可选）
├── core/
│   ├── config.py               # 全局配置
│   ├── llm.py                  # LLM / Embedding 工厂
│   ├── data_generator.py       # 合成数据生成 (CAAC 校准)
│   ├── rag_chains.py           # RAG 检索问答链
│   ├── agent_graph.py          # LangGraph Agent
│   ├── dl_models.py            # LSTM + Autoencoder 模型定义
│   └── train.py                # 模型训练脚本
├── ui/
│   └── app.py                  # Chainlit 前端
├── public/
│   └── custom.css               # DeepSeek 风格主题
├── .chainlit/
│   └── config.toml              # Chainlit 配置
├── data/
│   ├── regulations/            # 6 篇 CCAR 法规 .md
│   ├── flights.csv             # 航班时刻表 (10K 条, 3 个月)
│   └── flow_history.csv        # 机场流量时序 (5 机场, 365 天)
├── chroma_db/                  # Chroma 向量库（本地持久化）
├── models/                     # .pt 模型文件
└── docs/                       # 项目文档
```

## V1 功能清单（已实现）

| 功能 | 说明 | 状态 |
|------|------|------|
| 自然语言航班查询 | 按机场/日期/状态查航班，返回统计摘要 | ✓ |
| 空管法规检索 | RAG 检索 CCAR 法规，附条文来源 | ✓ |
| 机场流量分析 | 逐小时流量，均值/峰值/低谷 | ✓ |
| LSTM 流量预测 | 过去 24h → 未来 24h 预测 | ✓ |
| Autoencoder 异常检测 | z-score 双向判定 | ✓ |
| FastAPI 独立 API | Agent + DL 模型 REST 接口 | ✓ |

## V2 规划（待实施）

见 `docs/V2_plan.md`。核心：补齐"检测 → 评估 → 建议 → 通知 → 复盘"业务闭环。

## 数据校准参数 (CAAC 2024)

| 参数 | 数值 | 来源 |
|------|------|------|
| 全国航班正常率 | 87.1% | 2024 民航航班运行效率报告 |
| 天气原因占延误 | 58.31% | 同上 |
| 雷雨季(6-8月)正常率 | ~77% | 同上 |
| ZUUU 双跑道容量 | 48 架次/小时 | CCAR-93 第十二条 |
| ZUUU2 三跑道容量 | 70 架次/小时 | 同上 |
| 成渝机场群正常率 | 89.52% | 四大机场群最高 |

## API 端点

| 路径 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/agent/stream` | POST | Agent 对话流 |
| `/dl/predict` | POST | LSTM 流量预测 |
| `/dl/anomaly` | POST | Autoencoder 异常检测 |
| `/docs` | GET | Swagger UI |

## 已知局限与改进方向

| 局限 | 改进方向 |
|------|---------|
| DL 模型仅用 ZUUU 训练 | 多机场独立建模 |
| LSTM 与计划表功能重叠 | 纳入计划表/气象特征，做偏差修正 |
| Agent 角色仅为问答 | V2 升级为"评估+建议+通知"决策外脑 |
| 未覆盖完整业务闭环 | 补齐影响评估、处置建议、通知、复盘 |
| 对话记忆为进程内列表 | 滑动窗口 + 摘要压缩 |

## 注意事项

- API Key 需在阿里云百炼平台申请 DashScope API Key
- Embedding 使用本地模型 `shibing624/text2vec-base-chinese`，首次运行需下载
- 航班数据日期范围: 2025-03-01 ~ 2025-05-29
