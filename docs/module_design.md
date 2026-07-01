# 模块详细设计

> 最后更新: 2026-06-29

## 0. 项目定位

**用户**: 空管管制员、流量管理人员。决策窗口几十秒，信息分散在多个系统。

本系统是空管员的**第二双眼睛 + 决策外脑**，覆盖完整业务闭环：

```
检测异常 → 评估影响 → 推荐处置措施 → 生成通知清单 → 事后复盘
```

V1 实现了检测层（5 工具 Agent + DL 模型）。V2 补齐完整闭环，详见 `V2_plan.md`。

---

## 1. RAG 模块 (`core/rag_chains.py`)

### 1.1 架构

```
CCAR法规 .md 文件 (6篇)
    ↓ RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
文本块
    ↓ shibing624/text2vec-base-chinese (本地 Embedding, CPU 运行)
向量 → Chroma 持久化 (chroma_db/)
```

### 1.2 检索问答链

```
用户提问
    ↓
Embedding (同模型)
    ↓
Chroma.similarity_search(k=5, collection="atm_regulations")
    ↓
Prompt 模板:
  你是民航空管法规智能助手，精通 CCAR 规章。
  根据以下法规内容回答问题。如无相关信息请如实说明。
  {context}
  问题: {question}
    ↓
LLM (qwen-plus, DashScope) → 流式输出回答
```

### 1.3 关键设计决策

| 决策 | 内容 | 原因 |
|------|------|------|
| Embedding 换成本地模型 | `text2vec-base-chinese` | 避免 DashScope API 401 认证失败；Embedding 不需要 GPU；专为中文优化 |
| import 路径修复 | `langchain.chains` → `langchain_classic.chains` | LangChain 1.3+ 将 chains 移入 classic 命名空间 |
| sys.path 注入 | `sys.path.insert(0, parent_dir)` | Chainlit 启动时找不到 core 模块 |

### 1.4 向量库状态

- 构建命令: `python -c "from core.rag_chains import build_vectorstore; build_vectorstore()"`
- 持久化目录: `chroma_db/`
- Collection: `atm_regulations`
- 已验证问答: "成都双流机场跑道容量" → "48架次/小时" ✓

---

## 2. Agent 模块 (`core/agent_graph.py`)

### 2.1 LangGraph ReAct Agent

```
用户输入
    ↓
LLM 判断是否需要工具
    ↓ 是
调用工具 (tool_call)
    ↓
观察结果 (observation)
    ↓
继续判断 → 循环直到完成
    ↓ 否
最终回答
```

Agent 与 LLM 是双向循环关系——每次循环把 SYSTEM_PROMPT + 对话历史 + 工具返回结果发给 LLM，LLM 判断下一步行动。

### 2.2 工具清单

| 工具 | 功能 | 数据源 | 状态 |
|------|------|--------|------|
| `query_flights` | 按机场/日期/状态查航班 | flights.csv (10K 行) | 已验证 ✓ |
| `search_regulations` | RAG 检索法规 | Chroma 向量库 | 已验证 ✓ |
| `get_airport_flow` | 机场逐小时流量 | flow_history.csv (5机场) | 已验证 ✓ |
| `predict_flow` | LSTM 预测未来流量 | lstm_flow_model.pt | 已验证 ✓ |
| `detect_anomaly` | Autoencoder 异常检测 | autoencoder_model.pt | 已验证 ✓ |

### 2.3 已知问题 & 修复

| 问题 | 现象 | 修复 |
|------|------|------|
| LLM 日期幻觉 | Agent 自动传 `date="2023-10-05"`，数据里没有 | tool 描述加数据日期范围提示；查不到时返回可用日期 |
| ArrowDtype 类型不匹配 | `df["dep_airport"] == airport` 失败 | 统一 `.astype(str)` 处理 |
| empty 提示模糊 | 只说"未找到"不说原因 | 改为返回 `{最早日期}至{最晚日期}` |

---

## 3. 深度学习模块 (`core/dl_models.py` + `core/train.py`)

### 3.0 设计考量

**核心问题**：计划时刻表提前一周可知，气象预报和军航通告也有前置渠道。LSTM 用 24h 历史数据做纯时序外推，边际价值有限。Autoencoder 做的是模式识别而非预测——不管前置信息是什么，最终都会反映在实际流量里，偏离正常模式就报警。

### 3.1 LSTM 流量预测

```
输入: 过去24h逐时流量 [batch, 24, 1]
    ↓ LSTM (hidden=64, layers=2, dropout=0.2)
    ↓ Linear (64 → 24)
输出: 未来24h预测 [batch, 24]
```

训练数据: `flow_history.csv`，仅使用 ZUUU（成都双流）的数据

| 项目 | 值 |
|------|-----|
| 训练数据量 | 8,735 个滑动窗口 (ZUUU 365 天 × 24h) |
| 可用机场 | ZUUU, ZBAA, ZSPD, ZGGG, ZUUU2（5 个，仅 ZUUU 参与训练） |
| 最新 Loss | 4.81 (Epoch 50) |
| 模型路径 | `models/lstm_flow_model.pt` |

**定位**：当前为纯时序外推，demo 阶段展示用途。改进方向如下。

### 3.2 Autoencoder 异常检测

```
输入: 24h流量窗口 [batch, 24]
    ↓ Encoder (24→16→8→ReLU)
    ↓ Bottleneck (8)
    ↓ Decoder (8→16→24→Sigmoid)
输出: 重建窗口 [batch, 24]

异常判断: z-score = (reconstruction_error - mean) / std
         abs(z) > 2 → 异常（双向：偏高/偏低均可检测）
```

训练数据: `flow_history.csv`，仅使用 ZUUU（成都双流）的数据

| 项目 | 值 |
|------|-----|
| 训练数据量 | 8,735 个滑动窗口 |
| 误差均值 (mean) | 328.6 |
| 误差标准差 (std) | 82.9 |
| 存储格式 | `{"model": state_dict, "mean": float, "std": float}` |
| 模型路径 | `models/autoencoder_model.pt` |

**定位**：核心价值模块。不管有多少前置信息（计划表、气象、军航通告），所有因素最终都会体现在实际流量里。Autoencoder 不关心"为什么异常"，只检测"当前模式与历史正常模式是否偏离"——这是规则引擎和人工都难以实时做到的。

### 3.3 已知限制

**单机场训练**: 两个 DL 模型都只用 ZUUU 数据训练。对其他机场推理时，Autoencoder 会将不同流量模式误判为异常：

| 机场 | 重建误差 | z-score | 判定 |
|------|----------|---------|------|
| ZUUU | 340.6 | +0.1 | 正常 |
| ZBAA | 618.3 | +3.5 | 异常（实际是流量模式不同） |
| ZSPD | 526.5 | +2.4 | 异常 |
| ZGGG | 758.7 | +5.2 | 异常 |

改进方向：每机场独立建模，或归一化后混合训练。

**LSTM 与计划表功能重叠**：计划时刻表提前一周可知，LSTM 纯时序外推的独立价值有限。改进方向：将计划时刻表、气象预报、节假日标记作为额外特征拼入 LSTM 输入，模型从"纯时序外推"转变为"已知信息基础上的偏差修正"。

### 3.4 实验验证设计（待实施）

当前缺乏严格的模型验证。拟加入以下实验：

**合成异常回测**：在 `data_generator.py` 中插入已知异常事件（如第 180-185 天插入 6 小时"雷暴导致流量断崖"），用 Autoencoder 回测，验证：
- 异常事件发生时 z-score 是否标红（检出率）
- 正常时段误报率
- 异常持续时段与 z-score 超过阈值时段的吻合度

### 3.5 改进路线图

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| P1 | 合成异常回测验证 | 用可控合成数据验证 Autoencoder 检出能力 |
| P1 | 多机场独立模型 | 每机场训练独立 Autoencoder，消除跨机场泛化误差 |
| P2 | LSTM 输入加特征 | 计划表、气象、节假日标记作为额外特征 |
| P2 | 历史异常事件回测 | 拿真实运营中已知的不正常事件做验证 |

---

## 4. UI 模块 (`ui/app.py` + `public/custom.css`)

### 4.1 当前实现 (Chainlit 2.11)

单页 Agent 对话，启动时加载 Agent（含 LLM + 工具），输出欢迎消息。

```python
@cl.on_chat_start  → 创建 Agent, 发欢迎消息
@cl.on_message     → 处理消息，维护对话历史，传给 Agent
```

### 4.2 当前状态

Agent 模式，5 工具齐全，支持多轮对话记忆。对话历史为进程内列表（`history`），每次请求全量发送给 LLM——Demo 阶段够用，生产需加入滑动窗口或摘要压缩。

### 4.3 主题配置

```
public/custom.css        ← DeepSeek 风格主题（唯一的主题文件）
.chainlit/config.toml    ← custom_css = "/public/custom.css"
```

| 特性 | 实现 |
|------|------|
| 浅色主题 | `:root` — 白底灰字，蓝灰色调 |
| 深色主题 | `.dark` — 深灰底白字，自动切换 |
| 字体 | Inter (正文) + SF Mono/Cascadia Code (代码) |
| 输入框 | 圆角 8px，透明背景，focus 时亮色边框 |
| 发送按钮 | 圆形透明，hover 半透明底色 |

### 4.4 启动

```bash
chainlit run ui/app.py --port 8000
```

---

## 5. API 模块 (`app/server.py`)

### 5.1 端点清单

| 方法 | 路径 | 功能 | 状态 |
|------|------|------|------|
| GET | `/health` | 健康检查 | 已验证 ✓ |
| POST | `/agent/stream` | Agent 对话流 | 已验证 ✓ |
| POST | `/dl/predict` | LSTM 流量预测 | 已验证 ✓ |
| POST | `/dl/anomaly` | Autoencoder 异常检测 (z-score 双向) | 已验证 ✓ |

### 5.2 请求示例

```bash
# 预测 ZUUU 未来 12 小时流量
curl -X POST http://localhost:8000/dl/predict \
  -H "Content-Type: application/json" \
  -d '{"airport": "ZUUU", "hours": 12}'

# 检测 ZUUU 流量异常
curl -X POST http://localhost:8000/dl/anomaly \
  -H "Content-Type: application/json" \
  -d '{"airport": "ZUUU"}'
```

### 5.3 与 Chainlit 的关系

`server.py` 是可选独立模块。Chainlit 不依赖它启动，它存在的价值是：当其他系统（非对话场景）需要调用 Agent 或 DL 模型时，通过 REST API 访问，无需启动 Chainlit UI。

---

## 6. 数据模块 (core/config.py + core/data_generator.py)

### 6.1 数据概览

| 数据 | 大小 | 字段 |
|------|------|------|
| `flights.csv` | 10,000 行 | flight_no, airline, aircraft, dep/arr, scheduled, status, delay |
| `flow_history.csv` | 43,800 行 (5机场×365天×24h) | timestamp, airport, airport_name, flow |

### 6.2 数据用途

| 数据 | 用途 |
|------|------|
| `flights.csv` | `query_flights` 工具查询 |
| `flow_history.csv` | `get_airport_flow` 工具查询 + DL 模型训练 |

两个 DL 模型（LSTM、Autoencoder）**仅使用 ZUUU 的 8,760 小时数据训练**，其余 4 个机场（ZBAA、ZSPD、ZGGG、ZUUU2）仅用于工具查询。详见 §3.3。

### 6.3 CAAC 校准参数

所有关键数字来自 CAAC 2024 公报：

- 延误率 ~12.6% (生成验证: 正常率 87.4%)
- 延误分布: <30min 48.3%, 30-60min 29.3%, 1-2h 26.7%, >2h 5%
- 月度波动: 雷雨季(6-8月)正常率 75-79%, 成渝群偏高 2.4%
- 容量: ZUUU 48/h, ZUUU2 70/h, ZSPD 72/h
- 起降日内双峰分布: 早高峰 8-11点, 晚高峰 17-20点

### 6.4 bugfix

| 问题 | 修复 |
|------|------|
| `numpy.int64` 传不进 `timedelta(minutes=)` | `minute = int(np.random.choice(...))` |
| `DELAY_PROBS` 不归一化 | `probs = np.array(DELAY_PROBS) / sum(DELAY_PROBS)` |
