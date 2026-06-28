"""
LangGraph Agent - 空管智能助手
支持工具：航班查询、法规检索、流量查询、DL预测、异常检测
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

from core.config import FLIGHTS_CSV, FLOW_HISTORY_CSV, LSTM_MODEL_PATH, AUTOENCODER_MODEL_PATH
from core.llm import get_llm
from core.rag_chains import rag_query
from core.dl_models import create_lstm_model, predict_lstm, Autoencoder

checkpointer = MemorySaver()

flights_df = pd.read_csv(FLIGHTS_CSV)
flights_df["scheduled_dep"] = flights_df["scheduled_dep"].astype(str)
flow_df = pd.read_csv(FLOW_HISTORY_CSV)
flow_df["timestamp"] = pd.to_datetime(flow_df["timestamp"])

# 加载 DL 模型
_lstm = create_lstm_model()
_lstm.load_state_dict(torch.load(str(LSTM_MODEL_PATH), weights_only=True))
_ae = Autoencoder()
_ae_ckpt = torch.load(str(AUTOENCODER_MODEL_PATH), weights_only=True)
_ae.load_state_dict(_ae_ckpt["model"])
_ae_mean = _ae_ckpt["mean"]
_ae_std = _ae_ckpt["std"]

SYSTEM_PROMPT = """你是民航空管智能助手，专业于航班查询、流量分析、法规检索和深度学习预测。

## 身份
你服务于中国民用航空局(CAAC)，对机场ICAO代码(如ZUUU=成都双流)必须准确无误。

## 重要ICAO代码对照表
| ICAO | 中文名称 | 城市 |
|------|----------|------|
| ZUUU | 成都双流国际机场 | 成都 |
| ZUTF | 成都天府国际机场 | 成都 |
| ZUCK | 重庆江北国际机场 | 重庆 |
| ZBAA | 北京首都国际机场 | 北京 |
| ZBAD | 北京大兴国际机场 | 北京 |
| ZSPD | 上海浦东国际机场 | 上海 |
| ZSSS | 上海虹桥国际机场 | 上海 |
| ZGGG | 广州白云国际机场 | 广州 |
| ZLXY | 西安咸阳国际机场 | 西安 |
| ZPPP | 昆明长水国际机场 | 昆明 |

**禁止将ZUUU称为乌鲁木齐**，那是ZWWW。

## 数据说明
- 航班数据和流量数据均来自本地CSV文件，非实时API
- 数据日期范围: 2025-03-01 至 2025-05-29
- 查询当天日期请用工具返回的数据日期范围，不要自己猜测日期
- 深度学习模型（LSTM预测、Autoencoder异常检测）基于ZUUU历史数据训练

## 可用工具
1. query_flights — 查航班
2. search_regulations — 查法规
3. get_airport_flow — 查历史流量
4. predict_flow — LSTM预测未来24小时流量
5. detect_anomaly — Autoencoder检测流量异常

## 回答规则
- 先调用工具获取数据，再基于工具返回的内容回答问题
- 如果用户问"预测流量"、"未来流量"，用 predict_flow
- 如果用户问"是否异常"、"有没有异常"，用 detect_anomaly
- 如果工具返回的数据包含统计摘要，直接引用，不要再重复计算
- 如果工具返回"未找到"，如实告知用户并建议调整查询条件
- 回答要简洁，中英文数字混排时注意空格
"""


@tool
def query_flights(airport: str = "", status: str = "", date: str = "") -> str:
    """查询航班信息。airport=机场ICAO代码(如ZUUU)，status=状态(正常/延误/取消)，date=日期(YYYY-MM-DD, 如果不传默认用数据中最晚日期)"""
    dates = sorted(flights_df["scheduled_dep"].str[:10].unique())
    latest_date = dates[-1]
    df = flights_df.copy()
    # 不传 date 时默认用最新日期
    actual_date = str(date).strip() if date else latest_date
    df = df[df["scheduled_dep"].str.startswith(actual_date)]
    if airport:
        ap = str(airport).strip().upper()
        df = df[(df["dep_airport"].astype(str) == ap) | (df["arr_airport"].astype(str) == ap)]
    if status:
        df = df[df["status"].astype(str) == str(status).strip()]
    if df.empty:
        return f"未找到 {actual_date} 的航班记录。数据库日期范围: {dates[0]} 至 {dates[-1]}。"

    total = len(df)
    status_counts = df["status"].value_counts().to_dict()
    airline_counts = df["airline"].value_counts().head(5).to_dict()
    top_dest = df["arr_airport_name"].value_counts().head(5).to_dict()
    delayed = df[df["status"] == "延误"]
    avg_delay = f"{delayed['delay_minutes'].mean():.0f}分钟" if len(delayed) > 0 else "无延误"

    summary = (
        f"## 查询结果 ({actual_date})\n"
        f"- 航班总数: {total} 班\n"
        f"- 状态分布: " + " | ".join(f"{k}: {v}班" for k, v in status_counts.items()) + "\n"
        f"- 平均延误: {avg_delay}\n"
        f"- 主要航司(Top5): " + " | ".join(f"{k}: {v}班" for k, v in airline_counts.items()) + "\n"
        f"- 热门目的地(Top5): " + " | ".join(f"{k}: {v}班" for k, v in top_dest.items()) + "\n"
    )
    # 不输出完整表格到 Agent，只给摘要 + 少量示例
    detail_cols = ["flight_no", "airline", "dep_airport_name", "arr_airport_name",
                   "scheduled_dep", "status", "delay_minutes"]
    detail = df.head(8)[detail_cols].to_string(index=False)
    return summary + "\n## 示例航班（前8班）\n```\n{}\n```".format(detail)


@tool
def search_regulations(query: str) -> str:
    """检索空管法规知识库。参数：query=要查询的法规问题（如'尾流间隔标准'、'A类空域定义'）"""
    result = rag_query(query)
    sources = "、".join(result["sources"])
    return f"{result['answer']}\n\n参考法规: {sources}"


@tool
def get_airport_flow(airport: str = "ZUUU", hours: int = 24) -> str:
    """查询机场逐小时流量数据。参数：airport=机场ICAO代码(如ZUUU)，hours=回溯小时数(默认24)"""
    ap_flow = flow_df[flow_df["airport"] == airport.upper()].tail(hours)
    if ap_flow.empty:
        avail = flow_df["airport"].unique().tolist()
        return f"未找到机场 **{airport}** 的流量数据。可用机场: {', '.join(sorted(avail))}"
    name = ap_flow["airport_name"].iloc[0]
    values = ap_flow["flow"].tolist()
    timestamps = ap_flow["timestamp"].dt.strftime("%m-%d %H:00").tolist()
    avg = sum(values) / len(values)
    peak = max(values)
    peak_hour = timestamps[values.index(peak)]
    trough = min(values)
    trough_hour = timestamps[values.index(trough)]

    # 分时段统计
    daytime_vals = [v for i, v in enumerate(values) if 6 <= int(timestamps[i].split()[1][:2]) <= 22]
    daytime_avg = sum(daytime_vals) / len(daytime_vals) if daytime_vals else 0

    hourly_table = "\n".join(f"| {t} | {v} |" for t, v in zip(timestamps, values))
    return (
        f"## {name}（{airport.upper()}）最近 {hours} 小时流量\n\n"
        f"| 指标 | 数值 |\n|------|------|\n"
        f"| 平均流量 | **{avg:.1f}** 架次/小时 |\n"
        f"| 峰值 | **{peak}** 架次/小时 (_{peak_hour}_) |\n"
        f"| 低谷 | **{trough}** 架次/小时 (_{trough_hour}_) |\n"
        f"| 日间均值(6-22时) | **{daytime_avg:.1f}** 架次/小时 |\n\n"
        f"## 逐时数据\n\n"
        f"| 时段 | 流量(架次) |\n|------|------|\n{hourly_table}"
    )


@tool
def predict_flow(airport: str = "ZUUU", hours: int = 24) -> str:
    """预测机场未来流量（LSTM模型）。airport=机场ICAO代码(如ZUUU)，hours=预测时长(默认24小时)"""
    ap_data = flow_df[flow_df["airport"] == airport.upper()]
    if ap_data.empty:
        avail = flow_df["airport"].unique().tolist()
        return f"未找到机场 **{airport}** 的数据。可用机场: {', '.join(sorted(avail))}"
    recent = ap_data["flow"].tail(24).tolist()
    if len(recent) < 24:
        return f"机场 **{airport}** 数据不足24小时，无法预测"
    pred = predict_lstm(_lstm, recent)[:hours]
    name = ap_data["airport_name"].iloc[0]
    peak = max(pred)
    trough = min(pred)
    avg = sum(pred) / len(pred)
    peak_h = pred.index(peak)
    trough_h = pred.index(trough)
    table = "\n".join(f"| +{i+1}h | {v:.1f} |" for i, v in enumerate(pred))
    return (
        f"## {name}（{airport.upper()}）未来 {hours} 小时流量预测\n\n"
        f"- 预测均值: **{avg:.1f}** 架次/小时\n"
        f"- 预测峰值: **{peak:.1f}** 架次/小时 (+{peak_h+1}h)\n"
        f"- 预测低谷: **{trough:.1f}** 架次/小时 (+{trough_h+1}h)\n\n"
        f"| 时段 | 预测流量 |\n|------|------|\n{table}"
    )


@tool
def detect_anomaly(airport: str = "ZUUU") -> str:
    """检测机场流量是否异常（Autoencoder模型）。airport=机场ICAO代码"""
    from core.dl_models import detect_anomaly as _detect

    ap_data = flow_df[flow_df["airport"] == airport.upper()]
    if ap_data.empty:
        avail = flow_df["airport"].unique().tolist()
        return f"未找到机场 **{airport}** 的数据。可用机场: {', '.join(sorted(avail))}"
    window = ap_data["flow"].tail(24).tolist()
    if len(window) < 24:
        return f"机场 **{airport}** 数据不足24小时，无法检测"
    name = ap_data["airport_name"].iloc[0]
    is_anomaly, error, z = _detect(_ae, window, _ae_mean, _ae_std)
    status = "异常" if is_anomaly else "正常"
    direction = "偏高" if z > 0 else "偏低"
    return (
        f"## {name}（{airport.upper()}）异常检测结果\n\n"
        f"| 指标 | 数值 |\n|------|------|\n"
        f"| 重建误差 | **{error:.1f}** |\n"
        f"| z-score | **{z:+.1f}** ({direction}) |\n"
        f"| 状态 | **{status}** |"
    )


def create_agent():
    llm = get_llm(temperature=0.1, streaming=True)
    tools = [query_flights, search_regulations, get_airport_flow, predict_flow, detect_anomaly]
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT, checkpointer=checkpointer)
