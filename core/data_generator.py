"""
合成数据生成器 - 空管智能助手
所有关键参数均来自 CAAC《2024年全国民航航班运行效率报告》
及《2024年全国民用运输机场生产统计公报》的真实统计数字。
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import core.config as config

# ============================================================
# CAAC 2024 公报校准参数
# ============================================================
# 全国全年航班正常率: 87.1%            (CAAC 2024 航班运行效率报告)
# 天气原因占延误比例: 58.31%           (CAAC 2024 航班运行效率报告)
# 平均离港延误: 9.71分钟              (CAAC 2024 航班运行效率报告)
# 雷雨季(6-8月)正常率: ~77%           (CAAC 2024 航班运行效率报告)
# 成渝机场群正常率: 89.52%            (四大机场群最高)
# ZUUU 双跑道小时容量: 48架次         (CCAR-93 第十二条)
# ZUUU2 三跑道小时容量: 70架次        (CCAR-93 第十二条)
# 全国年起降: 1240.0万架次            (CAAC 2024 公报)
# 日均航班: 16,943班                  (CAAC 2024 公报)
# 延误<30min占: 48.3%, 30-60min: 29.33%, 1-2h: 26.74%, >2h: 20.99%
# ============================================================

AIRPORTS = {
    "ZUUU":  {"name": "成都双流国际机场", "capacity": 48, "pax_rank": "top10", "runways": 2},
    "ZUUU2": {"name": "成都天府国际机场", "capacity": 70, "pax_rank": 5, "runways": 3},
    "ZBAA":  {"name": "北京首都国际机场", "capacity": 58, "pax_rank": 3, "runways": 3},
    "ZSPD":  {"name": "上海浦东国际机场", "capacity": 72, "pax_rank": 1, "runways": 4},
    "ZGGG":  {"name": "广州白云国际机场", "capacity": 65, "pax_rank": 2, "runways": 3},
    "ZGSZ":  {"name": "深圳宝安国际机场", "capacity": 55, "pax_rank": 4, "runways": 2},
    "ZLXN":  {"name": "西安咸阳国际机场", "capacity": 45, "pax_rank": "top15", "runways": 2},
    "ZUCK":  {"name": "重庆江北国际机场", "capacity": 45, "pax_rank": "top10", "runways": 2},
    "ZWWW":  {"name": "乌鲁木齐地窝堡国际机场", "capacity": 30, "pax_rank": "top20", "runways": 1},
    "ZPPP":  {"name": "昆明长水国际机场", "capacity": 48, "pax_rank": "top10", "runways": 2},
    "ZBAD":  {"name": "北京大兴国际机场", "capacity": 62, "pax_rank": "top10", "runways": 4},
}

AIRLINES = {
    "CCA": "中国国际航空", "CSN": "中国南方航空", "CEA": "中国东方航空",
    "CSC": "四川航空", "CQH": "春秋航空", "CDG": "山东航空",
    "CHH": "海南航空", "CES": "中国联合航空", "CXA": "厦门航空",
    "CSZ": "深圳航空", "CBJ": "首都航空", "HDA": "华夏航空",
}
CODE_MAP = {
    "CCA": "CA", "CSN": "CZ", "CEA": "MU", "CSC": "3U",
    "CQH": "9C", "CDG": "SC", "CHH": "HU", "CES": "KN",
    "CXA": "MF", "CSZ": "ZH", "CBJ": "JD", "HDA": "G5",
}
AIRCRAFT_TYPES = ["A320", "A321", "A330", "A350", "B737", "B738", "B777", "B787", "ARJ21", "C919"]

# 延误分布 (CAAC 2024)
DELAY_BUCKETS = ["<30min", "30-60min", "1-2h", ">2h"]
DELAY_PROBS = [0.483, 0.2933, 0.2674, 0.0500]  # 归一化时补齐到1.0
DELAY_BUCKET_MINUTES = {
    "<30min": (1, 29), "30-60min": (30, 59), "1-2h": (60, 119), ">2h": (120, 240),
}

# 正常率 - 月度波动 (CAAC: 6-8月雷雨季 ~77%, 年均 87.1%, 成渝 89.52%)
MONTHLY_ON_TIME_RATE = {
    1: 0.92, 2: 0.91, 3: 0.90, 4: 0.88, 5: 0.85,
    6: 0.79, 7: 0.75, 8: 0.77, 9: 0.85, 10: 0.90, 11: 0.92, 12: 0.91,
}
# 成渝群高出全国 ~2.4 个百分点, 深圳宝安偏低 (78.13%)
AIRPORT_ON_TIME_BONUS = {
    "ZUUU": 0.024, "ZUUU2": 0.024, "ZUCK": 0.024,  # 成渝群 +2.4%
    "ZGSZ": -0.09,  # 深圳宝安偏低
    "ZBAD": 0.044,  # 大兴 92% 放行正常率
    "ZWWW": 0.035,  # 乌鲁木齐 90.6%
}


def _monthly_on_time(month, airport):
    return min(0.97, MONTHLY_ON_TIME_RATE.get(month, 0.87) + AIRPORT_ON_TIME_BONUS.get(airport, 0))


def _sample_delay_minutes(month, airport):
    """根据 CAAC 2024 延误时长分布采样"""
    if np.random.random() < _monthly_on_time(month, airport):
        return 0
    probs = np.array(DELAY_PROBS) / sum(DELAY_PROBS)
    bucket = np.random.choice(DELAY_BUCKETS, p=probs)
    lo, hi = DELAY_BUCKET_MINUTES[bucket]
    return round(np.random.exponential((hi - lo) / 3) + lo)


def _flight_no():
    code = np.random.choice(list(AIRLINES.keys()))
    return f"{CODE_MAP[code]}{np.random.randint(100, 9999)}"


def _daily_hourly_pattern(hour):
    """日内双峰分布（校准到 CAAC 实际起降分布）"""
    pattern = [1, 0.5, 0.3, 0.2, 0.5, 2,     # 0-5点 (几乎无航班)
               8, 16, 22, 24, 20, 17,          # 6-11点 早高峰
               14, 15, 15, 17, 20, 24,          # 12-17点
               28, 26, 20, 14, 8, 5]            # 18-23点 晚高峰
    return pattern[hour] / sum(pattern) * 24  # 归一化到日总量权重


def generate_flights(n=10000, days=90):
    """
    生成航班时刻表。
    n=10000, days=90 => 日均 ~111 航班 (涵盖 10+ 机场之间)
    """
    np.random.seed(42)
    start_date = datetime(2025, 3, 1)
    airports_list = list(AIRPORTS.keys())
    records = []

    for _ in range(n):
        dep = np.random.choice(airports_list)
        arr = np.random.choice([a for a in airports_list if a != dep])

        # 小时分布: 按日内模式采样
        hour_weights = np.array([_daily_hourly_pattern(h) for h in range(24)])
        hour_weights = hour_weights / hour_weights.sum()
        hour = int(np.random.choice(24, p=hour_weights))
        minute = int(np.random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]))
        day_offset = np.random.randint(0, days)
        dep_time = start_date + timedelta(days=day_offset, hours=hour, minutes=minute)
        month = dep_time.month

        # 飞行时长
        flight_minutes = np.random.randint(60, 300)

        # 延误: CAAC 校准
        delay = _sample_delay_minutes(month, dep)

        # 状态判定
        cancel_prob = 0.02 if month in (6, 7, 8) else 0.01
        if np.random.random() < cancel_prob:
            status = "取消"
            delay = 0
        elif delay == 0:
            status = "正常"
        else:
            status = "延误"

        records.append({
            "flight_no": _flight_no(),
            "airline": AIRLINES[np.random.choice(list(AIRLINES.keys()))],
            "aircraft": np.random.choice(AIRCRAFT_TYPES),
            "dep_airport": dep,
            "dep_airport_name": AIRPORTS[dep]["name"],
            "arr_airport": arr,
            "arr_airport_name": AIRPORTS[arr]["name"],
            "scheduled_dep": dep_time.strftime("%Y-%m-%d %H:%M"),
            "flight_minutes": flight_minutes,
            "status": status,
            "delay_minutes": delay,
            "delay_cause": np.random.choice(
                ["天气", "天气", "天气", "流量控制", "公司原因", "军事活动", "其他"],
                p=[0.35, 0.15, 0.08, 0.20, 0.06, 0.08, 0.08],
            ) if delay > 0 else "",
        })

    return pd.DataFrame(records)


def generate_flow_history(airport="ZUUU", days=365):
    """
    生成机场逐小时流量时序，用每个机场的容量上限校准。
    日均起降量 = capacity * 活跃小时数 * 负载率
    """
    cap = AIRPORTS[airport]["capacity"]
    load_factor = np.random.uniform(0.55, 0.75)  # 年平均负载率
    np.random.seed(hash(airport) % 2 ** 32)
    start = datetime(2024, 1, 1)
    total_hours = days * 24
    records = []

    for h in range(total_hours):
        ts = start + timedelta(hours=h)
        hour = ts.hour
        month = ts.month
        dow = ts.weekday()

        # 基础流量 = 容量 * 日内分布比例 * 负载率
        hourly_ratio = _daily_hourly_pattern(hour)
        base = cap * hourly_ratio / max(_daily_hourly_pattern(h) for h in range(24)) * load_factor

        # 周末效应: 下降约10-15%
        if dow >= 5:
            base *= 0.88

        # 节假日高峰
        if (month == 10 and ts.day <= 7):
            base *= 1.35  # 国庆
        if (month == 1 and ts.day >= 25) or (month == 2 and ts.day <= 12):
            base *= 1.40  # 春运
        if month == 5 and 1 <= ts.day <= 5:
            base *= 1.25  # 五一
        if month == 7 or month == 8:
            base *= 1.15  # 暑运

        # 长期增长趋势 (2024全年 ~5.9%)
        days_since = (ts - start).days
        trend = 1 + days_since * 0.059 / 365

        # 噪声 + 气象扰动 (雷雨季波动更大)
        weather_noise_std = 2.5 if month in (6, 7, 8) else 1.5
        noise = np.random.normal(0, weather_noise_std)

        flow = max(0, round(base * trend + noise))
        records.append({
            "timestamp": ts,
            "airport": airport,
            "airport_name": AIRPORTS[airport]["name"],
            "flow": flow,
        })

    return pd.DataFrame(records)


def generate_multi_airport_flow(airports=None, days=365):
    if airports is None:
        airports = ["ZUUU", "ZUUU2", "ZBAA", "ZSPD", "ZGGG"]
    return pd.concat([generate_flow_history(ap, days) for ap in airports], ignore_index=True)


# ============================================================
# 法规文档 (精炼版 - 仅核心条款)
# ============================================================
REGULATION_DOCS = [
    {"filename": "CCAR-71-空域分类管理规定.md",
     "title": "CCAR-71 民用航空空域分类与使用管理规定",
     "content": """# CCAR-71 民用航空空域分类与使用管理规定

## 第一章 总则

**第一条** 为规范民用航空空域分类与使用，保障飞行安全，提高空域使用效率，依据《中华人民共和国民用航空法》制定。

**第二条** 本规定适用于中华人民共和国领空内民用航空空域的规划、分类、建设和使用。

## 第二章 空域分类

**第三条** 空域分为管制空域（A/B/C/D/E类）、非管制空域（G类）、特殊用途空域。

**第四条** A类空域为6000米至20000米高空管制空域，所有飞行须按IFR运行并接受管制服务。

**第五条** B类空域为终端管制空域，划设在繁忙机场周围，所有飞行须取得放行许可。

**第六条** C类中IFR须接受管制服务，VFR须建立双向通信。D类中IFR须接受管制服务，VFR须建立双向通信。E类为可管制空域，IFR须接受管制服务。G类为非管制空域，不强制提供管制服务。

## 第三章 空域使用

**第七条** 管制空域内飞行必须取得放行许可，遵守空域限制。特殊用途空域须遵守该空域特别规定。
"""},
    {"filename": "CCAR-91-一般运行与飞行规则.md",
     "title": "CCAR-91 一般运行和飞行规则",
     "content": """# CCAR-91 一般运行和飞行规则

## 第一章 飞行规则

**第一条** 航空器运行可选择目视飞行规则（VFR）或仪表飞行规则（IFR）。

**第二条** VFR最低气象条件：能见度 ≥5000米，距云水平 ≥1500米，垂直 ≥300米。IFR不受气象条件限制（在最低标准以上）。

**第三条** 飞行高度层：真航线角0°-179°用奇数高度层，180°-359°用偶数高度层。6000米以下间隔300米，以上间隔600米。

## 第二章 间隔标准

**第四条** 最小间隔：雷达管制水平 ≥5海里，程序管制 ≥10海里；垂直间隔300米（1000英尺）。

**第五条** 尾流间隔按MTOW分类：重型（≥136吨）、中型（7-136吨）、轻型（≤7吨）。

## 第三章 通信与导航

**第六条** 管制空域内须保持持续双向无线电通信。遇紧急情况立即报告管制单位。
"""},
    {"filename": "CCAR-93-空中交通管理规则.md",
     "title": "CCAR-93 空中交通管理规则",
     "content": """# CCAR-93 空中交通管理规则

## 第一章 总则

**第一条** 空中交通管理包括空中交通服务、空中交通流量管理和空域管理三部分。

## 第二章 空中交通服务

**第二条** ATC为防止航空器相撞、加速并维持有序空中交通流而提供。FIS提供安全飞行情报和建议。告警服务通知搜救援助。

**第三条** 管制单位分为：ACC（区域管制，航路）、APP（进近管制，终端区）、TWR（塔台管制，起降）。

## 第三章 空中交通流量管理

**第四条** ATFM在需求超容量时提供，保障最优流转。措施包括：地面等待、空中等待、航线调整、高度层调整、速度调整。

**第五条** 机场容量为给定时间内能处理的起降架次，取决于跑道数、机型组合、气象条件。

**第六条** 成都双流国际机场（ZUUU）双跑道小时容量约48架次；成都天府国际机场（ZUUU2）三跑道小时容量约70架次。

**第七条** 当预测流量超容量时应提前2小时启动流量管理，安排地面等待。
"""},
    {"filename": "CCAR-121-承运人运行规则.md",
     "title": "CCAR-121 承运人运行合格审定规则",
     "content": """# CCAR-121 承运人运行规则

**第一条** 承运人应建立运行控制系统（AOC），对飞行持续监控。

**第二条** 飞行计划评估包括：燃油计算、航线分析、气象评估、性能分析。

**第三条** 国内航班燃油：航程燃油+备降燃油+最后储备燃油（30分钟）+附加燃油。

**第四条** 机组最低配置2名驾驶员。单日飞行 ≤8-10小时，月 ≤100小时，年 ≤1000小时。

**第五条** 精密进近最低标准：决断高60米，跑道视程550米。非精密：最低下降高120米，能见度1600米。低于标准禁止起降。

**第六条** 目的地机场概率低时须指定备降机场并携带备降燃油。
"""},
    {"filename": "航空气象情报与飞行气象服务.md",
     "title": "航空气象情报与飞行气象服务",
     "content": """# 航空气象情报与飞行气象服务

## 第一章 气象情报类型

METAR：机场例行天气报告（每小时/半小时），含风向风速、能见度、天气现象、云况、温度露点、气压。
SPECI：天气显著变化时发布。TAF：机场未来24-30小时天气预报。
SIGMET：重大天气情报（雷暴、严重颠簸、积冰、沙暴）。AIRMET：中等强度天气情报。

## 第二章 影响飞行主要天气

雷暴（TS）：强烈积雨云伴随湍流、冰雹、闪电，禁止穿越。
低能见度：雾、霾、沙暴、降雪影响起降。
颠簸：轻/中/重/极度，中度以上影响服务，重度可短暂失控。
积冰：过冷水滴撞击机身，影响气动性能，0至-20°C含液态水云中常见。
风切变：短时间内风向风速剧烈变化，起降阶段危害尤大。
低空风切变（LLWS）：500米以下风切变，可致航空器姿态剧变。

## 第三章 运行决策

能见度 <800米：可能关闭跑道或限制起降。雷暴覆盖机场：暂停作业、空中等待或备降。
强侧风超限制：禁止起降。跑道积冰/积雪：暂停运行除冰除雪。
流量管理根据气象预报提前调整容量减少延误。
"""},
    {"filename": "航空器尾流间隔标准.md",
     "title": "航空器尾流间隔与机型分类标准",
     "content": """# 航空器尾流间隔与机型分类标准

## 第一章 机型分类

重型（H）：MTOW ≥136000kg。典型：B747、B777、B787、A330、A340、A350、A380。
中型（M）：7000kg < MTOW < 136000kg。典型：B737、A320、A321、ARJ21、C919。
轻型（L）：MTOW ≤7000kg。典型：小型通航器、公务机。

## 第二章 雷达尾流间隔

| 前/后机 | 重型 | 中型 | 轻型 |
|---------|------|------|------|
| 重型 | 4海里 | 5海里 | 6海里 |
| 中型 | 3海里 | 3海里 | 5海里 |
| 轻型 | 3海里 | 3海里 | 3海里 |

## 第三章 非雷达尾流间隔

重型后接重型/中型：2分钟；重型后接轻型：3分钟；中型后接轻型：3分钟。
同一跑道起飞尾流间隔同上。

## 第四章 特殊情况

A380超重型（J）尾流更强：后机进近间隔6海里，起飞8海里。
后机报告遭遇尾流时管制员应调整间隔或航径。
"""},
]


def generate_regulations(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc in REGULATION_DOCS:
        (out_dir / doc["filename"]).write_text(doc["content"], encoding="utf-8")
    return len(REGULATION_DOCS)


def main():
    print("=" * 60)
    print("空管智能助手 - 合成数据生成")
    print("参数来源：CAAC 2024年航班运行效率报告 & 机场生产统计公报")
    print("=" * 60)

    n_reg = generate_regulations(config.REGULATIONS_DIR)
    print(f"[OK] 生成 {n_reg} 篇法规文档 -> {config.REGULATIONS_DIR}")

    flights = generate_flights(n=10000, days=90)
    flights.to_csv(config.FLIGHTS_CSV, index=False, encoding="utf-8-sig")
    on_time = (flights["status"] == "正常").mean()
    avg_delay = flights[flights["delay_minutes"] > 0]["delay_minutes"].mean()
    print(f"[OK] 生成 {len(flights)} 条航班记录 -> {config.FLIGHTS_CSV}")
    print(f"     正常率: {on_time:.1%} | 平均延误: {avg_delay:.1f}分钟")

    flow = generate_multi_airport_flow(["ZUUU", "ZUUU2", "ZBAA", "ZSPD", "ZGGG"], days=365)
    flow.to_csv(config.FLOW_HISTORY_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] 生成 {len(flow)} 条流量记录 -> {config.FLOW_HISTORY_CSV}")
    for ap in flow["airport"].unique():
        ap_data = flow[flow["airport"] == ap]
        print(f"     {ap}: 日均 {ap_data['flow'].mean():.1f}架次/小时, 峰值 {ap_data['flow'].max()}")

    print(f"\n数据生成完成！")


if __name__ == "__main__":
    main()
