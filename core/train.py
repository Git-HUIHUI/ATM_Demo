"""
模型训练脚本 - LSTM 流量预测 & Autoencoder 异常检测
训练数据来自 data_generator.py 生成的 flow_history.csv
"""
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import (
    FLOW_HISTORY_CSV, LSTM_MODEL_PATH, AUTOENCODER_MODEL_PATH,
    MODELS_DIR, SEQUENCE_LENGTH, LEARNING_RATE,
)
from core.dl_models import create_lstm_model, create_autoencoder, train_lstm, train_autoencoder


def build_sequences(values, seq_len=SEQUENCE_LENGTH):
    """滑动窗口构造训练样本，y 是下一时刻的真实值（shift 1）"""
    x, y = [], []
    for i in range(len(values) - seq_len - 1):
        x.append(values[i:i + seq_len])
        y.append(values[i + 1:i + 1 + seq_len])
    return np.array(x), np.array(y)


def main():
    print("=" * 60)
    print("空管智能助手 - 深度学习模型训练")
    print("=" * 60)

    if not FLOW_HISTORY_CSV.exists():
        print("[ERROR] 流量数据未找到，请先运行 python core/data_generator.py")
        sys.exit(1)

    df = pd.read_csv(FLOW_HISTORY_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 用 ZUUU(成都双流) 的数据训练
    zuuu = df[df["airport"] == "ZUUU"]["flow"].values
    print(f"[INFO] 加载 ZUUU 流量数据：{len(zuuu)} 条 ({len(zuuu)//24} 天)")

    x, y = build_sequences(zuuu, SEQUENCE_LENGTH)
    print(f"[INFO] 构建训练样本：{len(x)} 个 ({SEQUENCE_LENGTH}h 滑动窗口)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 训练 LSTM ---
    print("\n[训练] LSTM 流量预测模型...")
    lstm = create_lstm_model()
    lstm = train_lstm(lstm, x, y, epochs=50, batch_size=64)
    torch.save(lstm.state_dict(), str(LSTM_MODEL_PATH))
    print(f"[OK] LSTM 模型已保存 -> {LSTM_MODEL_PATH}")

    # --- 训练 Autoencoder ---
    print("\n[训练] Autoencoder 异常检测模型...")
    ae = create_autoencoder()
    ae, mean, std = train_autoencoder(ae, x, epochs=30, batch_size=64)
    torch.save({"model": ae.state_dict(), "mean": mean, "std": std}, str(AUTOENCODER_MODEL_PATH))
    print(f"[OK] Autoencoder 模型已保存 -> {AUTOENCODER_MODEL_PATH}")
    print(f"     误差均值: {mean:.4f}, 标准差: {std:.4f} (z-score 双向检测)")

    print("\n训练完成！")


if __name__ == "__main__":
    main()
