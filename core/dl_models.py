"""
深度学习模型 - LSTM 流量预测 & Autoencoder 异常检测
"""
import numpy as np
import torch
import torch.nn as nn

from core.config import SEQUENCE_LENGTH, PREDICT_HORIZON, LEARNING_RATE


class LSTMPredictor(nn.Module):
    """LSTM 机场流量预测"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, PREDICT_HORIZON)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # (batch, 24)


class Autoencoder(nn.Module):
    """Autoencoder 流量异常检测"""
    def __init__(self, input_size=SEQUENCE_LENGTH):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_size), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def create_lstm_model():
    return LSTMPredictor()


def create_autoencoder():
    return Autoencoder()


def train_lstm(model, x_data, y_data, epochs=50, batch_size=32):
    """训练 LSTM 模型，x_data/y_data 为 (n_samples, seq_len) 的 numpy 数组"""
    x = torch.tensor(x_data, dtype=torch.float32).unsqueeze(-1)  # (n, 24, 1)
    y = torch.tensor(y_data, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(len(x))
        total_loss = 0
        for i in range(0, len(x), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x[idx], y[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / max(1, len(x) // batch_size):.4f}")
    return model


def train_autoencoder(model, data, epochs=30, batch_size=32):
    """训练 Autoencoder，返回 (model, mean, std) 用于 z-score 双向异常检测"""
    x = torch.tensor(data, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(len(x))
        total_loss = 0
        for i in range(0, len(x), batch_size):
            idx = perm[i:i + batch_size]
            xb = x[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        reconstructed = model(x)
        errors = ((reconstructed - x) ** 2).mean(dim=1).numpy()
    mean = float(errors.mean())
    std = float(errors.std())
    return model, mean, std


def predict_lstm(model, sequence):
    """用 LSTM 预测未来流量，sequence shape: (seq_len,)"""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, 24, 1)
        pred = model(x).squeeze().tolist()
    return pred


def detect_anomaly(model, window, mean, std):
    """z-score 双向异常检测。abs(z) > 2 判定异常，返回 (is_anomaly, error, z_score)"""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        recon = model(x)
        error = ((recon - x) ** 2).mean().item()
    z = (error - mean) / std if std > 0 else 0
    return abs(z) > 2, error, z
