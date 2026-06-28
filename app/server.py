"""
FastAPI + LangServe 后端
"""
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pydantic import BaseModel

from core.agent_graph import create_agent
from core.dl_models import create_lstm_model, predict_lstm, Autoencoder, detect_anomaly
from core.config import (
    FLOW_HISTORY_CSV, FLIGHTS_CSV, CHROMA_PERSIST_DIR,
    LSTM_MODEL_PATH, AUTOENCODER_MODEL_PATH,
)

app = FastAPI(title="ATM Intelligence Suite", version="0.1.0")
flow_df = pd.read_csv(FLOW_HISTORY_CSV)

# DL 模型加载
_lstm = create_lstm_model()
_lstm.load_state_dict(torch.load(str(LSTM_MODEL_PATH), weights_only=True))
_ae = Autoencoder()
_ae_ckpt = torch.load(str(AUTOENCODER_MODEL_PATH), weights_only=True)
_ae.load_state_dict(_ae_ckpt["model"])
_ae_mean = _ae_ckpt["mean"]
_ae_std = _ae_ckpt["std"]


@app.get("/")
async def redirect():
    return RedirectResponse("/docs")


# --- Agent ---
agent = create_agent()
add_routes(app, agent, path="/agent")


# --- DL ---
class PredictRequest(BaseModel):
    airport: str = "ZUUU"
    hours: int = 24

class AnomalyRequest(BaseModel):
    airport: str = "ZUUU"


@app.post("/dl/predict")
async def predict(req: PredictRequest):
    ap_data = flow_df[flow_df["airport"] == req.airport.upper()]
    if ap_data.empty:
        return {"error": f"未找到机场 {req.airport} 的数据"}
    recent = ap_data["flow"].tail(24).tolist()
    if len(recent) < 24:
        return {"error": "数据不足24小时，无法预测"}
    pred = predict_lstm(_lstm, recent)[:req.hours]
    return {
        "airport": req.airport,
        "airport_name": ap_data["airport_name"].iloc[0],
        "predictions": [round(p, 1) for p in pred],
        "hours": len(pred),
    }


@app.post("/dl/anomaly")
async def anomaly(req: AnomalyRequest):
    ap_data = flow_df[flow_df["airport"] == req.airport.upper()]
    if ap_data.empty:
        return {"error": f"未找到机场 {req.airport} 的数据"}
    window = ap_data["flow"].tail(24).tolist()
    if len(window) < 24:
        return {"error": "数据不足24小时，无法检测"}
    is_anomaly, error, z = detect_anomaly(_ae, window, _ae_mean, _ae_std)
    return {
        "airport": req.airport,
        "airport_name": ap_data["airport_name"].iloc[0],
        "reconstruction_error": round(error, 3),
        "z_score": round(z, 3),
        "is_anomaly": bool(is_anomaly),
    }


@app.get("/health")
async def health():
    chroma_ok = CHROMA_PERSIST_DIR.exists() and any(CHROMA_PERSIST_DIR.iterdir())
    flights_count = len(pd.read_csv(FLIGHTS_CSV)) if FLIGHTS_CSV.exists() else 0
    flow_count = len(pd.read_csv(FLOW_HISTORY_CSV)) if FLOW_HISTORY_CSV.exists() else 0
    model_ok = LSTM_MODEL_PATH.exists()

    return {
        "status": "ok",
        "chroma_ok": chroma_ok,
        "flights_records": flights_count,
        "flow_records": flow_count,
        "model_loaded": model_ok,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
