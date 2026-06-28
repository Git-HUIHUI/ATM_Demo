"""
全局配置 - 空管智能助手 (ATM Intelligence Suite)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REGULATIONS_DIR = DATA_DIR / "regulations"
MODELS_DIR = PROJECT_ROOT / "models"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"

FLIGHTS_CSV = DATA_DIR / "flights.csv"
FLOW_HISTORY_CSV = DATA_DIR / "flow_history.csv"

LSTM_MODEL_PATH = MODELS_DIR / "lstm_flow_model.pt"
AUTOENCODER_MODEL_PATH = MODELS_DIR / "autoencoder_model.pt"

load_dotenv(PROJECT_ROOT / ".env")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RAG_TOP_K = 5

SEQUENCE_LENGTH = 24
PREDICT_HORIZON = 24
LEARNING_RATE = 0.001
