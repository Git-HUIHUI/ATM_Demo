"""
LLM & Embedding 工厂
"""
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from core.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL


def get_llm(temperature: float = 0.1, streaming: bool = False) -> ChatOpenAI:
    if not DASHSCOPE_API_KEY:
        raise ValueError("未设置 DASHSCOPE_API_KEY，请在 .env 中配置")
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        temperature=temperature,
        streaming=streaming,
    )


def get_embeddings():
    """本地 HuggingFace Embedding，不依赖云服务"""
    return HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
