"""
RAG 检索增强生成 - 空管法规知识库
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from core.config import (
    REGULATIONS_DIR, CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, RAG_TOP_K,
)
from core.llm import get_llm, get_embeddings

SYSTEM_PROMPT = (
    "你是民航空管法规智能助手，精通 CCAR 规章和空管业务知识。"
    "根据以下检索到的法规内容回答问题。"
    "如果法规中没有明确相关信息，请如实说明，不要编造。"
    "\n\n{context}"
)


def build_vectorstore() -> Chroma:
    """从法规 Markdown 构建 Chroma 向量库"""
    docs = []
    for md_path in sorted(REGULATIONS_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": md_path.name}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n", "。", "；", " "],
    )
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_PERSIST_DIR),
        collection_name="atm_regulations",
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=get_embeddings(),
        collection_name="atm_regulations",
    )


def create_rag_chain():
    """创建 RAG 检索问答链"""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RAG_TOP_K})
    llm = get_llm(temperature=0.1, streaming=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    combine_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_chain)


# 模块加载时初始化一次，后续调用复用
_rag_chain = None


def _get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = create_rag_chain()
    return _rag_chain


def rag_query(question: str) -> dict:
    """同步查询，返回 {answer, sources}"""
    result = _get_rag_chain().invoke({"input": question})
    sources = list({doc.metadata["source"] for doc in result.get("context", [])})
    return {"answer": result["answer"], "sources": sources}
