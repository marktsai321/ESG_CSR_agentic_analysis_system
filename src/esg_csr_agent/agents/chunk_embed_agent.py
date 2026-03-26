from __future__ import annotations

"""
Chunk + Embed Agent
===================
Splits extracted text into retrieval-ready chunks and stores them
with embeddings in the vector store.
"""

from pathlib import Path

from crewai import Agent, Task

from esg_csr_agent.config import OPENAI_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME


def create_chunk_embed_agent() -> Agent:
    return Agent(
        role="分塊嵌入代理",
        goal="將擷取的文字切分為適合檢索的區塊，並使用支援繁體中文的嵌入模型產生向量，存入向量資料庫。",
        backstory=(
            "你是專門負責文件分塊與嵌入的代理。"
            "你將長篇的報告書文字切分為固定大小且有重疊的區塊，"
            "使用支援繁體中文的嵌入模型產生向量表示，"
            "並將結果存入向量資料庫供分析代理進行 RAG 檢索。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count (approximating tokens)."""
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text chunks using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    return [emb.tolist() for emb in embeddings]


def chunk_and_embed(
    text_path: str,
    namespace: str,
) -> dict:
    """
    Full pipeline: read text -> chunk -> embed -> store in vector DB.

    Args:
        text_path: Path to extracted text file.
        namespace: Vector store namespace, e.g. '2330_2023_esg'.

    Returns:
        {"namespace": str, "chunk_count": int, "status": "ok"|"error", "error": str|None}
    """
    from esg_csr_agent.vector_store import get_vector_store

    vs = get_vector_store()

    # Idempotency
    if vs.namespace_exists(namespace):
        print(f"[EXIST] 向量命名空間已存在: {namespace}")
        return {"namespace": namespace, "chunk_count": -1, "status": "ok", "error": None}

    path = Path(text_path)
    if not path.exists():
        return {"namespace": namespace, "chunk_count": 0, "status": "error", "error": f"檔案不存在: {text_path}"}

    try:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        if not chunks:
            return {"namespace": namespace, "chunk_count": 0, "status": "error", "error": "無有效區塊"}

        print(f"[分塊] {namespace}: {len(chunks)} 個區塊，開始產生嵌入...")
        embeddings = generate_embeddings(chunks)

        metadatas = [{"namespace": namespace, "chunk_index": i} for i in range(len(chunks))]
        count = vs.add_documents(namespace, chunks, embeddings, metadatas)
        print(f"[OK] {namespace}: 已存入 {count} 個向量")

        return {"namespace": namespace, "chunk_count": count, "status": "ok", "error": None}

    except Exception as e:
        return {"namespace": namespace, "chunk_count": 0, "status": "error", "error": str(e)}


def chunk_and_embed_all(text_paths: dict[str, str]) -> list[dict]:
    """Process multiple text files."""
    results = []
    for namespace, text_path in text_paths.items():
        results.append(chunk_and_embed(text_path, namespace))
    return results


def create_chunk_embed_task(agent: Agent, text_paths: dict[str, str]) -> Task:
    paths_desc = "\n".join(f"  - {k}: {v}" for k, v in text_paths.items())
    return Task(
        description=(
            f"請將以下文字檔分塊並產生嵌入向量：\n{paths_desc}\n\n"
            f"分塊參數：大小={CHUNK_SIZE}，重疊={CHUNK_OVERLAP}\n"
            f"嵌入模型：{EMBEDDING_MODEL_NAME}\n"
            "結果存入向量資料庫，以 {{公司代號}}_{{年度}}_{{類型}} 作為命名空間。"
        ),
        expected_output="分塊嵌入結果摘要（每個命名空間的區塊數量及狀態）",
        agent=agent,
    )
