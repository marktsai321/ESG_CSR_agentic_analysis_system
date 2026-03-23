from __future__ import annotations

"""
ESG Analysis Agent
==================
Analyzes ESG content using RAG retrieval from the vector store.
"""

import json
from pathlib import Path

from crewai import Agent, Task

from esg_csr_agent.config import OPENAI_MODEL_NAME, ANALYSIS_DIR


def create_esg_analysis_agent() -> Agent:
    return Agent(
        role="ESG 分析代理",
        goal="針對指定公司的永續報告書（ESG）進行結構化分析，涵蓋環境、社會、治理三大面向。",
        backstory=(
            "你是專精於 ESG 永續報告書分析的代理。"
            "你使用 RAG 技術從向量資料庫中檢索相關段落，"
            "對報告書進行環境（碳排放、能源使用、水資源、廢棄物、氣候目標）、"
            "社會（員工福利、供應鏈、社區、人權）、"
            "治理（董事會組成、反貪腐、透明度、風險管理）三大面向的深度分析。"
            "你的分析結果包含量化指標、年度變化趨勢，以及每個分析面向的信心分數。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


ESG_QUERIES = {
    "environmental": [
        "碳排放量 溫室氣體 排放",
        "能源使用 再生能源 電力消耗",
        "水資源管理 用水量",
        "廢棄物管理 回收率",
        "氣候目標 淨零 碳中和",
    ],
    "social": [
        "員工福利 薪資 職業安全",
        "供應鏈管理 供應商",
        "社區投入 公益活動",
        "人權政策 多元包容",
    ],
    "governance": [
        "董事會組成 獨立董事",
        "反貪腐 誠信經營",
        "資訊透明 揭露",
        "風險管理 內部控制",
    ],
}


def _retrieve_context(namespace: str, queries: list[str], top_k: int = 5) -> list[str]:
    from esg_csr_agent.vector_store import get_vector_store

    vs = get_vector_store()

    # Skip retrieval if namespace has no data (no PDF was downloaded/embedded)
    if not vs.namespace_exists(namespace):
        return []

    from esg_csr_agent.agents.chunk_embed_agent import generate_embeddings

    all_texts: list[str] = []
    seen: set[str] = set()

    for query in queries:
        query_emb = generate_embeddings([query])[0]
        results = vs.query(namespace, query_emb, top_k=top_k)
        for r in results:
            text = r["text"]
            if text not in seen:
                seen.add(text)
                all_texts.append(text)

    return all_texts


def _compute_confidence(chunk_count: int, query_count: int) -> float:
    """Compute confidence score based on retrieval coverage.

    Heuristic: expect ~3 relevant chunks per query on average.
    Score scales from 0.0 (no chunks) to 1.0 (3+ chunks per query).
    """
    if query_count == 0:
        return 0.0
    expected = query_count * 3
    ratio = min(chunk_count / expected, 1.0) if expected > 0 else 0.0
    # Apply sigmoid-like curve: 0 chunks → 0.0, some → grows, saturates near 1.0
    return round(ratio, 2)


def analyze_esg(company_id: str, year: int, namespace: str) -> dict:
    output_path = ANALYSIS_DIR / f"{company_id}_{year}_esg.json"

    if output_path.exists():
        print(f"[EXIST] ESG 分析結果已存在: {output_path.name}")
        return json.loads(output_path.read_text(encoding="utf-8"))

    analysis: dict = {
        "company_id": company_id,
        "year": year,
        "type": "esg",
        "dimensions": {},
    }

    for dimension, queries in ESG_QUERIES.items():
        context_chunks = _retrieve_context(namespace, queries)
        confidence = _compute_confidence(len(context_chunks), len(queries))
        analysis["dimensions"][dimension] = {
            "retrieved_chunks": len(context_chunks),
            "context_summary": "\n".join(context_chunks[:10]),
            "findings": "",
            "metrics": {},
            "confidence": confidence,
        }

    output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] ESG 分析完成: {output_path.name}")
    return analysis


def create_esg_analysis_task(agent: Agent, company_id: str, year: int, namespace: str) -> Task:
    return Task(
        description=(
            f"請對公司 {company_id} 的 {year} 年度 ESG 永續報告書進行結構化分析。\n"
            f"向量命名空間：{namespace}\n\n"
            "分析面向：\n"
            "1. 環境（Environmental）：碳排放、能源使用、水資源、廢棄物、氣候目標\n"
            "2. 社會（Social）：員工福利、供應鏈、社區投入、人權\n"
            "3. 治理（Governance）：董事會組成、反貪腐、透明度、風險管理\n\n"
            "每個面向須包含：分析結論、量化指標、信心分數（0.0-1.0）。\n"
            f"結果儲存至 data/analysis/{company_id}_{year}_esg.json"
        ),
        expected_output="結構化 ESG 分析結果（JSON 格式，含各面向分析及信心分數）",
        agent=agent,
    )
