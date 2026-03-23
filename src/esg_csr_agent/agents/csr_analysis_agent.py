from __future__ import annotations

"""
CSR Analysis Agent
==================
Analyzes CSR content using RAG retrieval.
"""

import json
from pathlib import Path

from crewai import Agent, Task

from esg_csr_agent.config import OPENAI_MODEL_NAME, ANALYSIS_DIR


def create_csr_analysis_agent() -> Agent:
    return Agent(
        role="CSR 分析代理",
        goal="針對指定公司的企業社會責任報告書（CSR）進行結構化分析，依循 GRI 準則及台灣 CSR 報告指引。",
        backstory=(
            "你是專精於企業社會責任報告書分析的代理。"
            "你使用 RAG 技術從向量資料庫中檢索相關段落，"
            "依據 GRI Standards 及台灣 CSR 報告指引進行分析，"
            "涵蓋利害關係人溝通、重大議題、社區投入、員工關係、環境管理等面向。"
            "CSR 報告書主要涵蓋 2013–2021 年（ESG 強制揭露前），"
            "你會注意年度適用性。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


CSR_QUERIES = {
    "stakeholder_engagement": [
        "利害關係人 溝通 鑑別",
        "利害關係人 議合 回應",
    ],
    "material_topics": [
        "重大議題 鑑別 矩陣",
        "重大性分析 議題排序",
    ],
    "community_investment": [
        "社區投入 公益 捐贈",
        "社會參與 志工",
    ],
    "employee_relations": [
        "員工關係 薪資福利 人才發展",
        "職業安全 勞動權益 員工滿意度",
    ],
    "environmental_stewardship": [
        "環境管理 環保 污染防治",
        "資源利用 節能減碳 綠色採購",
    ],
}


def _retrieve_context(namespace: str, queries: list[str], top_k: int = 5) -> list[str]:
    from esg_csr_agent.vector_store import get_vector_store
    from esg_csr_agent.agents.chunk_embed_agent import generate_embeddings

    vs = get_vector_store()
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


def analyze_csr(company_id: str, year: int, namespace: str) -> dict:
    output_path = ANALYSIS_DIR / f"{company_id}_{year}_csr.json"

    if output_path.exists():
        print(f"[EXIST] CSR 分析結果已存在: {output_path.name}")
        return json.loads(output_path.read_text(encoding="utf-8"))

    if year >= 2022:
        print(f"[WARN] CSR 報告書自 2022 年起可能不存在（已改為 ESG 永續報告書）")

    analysis: dict = {
        "company_id": company_id,
        "year": year,
        "type": "csr",
        "dimensions": {},
    }

    for dimension, queries in CSR_QUERIES.items():
        context_chunks = _retrieve_context(namespace, queries)
        analysis["dimensions"][dimension] = {
            "retrieved_chunks": len(context_chunks),
            "context_summary": "\n".join(context_chunks[:10]),
            "findings": "",
            "metrics": {},
            "confidence": 0.0,
        }

    output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] CSR 分析完成: {output_path.name}")
    return analysis


def create_csr_analysis_task(agent: Agent, company_id: str, year: int, namespace: str) -> Task:
    return Task(
        description=(
            f"請對公司 {company_id} 的 {year} 年度 CSR 企業社會責任報告書進行結構化分析。\n"
            f"向量命名空間：{namespace}\n\n"
            "分析面向（依循 GRI Standards 及台灣 CSR 報告指引）：\n"
            "1. 利害關係人溝通\n2. 重大議題鑑別\n3. 社區投入\n4. 員工關係\n5. 環境管理\n\n"
            "每個面向須包含：分析結論、量化指標、信心分數（0.0-1.0）。\n"
            f"結果儲存至 data/analysis/{company_id}_{year}_csr.json"
        ),
        expected_output="結構化 CSR 分析結果（JSON 格式，含各面向分析及信心分數）",
        agent=agent,
    )
