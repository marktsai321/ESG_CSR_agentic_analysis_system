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
from esg_csr_agent.llm_client import chat_completion


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


CSR_DIM_LABELS = {
    "stakeholder_engagement": "利害關係人溝通與鑑別",
    "material_topics": "重大議題鑑別與排序",
    "community_investment": "社區投入與公益活動",
    "employee_relations": "員工關係（薪資福利、職業安全、人才發展）",
    "environmental_stewardship": "環境管理（污染防治、節能減碳、綠色採購）",
}


def _analyze_dimension_with_llm(
    company_id: str, year: int, dimension: str, context_chunks: list[str],
) -> dict:
    """Call the LLM to produce structured findings for one CSR dimension."""
    context_text = "\n\n---\n\n".join(context_chunks[:15])
    dim_label = CSR_DIM_LABELS.get(dimension, dimension)

    prompt = (
        f"你是企業社會責任（CSR）報告書分析專家。以下是公司 {company_id} 的 {year} 年度 CSR 報告書中"
        f"與「{dim_label}」相關的段落摘錄：\n\n"
        f"{context_text}\n\n"
        "請依據以下 ESG/CSR 評級 Rubric 進行評分與分析，並以 JSON 格式回傳：\n"
        "【五大維度】\n"
        "1. framework_compliance: 框架合規性（GRI/SASB/TCFD/ISSB 採用程度）\n"
        "2. data_completeness: 數據完整性（至少三年趨勢、範疇一/二/三碳排、全集團覆蓋）\n"
        "3. materiality_analysis: 重大性分析（重大性矩陣邏輯與產業對齊）\n"
        "4. targets_commitments: 目標與承諾（SBTi、RE100、淨零路徑與時間表）\n"
        "5. external_assurance: 外部確信（第三方驗證，如 SGS/BSI/會計師）\n\n"
        "【1-10 分級】\n"
        "1-3: 起步/合規；4-6: 發展/標準；7-8: 領先/透明；9-10: 卓越/轉型。\n\n"
        "【評分流程】\n"
        "先檢查第三方確信與指標索引（GRI Index），再檢視重大性與產業一致性，"
        "最後將質化轉量化。特別重視負面事件揭露與改善計畫的完整性。\n\n"
        "請輸出 JSON 欄位：\n"
        '1. "findings": 200-500 字中文結論，涵蓋主要發現與趨勢\n'
        '2. "metrics": 物件，關鍵量化指標 {指標名稱: 數值或描述}\n'
        '3. "rubric_scores": 物件，含五大維度分數（1-10，整數）\n'
        '4. "overall_score": 整體分數（1-10，可含小數點一位）\n'
        '5. "improvement_suggestions": 陣列，3-5 條可執行修正建議\n'
        '6. "confidence": 浮點數 0.0-1.0，依資料充分性給分\n\n'
        "請只回傳 JSON，不要加 markdown 標記或其他文字。"
    )

    raw = chat_completion(prompt, temperature=0.2, max_tokens=2000).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "findings": raw,
            "metrics": {},
            "rubric_scores": {},
            "overall_score": 0.0,
            "improvement_suggestions": [],
            "confidence": 0.4,
        }

    return result


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
        print(f"  [CSR] 分析 {dimension}...")
        context_chunks = _retrieve_context(namespace, queries)

        if not context_chunks:
            analysis["dimensions"][dimension] = {
                "retrieved_chunks": 0,
                "context_summary": "",
                "findings": "無相關內容可供分析。",
                "metrics": {},
                "rubric_scores": {},
                "overall_score": 0.0,
                "improvement_suggestions": [],
                "confidence": 0.0,
            }
            continue

        llm_result = _analyze_dimension_with_llm(company_id, year, dimension, context_chunks)
        analysis["dimensions"][dimension] = {
            "retrieved_chunks": len(context_chunks),
            "context_summary": "\n".join(context_chunks[:5]),
            "findings": llm_result.get("findings", ""),
            "metrics": llm_result.get("metrics", {}),
            "rubric_scores": llm_result.get("rubric_scores", {}),
            "overall_score": float(llm_result.get("overall_score", 0.0)),
            "improvement_suggestions": llm_result.get("improvement_suggestions", []),
            "confidence": float(llm_result.get("confidence", 0.0)),
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
            "每個面向須包含：分析結論、量化指標、五大維度 rubric_scores（1-10）、overall_score（1-10）、"
            "improvement_suggestions（3-5 條）、信心分數（0.0-1.0）。\n"
            f"結果儲存至 data/analysis/{company_id}_{year}_csr.json"
        ),
        expected_output="結構化 CSR 分析結果（JSON 格式，含各面向分析及信心分數）",
        agent=agent,
    )
