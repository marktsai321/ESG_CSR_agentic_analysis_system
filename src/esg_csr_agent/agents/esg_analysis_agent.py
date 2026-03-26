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
from esg_csr_agent.llm_client import chat_completion


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


ESG_DIM_LABELS = {
    "environmental": "環境（碳排放、能源使用、水資源、廢棄物、氣候目標）",
    "social": "社會（員工福利、供應鏈、社區、人權）",
    "governance": "治理（董事會組成、反貪腐、透明度、風險管理）",
}


def _analyze_dimension_with_llm(
    company_id: str, year: int, dimension: str, context_chunks: list[str],
) -> dict:
    """Call the LLM to produce structured findings for one ESG dimension."""
    context_text = "\n\n---\n\n".join(context_chunks[:15])
    dim_label = ESG_DIM_LABELS.get(dimension, dimension)

    prompt = (
        f"你是 ESG 永續報告書分析專家。以下是公司 {company_id} 的 {year} 年度永續報告書中"
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
    # Strip possible markdown fencing
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
        print(f"  [ESG] 分析 {dimension}...")
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
            "每個面向須包含：分析結論、量化指標、五大維度 rubric_scores（1-10）、overall_score（1-10）、"
            "improvement_suggestions（3-5 條）、信心分數（0.0-1.0）。\n"
            f"結果儲存至 data/analysis/{company_id}_{year}_esg.json"
        ),
        expected_output="結構化 ESG 分析結果（JSON 格式，含各面向分析及信心分數）",
        agent=agent,
    )
