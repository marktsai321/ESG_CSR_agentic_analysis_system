from __future__ import annotations

"""
Report Revision Agent
=====================
Deepens and improves analysis for the worst-scoring company,
focusing only on dimensions that received poor rubric scores.
Uses RAG to retrieve additional context and Gemini for re-analysis.
"""

import json
from pathlib import Path

from crewai import Agent, Task

from esg_csr_agent.config import (
    OPENAI_MODEL_NAME,
    ANALYSIS_DIR,
    REVISED_DIR,
    WEAK_SCORE_THRESHOLD,
)
from esg_csr_agent.llm_client import gemini_json_completion
from esg_csr_agent.pipeline_state import PipelineState


def create_report_revision_agent() -> Agent:
    return Agent(
        role="報告修訂代理",
        goal="針對最低分公司的弱項維度進行深化分析，透過 RAG 檢索額外資料並重新分析。",
        backstory=(
            "你是專門負責深化弱項分析的代理。"
            "你只對最低分公司的弱項維度進行修訂，"
            "透過 RAG 檢索初次分析可能遺漏的相關段落，"
            "並重新進行更深入的分析以提升品質。"
            "你不修訂分數良好的維度，也不處理其他公司。"
            "所有輸出皆為中文。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


# Dimension label mappings for readable output
ESG_DIM_NAMES = {
    "environmental": "環境（Environmental）",
    "social": "社會（Social）",
    "governance": "治理（Governance）",
}

CSR_DIM_NAMES = {
    "stakeholder_engagement": "利害關係人溝通",
    "material_topics": "重大議題鑑別",
    "community_investment": "社區投入",
    "employee_relations": "員工關係",
    "environmental_stewardship": "環境管理",
}

# RAG queries for deeper retrieval per dimension
_DEEP_QUERIES = {
    "environmental": [
        "碳排放量 溫室氣體 範疇一 範疇二 範疇三",
        "能源使用 再生能源 電力消耗 節能",
        "水資源管理 廢棄物 回收率 循環經濟",
        "氣候目標 淨零 碳中和 SBTi 路徑",
        "環境管理系統 ISO 14001 環保法規",
    ],
    "social": [
        "員工福利 薪資 職業安全 健康 職災率",
        "供應鏈管理 供應商審核 人權盡職調查",
        "社區投入 公益活動 社會影響",
        "人權政策 多元包容 性別平等 DEI",
        "教育訓練 人才發展 員工留任率",
    ],
    "governance": [
        "董事會組成 獨立董事 功能性委員會",
        "反貪腐 誠信經營 檢舉機制",
        "資訊透明 揭露 利害關係人溝通",
        "風險管理 內部控制 內部稽核",
        "永續治理架構 ESG 委員會 薪酬委員會",
    ],
    "stakeholder_engagement": [
        "利害關係人 溝通管道 鑑別 議合",
        "利害關係人 回應 申訴 意見回饋",
    ],
    "material_topics": [
        "重大議題 鑑別 矩陣 排序 衝擊評估",
        "重大性分析 雙重重大性 財務重大性",
    ],
    "community_investment": [
        "社區投入 公益 捐贈 志工 社會投資",
        "在地經濟 就業 社區發展",
    ],
    "employee_relations": [
        "員工關係 薪資福利 人才發展 培訓",
        "職業安全 勞動權益 員工滿意度 流動率",
    ],
    "environmental_stewardship": [
        "環境管理 環保 污染防治 排放",
        "資源利用 節能減碳 綠色採購 水資源",
    ],
}


def _load_analysis(company_id: str, year: int, rtype: str) -> dict | None:
    path = ANALYSIS_DIR / f"{company_id}_{year}_{rtype}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _retrieve_deep_context(namespace: str, queries: list[str], top_k: int = 8) -> list[str]:
    """Retrieve additional context chunks for deeper analysis."""
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


def _get_weak_rubric_names(dim_data: dict) -> list[str]:
    """Return names of rubric scores below the weak threshold."""
    weak = []
    for name, score in dim_data.get("rubric_scores", {}).items():
        if isinstance(score, (int, float)) and score < WEAK_SCORE_THRESHOLD:
            weak.append(name)
    return weak


def _revise_dimension_with_gemini(
    company_id: str,
    year: int,
    rtype: str,
    dim_key: str,
    dim_data: dict,
    context_chunks: list[str],
) -> dict | None:
    """Use Gemini to deepen analysis of a weak dimension."""
    all_dim_names = {**ESG_DIM_NAMES, **CSR_DIM_NAMES}
    dim_label = all_dim_names.get(dim_key, dim_key)
    weak_rubrics = _get_weak_rubric_names(dim_data)
    context_text = "\n\n---\n\n".join(context_chunks[:20])

    # Build context-aware prompt for both below-threshold and relatively-lowest cases
    if weak_rubrics:
        score_context = (
            f"「{dim_label}」維度的以下 rubric 項目得分偏低（低於 {WEAK_SCORE_THRESHOLD} 分）：\n"
            f"{', '.join(weak_rubrics)}"
        )
    else:
        # All scores are above threshold, but this is the relatively weakest dimension
        all_scores = dim_data.get("rubric_scores", {})
        score_summary = ", ".join(f"{k}: {v}" for k, v in all_scores.items())
        score_context = (
            f"「{dim_label}」維度是該公司相對最弱的維度（該公司為所有受分析公司中總分最低）。\n"
            f"各 rubric 分數：{score_summary}\n"
            f"雖均達門檻，但作為最低分公司仍需深化分析以提升品質。"
        )

    prompt = (
        f"你是 ESG/CSR 報告書深度分析專家。\n"
        f"公司 {company_id} 的 {year} 年度 {rtype.upper()} 報告中，\n"
        f"{score_context}\n\n"
        f"初次分析結果：\n{json.dumps(dim_data, ensure_ascii=False, indent=2)}\n\n"
        f"以下是從原始報告中額外檢索的相關段落：\n{context_text}\n\n"
        "請針對此維度進行深化分析，輸出 JSON：\n"
        '1. "original_scores": 物件，原始 rubric 項目及分數\n'
        '2. "revised_findings": 300-800 字中文深化分析，補充初次分析遺漏的證據與細節\n'
        '3. "additional_evidence": 陣列，從額外段落中發現的新證據（每項 50-100 字）\n'
        '4. "improvement_recommendations": 陣列，3-5 條具體可執行改善建議\n'
        '5. "revision_rationale": 100-200 字說明為何需要修訂及修訂重點\n\n'
        "請只回傳 JSON。"
    )

    try:
        result = gemini_json_completion(
            prompt,
            system_instruction="你是 ESG/CSR 報告修訂專家，專注於深化弱項維度的分析。所有輸出必須為中文。",
        )
        return result
    except Exception as e:
        print(f"[WARN] Gemini 深化分析失敗 ({dim_key}): {e}")
        return None


def revise_report(state: PipelineState) -> str | None:
    """Revise only the worst-scoring company's weak dimensions."""
    worst = state.worst_company
    if not worst:
        print("[SKIP] 無最低分公司，跳過修訂。")
        return None

    output_path = REVISED_DIR / f"{state.run_id}_{worst}_revision.md"
    if output_path.exists():
        print(f"[EXIST] 修訂報告已存在: {output_path.name}")
        return str(output_path)

    weak_dims = state.weak_dimensions.get(worst, [])
    if not weak_dims:
        # This should not happen — orchestrator always selects dimensions.
        # Fallback: pick all dimensions for revision.
        all_dims = list(state.scores.get(worst, {}).get("dimensions", {}).keys())
        weak_dims = all_dims
        state.weak_dimensions[worst] = weak_dims
        print(f"[INFO] 無預選維度，將修訂所有維度: {', '.join(weak_dims)}")

    worst_score = state.scores.get(worst, {})
    sections: list[str] = []

    sections.append(f"# 公司 {worst} — 弱項維度深化修訂報告\n")
    sections.append(f"**報告編號：** {state.run_id}  ")
    sections.append(f"**總評分：** {worst_score.get('total', 'N/A')}  ")
    sections.append(f"**弱項維度數：** {len(weak_dims)}  ")
    sections.append(f"**弱項門檻：** rubric 分數 < {WEAK_SCORE_THRESHOLD}\n")

    for dim_key in weak_dims:
        # Parse rtype and actual dimension name from dim_key (e.g. "esg_environmental")
        parts = dim_key.split("_", 1)
        if len(parts) != 2:
            continue
        rtype, dim_name = parts[0], parts[1]

        all_dim_names = {**ESG_DIM_NAMES, **CSR_DIM_NAMES}
        dim_label = all_dim_names.get(dim_name, dim_name)

        sections.append(f"\n---\n\n## {dim_label}（{rtype.upper()}）\n")

        # Load original analysis
        for year in state.years:
            analysis = _load_analysis(worst, year, rtype)
            if not analysis:
                continue
            dim_data = analysis.get("dimensions", {}).get(dim_name, {})
            if not dim_data:
                continue

            # Show original scores
            original_rubric = dim_data.get("rubric_scores", {})
            sections.append(f"### 原始評分\n")
            sections.append(f"- 整體分數: {dim_data.get('overall_score', 'N/A')}")
            for k, v in original_rubric.items():
                marker = " ⚠" if isinstance(v, (int, float)) and v < WEAK_SCORE_THRESHOLD else ""
                sections.append(f"- {k}: {v}{marker}")
            sections.append("")

            # Retrieve additional context via RAG
            ns = state.file_key(worst, year, rtype)
            queries = _DEEP_QUERIES.get(dim_name, [f"{dim_label} 分析 評估"])
            context_chunks = _retrieve_deep_context(ns, queries)

            if context_chunks:
                # Use Gemini for deepened analysis
                revision = _revise_dimension_with_gemini(
                    worst, year, rtype, dim_name, dim_data, context_chunks,
                )

                if revision:
                    sections.append("### 深化修訂分析\n")

                    rationale = revision.get("revision_rationale", "")
                    if rationale:
                        sections.append(f"**修訂原因：** {rationale}\n")

                    revised_findings = revision.get("revised_findings", "")
                    if revised_findings:
                        sections.append(f"**深化發現：**\n\n{revised_findings}\n")

                    evidence = revision.get("additional_evidence", [])
                    if evidence:
                        sections.append("**額外佐證：**\n")
                        for item in evidence:
                            sections.append(f"- {item}")
                        sections.append("")

                    recommendations = revision.get("improvement_recommendations", [])
                    if recommendations:
                        sections.append("**改善建議：**\n")
                        for item in recommendations:
                            sections.append(f"- {item}")
                        sections.append("")
                else:
                    sections.append("### 深化修訂分析\n")
                    sections.append("（Gemini 深化分析未成功，以下為原始改善建議）\n")
                    for s in dim_data.get("improvement_suggestions", []):
                        sections.append(f"- {s}")
                    sections.append("")
            else:
                sections.append("### 深化修訂分析\n")
                sections.append("（無額外可檢索段落，無法進行深化分析）\n")

    markdown = "\n".join(sections)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"[OK] 修訂報告已產生: {output_path.name}")
    return str(output_path)


def create_revision_task(agent: Agent, state: PipelineState) -> Task:
    worst = state.worst_company or "N/A"
    weak = state.weak_dimensions.get(worst, [])
    return Task(
        description=(
            f"請對最低分公司 {worst} 的弱項維度進行深化分析。\n"
            f"弱項維度：{', '.join(weak)}\n"
            f"弱項門檻：rubric 分數 < {WEAK_SCORE_THRESHOLD}\n"
            "使用 RAG 檢索額外段落，重新分析並產生改善建議。\n"
            f"輸出至 data/revised/{state.run_id}_{worst}_revision.md"
        ),
        expected_output="最低分公司弱項維度深化修訂報告（Markdown 格式）",
        agent=agent,
    )
