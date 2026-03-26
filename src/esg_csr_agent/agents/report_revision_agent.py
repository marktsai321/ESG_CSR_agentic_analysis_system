from __future__ import annotations

"""
Report Revision Agent
=====================
Transforms structured JSON analysis into polished Chinese Markdown.
"""

import json
from pathlib import Path

from crewai import Agent, Task

from esg_csr_agent.config import OPENAI_MODEL_NAME, ANALYSIS_DIR, REVISED_DIR
from esg_csr_agent.llm_client import gemini_json_completion
from esg_csr_agent.pipeline_state import PipelineState


def create_report_revision_agent() -> Agent:
    return Agent(
        role="報告修訂代理",
        goal="將結構化的 JSON 分析結果轉化為連貫、精鍊的中文報告文稿。",
        backstory=(
            "你是專門負責將分析結果轉化為高品質中文報告的代理。"
            "你確保報告具有一致的術語、語氣和格式，"
            "涵蓋摘要、ESG 分析、CSR 分析、交叉分析發現及量化數據表格。"
            "所有輸出皆為中文，除非是專有名詞或行業標準縮寫（如 GRI、TCFD、ESG）。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


def _load_analysis(company_id: str, year: int, rtype: str) -> dict | None:
    path = ANALYSIS_DIR / f"{company_id}_{year}_{rtype}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _format_dimension(name: str, data: dict) -> str:
    lines = [f"#### {name}\n"]
    findings = data.get("findings", "")
    if findings:
        lines.append(findings)
    else:
        summary = data.get("context_summary", "")
        if summary:
            lines.append(summary[:500] + ("..." if len(summary) > 500 else ""))
        else:
            lines.append("（無足夠資料進行分析）")

    metrics = data.get("metrics", {})
    if metrics:
        lines.append("\n**量化指標：**\n")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")

    overall_score = data.get("overall_score")
    if overall_score is not None:
        lines.append(f"\n**Rubric 分數（1-10）：** {overall_score}")

    rubric_scores = data.get("rubric_scores", {})
    if rubric_scores:
        lines.append("\n**五大維度評分：**")
        for k, v in rubric_scores.items():
            lines.append(f"- {k}: {v}")

    suggestions = data.get("improvement_suggestions", [])
    if suggestions:
        lines.append("\n**修正建議：**")
        for s in suggestions[:5]:
            lines.append(f"- {s}")

    conf = data.get("confidence", 0.0)
    lines.append(f"\n*信心分數: {conf:.2f}*\n")
    return "\n".join(lines)


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


def revise_report(state: PipelineState) -> str | None:
    output_path = REVISED_DIR / f"{state.run_id}.md"

    if output_path.exists():
        print(f"[EXIST] 修訂報告已存在: {output_path.name}")
        return str(output_path)

    sections: list[str] = []

    company_names = ", ".join(state.companies)
    year_range = ", ".join(str(y) for y in state.years)
    sections.append(f"# ESG/CSR 分析報告\n")
    sections.append(f"**公司代號：** {company_names}  ")
    sections.append(f"**報告年度：** {year_range}  ")
    sections.append(f"**報告類型：** {', '.join(state.report_types)}  ")
    sections.append(f"**報告編號：** {state.run_id}\n")

    sections.append("## 摘要\n")
    sections.append(
        f"本報告針對 {company_names} 公司的 {year_range} 年度"
        f" {'/'.join(t.upper() for t in state.report_types)} 報告書進行結構化分析，"
        "涵蓋環境、社會、治理等面向，並進行交叉比對分析。\n"
    )

    for company in state.companies:
        for year in state.years:
            sections.append(f"---\n\n## {company} — {year} 年度\n")

            if "esg" in state.report_types:
                esg = _load_analysis(company, year, "esg")
                if esg:
                    sections.append("### ESG 永續報告書分析\n")
                    for dim_key, dim_data in esg.get("dimensions", {}).items():
                        dim_name = ESG_DIM_NAMES.get(dim_key, dim_key)
                        sections.append(_format_dimension(dim_name, dim_data))

            if "csr" in state.report_types:
                csr = _load_analysis(company, year, "csr")
                if csr:
                    sections.append("### CSR 企業社會責任報告書分析\n")
                    for dim_key, dim_data in csr.get("dimensions", {}).items():
                        dim_name = CSR_DIM_NAMES.get(dim_key, dim_key)
                        sections.append(_format_dimension(dim_name, dim_data))

            if len(state.report_types) >= 2:
                cross = _load_analysis(company, year, "cross")
                if cross:
                    sections.append("### 交叉分析結果\n")
                    for c in cross.get("contradictions", []):
                        sections.append(f"- **矛盾：** {c.get('note', '')}\n")
                    for a in cross.get("alignments", []):
                        sections.append(f"- **一致性：** {a.get('dimension', '')}: {a.get('note', '')}\n")
                    for g in cross.get("gaps", []):
                        sections.append(f"- **缺口：** {g.get('dimension', '')}: {g.get('note', '')}\n")

    markdown = "\n".join(sections)
    revised_markdown, suggestions_markdown = _revise_markdown_with_gemini(state, markdown)
    if revised_markdown:
        markdown = _sanitize_revised_markdown(revised_markdown)
    output_path.write_text(markdown, encoding="utf-8")
    _write_rewrite_suggestions(state, suggestions_markdown)
    print(f"[OK] 修訂報告已產生: {output_path.name}")
    return str(output_path)


def _revision_payload_from_state(state: PipelineState) -> dict:
    payload: dict = {
        "companies": state.companies,
        "years": state.years,
        "report_types": state.report_types,
        "analysis": {},
    }
    for company in state.companies:
        for year in state.years:
            base_key = f"{company}_{year}"
            payload["analysis"][base_key] = {
                "esg": _load_analysis(company, year, "esg"),
                "csr": _load_analysis(company, year, "csr"),
                "cross": _load_analysis(company, year, "cross"),
            }
    return payload


def _source_text_payload_from_state(state: PipelineState, max_chars_per_doc: int = 12000) -> dict:
    """Load original extracted txt contents as revision source."""
    payload: dict = {}
    extracted = state.files.get("extracted_text", {})
    for key, text_path in extracted.items():
        try:
            path = Path(text_path)
            text = path.read_text(encoding="utf-8")
            payload[key] = {
                "path": str(path),
                "text": text[:max_chars_per_doc],
                "truncated": len(text) > max_chars_per_doc,
            }
        except Exception as e:
            payload[key] = {
                "path": str(text_path),
                "text": "",
                "truncated": False,
                "error": str(e),
            }
    return payload


def _revise_markdown_with_gemini(state: PipelineState, draft_markdown: str) -> tuple[str | None, str | None]:
    payload = _revision_payload_from_state(state)
    source_texts = _source_text_payload_from_state(state)
    system_instruction = (
        "你是 ESG/CSR 報告修訂專家。"
        "你必須先產生修訂建議，再依建議重寫完整報告。"
        "修訂內容必須以原始 extracted txt 內容為主要依據，分析 JSON 僅作輔助。"
    )

    # Step 1: generate rewrite suggestions markdown first.
    suggestion_prompt = (
        "請先根據以下資料輸出 JSON：\n"
        '1. "rewrite_suggestions_markdown": 改寫建議 Markdown（需含優先序、負責角色、時程）\n'
        '2. "top_priorities": 長度 3 的陣列，列出最高優先改善項目\n\n'
        "【強制要求】\n"
        "- 建議必須具體可執行，不可空泛。\n"
        "- 需覆蓋低分面向，並對應改善時程與責任單位。\n\n"
        "【原始 extracted txt（主要修訂依據）】\n"
        f"{json.dumps(source_texts, ensure_ascii=False)}\n\n"
        "【草稿 Markdown】\n"
        f"{draft_markdown}\n\n"
        "【結構化分析 JSON】\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "請僅輸出 JSON。"
    )
    try:
        suggestion_result = gemini_json_completion(suggestion_prompt, system_instruction=system_instruction)
    except Exception as e:
        print(f"[WARN] Gemini 修訂失敗，改用本地草稿: {e}")
        return None, _fallback_suggestions_markdown(state, [])

    suggestions = suggestion_result.get("rewrite_suggestions_markdown", "")
    top_priorities = suggestion_result.get("top_priorities", [])
    suggestions_md = suggestions if isinstance(suggestions, str) and suggestions.strip() else _fallback_suggestions_markdown(state, top_priorities)

    # Step 2: generate final revised report strictly based on suggestions markdown.
    revised_prompt = (
        "請根據以下資料輸出 JSON：\n"
        '1. "revised_markdown": 修訂後完整 Markdown 報告字串（可直接交付）\n\n'
        "【強制要求】\n"
        "- revised_markdown 必須依據「改寫建議 Markdown」逐項完成改寫。\n"
        "- 不是摘要，必須是完整報告書正文。\n"
        "- 不可包含『改寫建議』或『建議清單』章節。\n\n"
        "【改寫建議 Markdown（唯一修訂依據）】\n"
        f"{suggestions_md}\n\n"
        "【原始 extracted txt（事實依據）】\n"
        f"{json.dumps(source_texts, ensure_ascii=False)}\n\n"
        "【草稿 Markdown】\n"
        f"{draft_markdown}\n\n"
        "請僅輸出 JSON。"
    )
    try:
        revised_result = gemini_json_completion(revised_prompt, system_instruction=system_instruction)
    except Exception as e:
        print(f"[WARN] Gemini 完整改寫失敗，改用草稿: {e}")
        return None, suggestions_md

    revised = revised_result.get("revised_markdown", "")
    if not isinstance(revised, str) or not revised.strip():
        return None, suggestions_md
    return revised, suggestions_md


def _fallback_suggestions_markdown(state: PipelineState, top_priorities: list) -> str:
    lines = [
        "# 改寫建議",
        "",
        f"報告編號：`{state.run_id}`",
        "",
        "## 優先改善項目",
    ]
    if isinstance(top_priorities, list) and top_priorities:
        for i, item in enumerate(top_priorities[:5], start=1):
            lines.append(f"{i}. {item}")
    else:
        lines.append("1. 補強框架合規性與指標索引揭露（GRI/SASB/TCFD/ISSB）。")
        lines.append("2. 補足三年以上趨勢數據與範疇一/二/三數據完整性。")
        lines.append("3. 補充重大性矩陣邏輯、負面事件揭露與改善時程。")
    lines.extend([
        "",
        "## 建議執行方式",
        "- 負責角色：永續單位、財會單位、風險管理單位、法遵單位。",
        "- 時程建議：30/60/90 天分階段完成資料補齊與改版。",
        "- 交付格式：更新後指標表、確信聲明、修訂後正式報告。",
    ])
    return "\n".join(lines) + "\n"


def _write_rewrite_suggestions(state: PipelineState, suggestions_markdown: str | None) -> None:
    suggestion_path = REVISED_DIR / f"{state.run_id}_rewrite_suggestions.md"
    content = suggestions_markdown or _fallback_suggestions_markdown(state, [])
    suggestion_path.write_text(content, encoding="utf-8")
    print(f"[OK] 改寫建議已產生: {suggestion_path.name}")


def _sanitize_revised_markdown(markdown: str) -> str:
    """Ensure revised output remains full report正文, not suggestion list."""
    lowered = markdown.lower()
    if "# 改寫建議" in markdown or "## 改寫建議" in markdown or "rewrite suggestions" in lowered:
        cut_points = []
        for marker in ("# 改寫建議", "## 改寫建議", "# Rewrite Suggestions", "## Rewrite Suggestions"):
            idx = markdown.find(marker)
            if idx != -1:
                cut_points.append(idx)
        if cut_points:
            return markdown[: min(cut_points)].rstrip() + "\n"
    return markdown


def create_revision_task(agent: Agent, state: PipelineState) -> Task:
    return Task(
        description=(
            "請根據分析結果，重寫一份修訂後的完整中文報告書（非摘要）。\n"
            f"公司：{', '.join(state.companies)}\n"
            f"年度：{', '.join(str(y) for y in state.years)}\n"
            "請根據各面向 rubric_scores 與 overall_score，將低分項目的修正建議直接寫入新版報告正文。\n"
            f"輸出至 data/revised/{state.run_id}.md"
        ),
        expected_output="修訂後完整中文 Markdown 報告書（已納入改善建議）",
        agent=agent,
    )
