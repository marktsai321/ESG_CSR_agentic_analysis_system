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
    output_path.write_text(markdown, encoding="utf-8")
    print(f"[OK] 修訂報告已產生: {output_path.name}")
    return str(output_path)


def create_revision_task(agent: Agent, state: PipelineState) -> Task:
    return Task(
        description=(
            "請將所有分析結果整合為一份連貫的中文報告文稿。\n"
            f"公司：{', '.join(state.companies)}\n"
            f"年度：{', '.join(str(y) for y in state.years)}\n"
            f"輸出至 data/revised/{state.run_id}.md"
        ),
        expected_output="完整的中文 Markdown 報告文稿",
        agent=agent,
    )
