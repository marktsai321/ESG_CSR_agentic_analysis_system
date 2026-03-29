from __future__ import annotations

"""
Output Delivery Agent
=====================
Generates the final PDF report containing:
  1. Comparative score summary for ALL companies
  2. Per-company analysis review for ALL companies
  3. Revised deep-dive for the worst-scoring company's weak dimensions
"""

import datetime
import json
from pathlib import Path

from crewai import Agent, Task

from esg_csr_agent.config import (
    OPENAI_MODEL_NAME,
    REVISED_PDF_DIR,
    REVISED_DIR,
    ANALYSIS_DIR,
    WEAK_SCORE_THRESHOLD,
)
from esg_csr_agent.pipeline_state import PipelineState


def create_output_delivery_agent() -> Agent:
    return Agent(
        role="輸出交付代理",
        goal="產生包含所有公司評分與分析、以及最低分公司深化修訂的最終 PDF 報告。",
        backstory=(
            "你是負責最終報告產出的代理。"
            "你將所有公司的評分摘要、分析結果、"
            "以及最低分公司的弱項深化修訂整合為一份完整的 PDF 報告。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


# Dimension label mappings
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


def _load_analysis(company_id: str, year: int, rtype: str) -> dict | None:
    path = ANALYSIS_DIR / f"{company_id}_{year}_{rtype}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _format_dimension_review(name: str, data: dict) -> str:
    """Format a single dimension's analysis for the per-company review section."""
    lines = [f"#### {name}\n"]
    findings = data.get("findings", "")
    if findings:
        lines.append(findings)
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
            marker = " ⚠" if isinstance(v, (int, float)) and v < WEAK_SCORE_THRESHOLD else ""
            lines.append(f"- {k}: {v}{marker}")

    suggestions = data.get("improvement_suggestions", [])
    if suggestions:
        lines.append("\n**修正建議：**")
        for s in suggestions[:5]:
            lines.append(f"- {s}")

    conf = data.get("confidence", 0.0)
    lines.append(f"\n*信心分數: {conf:.2f}*\n")
    return "\n".join(lines)


def _build_full_markdown(state: PipelineState) -> str:
    """Build the complete report Markdown with all three sections."""
    sections: list[str] = []

    company_names = ", ".join(state.companies)
    year_range = ", ".join(str(y) for y in state.years)
    worst = state.worst_company or "N/A"

    sections.append("# ESG/CSR 分析報告\n")
    sections.append(f"**公司代號：** {company_names}  ")
    sections.append(f"**報告年度：** {year_range}  ")
    sections.append(f"**最低分公司：** {worst}  ")
    sections.append(f"**報告編號：** {state.run_id}\n")

    # ── Section 1: Comparative Score Summary ──
    sections.append("## 第一部分：各公司評分總覽\n")

    # Score summary table header
    sections.append("| 公司代號 | 總分 | 可用報告 | 備註 |")
    sections.append("|---------|------|---------|------|")

    for company in state.companies:
        score_info = state.scores.get(company, {})
        total = score_info.get("total", 0)
        avail = ", ".join(r.upper() for r in state.available_reports.get(company, []))
        note = "⚠ 最低分 — 已進行深化修訂" if company == worst else ""
        sections.append(f"| {company} | {total} | {avail} | {note} |")

    sections.append("")

    # Per-dimension breakdown table for each company
    for company in state.companies:
        score_info = state.scores.get(company, {})
        dims = score_info.get("dimensions", {})
        if not dims:
            continue

        sections.append(f"### {company} 維度評分明細\n")
        sections.append("| 維度 | 小計 | framework | data | materiality | targets | assurance |")
        sections.append("|------|------|-----------|------|-------------|---------|-----------|")

        for dim_key, dim_detail in dims.items():
            rubric = dim_detail.get("rubric_scores", {})
            subtotal = dim_detail.get("subtotal", 0)
            fc = rubric.get("framework_compliance", "-")
            dc = rubric.get("data_completeness", "-")
            ma = rubric.get("materiality_analysis", "-")
            tc = rubric.get("targets_commitments", "-")
            ea = rubric.get("external_assurance", "-")
            sections.append(f"| {dim_key} | {subtotal} | {fc} | {dc} | {ma} | {tc} | {ea} |")

        sections.append("")

    # ── Section 2: Per-Company Analysis Review ──
    sections.append("---\n\n## 第二部分：各公司分析結果\n")

    for company in state.companies:
        for year in state.years:
            sections.append(f"### {company} — {year} 年度\n")

            for rtype in ["esg", "csr"]:
                analysis = _load_analysis(company, year, rtype)
                if not analysis:
                    continue

                type_label = "ESG 永續報告書" if rtype == "esg" else "CSR 企業社會責任報告書"
                dim_names = ESG_DIM_NAMES if rtype == "esg" else CSR_DIM_NAMES
                sections.append(f"#### {type_label}分析\n")

                for dim_key, dim_data in analysis.get("dimensions", {}).items():
                    dim_name = dim_names.get(dim_key, dim_key)
                    sections.append(_format_dimension_review(dim_name, dim_data))

    # ── Section 3: Revised Analysis for Worst-Scoring Company ──
    sections.append("---\n\n## 第三部分：最低分公司弱項深化修訂\n")

    if worst and worst != "N/A":
        revision_path = REVISED_DIR / f"{state.run_id}_{worst}_revision.md"
        if revision_path.exists():
            revision_content = revision_path.read_text(encoding="utf-8")
            # Strip the top-level heading from revision (we provide our own)
            lines = revision_content.split("\n")
            skip_first_heading = False
            for i, line in enumerate(lines):
                if line.startswith("# ") and not skip_first_heading:
                    skip_first_heading = True
                    continue
                break
            if skip_first_heading:
                revision_content = "\n".join(lines[i:])
            sections.append(revision_content)
        else:
            sections.append(f"（修訂報告檔案不存在：{revision_path.name}）\n")
    else:
        sections.append("（無最低分公司需要修訂）\n")

    return "\n".join(sections)


def _build_html(markdown_text: str, state: PipelineState) -> str:
    import markdown as md_lib

    body = md_lib.markdown(markdown_text, extensions=["tables", "toc"])
    today = datetime.date.today().strftime("%Y-%m-%d")
    worst = state.worst_company or "N/A"

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<style>
@page {{ size: A4; margin: 2cm; }}
body {{ font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif;
       font-size: 11pt; line-height: 1.6; color: #333; }}
h1 {{ text-align: center; border-bottom: 2px solid #2c3e50;
     padding-bottom: 10px; color: #2c3e50; }}
h2 {{ color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
h3 {{ color: #34495e; }}
h4 {{ color: #7f8c8d; }}
table {{ width: 100%; border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #2c3e50; color: white; }}
hr {{ border: none; border-top: 1px solid #eee; margin: 2em 0; }}
.cover {{ text-align: center; padding: 100px 0; page-break-after: always; }}
.cover h1 {{ font-size: 28pt; border: none; }}
.cover p {{ font-size: 14pt; color: #555; }}
</style>
</head>
<body>
<div class="cover">
    <h1>ESG/CSR 分析報告</h1>
    <p>公司代號：{', '.join(state.companies)}</p>
    <p>報告年度：{', '.join(str(y) for y in state.years)}</p>
    <p>最低分公司：{worst}</p>
    <p>生成日期：{today}</p>
    <p>報告編號：{state.run_id}</p>
</div>
{body}
</body>
</html>"""


def _build_pdf_with_reportlab(markdown_text: str, output_path: Path) -> None:
    """Pure-Python PDF fallback for environments missing WeasyPrint system libs."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    left = 40
    top = height - 40
    line_height = 16
    y = top

    c.setFont("STSong-Light", 12)
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            y -= line_height
        else:
            chunk = line
            while chunk:
                part = chunk[:70]
                chunk = chunk[70:]
                c.drawString(left, y, part)
                y -= line_height
                if y < 40:
                    c.showPage()
                    c.setFont("STSong-Light", 12)
                    y = top
        if y < 40:
            c.showPage()
            c.setFont("STSong-Light", 12)
            y = top

    c.save()


def generate_pdf(state: PipelineState) -> str | None:
    year_suffix = "_".join(str(y) for y in state.years)
    output_filename = f"{state.run_id}_{year_suffix}.pdf"
    output_path = REVISED_PDF_DIR / output_filename

    if output_path.exists():
        print(f"[EXIST] PDF 已存在: {output_path.name}")
        return str(output_path)

    # Build the full markdown report with all three sections
    md_text = _build_full_markdown(state)

    # Also save the combined markdown for reference
    combined_md_path = REVISED_DIR / f"{state.run_id}.md"
    combined_md_path.write_text(md_text, encoding="utf-8")

    html_content = _build_html(md_text, state)

    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(str(output_path))
        print(f"[OK] PDF 報告已產生: {output_path.name}")
        return str(output_path)
    except Exception as e:
        print(f"[WARN] WeasyPrint 產生 PDF 失敗，改用 ReportLab 後備方案: {e}")
        try:
            _build_pdf_with_reportlab(md_text, output_path)
            print(f"[OK] PDF（ReportLab 後備）已產生: {output_path.name}")
            return str(output_path)
        except Exception as e2:
            html_path = output_path.with_suffix(".html")
            html_path.write_text(html_content, encoding="utf-8")
            print(f"[WARN] ReportLab 也失敗，改儲存為 HTML: {html_path.name}")
            print(f"[WARN] 失敗原因: {e2}")
            return str(html_path)


def create_delivery_task(agent: Agent, state: PipelineState) -> Task:
    return Task(
        description=(
            "請產生最終 PDF 報告，包含：\n"
            "1. 所有公司評分總覽\n"
            "2. 各公司分析結果\n"
            "3. 最低分公司弱項深化修訂\n"
            f"輸出至 data/revised_pdfs/{state.run_id}_*.pdf"
        ),
        expected_output="最終 PDF 報告路徑（data/revised_pdfs/）",
        agent=agent,
    )
