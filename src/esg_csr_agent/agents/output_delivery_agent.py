from __future__ import annotations

"""
Output Delivery Agent
=====================
Generates the final PDF report from Markdown and delivers it.
"""

import datetime

from crewai import Agent, Task

from esg_csr_agent.config import OPENAI_MODEL_NAME, OUTPUTS_DIR, REVISED_DIR
from esg_csr_agent.pipeline_state import PipelineState


def create_output_delivery_agent() -> Agent:
    return Agent(
        role="輸出交付代理",
        goal="將修訂後的 Markdown 文稿轉換為結構化 PDF 報告並交付給使用者。",
        backstory=(
            "你是負責最終報告產出的代理。"
            "你將修訂好的 Markdown 報告轉換為格式完整的 PDF，"
            "包含封面、目錄、分析章節、數據表格及來源引用。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


def _build_html(markdown_text: str, state: PipelineState) -> str:
    import markdown as md_lib

    body = md_lib.markdown(markdown_text, extensions=["tables", "toc"])
    today = datetime.date.today().strftime("%Y-%m-%d")

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
    <p>報告類型：{', '.join(t.upper() for t in state.report_types)}</p>
    <p>生成日期：{today}</p>
    <p>報告編號：{state.run_id}</p>
</div>
{body}
</body>
</html>"""


def generate_pdf(state: PipelineState) -> str | None:
    md_path = REVISED_DIR / f"{state.run_id}.md"
    if not md_path.exists():
        print(f"[ERR] 修訂報告不存在: {md_path}")
        return None

    company_suffix = "_".join(state.companies)
    year_suffix = "_".join(str(y) for y in state.years)
    output_filename = f"{state.run_id}_{company_suffix}_{year_suffix}.pdf"
    output_path = OUTPUTS_DIR / output_filename

    if output_path.exists():
        print(f"[EXIST] PDF 已存在: {output_path.name}")
        return str(output_path)

    md_text = md_path.read_text(encoding="utf-8")
    html_content = _build_html(md_text, state)

    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(str(output_path))
        print(f"[OK] PDF 報告已產生: {output_path.name}")
        return str(output_path)
    except ImportError:
        html_path = output_path.with_suffix(".html")
        html_path.write_text(html_content, encoding="utf-8")
        print(f"[WARN] weasyprint 未安裝，改儲存為 HTML: {html_path.name}")
        return str(html_path)
    except Exception as e:
        print(f"[ERR] PDF 產生失敗: {e}")
        return None


def create_delivery_task(agent: Agent, state: PipelineState) -> Task:
    return Task(
        description=(
            "請將修訂後的報告文稿轉換為最終 PDF。\n"
            f"修訂文稿路徑：data/revised/{state.run_id}.md\n"
            f"輸出至 outputs/{state.run_id}_*.pdf"
        ),
        expected_output="最終 PDF 報告路徑",
        agent=agent,
    )
