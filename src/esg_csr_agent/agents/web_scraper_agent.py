from __future__ import annotations

"""
Web Scraper Agent
=================
Downloads ESG and CSR PDF reports from official Taiwan regulatory platforms.
Delegates actual downloading to download_reports.py.
"""

from crewai import Agent, Task

from esg_csr_agent.config import OPENAI_MODEL_NAME, RAW_PDF_DIR, LOGS_DIR


def create_web_scraper_agent() -> Agent:
    return Agent(
        role="網頁爬蟲代理",
        goal="從官方平台（TWSE ESG+ 及 MOPS）下載指定公司的 ESG 及 CSR 報告書 PDF。",
        backstory=(
            "你是專門負責從台灣官方平台下載永續報告書的代理。"
            "你使用 TWSE ESG+ 平台下載 ESG 報告，使用 MOPS 公開資訊觀測站下載 CSR 報告。"
            "下載前會先檢查本地是否已有檔案，避免重複下載。"
            "所有下載失敗的紀錄會寫入 logs/ 供失敗處理代理使用。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


def run_download(
    companies: list[str],
    report_types: list[str],
    years: list[int],
    fallback_url: bool = False,
    jobs: int = 8,
) -> dict:
    """Programmatic entry point called by the Orchestrator."""
    from esg_csr_agent import download_reports

    all_results: dict = {}

    for year in years:
        for rtype in report_types:
            result = download_reports.run(
                report_type=rtype,
                year=year,
                company_codes=companies if companies else None,
                jobs=jobs,
                fallback_url=fallback_url,
            )
            for key, info in result.items():
                if key not in all_results:
                    all_results[key] = {"downloaded": [], "failed": []}
                all_results[key]["downloaded"].extend(info["downloaded"])
                all_results[key]["failed"].extend(info["failed"])

    return all_results


def create_download_task(
    agent: Agent,
    companies: list[str],
    report_types: list[str],
    years: list[int],
) -> Task:
    return Task(
        description=(
            f"下載以下公司的報告書 PDF：\n"
            f"公司代號：{', '.join(companies)}\n"
            f"報告類型：{', '.join(report_types)}\n"
            f"年度：{', '.join(str(y) for y in years)}\n\n"
            "請使用 download_reports.py 進行下載，並回報：\n"
            "1. 成功下載的檔案路徑\n"
            "2. 下載失敗的公司代號\n"
            "3. 失敗原因"
        ),
        expected_output="下載結果摘要（JSON 格式，含成功與失敗清單）",
        agent=agent,
    )
