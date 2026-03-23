from __future__ import annotations

"""
User Interaction (UI) Agent
===========================
Extracts structured requirements from the user before the pipeline begins,
and presents the final output at the end.

All user-facing communication is in Chinese (zh).
"""

from crewai import Agent, Task

from esg_csr_agent.config import OPENAI_MODEL_NAME


def create_ui_agent() -> Agent:
    return Agent(
        role="使用者互動代理",
        goal="從使用者取得完整的分析需求，包括公司代號、報告年度、報告範圍，並在管線結束後交付結果。",
        backstory=(
            "你是一位專業的ESG/CSR報告分析系統的使用者介面代理。"
            "你的職責是用中文與使用者溝通，收集分析所需的完整資訊，"
            "包括目標公司代號（公司代號）、報告年度、報告範圍（ESG/CSR/兩者皆是），"
            "以及任何特定的分析重點。當資訊不完整時，你會向使用者澄清。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


def create_collect_requirements_task(agent: Agent, user_input: str) -> Task:
    return Task(
        description=(
            f"使用者輸入：{user_input}\n\n"
            "請從上述輸入中提取以下資訊：\n"
            "1. 公司代號列表（如 2330、2317）\n"
            "2. 報告年度列表\n"
            "3. 報告範圍（esg / csr / both）\n"
            "4. 特定分析重點（如有）\n\n"
            "以 JSON 格式回傳結果，包含以下欄位：\n"
            "- companies: 公司代號列表\n"
            "- years: 年度列表\n"
            "- report_types: 報告類型列表 (esg/csr)\n"
            "- focus_areas: 特定分析重點列表（可為空）"
        ),
        expected_output="JSON 格式的結構化使用者需求",
        agent=agent,
    )


def create_deliver_result_task(agent: Agent, output_path: str, summary: str) -> Task:
    return Task(
        description=(
            f"分析管線已完成。最終報告路徑：{output_path}\n"
            f"分析摘要：{summary}\n\n"
            "請向使用者報告分析結果，包括：\n"
            "1. 報告已完成的確認\n"
            "2. 報告檔案的路徑\n"
            "3. 分析的簡要摘要"
        ),
        expected_output="中文的使用者友善結果摘要",
        agent=agent,
    )
