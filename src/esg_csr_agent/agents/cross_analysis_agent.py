from __future__ import annotations

"""
Cross Analysis Agent
====================
Compares ESG and CSR findings for the same company.
"""

import json

from crewai import Agent, Task

from esg_csr_agent.config import OPENAI_MODEL_NAME, ANALYSIS_DIR


def create_cross_analysis_agent() -> Agent:
    return Agent(
        role="交叉分析代理",
        goal="比對同一公司的 ESG 與 CSR 分析結果，找出一致性、矛盾與缺口。",
        backstory=(
            "你是專門負責交叉比對 ESG 與 CSR 報告書分析結果的代理。"
            "你會檢查兩份報告中的事實矛盾（如不同的排放數據）、"
            "重大議題的對齊程度、僅出現在一份報告中的聲明、"
            "以及多年度數據的趨勢一致性。"
            "高嚴重度的矛盾會被標記供驗證閘門代理審查。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


def cross_analyze(company_id: str, year: int) -> dict:
    output_path = ANALYSIS_DIR / f"{company_id}_{year}_cross.json"

    if output_path.exists():
        print(f"[EXIST] 交叉分析結果已存在: {output_path.name}")
        return json.loads(output_path.read_text(encoding="utf-8"))

    esg_path = ANALYSIS_DIR / f"{company_id}_{year}_esg.json"
    csr_path = ANALYSIS_DIR / f"{company_id}_{year}_csr.json"

    esg_data = json.loads(esg_path.read_text(encoding="utf-8")) if esg_path.exists() else None
    csr_data = json.loads(csr_path.read_text(encoding="utf-8")) if csr_path.exists() else None

    cross_result: dict = {
        "company_id": company_id,
        "year": year,
        "type": "cross",
        "esg_available": esg_data is not None,
        "csr_available": csr_data is not None,
        "contradictions": [],
        "alignments": [],
        "gaps": [],
        "flags": [],
    }

    if esg_data and csr_data:
        esg_env = esg_data.get("dimensions", {}).get("environmental", {})
        csr_env = csr_data.get("dimensions", {}).get("environmental_stewardship", {})
        if esg_env and csr_env:
            cross_result["alignments"].append({
                "dimension": "環境",
                "note": "ESG 與 CSR 報告均涵蓋環境面向",
                "esg_chunks": esg_env.get("retrieved_chunks", 0),
                "csr_chunks": csr_env.get("retrieved_chunks", 0),
            })

        esg_social = esg_data.get("dimensions", {}).get("social", {})
        csr_employee = csr_data.get("dimensions", {}).get("employee_relations", {})
        if esg_social and csr_employee:
            cross_result["alignments"].append({
                "dimension": "社會/員工",
                "note": "ESG 社會面向與 CSR 員工關係可交叉比對",
                "esg_chunks": esg_social.get("retrieved_chunks", 0),
                "csr_chunks": csr_employee.get("retrieved_chunks", 0),
            })

        if not csr_data.get("dimensions", {}).get("stakeholder_engagement", {}).get("retrieved_chunks"):
            cross_result["gaps"].append({
                "dimension": "利害關係人溝通",
                "source": "csr",
                "note": "CSR 報告中未找到相關內容",
            })
    elif not esg_data:
        cross_result["gaps"].append({
            "dimension": "全部", "source": "esg",
            "note": f"ESG 分析結果不存在: {company_id}_{year}_esg.json",
        })
    elif not csr_data:
        cross_result["gaps"].append({
            "dimension": "全部", "source": "csr",
            "note": f"CSR 分析結果不存在: {company_id}_{year}_csr.json",
        })

    output_path.write_text(json.dumps(cross_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 交叉分析完成: {output_path.name}")
    return cross_result


def create_cross_analysis_task(agent: Agent, company_id: str, year: int) -> Task:
    return Task(
        description=(
            f"請對公司 {company_id} 的 {year} 年度進行 ESG 與 CSR 交叉分析。\n\n"
            "比對項目：\n"
            "1. 事實矛盾\n2. 重大議題對齊程度\n3. 僅出現在一份報告中的聲明\n4. 趨勢一致性\n\n"
            f"結果儲存至 data/analysis/{company_id}_{year}_cross.json"
        ),
        expected_output="交叉分析結果（JSON 格式）",
        agent=agent,
    )
