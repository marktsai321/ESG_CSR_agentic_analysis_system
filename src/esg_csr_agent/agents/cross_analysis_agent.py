from __future__ import annotations

"""
Cross Analysis Agent
====================
Compares ESG and CSR findings for the same company.
"""

import json

from crewai import Agent, Task

from esg_csr_agent.config import OPENAI_MODEL_NAME, ANALYSIS_DIR
from esg_csr_agent.llm_client import chat_completion


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


def _cross_analyze_with_llm(company_id: str, year: int, esg_data: dict, csr_data: dict) -> dict:
    """Use LLM to compare ESG and CSR analysis findings."""
    esg_summary = {}
    for dim, info in esg_data.get("dimensions", {}).items():
        esg_summary[dim] = {"findings": info.get("findings", ""), "metrics": info.get("metrics", {})}

    csr_summary = {}
    for dim, info in csr_data.get("dimensions", {}).items():
        csr_summary[dim] = {"findings": info.get("findings", ""), "metrics": info.get("metrics", {})}

    prompt = (
        f"你是 ESG/CSR 交叉分析專家。以下是公司 {company_id} 的 {year} 年度分析結果：\n\n"
        f"=== ESG 分析 ===\n{json.dumps(esg_summary, ensure_ascii=False, indent=2)}\n\n"
        f"=== CSR 分析 ===\n{json.dumps(csr_summary, ensure_ascii=False, indent=2)}\n\n"
        "請比對兩份分析結果，以 JSON 格式回傳：\n"
        '1. "contradictions": 列表，每項包含 {"dimension": str, "note": str, "severity": "high"|"medium"|"low"}\n'
        '2. "alignments": 列表，每項包含 {"dimension": str, "note": str}\n'
        '3. "gaps": 列表，每項包含 {"dimension": str, "source": "esg"|"csr", "note": str}\n\n'
        "請只回傳 JSON，不要加 markdown 標記或其他文字。"
    )

    raw = chat_completion(prompt, temperature=0.2, max_tokens=2000).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"contradictions": [], "alignments": [], "gaps": []}


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
        llm_result = _cross_analyze_with_llm(company_id, year, esg_data, csr_data)
        cross_result["contradictions"] = llm_result.get("contradictions", [])
        cross_result["alignments"] = llm_result.get("alignments", [])
        cross_result["gaps"] = llm_result.get("gaps", [])
        # Propagate high-severity contradictions as flags
        cross_result["flags"] = [
            c for c in cross_result["contradictions"]
            if c.get("severity") == "high"
        ]
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
