from __future__ import annotations

"""
Validation Gate Agent
=====================
Quality checkpoint — go/no-go decision only.
Validates analysis completeness, confidence thresholds, and scoring consistency.
No cross-analysis checks (Cross Analysis Agent has been removed).
"""

import json

from crewai import Agent, Task

from esg_csr_agent.config import OPENAI_MODEL_NAME, ANALYSIS_DIR, CONFIDENCE_THRESHOLD
from esg_csr_agent.pipeline_state import PipelineState


def create_validation_gate_agent() -> Agent:
    return Agent(
        role="驗證閘門代理",
        goal="在報告產生前進行品質檢查，確保所有分析結果完整且達到信心門檻。",
        backstory=(
            "你是管線的品質守門員。"
            "你驗證所有請求的公司都有至少一份分析輸出（ESG 或 CSR）、"
            "沒有分析面向的信心分數低於門檻值、"
            "且評分計算與最低分公司選取一致。"
            "你只做通過/不通過的決定，不修復問題。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


def validate(state: PipelineState) -> dict:
    checks: list[dict] = []
    extracted = state.files.get("extracted_text", {})

    # Check 1: Every company has at least one analysis output
    for company in state.companies:
        has_any = False
        for year in state.years:
            for rtype in ["esg", "csr"]:
                key = state.file_key(company, year, rtype)
                if key not in extracted:
                    continue
                analysis_path = ANALYSIS_DIR / f"{key}.json"
                if analysis_path.exists():
                    has_any = True
        checks.append({
            "name": f"公司分析存在: {company}",
            "passed": has_any,
            "detail": "至少一份分析結果存在" if has_any else f"公司 {company} 無任何分析結果",
        })

    # Check 2: Confidence thresholds per dimension
    for company in state.companies:
        for year in state.years:
            for rtype in ["esg", "csr"]:
                key = state.file_key(company, year, rtype)
                if key not in extracted:
                    continue
                analysis_path = ANALYSIS_DIR / f"{key}.json"
                if not analysis_path.exists():
                    continue
                try:
                    data = json.loads(analysis_path.read_text(encoding="utf-8"))
                    for dim_name, dim_data in data.get("dimensions", {}).items():
                        conf = dim_data.get("confidence", 0.0)
                        passed = conf >= CONFIDENCE_THRESHOLD
                        checks.append({
                            "name": f"信心分數: {key}/{dim_name}",
                            "passed": passed,
                            "detail": f"confidence={conf:.2f} (門檻={CONFIDENCE_THRESHOLD})",
                        })
                except Exception as e:
                    checks.append({
                        "name": f"讀取分析結果: {key}",
                        "passed": False,
                        "detail": f"讀取失敗: {e}",
                    })

    # Check 3: Scoring consistency — verify worst_company matches scores
    if state.scores and state.worst_company:
        actual_worst = min(state.scores, key=lambda c: state.scores[c]["total"])
        consistent = actual_worst == state.worst_company
        checks.append({
            "name": "評分一致性: worst_company",
            "passed": consistent,
            "detail": (
                f"最低分公司 {state.worst_company} (總分: {state.scores[state.worst_company]['total']})"
                if consistent
                else f"不一致: state={state.worst_company}, 計算={actual_worst}"
            ),
        })
    elif not state.scores:
        checks.append({
            "name": "評分一致性: scores",
            "passed": False,
            "detail": "無評分資料",
        })

    all_passed = all(c["passed"] for c in checks)
    result = {"passed": all_passed, "checks": checks}
    print(f"[驗證] {'通過' if all_passed else '未通過'} ({sum(1 for c in checks if c['passed'])}/{len(checks)} 項通過)")
    return result


def create_validation_task(agent: Agent, state: PipelineState) -> Task:
    return Task(
        description=(
            "請執行管線品質驗證。\n"
            f"公司：{', '.join(state.companies)}\n"
            f"年度：{', '.join(str(y) for y in state.years)}\n\n"
            f"信心門檻：{CONFIDENCE_THRESHOLD}"
        ),
        expected_output="驗證結果（通過/不通過）",
        agent=agent,
    )
