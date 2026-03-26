from __future__ import annotations

"""
Validation Gate Agent
=====================
Quality checkpoint — go/no-go decision only.
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
            "你驗證所有請求的公司和年度都有分析輸出、"
            "沒有分析面向的信心分數低於門檻值、"
            "且交叉分析中沒有未解決的高嚴重度矛盾。"
            "你只做通過/不通過的決定，不修復問題。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


def validate(state: PipelineState) -> dict:
    checks: list[dict] = []
    extracted = state.files.get("extracted_text", {})

    for company in state.companies:
        for year in state.years:
            for rtype in state.report_types:
                key = state.file_key(company, year, rtype)
                # Skip validation for report types that had no source PDF/text
                if key not in extracted:
                    print(f"[驗證] 跳過 {key}（無原始資料可供分析）")
                    continue
                analysis_path = ANALYSIS_DIR / f"{key}.json"
                exists = analysis_path.exists()
                checks.append({
                    "name": f"分析檔案存在: {key}",
                    "passed": exists,
                    "detail": str(analysis_path) if exists else f"缺少檔案: {analysis_path.name}",
                })

    for company in state.companies:
        for year in state.years:
            for rtype in state.report_types:
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

    for company in state.companies:
        for year in state.years:
            esg_key = state.file_key(company, year, "esg")
            csr_key = state.file_key(company, year, "csr")
            has_esg = esg_key in state.files.get("analysis", {})
            has_csr = csr_key in state.files.get("analysis", {})

            if not (has_esg and has_csr):
                print(f"[驗證] 跳過 {company}_{year} 交叉分析檢查（ESG/CSR 非同時可用）")
                continue
            cross_path = ANALYSIS_DIR / f"{company}_{year}_cross.json"
            if not cross_path.exists():
                checks.append({
                    "name": f"交叉分析: {company}_{year}",
                    "passed": False,
                    "detail": "交叉分析結果不存在",
                })
                continue
            try:
                cross = json.loads(cross_path.read_text(encoding="utf-8"))
                high_sev = [f for f in cross.get("flags", []) if f.get("severity") == "high"]
                passed = len(high_sev) == 0
                checks.append({
                    "name": f"交叉分析高嚴重度: {company}_{year}",
                    "passed": passed,
                    "detail": f"{len(high_sev)} 個高嚴重度矛盾" if not passed else "無高嚴重度矛盾",
                })
            except Exception as e:
                checks.append({
                    "name": f"讀取交叉分析: {company}_{year}",
                    "passed": False,
                    "detail": f"讀取失敗: {e}",
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
            f"年度：{', '.join(str(y) for y in state.years)}\n"
            f"報告類型：{', '.join(state.report_types)}\n\n"
            f"信心門檻：{CONFIDENCE_THRESHOLD}"
        ),
        expected_output="驗證結果（通過/不通過）",
        agent=agent,
    )
