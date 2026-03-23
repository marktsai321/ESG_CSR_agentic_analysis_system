from __future__ import annotations

"""
Fail Handler Agent
==================
Diagnoses failures and proposes recovery actions. Advisory only.
"""

from crewai import Agent, Task

from esg_csr_agent.config import OPENAI_MODEL_NAME


def create_fail_handler_agent() -> Agent:
    return Agent(
        role="失敗處理代理",
        goal="診斷管線中任何階段的失敗原因，並向協調者提出結構化的復原建議。",
        backstory=(
            "你是管線的故障診斷專家。"
            "當任何代理逾時、發生例外或反覆重試時，協調者會啟動你。"
            "你會將失敗分類並提出具體的復原建議。"
            "你只提供建議，不直接呼叫其他代理。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


FAILURE_CATEGORIES = {
    "scrape_failure": {
        "pattern_keywords": ["download", "scrape", "HTTP", "timeout", "網頁", "下載"],
        "description": "PDF 無法從平台下載",
        "proposals": [
            {"action": "skip", "detail": "跳過此公司，繼續處理其他公司"},
            {"action": "fallback_url", "detail": "啟用 --fallback-url 從公司網站下載"},
            {"action": "retry", "detail": "等待後重試下載"},
        ],
    },
    "ocr_failure": {
        "pattern_keywords": ["extract", "OCR", "text", "擷取", "掃描"],
        "description": "PDF 為掃描檔或損毀，無法擷取文字",
        "proposals": [
            {"action": "flag_manual", "detail": "標記為需人工審閱"},
            {"action": "skip", "detail": "跳過此檔案"},
        ],
    },
    "analysis_loop": {
        "pattern_keywords": ["analysis", "loop", "cycling", "分析", "迴圈"],
        "description": "分析代理反覆執行但未產出結果",
        "proposals": [
            {"action": "re_prompt_narrow", "detail": "以較窄的上下文視窗重新執行分析"},
            {"action": "skip_dimension", "detail": "跳過問題面向"},
        ],
    },
    "confidence_failure": {
        "pattern_keywords": ["confidence", "threshold", "信心", "門檻"],
        "description": "分析信心分數低於門檻",
        "proposals": [
            {"action": "re_run_retrieval", "detail": "使用不同的檢索策略重新執行"},
            {"action": "lower_threshold", "detail": "暫時降低門檻值（需人工確認）"},
        ],
    },
    "dependency_failure": {
        "pattern_keywords": ["missing", "upstream", "file not found", "缺少", "不存在"],
        "description": "上游必要檔案缺失",
        "proposals": [
            {"action": "re_run_upstream", "detail": "重新執行上游代理"},
        ],
    },
}


def diagnose(failure: dict) -> dict:
    error_lower = (failure.get("error", "") + " " + failure.get("step", "")).lower()

    matched_category = "unknown"
    matched_info = {
        "description": "未知失敗類型",
        "proposals": [{"action": "escalate", "detail": "無法自動分類，建議人工介入"}],
    }

    for cat_name, cat_info in FAILURE_CATEGORIES.items():
        for keyword in cat_info["pattern_keywords"]:
            if keyword.lower() in error_lower:
                matched_category = cat_name
                matched_info = cat_info
                break
        if matched_category != "unknown":
            break

    result = {
        "category": matched_category,
        "description": matched_info["description"],
        "proposals": matched_info["proposals"],
        "original_failure": failure,
    }

    print(f"[診斷] 類別={matched_category}: {matched_info['description']}")
    for p in matched_info["proposals"]:
        print(f"  → 建議: {p['action']} — {p['detail']}")

    return result


def create_diagnosis_task(agent: Agent, failure: dict) -> Task:
    return Task(
        description=(
            f"請診斷以下管線失敗：\n\n"
            f"失敗代理：{failure.get('agent', '未知')}\n"
            f"失敗步驟：{failure.get('step', '未知')}\n"
            f"錯誤訊息：{failure.get('error', '未知')}\n\n"
            "請分類失敗原因並提出復原建議。"
        ),
        expected_output="失敗診斷結果及復原建議",
        agent=agent,
    )
