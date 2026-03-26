from __future__ import annotations

"""
ESG/CSR Report Analysis System — Main Entry Point
==================================================

Usage:
    # After installation:
    esg-csr-agent                                        # interactive mode
    esg-csr-agent --companies 2330 2317 --years 2023     # direct mode
    esg-csr-agent --companies 2330 --years 2023 --types esg

    # Or via Python module:
    python -m esg_csr_agent --companies 2330 --years 2023
"""

import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv


def _upsert_env_key(env_path: str, key_name: str, key_value: str) -> None:
    """Insert or update a single key=value entry in .env."""
    if os.path.exists(env_path):
        content = open(env_path, "r", encoding="utf-8").read()
        if re.search(rf"^{key_name}\s*=", content, flags=re.MULTILINE):
            content = re.sub(
                rf"^{key_name}\s*=.*$",
                f"{key_name}={key_value}",
                content,
                flags=re.MULTILINE,
            )
        else:
            if content and not content.endswith("\n"):
                content += "\n"
            content += f"{key_name}={key_value}\n"
        open(env_path, "w", encoding="utf-8").write(content)
    else:
        open(env_path, "w", encoding="utf-8").write(f"{key_name}={key_value}\n")


def ensure_api_key() -> None:
    """Check for required API keys and prompt if missing."""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    env_path = os.path.join(os.getcwd(), ".env")

    if not openai_key or openai_key == "your-api-key-here":
        print("=" * 60)
        print(" OPENAI_API_KEY 尚未設定")
        print("=" * 60)
        print()
        print("本系統需要 OpenAI/Anthropic 金鑰執行分析。")
        print("支援 OpenAI (sk-...) 及 Anthropic (sk-ant-...) 金鑰。")
        print()
        openai_key = input("請輸入 OPENAI_API_KEY（直接按 Enter 跳過）：").strip()
        if not openai_key:
            print("\n[錯誤] 未提供 OPENAI_API_KEY，無法繼續。")
            print("請設定後重新執行：")
            print("  OPENAI_API_KEY=sk-...")
            print("  esg-csr-agent")
            sys.exit(1)
        _upsert_env_key(env_path, "OPENAI_API_KEY", openai_key)
        os.environ["OPENAI_API_KEY"] = openai_key
        print("[OK] OPENAI_API_KEY 已儲存至 .env\n")

    if not gemini_key or gemini_key == "your-gemini-api-key-here":
        print("=" * 60)
        print(" GEMINI_API_KEY 尚未設定")
        print("=" * 60)
        print()
        print("本系統 Revision 階段使用 Gemini 模型。")
        print()
        gemini_key = input("請輸入 GEMINI_API_KEY（直接按 Enter 跳過）：").strip()
        if not gemini_key:
            print("\n[錯誤] 未提供 GEMINI_API_KEY，無法繼續。")
            print("請設定後重新執行：")
            print("  GEMINI_API_KEY=AIza...")
            print("  esg-csr-agent")
            sys.exit(1)
        _upsert_env_key(env_path, "GEMINI_API_KEY", gemini_key)
        os.environ["GEMINI_API_KEY"] = gemini_key
        print("[OK] GEMINI_API_KEY 已儲存至 .env\n")

    if "REVISION_MODEL_NAME" not in os.environ:
        _upsert_env_key(env_path, "REVISION_MODEL_NAME", "gemini-2.5-flash")
        os.environ["REVISION_MODEL_NAME"] = "gemini-2.5-flash"
        print("[OK] REVISION_MODEL_NAME 已設定為 gemini-2.5-flash\n")

    if not openai_key or not gemini_key:
        print("\n[錯誤] API Key 設定不完整，無法繼續。")
        print("請設定後重新執行：")
        print("  OPENAI_API_KEY=sk-...")
        print("  GEMINI_API_KEY=AIza...")
        print("  esg-csr-agent")
        sys.exit(1)


def interactive_mode():
    """Collect requirements interactively (simplified UI Agent behaviour)."""
    from esg_csr_agent.pipeline_state import PipelineState

    print("=" * 60)
    print("ESG/CSR 報告書分析系統")
    print("=" * 60)
    print()

    while True:
        raw = input("請輸入公司代號（空白分隔，如 2330 2317）：").strip()
        if raw:
            companies = raw.split()
            break
        print("請至少輸入一個公司代號。")

    while True:
        raw = input("請輸入報告年度（空白分隔，如 2023）：").strip()
        if raw:
            try:
                years = [int(y) for y in raw.split()]
                break
            except ValueError:
                print("請輸入有效的年度數字。")
        else:
            print("請至少輸入一個年度。")

    raw = input("報告範圍 [esg/csr/both]（預設 both）：").strip().lower()
    if raw in ("esg", "csr"):
        report_types = [raw]
    else:
        report_types = ["esg", "csr"]

    state = PipelineState(
        companies=companies,
        years=years,
        report_types=report_types,
    )

    print(f"\n確認分析需求：")
    print(f"  公司代號：{', '.join(companies)}")
    print(f"  報告年度：{', '.join(str(y) for y in years)}")
    print(f"  報告類型：{', '.join(report_types)}")
    print()

    return state


def direct_mode(args: argparse.Namespace):
    """Build state directly from CLI arguments."""
    from esg_csr_agent.pipeline_state import PipelineState

    report_types = []
    if args.types in ("esg", "both"):
        report_types.append("esg")
    if args.types in ("csr", "both"):
        report_types.append("csr")

    return PipelineState(
        companies=args.companies,
        years=args.years,
        report_types=report_types,
    )


def main() -> None:
    # Validate API key first (before heavy imports)
    ensure_api_key()

    from esg_csr_agent.agents.orchestrator import Pipeline
    from esg_csr_agent.config import LOGS_DIR

    parser = argparse.ArgumentParser(
        description="ESG/CSR 報告書分析系統",
    )
    parser.add_argument(
        "--companies", nargs="+", metavar="CODE",
        help="公司代號（空白分隔）",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, metavar="YEAR",
        help="報告年度",
    )
    parser.add_argument(
        "--types", default="both", choices=["esg", "csr", "both"],
        help="報告類型（預設 both）",
    )
    args = parser.parse_args()

    # Choose mode
    if args.companies and args.years:
        state = direct_mode(args)
    else:
        state = interactive_mode()

    # Run pipeline
    pipeline = Pipeline(state)
    final_state = pipeline.run()

    # Summary
    print("\n" + "=" * 60)
    print("管線執行完成")
    print("=" * 60)
    print(f"  狀態：{final_state.stage}")
    print(f"  驗證：{'通過' if final_state.validation_passed else '未通過'}")
    print(f"  失敗次數：{len(final_state.failures)}")
    if final_state.output_path:
        print(f"  輸出報告：{final_state.output_path}")
    else:
        print("  輸出報告：未產生")

    if final_state.failures:
        print(f"\n失敗紀錄：")
        for f in final_state.failures:
            print(f"  - [{f['agent']}] {f['step']}: {f['error']}")

    # Dump state for debugging
    state_path = LOGS_DIR / f"pipeline_state_{final_state.run_id}.json"
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(final_state.to_dict(), fh, ensure_ascii=False, indent=2)
    print(f"\n管線狀態已儲存至 {state_path}")


if __name__ == "__main__":
    main()
