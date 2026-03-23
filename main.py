from __future__ import annotations

"""
ESG/CSR Report Analysis System — Main Entry Point
==================================================

Usage:
    # Interactive mode (UI Agent collects requirements)
    python main.py

    # Direct mode (skip UI Agent)
    python main.py --companies 2330 2317 --years 2023 --types both

    # ESG only for a single company
    python main.py --companies 2330 --years 2023 --types esg
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv


def ensure_api_key() -> None:
    """Check for OPENAI_API_KEY; prompt the user if it is missing or placeholder."""
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY", "")
    if key and key != "your-api-key-here":
        return

    print("=" * 60)
    print(" OPENAI_API_KEY 尚未設定")
    print("=" * 60)
    print()
    print("本系統需要 OpenAI API Key 才能執行分析。")
    print("您可以：")
    print("  1. 將 Key 寫入 .env 檔案（建議）")
    print("  2. 設定環境變數 export OPENAI_API_KEY=sk-...")
    print("  3. 現在直接輸入")
    print()

    api_key = input("請輸入 OpenAI API Key（直接按 Enter 跳過）：").strip()
    if not api_key:
        print("\n[錯誤] 未提供 API Key，無法繼續。")
        print("請設定後重新執行：")
        print("  export OPENAI_API_KEY=sk-...")
        print("  python main.py")
        sys.exit(1)

    # Persist to .env so the user doesn't have to enter it again
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        content = open(env_path, "r", encoding="utf-8").read()
        if "OPENAI_API_KEY" in content:
            content = content.replace("your-api-key-here", api_key)
            # Also handle empty value case
            import re
            content = re.sub(r"OPENAI_API_KEY=\s*\n", f"OPENAI_API_KEY={api_key}\n", content)
        else:
            content += f"\nOPENAI_API_KEY={api_key}\n"
        open(env_path, "w", encoding="utf-8").write(content)
    else:
        open(env_path, "w", encoding="utf-8").write(f"OPENAI_API_KEY={api_key}\n")

    os.environ["OPENAI_API_KEY"] = api_key
    print("[OK] API Key 已儲存至 .env\n")


# Validate API key before importing config (which reads .env at import time)
ensure_api_key()

from pipeline_state import PipelineState
from agents.orchestrator import Pipeline


def interactive_mode() -> PipelineState:
    """Collect requirements interactively (simplified UI Agent behaviour)."""
    print("=" * 60)
    print("ESG/CSR 報告書分析系統")
    print("=" * 60)
    print()

    # Collect company codes
    while True:
        raw = input("請輸入公司代號（空白分隔，如 2330 2317）：").strip()
        if raw:
            companies = raw.split()
            break
        print("請至少輸入一個公司代號。")

    # Collect years
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

    # Collect report types
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


def direct_mode(args: argparse.Namespace) -> PipelineState:
    """Build state directly from CLI arguments."""
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
    state_path = f"logs/pipeline_state_{final_state.run_id}.json"
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(final_state.to_dict(), fh, ensure_ascii=False, indent=2)
    print(f"\n管線狀態已儲存至 {state_path}")


if __name__ == "__main__":
    main()
