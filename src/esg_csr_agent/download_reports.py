from __future__ import annotations

"""
Unified entry point for the agentic ESG/CSR report download pipeline.

  python -m esg_csr_agent.download_reports --type both --year 2023 --companies 2330 2317
"""

import argparse
import time
from pathlib import Path

from esg_csr_agent.report_utils import ROOT


def run(
    report_type: str = "both",
    year: int | None = None,
    company_codes: list[str] | None = None,
    jobs: int = 8,
    fallback_url: bool = False,
) -> dict:
    """
    Programmatic entry point for the orchestrator agent.

    Returns a summary dict:
    {
        "esg": {"downloaded": [paths], "failed": [company_ids]},
        "csr": {"downloaded": [paths], "failed": [company_ids]},
    }
    """
    results: dict = {}

    if report_type in ("esg", "both"):
        from esg_csr_agent import download_esg_pdfs as esg_mod
        esg_year = year if year is not None else 2024
        print(f"\n{'='*60}")
        print(f"ESG 報告書下載  年度={esg_year}  公司代號={company_codes or '全部'}")
        print(f"{'='*60}")
        rows = esg_mod._fetch_rows_from_page(year=esg_year)
        if company_codes:
            rows = [r for r in rows if r.get("company_id") in company_codes]
        downloaded, failed = _run_downloads(rows, esg_mod.download_one_with_retry, jobs, fallback_url)
        results["esg"] = {"downloaded": downloaded, "failed": failed}
        _write_failed_csv(failed, "esg", esg_year)

    if report_type in ("csr", "both"):
        from esg_csr_agent import download_csr_pdfs as csr_mod
        csr_year = year if year is not None else 2020
        print(f"\n{'='*60}")
        print(f"CSR 報告書下載  年度={csr_year}  公司代號={company_codes or '全部'}")
        print(f"{'='*60}")
        rows = csr_mod._fetch_rows_from_page(year=csr_year)
        if company_codes:
            rows = [r for r in rows if r.get("company_id") in company_codes]
        downloaded, failed = _run_downloads(rows, csr_mod.download_one_with_retry, jobs, fallback_url)
        results["csr"] = {"downloaded": downloaded, "failed": failed}
        _write_failed_csv(failed, "csr", csr_year)

    return results


def _run_downloads(
    rows: list[dict],
    download_fn,
    jobs: int,
    fallback_url: bool,
) -> tuple[list[str], list[str]]:
    downloaded: list[str] = []
    failed:     list[str] = []

    def _do(row: dict):
        result = download_fn(row, platform_only=not fallback_url)
        if result is not None:
            downloaded.append(str(result))
        else:
            failed.append(row.get("company_id", "?"))

    t0 = time.perf_counter()
    if jobs <= 1:
        for row in rows:
            _do(row)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            list(ex.map(_do, rows))

    elapsed = time.perf_counter() - t0
    print(f"\n  → 完成 {len(rows)} 筆，成功 {len(downloaded)}，失敗 {len(failed)}，耗時 {elapsed:.1f} 秒")
    return downloaded, failed


def _write_failed_csv(failed_ids: list[str], report_type: str, year: int) -> None:
    if not failed_ids:
        return
    path = ROOT / "logs" / f"failed_{report_type}_{year}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company_id"])
        for cid in failed_ids:
            w.writerow([cid])
    print(f"  → 失敗清單已存至 {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="統一入口：下載 ESG（TWSE ESG+）或 CSR（MOPS）報告書 PDF",
    )
    parser.add_argument("--type", dest="report_type", default="both",
                        choices=["esg", "csr", "both"], help="下載類型（預設 both）")
    parser.add_argument("--year", type=int, default=None, help="西元年度")
    parser.add_argument("--companies", nargs="+", metavar="CODE", help="公司代號")
    parser.add_argument("--fallback-url", action="store_true", help="平台無檔時改抓公司網址")
    parser.add_argument("-j", "--jobs", type=int, default=8, metavar="N", help="並行下載數（預設 8）")
    args = parser.parse_args()

    summary = run(
        report_type=args.report_type,
        year=args.year,
        company_codes=args.companies,
        jobs=args.jobs,
        fallback_url=args.fallback_url,
    )

    print("\n" + "=" * 60)
    print("下載摘要")
    print("=" * 60)
    for rtype, info in summary.items():
        print(f"  {rtype.upper()}: 成功 {len(info['downloaded'])} 筆，失敗 {len(info['failed'])} 筆")


if __name__ == "__main__":
    main()
