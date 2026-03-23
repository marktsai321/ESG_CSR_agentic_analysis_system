from __future__ import annotations

"""
下載 MOPS 永續報告書 / CSR 報告書。

平台：公開資訊觀測站 (MOPS) — mopsov.twse.com.tw
涵蓋年度：~2013（ROC 102）至今（包含舊版 CSR 與新版永續報告書）
API 流程：
  1. POST https://mops.twse.com.tw/mops/api/redirectToOld
       body: {"apiName": "ajax_t100sb11", "parameters": {...}}
     → 回傳 {"result": {"url": "https://mopsov.twse.com.tw/mops/web/ajax_t100sb11?parameters=HASH"}}
  2. GET mopsov URL → HTML 表格，內含各公司的 PDF 下載連結
  3. GET https://mopsov.twse.com.tw/server-java/FileDownLoad?step=9&filePath=...&fileName=...
     → PDF 位元流

  python download_csr_pdfs.py              # 取上市+上櫃，預設 2020 年度
  python download_csr_pdfs.py --csv        # 使用既有 CSV 清單
  python download_csr_pdfs.py --year 2019  # 指定年度（西元年）
  python download_csr_pdfs.py --fallback-url  # 失敗時改抓公司網址
  python download_csr_pdfs.py -j 8         # 並行下載
"""

import argparse
import csv
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from report_utils import ROOT, get_data_dir, iter_rows, safe_filename

# ── Constants ──────────────────────────────────────────────────────────────────

MOPS_BASE    = "https://mops.twse.com.tw"
MOPSOV_BASE  = "https://mopsov.twse.com.tw"

MOPS_API_REDIRECT  = f"{MOPS_BASE}/mops/api/redirectToOld"
MOPSOV_FILE_STREAM = f"{MOPSOV_BASE}/server-java/FileDownLoad"
MOPSOV_FILE_PATH   = "/home/html/nas/protect/t100/"

MOPS_HEADERS = {
    "Accept":       "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Origin":       MOPS_BASE,
    "Referer":      f"{MOPS_BASE}/mops/t100sb11",
    "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
FILE_STREAM_HEADERS = {
    "Accept":     "application/pdf, */*",
    "Origin":     MOPSOV_BASE,
    "Referer":    f"{MOPSOV_BASE}/mops/web/ajax_t100sb11",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

# Market types: sii=上市, otc=上櫃, rotc=興櫃, pub=公開發行
MARKET_TYPES = ["sii", "otc"]

DEFAULT_CSR_CSV = ROOT / "csr_sources_mops.csv"
DATA_DIR = get_data_dir("csr")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ce_to_roc(ce_year: int) -> str:
    """Convert CE (Gregorian) year to ROC year string."""
    return str(ce_year - 1911)


# ── API / HTML fetching ────────────────────────────────────────────────────────

def _fetch_html_for_year_and_type(year_roc: str, typek: str, timeout: int = 60) -> str | None:
    """
    Two-step fetch:
      1. POST redirectToOld → get mopsov redirect URL
      2. GET redirect URL → return HTML string
    Returns None on any failure.
    """
    payload = {
        "apiName": "ajax_t100sb11",
        "parameters": {
            "encodeURIComponent": 1,
            "step": 1,
            "firstin": True,
            "TYPEK": typek,
            "year": year_roc,
            "co_id": "",
            "skind": "",
        },
    }
    try:
        r = requests.post(MOPS_API_REDIRECT, json=payload, headers=MOPS_HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != 200:
            print(f"  [WARN] redirectToOld 回傳: {data.get('message')}")
            return None
        redirect_url = data["result"]["url"]
    except Exception as e:
        print(f"  [ERR] redirectToOld 呼叫失敗: {e}")
        return None

    try:
        r2 = requests.get(
            redirect_url,
            headers={"User-Agent": FILE_STREAM_HEADERS["User-Agent"]},
            timeout=timeout,
        )
        r2.raise_for_status()
        r2.encoding = "utf-8"
        return r2.text
    except Exception as e:
        print(f"  [ERR] mopsov 取 HTML 失敗: {e}")
        return None


def _parse_html_rows(html: str, year_ce: int) -> list[dict]:
    """
    Parse the mopsov HTML table and return normalised row dicts.

    Table column layout (0-indexed, after expanding rowspan/colspan):
      0  公司代號    1  公司名稱    2  英文簡稱
      3  申報原因    4  產業類別    5  報告書內容涵蓋期間
      6  編製依循準則
      7  驗證單位    8  單位名稱    9  採用標準
      10 公司網站報告書相關資訊(中文版)
      11 中文版永續報告書檔案          ← ZH PDF (FileDownLoad href)
      12 中文版上傳日期
      13 英文版永續報告書網址           ← EN company URL
      14 英文版永續報告書檔案           ← EN PDF (FileDownLoad href)
      15 英文版上傳日期
      16 中文版永續報告書(修正後版本)  ← ZH revised PDF
      17 上傳日期(中文修正後版本)
      18 英文版永續報告書(修正後版本)  ← EN revised PDF
      19 上傳日期(英文修正後版本)
      20 報告書聯絡資訊    21 備註
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    table = soup.find("table")
    if not table:
        return rows

    def _extract_filename(td) -> str:
        """Get the fileName param from a FileDownLoad href, or empty string."""
        if td is None:
            return ""
        link = td.find("a")
        if not link:
            return ""
        href = link.get("href", "")
        m = re.search(r"fileName=([^&\s]+)", href)
        return m.group(1).strip() if m else ""

    def _extract_href(td) -> str:
        if td is None:
            return ""
        link = td.find("a")
        return (link.get("href") or "").strip() if link else ""

    # Skip header rows (class="tblHead")
    data_rows = [
        tr for tr in table.find_all("tr")
        if "tblHead" not in (tr.get("class") or [])
    ]

    for tr in data_rows:
        tds = tr.find_all("td")
        if len(tds) < 12:
            continue

        company_id   = tds[0].get_text(strip=True)
        company_name = tds[1].get_text(strip=True)
        sector       = tds[4].get_text(strip=True) if len(tds) > 4 else ""

        zh_filename = _extract_filename(tds[11] if len(tds) > 11 else None)
        en_url      = _extract_href(tds[13] if len(tds) > 13 else None)
        en_filename = _extract_filename(tds[14] if len(tds) > 14 else None)
        # Prefer revised version (col 16 / 18) if original is missing
        if not zh_filename and len(tds) > 16:
            zh_filename = _extract_filename(tds[16])
        if not en_filename and len(tds) > 18:
            en_filename = _extract_filename(tds[18])

        if zh_filename:
            rows.append({
                "source":          "mops",
                "company_id":      company_id,
                "company_name":    company_name,
                "sector":          sector,
                "year":            str(year_ce),
                "lang":            "zh",
                "url":             "",
                "mops_download_id": zh_filename,
            })
        if en_filename:
            rows.append({
                "source":          "mops",
                "company_id":      company_id,
                "company_name":    company_name,
                "sector":          sector,
                "year":            str(year_ce),
                "lang":            "en",
                "url":             en_url,
                "mops_download_id": en_filename,
            })

    return rows


def _fetch_rows_from_page(year: int = 2020) -> list[dict]:
    """
    Fetch all CSR/sustainability report rows for a given CE year
    across all configured market types.
    """
    year_roc = _ce_to_roc(year)
    all_rows: list[dict] = []
    for typek in MARKET_TYPES:
        print(f"  [MOPS] 取得 {year} 年度 {typek} 清單 (ROC {year_roc})…")
        html = _fetch_html_for_year_and_type(year_roc, typek)
        if html:
            rows = _parse_html_rows(html, year)
            print(f"    → {len(rows)} 筆")
            all_rows.extend(rows)
    return all_rows


# ── Download ───────────────────────────────────────────────────────────────────

def _try_mops_download(filename: str, timeout: int = 60) -> tuple[bytes, str] | tuple[None, str]:
    """
    Download a PDF from mopsov FileDownLoad endpoint.
    Returns (content, content_type) on success, (None, reason) on failure.
    """
    if not filename:
        return (None, "無 filename")
    try:
        resp = requests.get(
            MOPSOV_FILE_STREAM,
            params={"step": "9", "filePath": MOPSOV_FILE_PATH, "fileName": filename},
            headers=FILE_STREAM_HEADERS,
            timeout=timeout,
        )
        if resp.status_code != 200:
            return (None, f"HTTP {resp.status_code}")
        if len(resp.content) < 100:
            return (None, "回傳內容過短")
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "pdf" in ctype or "octet-stream" in ctype or resp.content[:4] == b"%PDF":
            return (resp.content, ctype)
        return (None, f"回傳非 PDF (ctype={ctype[:40]})")
    except requests.exceptions.Timeout:
        return (None, "逾時")
    except Exception as e:
        return (None, str(e)[:50])


def download_one(row: dict, timeout: int = 60, platform_only: bool = True) -> Path | None:
    mops_id = (row.get("mops_download_id") or "").strip()
    url     = (row.get("url") or "").strip()
    if not mops_id and not url:
        print(f"[SKIP] {row.get('company_name')} {row.get('year')} 沒有 URL 或 mops_download_id")
        return None

    target = safe_filename(row, DATA_DIR)
    if target.exists():
        print(f"[EXIST] {target.name}")
        return target

    content = None
    ctype   = ""

    # Primary: platform download via mopsov FileDownLoad
    if mops_id:
        print(f"[平台] {row.get('company_name')} {row.get('year')} {row.get('lang')} file={mops_id}")
        out, msg = _try_mops_download(mops_id, timeout=timeout)
        if out is not None:
            content, ctype = out, msg
            print(f"      → 成功 ({len(content)} bytes)")
        else:
            print(f"      → 失敗: {msg}")

    # Fallback: direct company URL (opt-in with --fallback-url)
    if content is None and url and not platform_only:
        print(f"[網址] {row.get('company_name')} {row.get('year')} {row.get('lang')} ← {url[:60]}...")
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            content = resp.content
            ctype   = (resp.headers.get("Content-Type") or "").lower()
            print(f"      → 成功 ({len(content)} bytes)")
        except Exception as e:
            print(f"      → 失敗: {e}")
            return None

    if content is None:
        return None

    ext = ".pdf" if "pdf" in ctype else (".html" if "html" in ctype or "text/" in ctype else ".bin")
    if not target.name.endswith(ext):
        target = target.with_suffix(ext)

    target.write_bytes(content)
    print(f"[OK  ] → {target}")
    return target


def download_one_with_retry(row: dict, retries: int = 2, **kwargs) -> Path | None:
    for attempt in range(retries + 1):
        result = download_one(row, **kwargs)
        if result is not None:
            return result
        if attempt < retries:
            time.sleep(2 ** attempt)  # 1s, 2s back-off
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="下載 MOPS 永續/CSR 報告書 PDF")
    parser.add_argument("--csv", action="store_true", help="使用既有 CSV 清單，不從 MOPS 取")
    parser.add_argument("--year", type=int, default=2020, help="CE 年度（預設 2020；CSR 有效範圍約 2013–2021）")
    parser.add_argument("--fallback-url", action="store_true", help="平台無檔時改抓公司網址（預設不啟用）")
    parser.add_argument("-j", "--jobs", type=int, default=8, metavar="N", help="並行下載數（預設 8；設 1 則依序）")
    args = parser.parse_args()

    if args.csv:
        csv_path = DEFAULT_CSR_CSV
        if not csv_path.exists():
            raise SystemExit(f"找不到 CSV：{csv_path}。不加 --csv 會從 MOPS 取清單。")
        rows = list(iter_rows(csv_path))
        print(f"使用既有清單 {csv_path.name}，共 {len(rows)} 筆。\n")
    else:
        print(f"[MOPS] 取得 {args.year} 年度清單…")
        rows = _fetch_rows_from_page(year=args.year)
        DEFAULT_CSR_CSV.parent.mkdir(parents=True, exist_ok=True)
        with DEFAULT_CSR_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["source", "company_id", "company_name", "sector",
                            "year", "lang", "url", "mops_download_id"],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"  → 找到 {len(rows)} 筆可下載 PDF，CSV 已存至 {DEFAULT_CSR_CSV.name}，開始下載。\n")

    platform_only = not args.fallback_url
    if platform_only:
        print("模式：只從 MOPS 平台下載，不連公司網站。\n")
    if args.jobs > 1:
        print(f"並行下載：{args.jobs} 個連線。\n")

    t0 = time.perf_counter()
    if args.jobs <= 1:
        for row in rows:
            download_one_with_retry(row, platform_only=platform_only)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(
                lambda row: download_one_with_retry(row, platform_only=platform_only),
                rows,
            ))
    print(f"\n完成，共 {len(rows)} 筆。總耗時：{time.perf_counter() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
