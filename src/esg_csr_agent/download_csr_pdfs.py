from __future__ import annotations

"""
CSR report downloader — MOPS platform.

  python -m esg_csr_agent.download_csr_pdfs              # default
  python -m esg_csr_agent.download_csr_pdfs --year 2019  # specify year
"""

import argparse
import csv
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from esg_csr_agent.report_utils import ROOT, get_data_dir, iter_rows, safe_filename

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

MARKET_TYPES = ["sii", "otc"]

DEFAULT_CSR_CSV = ROOT / "csr_sources_mops.csv"
DATA_DIR = get_data_dir("csr")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ce_to_roc(ce_year: int) -> str:
    return str(ce_year - 1911)


# ── API / HTML fetching ────────────────────────────────────────────────────────

def _fetch_html_for_year_and_type(year_roc: str, typek: str, timeout: int = 60) -> str | None:
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
    
    # 1. Initialize variables to avoid UnboundLocalError
    data = None 
    
    try:
        r = requests.post(MOPS_API_REDIRECT, json=payload, headers=MOPS_HEADERS, timeout=timeout)
        r.raise_for_status()
        
        # 2. Check if the response is actually JSON before parsing
        if "application/json" not in r.headers.get("Content-Type", "").lower():
            print(f"  [ERR] Expected JSON but got: {r.headers.get('Content-Type')}")
            print(f"  [DEBUG] Response body starts with: {r.text[:100]}")
            return None
            
        data = r.json()
        
        if data.get("code") != 200:
            print(f"  [WARN] redirectToOld returned error: {data.get('message')}")
            return None
            
        redirect_url = data["result"]["url"]
        
    except Exception as e:
        print(f"  [ERR] redirectToOld call failed: {e}")
        return None # Crucial: Stop here if there's an error

    # ... rest of the code to fetch r2 (the actual HTML) ...
    try:
        r2 = requests.get(redirect_url, headers=MOPS_HEADERS, timeout=timeout)
        r2.raise_for_status()
        r2.encoding = "utf-8"
        return r2.text
    except Exception as e:
        print(f"  [ERR] Failed to fetch target HTML: {e}")
        return None


def _parse_html_rows(html: str, year_ce: int) -> list[dict]:
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    data_rows = []  # Initialize early to prevent UnboundLocalError

    # MOPS tables often use 'hasBorder' or are the first table in the main content area
    table = soup.find("table", {"class": "hasBorder"}) or soup.find("table")
    
    if not table:
        print("  [DEBUG] No <table> found in the HTML.")
        # Optional: save the HTML to a file to see what MOPS actually sent
        # Path("debug_mops.html").write_text(html, encoding="utf-8")
        return rows

    # Helpers for extraction
    def _extract_filename(td) -> str:
        if td is None: return ""
        link = td.find("a")
        if not link: return ""
        href = link.get("href", "")
        m = re.search(r"fileName=([^&\s]+)", href)
        return m.group(1).strip() if m else ""

    def _extract_href(td) -> str:
        if td is None: return ""
        link = td.find("a")
        return (link.get("href") or "").strip() if link else ""

    # Filter out header rows
    data_rows = [
        tr for tr in table.find_all("tr")
        if "tblHead" not in (tr.get("class") or [])
    ]
    
    if not data_rows:
        print("  [DEBUG] Table found, but no <tr> data rows identified.")
        return rows

    print(f"  [DEBUG] Found {len(data_rows)} rows. Processing...")

    for tr in data_rows:
        tds = tr.find_all("td")
        if len(tds) < 10:  # Adjust based on observed table width
            continue

        company_id   = tds[0].get_text(strip=True)
        company_name = tds[1].get_text(strip=True)
        
        # NOTE: Column indices change frequently on MOPS. 
        # If company_id looks like text instead of a number, the index is wrong.
        
        # Basic parsing (adjust indices as needed for your specific year)
        zh_filename = _extract_filename(tds[11] if len(tds) > 11 else None)
        
        if zh_filename:
            rows.append({
                "source": "mops", "company_id": company_id,
                "company_name": company_name, "year": str(year_ce), 
                "lang": "zh", "mops_download_id": zh_filename,
            })

    return rows


def _fetch_rows_from_page(year: int = 2020) -> list[dict]:
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
    if not filename:
        return (None, "無 filename")
    
    # Construct the download parameters
    params = {
        "step": "9", 
        "filePath": MOPSOV_FILE_PATH, 
        "fileName": filename
    }
    
    try:
        # Use the actual variables instead of (...)
        resp = requests.get(
            MOPSOV_FILE_STREAM,
            params=params,
            headers=FILE_STREAM_HEADERS,
            timeout=timeout,
        )
        
        # Debugging: let's see what the server says if it's not a 200
        if resp.status_code != 200:
            return (None, f"HTTP {resp.status_code}")
            
        if len(resp.content) < 100:
            return (None, "回傳內容過短 (可能是錯誤頁面)")
            
        ctype = (resp.headers.get("Content-Type") or "").lower()
        
        # Verify if it's actually a PDF
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

    if mops_id:
        print(f"[平台] {row.get('company_name')} {row.get('year')} {row.get('lang')} file={mops_id}")
        out, msg = _try_mops_download(mops_id, timeout=timeout)
        if out is not None:
            content, ctype = out, msg
            print(f"      → 成功 ({len(content)} bytes)")
        else:
            print(f"      → 失敗: {msg}")

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
            time.sleep(2 ** attempt)
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="下載 MOPS 永續/CSR 報告書 PDF")
    parser.add_argument("--csv", action="store_true", help="使用既有 CSV 清單")
    parser.add_argument("--year", type=int, default=2020, help="CE 年度（預設 2020）")
    parser.add_argument("--fallback-url", action="store_true", help="平台無檔時改抓公司網址")
    parser.add_argument("-j", "--jobs", type=int, default=8, metavar="N", help="並行下載數（預設 8）")
    args = parser.parse_args()

    if args.csv:
        csv_path = DEFAULT_CSR_CSV
        if not csv_path.exists():
            raise SystemExit(f"找不到 CSV：{csv_path}")
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
        print(f"  → 找到 {len(rows)} 筆，開始下載。\n")

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
