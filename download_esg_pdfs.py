from __future__ import annotations

"""
那一頁就有全部公司的「下載 PDF」按鈕，這支程式就是：從那頁取得全部 → 一個一個按下載。

做法：用該頁背後的查詢 API 拿到整頁清單（畫面上每一筆對應一個下載按鈕），
再對每一筆呼叫同一個「下載 PDF」API（等同按那顆按鈕），把檔案存到 data/raw_pdfs/esg/。

  python download_esg_pdfs.py              # 從那頁取全部、一直下載 PDF（預設）
  python download_esg_pdfs.py --csv         # 改用既有 CSV 清單
  python download_esg_pdfs.py --year 2023   # 指定年度（預設 2024）
"""

import argparse
import csv
import time
from pathlib import Path

import requests

from report_utils import ROOT, get_data_dir, iter_rows, safe_filename

DATA_DIR = get_data_dir("esg")

DEFAULT_CSV = ROOT / "esg_sources_twse.csv"

TWSE_BASE = "https://esggenplus.twse.com.tw"
TWSE_DATA_API = f"{TWSE_BASE}/api/api/MopsSustainReport/data"
TWSE_FILE_STREAM_URL = f"{TWSE_BASE}/api/api/MopsSustainReport/data/FileStream"
NULL_UUID = "00000000-0000-0000-0000-000000000000"
TWSE_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Origin": TWSE_BASE,
    "Referer": f"{TWSE_BASE}/inquiry/report?lang=zh-TW",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}
FILE_STREAM_HEADERS = {
    "Accept": "application/pdf, */*",
    "Origin": TWSE_BASE,
    "Referer": f"{TWSE_BASE}/inquiry/report?lang=zh-TW",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}


def _try_twse_download(download_id: str, timeout: int = 60) -> tuple[bytes, str] | tuple[None, str]:
    """
    用公開資訊網「下載 PDF」API（FileStream?id=）直接取檔案。
    成功回傳 (content, content_type)，失敗回傳 (None, 失敗原因字串)。
    """
    if not download_id or download_id == "00000000-0000-0000-0000-000000000000":
        return (None, "無下載 ID")
    try:
        resp = requests.get(
            TWSE_FILE_STREAM_URL,
            params={"id": download_id},
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
        return (None, "回傳非 PDF")
    except requests.exceptions.Timeout:
        return (None, "逾時")
    except Exception as e:
        return (None, str(e)[:50])


def download_one(row: dict, timeout: int = 60, platform_only: bool = True) -> Path | None:
    twse_id = (row.get("twse_download_id") or "").strip()
    url = (row.get("url") or "").strip()
    if not twse_id and not url:
        print(f"[SKIP] {row.get('company_name')} {row.get('year')} 沒有 URL 或 twse_download_id")
        return None

    target = safe_filename(row, DATA_DIR)
    if target.exists():
        print(f"[EXIST] {target.name}")
        return target

    content = None
    ctype = ""

    # 方式一：公開資訊網直接下載（等同網頁「下載 PDF」）
    if twse_id:
        print(f"[平台] {row.get('company_name')} {row.get('year')} {row.get('lang')} id={twse_id[:8]}...")
        out, msg = _try_twse_download(twse_id, timeout=timeout)
        if out is not None:
            content, ctype = out, msg
            print(f"      -> 成功 ({len(content)} bytes)")
        else:
            print(f"      -> 失敗: {msg}")

    # 選用：失敗時改抓公司網址（預設不啟用，只解析這頁、只從這頁下載 PDF）
    if content is None and url and platform_only is False:
        print(f"[網址] {row.get('company_name')} {row.get('year')} {row.get('lang')} <- {url[:50]}...")
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            content = resp.content
            ctype = (resp.headers.get("Content-Type") or "").lower()
            print(f"      -> 成功 ({len(content)} bytes)")
        except Exception as e:
            print(f"      -> 失敗: {e}")
            return None

    if content is None:
        return None

    ext = ".pdf" if "pdf" in ctype else (".html" if "html" in ctype or "text/" in ctype else ".bin")
    if not target.name.endswith(ext):
        target = target.with_suffix(ext)

    target.write_bytes(content)
    print(f"[OK  ] -> {target}")
    return target


def download_one_with_retry(row: dict, retries: int = 2, **kwargs) -> Path | None:
    for attempt in range(retries + 1):
        result = download_one(row, **kwargs)
        if result is not None:
            return result
        if attempt < retries:
            time.sleep(2 ** attempt)
    return None


def _fetch_rows_from_page(year: int = 2024) -> list[dict]:
    """從該頁查詢 API 取得整頁清單（每一筆對應一顆下載 PDF 按鈕）。"""
    payload = {
        "marketType": 0,
        "year": year,
        "industryNameList": [],
        "companyCodeList": [],
        "industryName": "all",
        "companyCode": "all",
    }
    resp = requests.post(TWSE_DATA_API, json=payload, headers=TWSE_HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or "data" not in data:
        raise RuntimeError(f"查詢 API 回傳格式異常: {data!r}")
    raw = list(data["data"])
    rows: list[dict[str, str]] = []
    for item in raw:
        code = str(item.get("code") or "").strip()
        name = str(item.get("name") or "").strip()
        sector = str(item.get("sector") or "").strip()
        tw_url = (item.get("twDocLink") or "").strip()
        en_url = (item.get("enDocLink") or "").strip()
        tw_id = (item.get("twFirstReportDownloadId") or "").strip()
        en_id = (item.get("enFirstReportDownloadId") or "").strip()
        if tw_id == NULL_UUID:
            tw_id = ""
        if en_id == NULL_UUID:
            en_id = ""
        if tw_url or tw_id:
            rows.append({"source": "twse", "company_id": code, "company_name": name, "sector": sector, "year": str(year), "lang": "zh", "url": tw_url, "twse_download_id": tw_id})
        if en_url or en_id:
            rows.append({"source": "twse", "company_id": code, "company_name": name, "sector": sector, "year": str(year), "lang": "en", "url": en_url, "twse_download_id": en_id})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="在那一頁找所有下載 PDF，然後一直下載")
    parser.add_argument("--csv", action="store_true", help="改用既有 CSV 清單，不從該頁取（預設是從該頁取）")
    parser.add_argument("--year", type=int, default=2024, help="從該頁取清單時使用的年度（預設 2024）")
    parser.add_argument("--fallback-url", action="store_true", help="平台無檔時改抓公司網址（預設不啟用）")
    parser.add_argument("-j", "--jobs", type=int, default=8, metavar="N", help="並行下載數（預設 8；設 1 則依序）")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(DEFAULT_CSV)
        if not csv_path.exists():
            raise SystemExit(f"找不到 CSV：{csv_path}。不加 --csv 會從該頁取清單。")
        rows = list(iter_rows(csv_path))
        print(f"使用既有清單 {csv_path.name}，共 {len(rows)} 筆。\n")
    else:
        print(f"[該頁] 取得 {args.year} 年度清單…")
        rows = _fetch_rows_from_page(year=args.year)
        csv_path = Path(DEFAULT_CSV)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source", "company_id", "company_name", "sector", "year", "lang", "url", "twse_download_id"])
            w.writeheader()
            w.writerows(rows)
        print(f"  - 找到 {len(rows)} 筆可下載 PDF，開始下載。\n")
    platform_only = not args.fallback_url
    if platform_only:
        print("模式：只從公開資訊網「下載 PDF」下載，不連公司網站。\n")
    if args.jobs > 1:
        print(f"並行下載：{args.jobs} 個連線。\n")

    t0 = time.perf_counter()
    if args.jobs <= 1:
        for row in rows:
            download_one_with_retry(row, platform_only=platform_only)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(lambda row: download_one_with_retry(row, platform_only=platform_only), rows))
    print(f"\n完成，共 {len(rows)} 筆。總耗時：{time.perf_counter() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
