from __future__ import annotations

"""Shared utilities for ESG and CSR report downloaders."""

import csv
import os
from pathlib import Path
from typing import Iterable

# Project root: resolved at runtime so it works whether run from source or
# from an installed package.  Prefer the CWD when a data/ directory already
# exists there (typical project checkout); fall back to the directory that
# contains this file (editable / source install).
_THIS_DIR = Path(__file__).resolve().parent

def _resolve_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "data").is_dir() or (cwd / "CLAUDE.md").is_file():
        return cwd
    return _THIS_DIR

ROOT = _resolve_root()


def get_data_dir(report_type: str) -> Path:
    """Return (and create) data/raw_pdfs/{report_type}/ inside the project dir."""
    d = ROOT / "data" / "raw_pdfs" / report_type
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_filename(row: dict, data_dir: Path) -> Path:
    """Produce a collision-free filename: {company_id}_{year}_{lang}.pdf inside data_dir."""
    company_id = (row.get("company_id") or "").strip()
    name = (row.get("company_name") or "").strip()
    year = (row.get("year") or "").strip() or "unknown"
    lang = (row.get("lang") or "").strip() or "unk"

    base = f"{company_id or name}_{year}_{lang}".strip("_")
    base = "".join(c for c in base if c.isalnum() or c in ("-", "_"))

    return data_dir / (base + ".pdf")


def iter_rows(csv_path: Path) -> Iterable[dict]:
    """Yield dicts from a UTF-8-BOM CSV."""
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row
