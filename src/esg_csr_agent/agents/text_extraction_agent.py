from __future__ import annotations

"""
Text Extraction Agent
=====================
Converts downloaded PDFs into clean plain text for downstream processing.
"""

from pathlib import Path

from crewai import Agent, Task

from esg_csr_agent.config import OPENAI_MODEL_NAME, EXTRACTED_TEXT_DIR


def create_text_extraction_agent() -> Agent:
    return Agent(
        role="文字擷取代理",
        goal="將下載的 PDF 報告書轉換為乾淨的純文字檔，供後續分析使用。",
        backstory=(
            "你是專門負責從 PDF 中擷取文字的代理。"
            "你使用 pdfplumber 從 PDF 中提取文字內容，"
            "處理包含表格、多欄排版的複雜 PDF。"
            "擷取失敗時會記錄錯誤但不中斷管線。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=OPENAI_MODEL_NAME,
    )


def extract_text_from_pdf(pdf_path: str, output_key: str) -> str | None:
    """
    Extract text from a PDF file and save to extracted_text directory.

    Args:
        pdf_path: Path to the input PDF.
        output_key: Key like '2330_2023_esg' for the output filename.

    Returns:
        Path to the extracted text file, or None on failure.
    """
    output_path = EXTRACTED_TEXT_DIR / f"{output_key}.txt"

    # Idempotency: skip if already extracted
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[EXIST] {output_path.name}")
        return str(output_path)

    pdf = Path(pdf_path)
    if not pdf.exists():
        print(f"[ERR] PDF 不存在: {pdf_path}")
        return None

    try:
        import pdfplumber

        text_parts: list[str] = []
        with pdfplumber.open(pdf) as reader:
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- 第 {i + 1} 頁 ---\n{page_text}")

        if not text_parts:
            print(f"[WARN] PDF 無法擷取文字（可能為掃描檔）: {pdf_path}")
            return None

        full_text = "\n\n".join(text_parts)
        output_path.write_text(full_text, encoding="utf-8")
        print(f"[OK] 擷取完成: {output_path.name} ({len(text_parts)} 頁)")
        return str(output_path)

    except Exception as e:
        print(f"[ERR] 擷取失敗 {pdf_path}: {e}")
        return None


def extract_all(pdf_paths: dict[str, str]) -> dict[str, str | None]:
    """Extract text from multiple PDFs."""
    results: dict[str, str | None] = {}
    for key, pdf_path in pdf_paths.items():
        results[key] = extract_text_from_pdf(pdf_path, key)
    return results


def create_extraction_task(agent: Agent, pdf_paths: dict[str, str]) -> Task:
    paths_desc = "\n".join(f"  - {k}: {v}" for k, v in pdf_paths.items())
    return Task(
        description=(
            f"請從以下 PDF 檔案擷取文字：\n{paths_desc}\n\n"
            "擷取的純文字檔儲存到 data/extracted_text/ 目錄。\n"
            "若 PDF 為掃描檔或擷取失敗，記錄錯誤但繼續處理其他檔案。"
        ),
        expected_output="擷取結果摘要（每個檔案的成功/失敗狀態及輸出路徑）",
        agent=agent,
    )
