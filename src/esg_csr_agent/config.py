from __future__ import annotations

"""Global configuration loaded from environment / .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
# Use the same ROOT logic as report_utils so all modules agree on paths.
from esg_csr_agent.report_utils import ROOT

DATA_DIR = ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
EXTRACTED_TEXT_DIR = DATA_DIR / "extracted_text"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
ANALYSIS_DIR = DATA_DIR / "analysis"
REVISED_DIR = DATA_DIR / "revised"
REVISED_PDF_DIR = DATA_DIR / "revised_pdfs"
OUTPUTS_DIR = ROOT / "outputs"
LOGS_DIR = ROOT / "logs"

for d in (
    RAW_PDF_DIR / "esg", RAW_PDF_DIR / "csr",
    EXTRACTED_TEXT_DIR, VECTOR_STORE_DIR,
    ANALYSIS_DIR, REVISED_DIR, REVISED_PDF_DIR, OUTPUTS_DIR, LOGS_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
REVISION_MODEL_NAME = os.getenv("REVISION_MODEL_NAME", "gemini-2.5-flash")

# Auto-detect Anthropic key stored as OPENAI_API_KEY
_is_anthropic_key = OPENAI_API_KEY.startswith("sk-ant-")
if _is_anthropic_key:
    ANTHROPIC_API_KEY = OPENAI_API_KEY
    ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "claude-sonnet-4-20250514")
    LLM_PROVIDER = "anthropic"
else:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "claude-sonnet-4-20250514")
    LLM_PROVIDER = "openai"

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")

# ── Vector store ──────────────────────────────────────────────────────────────
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "chromadb")

# ── Analysis ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
