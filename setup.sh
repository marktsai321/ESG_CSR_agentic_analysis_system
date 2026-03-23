#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# ESG/CSR Report Analysis System — One-line Setup
#
#   bash setup.sh
#
# What it does:
#   1. Creates a Python virtual environment (.venv)
#   2. Installs the package and all dependencies (pip install .)
#   3. Creates .env from .env.example if it doesn't exist
#   4. Prompts for OPENAI_API_KEY if not already set in .env
#   5. Creates the required directory structure
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " ESG/CSR 報告書分析系統 — 環境設定"
echo "============================================================"
echo

# ── 1. Virtual environment ────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "[1/4] 建立 Python 虛擬環境 (.venv)..."
    python3 -m venv .venv
else
    echo "[1/4] 虛擬環境已存在，跳過建立。"
fi

source .venv/bin/activate
echo "      Python: $(python --version) @ $(which python)"

# ── 2. Install package ────────────────────────────────────────
echo "[2/4] 安裝 esg-csr-agent 及所有相依套件..."
pip install --upgrade pip -q
pip install -e . -q
echo "      完成。已安裝指令: esg-csr-agent"

# ── 3. Create .env ────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo "[3/4] 建立 .env 設定檔..."
    cp .env.example .env

    # Prompt for API key
    if grep -q "your-api-key-here" .env; then
        echo
        read -rp "      請輸入您的 OpenAI API Key: " api_key
        if [ -n "$api_key" ]; then
            sed -i "s|your-api-key-here|${api_key}|" .env
            echo "      API Key 已寫入 .env"
        else
            echo "      [警告] 未輸入 API Key，稍後執行時系統會再次詢問。"
        fi
    fi
else
    echo "[3/4] .env 已存在，跳過建立。"
fi

# ── 4. Directory structure ────────────────────────────────────
echo "[4/4] 建立目錄結構..."
mkdir -p data/raw_pdfs/esg data/raw_pdfs/csr data/extracted_text \
         data/vector_store data/analysis data/revised outputs logs
echo "      完成。"

echo
echo "============================================================"
echo " 設定完成！使用方式："
echo ""
echo "   source .venv/bin/activate"
echo "   esg-csr-agent                                   # 互動模式"
echo "   esg-csr-agent --companies 2330 --years 2023     # 直接模式"
echo "============================================================"
