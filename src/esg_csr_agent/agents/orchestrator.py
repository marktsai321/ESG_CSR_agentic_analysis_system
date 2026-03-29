from __future__ import annotations

"""
Orchestrator Agent
==================
Central coordinator — pipeline state machine.
Computes total rubric scores after analysis and routes only the worst-scoring
company to the Report Revision Agent for deep-dive on weak dimensions.
"""

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from crewai import Agent

from esg_csr_agent.config import (
    OPENAI_MODEL_NAME,
    ANALYSIS_DIR,
    WEAK_SCORE_THRESHOLD,
    MIN_COMPANIES,
)
from esg_csr_agent.pipeline_state import PipelineState


def create_orchestrator_agent() -> Agent:
    return Agent(
        role="協調者代理",
        goal="管理整個 ESG/CSR 分析管線的狀態機，依序啟動各代理並監控進度。",
        backstory=(
            "你是整個分析管線的中央協調者。"
            "你負責初始化管線狀態、依序啟動各下游代理、監控進度、"
            "並在代理失敗時啟動失敗處理代理。"
            "你不直接執行分析、不與使用者互動、不寫入檔案。"
        ),
        verbose=True,
        allow_delegation=True,
        llm=OPENAI_MODEL_NAME,
    )


class Pipeline:
    """Imperative pipeline runner — the Orchestrator's execution engine."""

    def __init__(self, state: PipelineState):
        self.state = state

    def run(self) -> PipelineState:
        # Validate minimum companies
        if len(self.state.companies) < MIN_COMPANIES:
            print(f"[錯誤] 至少需要 {MIN_COMPANIES} 家公司，目前僅 {len(self.state.companies)} 家。")
            self.state.stage = "error"
            return self.state

        stages = [
            ("web_scraper", self._stage_download),
            ("text_extraction", self._stage_extract),
            ("chunk_embed", self._stage_chunk_embed),
            ("analysis", self._stage_analysis),
            ("scoring", self._stage_scoring),
            ("validation", self._stage_validation),
            ("revision", self._stage_revision),
            ("output", self._stage_output),
        ]

        for stage_name, stage_fn in stages:
            self.state.stage = stage_name
            print(f"\n{'='*60}")
            print(f"[管線] 階段: {stage_name}")
            print(f"{'='*60}")

            try:
                stage_fn()
            except Exception as e:
                print(f"[管線] 階段 {stage_name} 失敗: {e}")
                traceback.print_exc()
                self.state.add_failure("orchestrator", stage_name, str(e))
                self._handle_failure(self.state.failures[-1])

        self.state.stage = "complete"
        return self.state

    def _stage_download(self) -> None:
        from esg_csr_agent.agents.web_scraper_agent import run_download

        for year in self.state.years:
            if year >= 2022:
                print(f"[WARN] {year} 年度 CSR 報告書可能不存在（已改為 ESG 永續報告書）")

        # Always try both ESG and CSR
        results = run_download(
            companies=self.state.companies,
            report_types=["esg", "csr"],
            years=self.state.years,
        )

        for rtype, info in results.items():
            for path in info["downloaded"]:
                self.state.add_file("raw_pdfs", path, path)
            # Only log true failures, not missing report types
            for cid in info["failed"]:
                self.state.add_failure("web_scraper", "download", f"下載失敗: {cid}")

        # Track which report types are available per company
        for company in self.state.companies:
            available = []
            for year in self.state.years:
                for rtype in ["esg", "csr"]:
                    pdf_path = self._find_pdf(company, year, rtype)
                    if pdf_path:
                        available.append(rtype)
            self.state.available_reports[company] = list(set(available))

        # Check that every company has at least one report
        for company in self.state.companies:
            if not self.state.available_reports.get(company):
                self.state.add_failure(
                    "web_scraper", "download",
                    f"公司 {company} 無任何可用報告（ESG 及 CSR 均不存在）",
                )
            else:
                avail = ", ".join(r.upper() for r in self.state.available_reports[company])
                print(f"[INFO] 公司 {company} 可用報告: {avail}")

    def _stage_extract(self) -> None:
        from esg_csr_agent.agents.text_extraction_agent import extract_text_from_pdf

        for company in self.state.companies:
            for year in self.state.years:
                for rtype in ["esg", "csr"]:
                    key = self.state.file_key(company, year, rtype)
                    pdf_path = self._find_pdf(company, year, rtype)
                    if not pdf_path:
                        # Missing report type is normal, not a failure
                        continue
                    result = extract_text_from_pdf(pdf_path, key)
                    if result:
                        self.state.add_file("extracted_text", key, result)
                    else:
                        self.state.add_failure("text_extraction", key, f"擷取失敗: {pdf_path}")

    def _stage_chunk_embed(self) -> None:
        from esg_csr_agent.agents.chunk_embed_agent import chunk_and_embed

        for key, text_path in self.state.files.get("extracted_text", {}).items():
            result = chunk_and_embed(text_path, key)
            if result["status"] != "ok":
                self.state.add_failure("chunk_embed", key, result.get("error", "未知錯誤"))

    def _stage_analysis(self) -> None:
        """Run ESG and CSR analysis in parallel, only for available report types."""
        from esg_csr_agent.agents.esg_analysis_agent import analyze_esg
        from esg_csr_agent.agents.csr_analysis_agent import analyze_csr

        extracted = self.state.files.get("extracted_text", {})
        tasks = []
        for company in self.state.companies:
            for year in self.state.years:
                for rtype in ["esg", "csr"]:
                    ns = self.state.file_key(company, year, rtype)
                    if ns in extracted:
                        tasks.append((rtype, company, year, ns))
                    # Not in extracted = no PDF available, skip silently

        def _run_analysis(task_info):
            rtype, cid, yr, ns = task_info
            try:
                if rtype == "esg":
                    return analyze_esg(cid, yr, ns)
                else:
                    return analyze_csr(cid, yr, ns)
            except Exception as e:
                return {"error": str(e), "type": rtype, "company_id": cid, "year": yr}

        with ThreadPoolExecutor(max_workers=len(tasks) or 1) as ex:
            futures = {ex.submit(_run_analysis, t): t for t in tasks}
            for future in as_completed(futures):
                task_info = futures[future]
                result = future.result()
                if "error" in result and result.get("dimensions") is None:
                    rtype, cid, yr, _ = task_info
                    self.state.add_failure(f"{rtype}_analysis", f"{cid}_{yr}", result["error"])
                else:
                    key = self.state.file_key(task_info[1], task_info[2], task_info[0])
                    self.state.add_file("analysis", key, str(result))

    def _stage_scoring(self) -> None:
        """Compute total rubric scores per company and identify the worst."""
        print("[評分] 計算各公司總評分...")

        for company in self.state.companies:
            company_total = 0
            dimensions_detail: dict = {}

            for year in self.state.years:
                for rtype in ["esg", "csr"]:
                    key = self.state.file_key(company, year, rtype)
                    analysis_path = ANALYSIS_DIR / f"{key}.json"
                    if not analysis_path.exists():
                        continue

                    try:
                        data = json.loads(analysis_path.read_text(encoding="utf-8"))
                    except Exception:
                        continue

                    for dim_name, dim_data in data.get("dimensions", {}).items():
                        rubric_scores = dim_data.get("rubric_scores", {})
                        subtotal = sum(int(v) for v in rubric_scores.values() if isinstance(v, (int, float)))
                        dim_key = f"{rtype}_{dim_name}"
                        dimensions_detail[dim_key] = {
                            "rubric_scores": rubric_scores,
                            "subtotal": subtotal,
                            "overall_score": dim_data.get("overall_score", 0.0),
                            "confidence": dim_data.get("confidence", 0.0),
                        }
                        company_total += subtotal

            self.state.scores[company] = {
                "total": company_total,
                "dimensions": dimensions_detail,
            }
            print(f"  公司 {company}: 總分 {company_total}")

        # Identify worst company
        if self.state.scores:
            worst = min(self.state.scores, key=lambda c: self.state.scores[c]["total"])
            self.state.worst_company = worst
            print(f"[評分] 最低分公司: {worst} (總分: {self.state.scores[worst]['total']})")

            # Identify weak dimensions for the worst company
            # First: find dimensions with any rubric score below threshold
            weak_dims = []
            for dim_key, dim_info in self.state.scores[worst]["dimensions"].items():
                for rubric_name, score in dim_info["rubric_scores"].items():
                    if isinstance(score, (int, float)) and score < WEAK_SCORE_THRESHOLD:
                        if dim_key not in weak_dims:
                            weak_dims.append(dim_key)
                            break

            # If no dimensions are below threshold, still select the lowest-scoring
            # dimensions for revision (worst company always gets revised)
            if not weak_dims:
                dims_by_subtotal = sorted(
                    self.state.scores[worst]["dimensions"].items(),
                    key=lambda x: x[1]["subtotal"],
                )
                # Pick the bottom half (at least 1) for revision
                n_to_revise = max(1, len(dims_by_subtotal) // 2)
                weak_dims = [dim_key for dim_key, _ in dims_by_subtotal[:n_to_revise]]
                print(f"[評分] 所有維度均達門檻，選取最低分 {n_to_revise} 個維度進行修訂")

            self.state.weak_dimensions[worst] = weak_dims
            print(f"[評分] {worst} 修訂維度: {', '.join(weak_dims)}")

    def _stage_validation(self) -> None:
        from esg_csr_agent.agents.validation_gate_agent import validate

        result = validate(self.state)
        self.state.validation_passed = result["passed"]

        if not result["passed"]:
            failed_checks = [c for c in result["checks"] if not c["passed"]]
            print(f"[驗證] {len(failed_checks)} 項未通過:")
            for c in failed_checks:
                print(f"  - {c['name']}: {c['detail']}")

    def _stage_revision(self) -> None:
        from esg_csr_agent.agents.report_revision_agent import revise_report

        if not self.state.worst_company:
            print("[SKIP] 無最低分公司，跳過修訂階段。")
            return

        result = revise_report(self.state)
        if not result:
            self.state.add_failure("revision", "generate_markdown", "修訂報告產生失敗")

    def _stage_output(self) -> None:
        from esg_csr_agent.agents.output_delivery_agent import generate_pdf

        result = generate_pdf(self.state)
        if result:
            self.state.output_path = result
        else:
            self.state.add_failure("output", "generate_pdf", "PDF 產生失敗")

    def _handle_failure(self, failure: dict) -> None:
        from esg_csr_agent.agents.fail_handler_agent import diagnose

        diagnosis = diagnose(failure)
        print(f"[失敗處理] 建議: {diagnosis['proposals']}")

    def _find_pdf(self, company_id: str, year: int, report_type: str) -> str | None:
        from esg_csr_agent.config import RAW_PDF_DIR

        pdf_dir = RAW_PDF_DIR / report_type
        if not pdf_dir.exists():
            return None

        pattern = f"{company_id}_{year}_*.pdf"
        matches = list(pdf_dir.glob(pattern))
        if matches:
            # Prefer zh (Chinese) file for analysis
            zh_matches = [m for m in matches if "_zh" in m.name]
            if zh_matches:
                return str(zh_matches[0])
            return str(matches[0])

        for f in pdf_dir.iterdir():
            if f.suffix == ".pdf" and company_id in f.name and str(year) in f.name:
                return str(f)

        return None
