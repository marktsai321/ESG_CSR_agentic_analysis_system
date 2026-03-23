from __future__ import annotations

"""Shared pipeline state schema used by the Orchestrator and all agents."""

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineState:
    """Mutable state object that the Orchestrator passes between agents."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    companies: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    report_types: list[str] = field(default_factory=list)
    stage: str = "init"
    files: dict[str, dict[str, str]] = field(default_factory=lambda: {
        "raw_pdfs": {},
        "extracted_text": {},
        "analysis": {},
    })
    failures: list[dict[str, Any]] = field(default_factory=list)
    validation_passed: bool = False
    output_path: str | None = None

    def file_key(self, company_id: str, year: int, report_type: str) -> str:
        return f"{company_id}_{year}_{report_type}"

    def add_file(self, category: str, key: str, path: str) -> None:
        self.files.setdefault(category, {})[key] = path

    def get_file(self, category: str, key: str) -> str | None:
        return self.files.get(category, {}).get(key)

    def add_failure(self, agent: str, step: str, error: str, context: dict | None = None) -> None:
        self.failures.append({
            "agent": agent,
            "step": step,
            "error": error,
            "context": context or {},
        })

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "companies": self.companies,
            "years": self.years,
            "report_types": self.report_types,
            "stage": self.stage,
            "files": self.files,
            "failures": self.failures,
            "validation_passed": self.validation_passed,
            "output_path": self.output_path,
        }
