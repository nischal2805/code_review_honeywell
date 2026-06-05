from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional

from rag_engine.config import Config
from rag_engine.document_processor.template_engine import render
from rag_engine.features.coupling_analysis import CouplingAnalysisResult
from rag_engine.features.dead_code_detector import DeadCodeReport
from rag_engine.features.standards_validator import ComplianceReport
from rag_engine.features.virtual_analysis import VirtualAnalysisResult
from rag_engine.reporting.output_formatter import markdown_to_docx, write_text_file

_FORMAT = Literal['markdown', 'docx']


class ReportGenerator:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def generate_all(self, virtual: VirtualAnalysisResult, coupling: CouplingAnalysisResult,
                     dead: DeadCodeReport, standards: ComplianceReport,
                     format: _FORMAT = 'docx',
                     llm_narratives: Optional[Dict[str, str]] = None) -> List[str]:
        n = llm_narratives or {}
        return [
            self.generate_virtual_analysis(virtual, format, n.get('virtual')),
            self.generate_coupling_analysis(coupling, format, n.get('coupling')),
            self.generate_dead_code(dead, format, n.get('dead_code')),
            self.generate_standards(standards, format, n.get('standards')),
        ]

    def generate_virtual_analysis(self, result: VirtualAnalysisResult, format: _FORMAT = 'docx',
                                   llm_narrative: Optional[str] = None) -> str:
        md = render('virtual_analysis.jinja2', {
            'report_date': date.today().isoformat(),
            'dal_level': self._cfg.dal_level,
            'changes': result.changes,
            'summary': result.summary,
            'base_virtual_count': result.base_virtual_count,
            'current_virtual_count': result.current_virtual_count,
            'llm_narrative': llm_narrative,
        })
        return self._save('virtual_analysis', md, format)

    def generate_coupling_analysis(self, result: CouplingAnalysisResult, format: _FORMAT = 'docx',
                                    llm_narrative: Optional[str] = None) -> str:
        md = render('coupling_analysis.jinja2', {
            'report_date': date.today().isoformat(),
            'dal_level': self._cfg.dal_level,
            'lru_impacts': result.lru_impacts,
            'llm_narrative': llm_narrative,
        })
        return self._save('coupling_analysis', md, format)

    def generate_dead_code(self, result: DeadCodeReport, format: _FORMAT = 'docx',
                            llm_narrative: Optional[str] = None) -> str:
        md = render('dead_code_report.jinja2', {
            'report_date': date.today().isoformat(),
            'dal_level': self._cfg.dal_level,
            'items': result.items,
            'coverage_impact': result.structural_coverage_impact,
            'llm_narrative': llm_narrative,
        })
        return self._save('dead_code_report', md, format)

    def generate_standards(self, result: ComplianceReport, format: _FORMAT = 'docx',
                            llm_narrative: Optional[str] = None) -> str:
        md = render('standards_report.jinja2', {
            'report_date': date.today().isoformat(),
            'dal_level': self._cfg.dal_level,
            'violations': result.violations,
            'compliance_score': round(result.compliance_score, 1),
            'llm_narrative': llm_narrative,
        })
        return self._save('standards_report', md, format)

    def _save(self, name: str, md: str, format: _FORMAT) -> str:
        base = Path(self._cfg.output_dir)
        if format == 'docx':
            path = str(base / f"{name}.docx")
            return markdown_to_docx(md, path)
        path = str(base / f"{name}.md")
        return write_text_file(md, path)
