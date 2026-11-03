from __future__ import annotations
from datetime import date
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

from rag_engine.config import Config
from rag_engine.document_processor.template_engine import render
from rag_engine.features.function_effect import FunctionEffect
from rag_engine.features.dead_code_detector import DeadCodeReport
from rag_engine.features.standards_validator import ComplianceReport
from rag_engine.features.sensor_analysis import SensorAnalysisResult
from rag_engine.features.virtual_analysis import VirtualAnalysisResult
from rag_engine.reporting.output_formatter import markdown_to_docx, write_text_file

if TYPE_CHECKING:   # annotation only; importing it would load the embedding models
    from rag_engine.features.coupling_analysis import CouplingAnalysisResult

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
            'inventory': result.inventory,
            'hierarchy_trees': result.hierarchy_trees,
            'entry_points': result.entry_points,
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

    def generate_function_effect(self, effects: List[FunctionEffect], format: _FORMAT = 'docx',
                                  query: Optional[str] = None) -> str:
        """*query* names the report when *effects* is empty; otherwise the first function's short name does."""
        short = query or (effects[0].function.split('(')[0].rsplit('::', 1)[-1] if effects else 'unknown')
        md = render('function_effect.jinja2', {
            'report_date': date.today().isoformat(),
            'dal_level': self._cfg.dal_level,
            'query': short,
            'effects': effects,
        })
        safe = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in short)
        return self._save(f"function_effect_{safe}", md, format)

    def generate_dead_code(self, result: DeadCodeReport, format: _FORMAT = 'docx',
                            llm_narrative: Optional[str] = None) -> str:
        md = render('dead_code_report.jinja2', {
            'report_date': date.today().isoformat(),
            'dal_level': self._cfg.dal_level,
            'items': result.items,
            'coverage_impact': result.structural_coverage_impact,
            'warnings': result.warnings,
            'entry_points': result.entry_points_used,
            'total_functions': result.total_functions,
            'unresolved_calls': result.unresolved_calls,
            'counts_by_category': result.counts_by_category,
            'counts_by_confidence': {c: sum(1 for i in result.items if i.confidence == c)
                                     for c in ('high', 'medium', 'low')},
            'llm_narrative': llm_narrative,
        })
        findings = {'items': [vars(i) for i in result.items],   # vars(): the dataclass fields
                    'warnings': result.warnings, 'entry_points': result.entry_points_used,
                    'unresolved_calls': result.unresolved_calls}
        (Path(self._cfg.output_dir) / 'dead_code_findings.json').write_text(
            json.dumps(findings, indent=2), encoding='utf-8')
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

    def generate_sensor_analysis(self, result: SensorAnalysisResult, format: _FORMAT = 'docx',
                                  llm_narrative: Optional[str] = None) -> str:
        md = render('sensor_analysis.jinja2', {
            'report_date': date.today().isoformat(),
            'dal_level': self._cfg.dal_level,
            'findings': result.findings,
            'unmapped': result.unmapped_sensor_classes,
            'llm_narrative': llm_narrative,
        })
        return self._save('sensor_analysis', md, format)

    def _save(self, name: str, md: str, format: _FORMAT) -> str:
        base = Path(self._cfg.output_dir)
        if format == 'docx':
            path = str(base / f"{name}.docx")
            return markdown_to_docx(md, path)
        path = str(base / f"{name}.md")
        return write_text_file(md, path)
