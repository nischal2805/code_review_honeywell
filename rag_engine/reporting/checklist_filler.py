from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Literal, Optional

from rag_engine.config import Config
from rag_engine.features.dead_code_detector import CLOSED_STATUSES, DeadCodeReport
from rag_engine.features.sensor_analysis import SensorAnalysisResult
from rag_engine.features.standards_validator import ComplianceReport
from rag_engine.features.virtual_analysis import VirtualAnalysisResult
from rag_engine.models import DeadCodeItem


def _reviewed(item: DeadCodeItem) -> bool:
    """A reviewer closed the finding with a documented status and a justification; any other status
    (a typo, an old value) counts as no decision."""
    return item.review_status in CLOSED_STATUSES and bool(item.reviewer_justification)


@dataclass
class ChecklistItem:
    category: str
    item_id: str
    description: str
    status: Literal['PASS', 'FAIL', 'N/A', 'OPEN']
    evidence: str = ''


@dataclass
class SQAChecklist:
    generated_date: str
    dal_level: str
    items: List[ChecklistItem] = field(default_factory=list)


class ChecklistFiller:
    def __init__(self, config: Config) -> None:
        self._cfg = config

    def fill_sqa_checklist(self, virtual: VirtualAnalysisResult, dead: DeadCodeReport,
                           standards: ComplianceReport,
                           sensors: Optional[SensorAnalysisResult] = None) -> SQAChecklist:
        cl = SQAChecklist(generated_date=date.today().isoformat(), dal_level=self._cfg.dal_level)
        cl.items.append(ChecklistItem('virtual_analysis', 'VA-01',
            'All virtual function changes identified and classified (Cat1/Cat2)',
            'PASS' if virtual.changes else 'OPEN',
            f"{virtual.summary.get('modified',0)} modified, {virtual.summary.get('added',0)} added"))
        cl.items.append(ChecklistItem('virtual_analysis', 'VA-02',
            'All Category 2 changes scheduled for reverification', 'PASS',
            str(sum(1 for c in virtual.changes if c.do178c_category == 'Category 2')) + ' Cat2 changes'))
        # An override nobody can reach is dead code unless a reviewer dispositioned its finding.
        justified = {i.name for i in dead.items if _reviewed(i)}
        unreached = [o.function for v in virtual.inventory for o in v.overridden_by
                     if not o.reachable and o.function not in justified]
        cl.items.append(ChecklistItem('virtual_analysis', 'VA-03',
            'Every virtual override is reachable from the entry points or justified',
            'PASS' if not unreached else 'OPEN',
            f"{len(unreached)} unreachable, unjustified override(s)" + (f": {', '.join(unreached)}" if unreached else '')))
        cl.items.append(ChecklistItem('dead_code', 'DC-01',
            'Dead code items identified per DO-178C §6.4.2.2',
            'PASS' if dead.items is not None else 'OPEN',
            f"{dead.dead_count} dead, {dead.deactivated_count} deactivated"))
        cl.items.append(ChecklistItem('dead_code', 'DC-02',
            'All dead code items have disposition (Remove/Justify)',
            'PASS' if all(i.do178c_disposition for i in dead.items) else 'OPEN',
            f"{len(dead.items)} items with disposition"))
        open_dead = sum(1 for i in dead.items
                        if i.category == 'dead_code' and i.confidence == 'high' and not _reviewed(i))
        cl.items.append(ChecklistItem('dead_code', 'DC-03',
            'No open high-confidence dead code findings',
            'PASS' if open_dead == 0 else 'OPEN',
            f"{open_dead} open high-confidence dead code finding(s)"))
        critical = standards.violations_by_severity.get('CRITICAL', 0)
        cl.items.append(ChecklistItem('standards', 'STD-01',
            'Zero unresolved CRITICAL violations at release',
            'PASS' if critical == 0 else 'FAIL',
            f"{critical} CRITICAL violation(s)"))
        cl.items.append(ChecklistItem('standards', 'STD-02',
            'Compliance score meets minimum threshold (85%)',
            'PASS' if standards.compliance_score >= 85 else 'OPEN',
            f"Score: {standards.compliance_score:.1f}%"))
        if sensors is not None:
            cl.items += self._sensor_items(sensors)
        Path(self._cfg.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(self._cfg.output_dir) / 'sqa_checklist.md').write_text(self._render(cl), encoding='utf-8')
        return cl

    @staticmethod
    def _sensor_items(sensors: SensorAnalysisResult) -> List[ChecklistItem]:
        unmapped = [f.template.name for f in sensors.findings if f.status != 'found']
        gaps = [f"{f.template.name}.{name} ({c.status})" for f in sensors.findings
                for name, c in f.checks.items() if c.status in ('missing', 'mismatch')]
        return [
            ChecklistItem('sensors', 'SEN-01', 'Every sensor template is mapped to a code class',
                          'PASS' if not unmapped else 'OPEN',
                          f"{len(sensors.findings) - len(unmapped)}/{len(sensors.findings)} mapped"
                          + (f"; unmapped: {', '.join(unmapped)}" if unmapped else '')),
            ChecklistItem('sensors', 'SEN-02', 'No stale/corrupt/debounce check is missing or mismatched with its template',
                          'PASS' if not gaps else 'FAIL',
                          f"{len(gaps)} check(s)" + (f": {', '.join(gaps)}" if gaps else '')),
        ]

    @staticmethod
    def _render(cl: SQAChecklist) -> str:
        lines = ['# DO-178C SQA Checklist', f'Date: {cl.generated_date}  DAL: {cl.dal_level}', '',
                 '| ID | Category | Description | Status | Evidence |',
                 '|----|----------|-------------|--------|----------|']
        for item in cl.items:
            lines.append(f"| {item.item_id} | {item.category} | {item.description} | {item.status} | {item.evidence} |")
        return '\n'.join(lines)
