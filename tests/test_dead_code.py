from pathlib import Path
from rag_engine.core.parser import CodeParser
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.features.dead_code_detector import DeadCodeDetector

FIXTURES = Path(__file__).parent / 'fixtures' / 'cpp'


def _detector():
    results = CodeParser().parse_directory(str(FIXTURES / 'base_build'))
    gb = CallGraphBuilder(results)
    gb.build()
    return DeadCodeDetector(gb, results)


def test_analyze_returns_report():
    assert _detector().analyze().items is not None


def test_orphan_func_detected():
    report = _detector().analyze()
    assert 'orphanFunc' in {i.name for i in report.items}


def test_orphan_is_dead_code():
    report = _detector().analyze()
    item_map = {i.name: i for i in report.items}
    if 'orphanFunc' in item_map:
        assert item_map['orphanFunc'].category == 'dead_code'
        assert item_map['orphanFunc'].do178c_disposition == 'Remove'


def test_coverage_impact_populated():
    for item in _detector().analyze().items:
        assert item.coverage_impact


def test_entry_points_not_in_dead():
    report = _detector().analyze()
    assert 'main' not in {i.name for i in report.items}
