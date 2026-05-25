from pathlib import Path
from rag_engine.core.parser import CodeParser
from rag_engine.features.virtual_analysis import VirtualAnalyzer

FIXTURES = Path(__file__).parent / 'fixtures' / 'cpp'


def _analyzer():
    base = CodeParser().parse_directory(str(FIXTURES / 'base_build'))
    current = CodeParser().parse_directory(str(FIXTURES / 'current_build'))
    return VirtualAnalyzer(base, current)


def test_analyze_returns_result():
    assert _analyzer().analyze() is not None


def test_detects_added_virtual():
    result = _analyzer().analyze()
    added = {c.function.name for c in result.changes if c.change_type == 'added'}
    assert 'brake' in added


def test_detects_modified_virtual():
    result = _analyzer().analyze()
    modified = {c.function.name for c in result.changes if c.change_type == 'modified'}
    assert 'start' in modified


def test_unchanged_virtual_present():
    result = _analyzer().analyze()
    unchanged = {c.function.name for c in result.changes if c.change_type == 'unchanged'}
    assert 'getSpeed' in unchanged


def test_added_is_category2():
    result = _analyzer().analyze()
    for c in result.changes:
        if c.change_type == 'added':
            assert c.do178c_category == 'Category 2'


def test_modified_param_change_is_category2():
    result = _analyzer().analyze()
    for c in result.changes:
        if c.change_type == 'modified' and c.function.name == 'start':
            assert c.do178c_category == 'Category 2'


def test_summary_counts():
    s = _analyzer().analyze().summary
    assert s['added'] >= 1
    assert s['modified'] >= 1
    assert s['unchanged'] >= 1


def test_non_virtual_not_in_changes():
    result = _analyzer().analyze()
    names = {c.function.name for c in result.changes}
    assert 'stop' not in names
    assert 'utilityFunc' not in names
