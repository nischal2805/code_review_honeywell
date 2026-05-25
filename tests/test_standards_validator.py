from pathlib import Path
from rag_engine.core.parser import CodeParser
from rag_engine.config import Config
from rag_engine.features.standards_validator import StandardsValidator

FIXTURES = Path(__file__).parent / 'fixtures' / 'cpp'


def _validator():
    return StandardsValidator(Config(), CodeParser().parse_directory(str(FIXTURES / 'base_build')))


def test_analyze_returns_report():
    assert _validator().analyze() is not None


def test_violations_are_violation_objects():
    from rag_engine.models import Violation
    for v in _validator().analyze().violations:
        assert isinstance(v, Violation)


def test_compliance_score_range():
    score = _validator().analyze().compliance_score
    assert 0 <= score <= 100


def test_goto_flagged(tmp_path):
    cpp = tmp_path / 'bad.cpp'
    cpp.write_text('void badFunc() {\n    goto end;\n    end: return;\n}\n')
    v = StandardsValidator(Config(), CodeParser().parse_directory(str(tmp_path)))
    rules = {x.rule for x in v.analyze().violations}
    assert 'DO178-GOTO' in rules


def test_dynamic_memory_flagged(tmp_path):
    cpp = tmp_path / 'dynmem.cpp'
    cpp.write_text('void allocFunc() {\n    int* p = new int(5);\n    delete p;\n}\n')
    v = StandardsValidator(Config(), CodeParser().parse_directory(str(tmp_path)))
    rules = {x.rule for x in v.analyze().violations}
    assert 'DO178-DYNAMIC-MEM' in rules


def test_high_complexity_flagged(tmp_path):
    conditions = '\n'.join(f'    if (x == {i}) {{ x++; }}' for i in range(12))
    cpp = tmp_path / 'complex.cpp'
    cpp.write_text(f'void complexFunc(int x) {{\n{conditions}\n}}\n')
    v = StandardsValidator(Config(), CodeParser().parse_directory(str(tmp_path)))
    rules = {x.rule for x in v.analyze().violations}
    assert 'DO178-CC' in rules
