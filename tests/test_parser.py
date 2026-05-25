import pytest
from pathlib import Path
from rag_engine.core.parser import CodeParser
from rag_engine.models import FunctionDef, ParseResult

FIXTURES = Path(__file__).parent / 'fixtures' / 'cpp'
BASE = FIXTURES / 'base_build'
CURRENT = FIXTURES / 'current_build'


def test_parse_file_returns_parse_result():
    result = CodeParser().parse_file(str(BASE / 'vehicle.cpp'))
    assert isinstance(result, ParseResult)


def test_parse_file_extracts_functions():
    result = CodeParser().parse_file(str(BASE / 'vehicle.cpp'))
    names = {f.name for f in result.functions}
    assert {'start', 'getSpeed', 'stop', 'utilityFunc', 'orphanFunc'}.issubset(names)


def test_parse_file_detects_virtual():
    result = CodeParser().parse_file(str(BASE / 'vehicle.cpp'))
    func_map = {f.name: f for f in result.functions}
    assert func_map['start'].is_virtual is True
    assert func_map['getSpeed'].is_virtual is True
    assert func_map['stop'].is_virtual is False
    assert func_map['utilityFunc'].is_virtual is False


def test_parse_file_line_numbers_positive():
    result = CodeParser().parse_file(str(BASE / 'vehicle.cpp'))
    for f in result.functions:
        assert f.line_number > 0, f"{f.name} has non-positive line number"


def test_parse_file_extracts_classes():
    result = CodeParser().parse_file(str(BASE / 'vehicle.cpp'))
    assert 'Vehicle' in result.classes


def test_parse_file_extracts_includes():
    result = CodeParser().parse_file(str(BASE / 'vehicle.cpp'))
    assert 'string' in result.includes


def test_parse_directory_returns_dict():
    results = CodeParser().parse_directory(str(BASE))
    assert isinstance(results, dict)
    assert len(results) >= 1
    assert any('vehicle.cpp' in k for k in results)


def test_parse_current_build_has_brake():
    result = CodeParser().parse_file(str(CURRENT / 'vehicle.cpp'))
    names = {f.name for f in result.functions}
    assert 'brake' in names


def test_cyclomatic_complexity_brake():
    result = CodeParser().parse_file(str(CURRENT / 'vehicle.cpp'))
    func_map = {f.name: f for f in result.functions}
    assert func_map['brake'].cyclomatic_complexity >= 2
