from rag_engine.models import (
    FunctionDef, Parameter, VirtualChange, Violation, DeadCodeItem, LRUCoupling, ParseResult
)


def test_function_def_qualified_name():
    func = FunctionDef(
        name='compute', file_path='src/engine.cpp', line_number=10,
        return_type='void', parameters=[], is_virtual=False,
        is_inline=False, is_static=False, body='{}', docstring=None,
        cyclomatic_complexity=1, line_count=3, nesting_depth=0,
    )
    assert func.qualified_name == 'src/engine.cpp::compute'


def test_function_def_signature():
    func = FunctionDef(
        name='start', file_path='src/vehicle.cpp', line_number=5,
        return_type='void', parameters=[Parameter(name='speed', type_='int')],
        is_virtual=True, is_inline=False, is_static=False,
        body='{}', docstring=None,
        cyclomatic_complexity=1, line_count=3, nesting_depth=0,
    )
    assert func.signature == 'void start(int speed)'


def test_violation_fields():
    v = Violation(rule='R1', misra_ref='M5-0-1', file='a.cpp', line=1,
                  element='foo', message='bad', severity='CRITICAL')
    assert v.severity == 'CRITICAL'
    assert v.disposition is None


def test_dead_code_item():
    item = DeadCodeItem(
        name='unused_fn', file_path='a.cpp', line_number=5,
        category='dead_code', do178c_disposition='Remove',
        coverage_impact='0% statement coverage loss',
    )
    assert item.do178c_disposition == 'Remove'


def test_lru_coupling_defaults():
    lru = LRUCoupling(lru_name='FCS')
    assert lru.risk_level == 'low'
    assert lru.control_coupling == []
