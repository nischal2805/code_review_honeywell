from pathlib import Path
from rag_engine.core.parser import CodeParser
from rag_engine.core.graph_builder import CallGraphBuilder

FIXTURES = Path(__file__).parent / 'fixtures' / 'cpp'


def _builder():
    results = CodeParser().parse_directory(str(FIXTURES / 'base_build'))
    b = CallGraphBuilder(results)
    b.build()
    return b


def test_build_returns_graph():
    b = _builder()
    assert b._graph is not None
    assert len(b._graph.nodes) > 0


def test_find_unreachable_includes_orphan():
    b = _builder()
    unreachable = b.find_unreachable(['utilityFunc', 'start', 'getSpeed', 'stop'])
    assert 'orphanFunc' in unreachable


def test_find_reachable_from_returns_set():
    b = _builder()
    reachable = b.find_reachable_from('utilityFunc')
    assert isinstance(reachable, set)


def test_get_function_returns_function_def():
    from rag_engine.models import FunctionDef
    b = _builder()
    fn = b.get_function('start')
    assert fn is None or isinstance(fn, FunctionDef)
