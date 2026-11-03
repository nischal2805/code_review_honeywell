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
    unreachable = b.find_unreachable(['utilityFunc', 'Vehicle::start', 'Vehicle::getSpeed', 'Vehicle::stop'])
    assert 'orphanFunc()' in unreachable


def test_find_reachable_from_returns_set():
    b = _builder()
    reachable = b.find_reachable_from('utilityFunc')
    assert isinstance(reachable, set)


def test_get_function_returns_function_def():
    from rag_engine.models import FunctionDef
    b = _builder()
    fn = b.get_function('Vehicle::start')
    assert isinstance(fn, FunctionDef)


def _build(tmp_path, source):
    (tmp_path / 'm.cpp').write_text(source)
    b = CallGraphBuilder(CodeParser().parse_directory(str(tmp_path)))
    b.build()
    return b


def test_free_function_self_call_is_kept(tmp_path):
    b = _build(tmp_path, 'int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }\n')
    assert b._graph.has_edge('fact(int)', 'fact(int)')
    assert 'fact(int)' in b.get_function('fact(int)').calls


def test_qualified_self_call_is_kept(tmp_path):
    b = _build(tmp_path, 'class C { public: int run(int n); };\n'
                         'int C::run(int n) { return n ? C::run(n - 1) : 0; }\n')
    assert b._graph.has_edge('C::run(int)', 'C::run(int)')


def test_overload_call_is_not_self_recursion(tmp_path):
    b = _build(tmp_path, 'int f(int a) { return a; }\n'
                         'int f(int a, int c) { return f(a) + c; }\n')
    assert b._graph.has_edge('f(int, int)', 'f(int)')
    assert not b._graph.has_edge('f(int, int)', 'f(int, int)')
    assert not b._graph.has_edge('f(int)', 'f(int)')


def test_same_named_member_call_is_not_self_recursion(tmp_path):
    b = _build(tmp_path, 'struct Foo { int count() const { return 1; } };\n'
                         'int count(const Foo& f) { return f.count(); }\n')
    assert not b._graph.has_edge('count(const Foo &)', 'count(const Foo &)')


def test_unknown_receiver_fan_out_has_no_self_edge(tmp_path):
    b = _build(tmp_path, 'class A { public: void run() { other.run(); } };\n')
    assert not b._graph.has_edge('A::run()', 'A::run()')

