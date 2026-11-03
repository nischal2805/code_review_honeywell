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



# ── Class hierarchy, virtual dispatch, call sites ──────────────────────────────

SENSORS = FIXTURES / 'sensors'


def _sensors():
    b = CallGraphBuilder(CodeParser().parse_directory(str(SENSORS)))
    b.build()
    return b


def _edges(b, caller):
    return {v: d['call_type'] for _, v, d in b._graph.out_edges(caller, data=True)}


def _line_of(filename, text):
    lines = (SENSORS / filename).read_text().splitlines()
    return next(i for i, line in enumerate(lines, 1) if text in line)


def test_pointer_call_dispatches_to_instantiated_overrides_only():
    b = _sensors()
    assert _edges(b, 'SensorManager::update()') == {
        'Sensor::process()': 'direct',
        'TemperatureSensor::process()': 'virtual',
        'PressureSensor::process()': 'virtual',
    }
    assert not any(v.startswith('AltitudeSensor::') for _, v in b._graph.edges)


def test_implicit_this_call_to_pure_virtual_fans_out_over_instantiated_classes():
    b = _sensors()
    assert _edges(b, 'Sensor::process()') == {
        'TemperatureSensor::read()': 'virtual',
        'PressureSensor::read()': 'virtual',
    }
    assert 'Sensor::read()' not in b._graph        # a pure virtual has no body, so no node


def test_field_receivers_are_typed_by_the_enclosing_class():
    b = _sensors()
    assert _edges(b, 'SensorManager::updateTemperature()') == {
        'TemperatureSensor::read()': 'direct',
        'Validator::check(float)': 'direct',
        'Display::update(float)': 'direct',
    }


def test_reachability_follows_dispatch_edges():
    b = _sensors()
    reachable = b.find_reachable_from('main')
    assert {'Sensor::process()', 'TemperatureSensor::process()', 'PressureSensor::process()',
            'TemperatureSensor::read()', 'PressureSensor::read()'} <= reachable
    assert not reachable & {'AltitudeSensor::read()', 'AltitudeSensor::selfTest()', 'Sensor::selfTest()'}


def test_instantiated_classes():
    assert _sensors().instantiated_classes == {
        'SensorManager', 'TemperatureSensor', 'PressureSensor', 'Validator', 'Display'}


def test_call_sites_carry_line_and_receiver():
    b = _sensors()
    sites = {(c.caller, c.target): c for c in b.call_sites}
    update = sites[('SensorManager::update()', 'Sensor::process()')]
    assert update.file_path.endswith('sensor_manager.cpp')
    assert update.line == _line_of('sensor_manager.cpp', 's->process();')
    assert update.receiver == 's'
    assert update.call_type == 'virtual'            # static target Sensor::process() is virtual
    assert len([c for c in b.call_sites if c.caller == 'SensorManager::update()']) == 1   # one per call, not per override

    temp = [c for c in b.call_sites if c.caller == 'SensorManager::updateTemperature()']
    assert [(c.target, c.receiver, c.call_type) for c in sorted(temp, key=lambda c: c.line)] == [
        ('TemperatureSensor::read()', 'sensor', 'direct'),      # value field: bound statically
        ('Validator::check(float)', 'validator', 'direct'),
        ('Display::update(float)', 'display', 'direct'),
    ]
    assert [c.line for c in sorted(temp, key=lambda c: c.line)] == [
        _line_of('sensor_manager.cpp', 'sensor.read()'),
        _line_of('sensor_manager.cpp', 'validator.check('),
        _line_of('sensor_manager.cpp', 'display.update(temperature)'),
    ]


def test_call_site_to_pure_virtual_is_recorded_without_a_node():
    site = next(c for c in _sensors().call_sites if c.target == 'Sensor::read()')
    assert site.caller == 'Sensor::process()' and site.call_type == 'virtual' and site.receiver is None
    assert site.file_path.endswith('sensor.cpp') and site.line == _line_of('sensor.cpp', 'read();')


def test_edges_carry_call_type_and_lines():
    b = _sensors()
    update_line = _line_of('sensor_manager.cpp', 's->process();')
    for target in ('Sensor::process()', 'TemperatureSensor::process()', 'PressureSensor::process()'):
        assert b._graph['SensorManager::update()'][target]['lines'] == [update_line]
    assert b._graph['main()']['SensorManager::update()']['lines'] == [_line_of('main.cpp', 'mgr.update();')]
    assert _edges(b, 'main()')['onTimerTick()'] == 'callback'


def test_lines_accumulate_per_edge(tmp_path):
    b = _build(tmp_path, 'void a() {}\nvoid run() {\n    a();\n    a();\n}\n')
    assert b._graph['run()']['a()']['lines'] == [3, 4]
    assert [c.line for c in b.call_sites if c.caller == 'run()'] == [3, 4]


def test_qualified_call_is_never_dispatched(tmp_path):
    b = _build(tmp_path, 'class B { public: virtual void f(); };\nvoid B::f() {}\n'
                         'class D : public B { public: void f() { B::f(); } };\n'
                         'class E : public D { public: void f() {} };\n'
                         'void make() { E e; }\n')
    assert _edges(b, 'D::f()') == {'B::f()': 'direct'}


def test_class_that_is_never_instantiated_gets_no_virtual_edge(tmp_path):
    src = ('class B { public: virtual void f() {} };\nclass D : public B { public: void f() {} };\n'
           'void run(B* b) { b->f(); }\n')
    assert _edges(_build(tmp_path, src), 'run(B *)') == {'B::f()': 'direct'}
    (tmp_path / 'm.cpp').write_text(src + 'void make() { D d; }\n')
    b = CallGraphBuilder(CodeParser().parse_directory(str(tmp_path)))
    b.build()
    assert _edges(b, 'run(B *)') == {'B::f()': 'direct', 'D::f()': 'virtual'}


def test_new_expression_instantiates_the_class(tmp_path):
    b = _build(tmp_path, 'class B { public: virtual void f() {} };\nclass D : public B { public: void f() {} };\n'
                         'void run(B* b) { b->f(); }\nB* make() { return new D(); }\n')
    assert b.instantiated_classes == {'D'}
    assert _edges(b, 'run(B *)')['D::f()'] == 'virtual'


def test_indexed_receiver_and_inherited_field(tmp_path):
    b = _build(tmp_path, 'class S { public: virtual void p() {} };\nclass T : public S { public: void p() {} };\n'
                         'class Dev { public: void go() {} };\n'
                         'class Base { protected: Dev dev; S* items[2]; };\n'
                         'class M : public Base { public: void run() { items[0]->p(); dev.go(); } };\n'
                         'void make() { T t; }\n')
    assert _edges(b, 'M::run()') == {'S::p()': 'direct', 'T::p()': 'virtual', 'Dev::go()': 'direct'}
    assert {c.receiver for c in b.call_sites if c.caller == 'M::run()'} == {'items', 'dev'}


def test_this_call_has_no_receiver(tmp_path):
    b = _build(tmp_path, 'class C { public: void a() {} void b() { this->a(); a(); } };\n')
    assert _edges(b, 'C::b()') == {'C::a()': 'direct'}
    assert [(c.target, c.receiver) for c in b.call_sites] == [('C::a()', None)]    # same target, same line: one site


def test_member_beats_free_function_inside_a_method(tmp_path):
    b = _build(tmp_path, 'void log() {}\nclass C { public: void log() {} void run() { log(); } };\n')
    assert _edges(b, 'C::run()') == {'C::log()': 'direct'}


def test_unknown_receiver_is_unresolved(tmp_path):
    b = _build(tmp_path, 'class A { public: void go() {} };\nclass B { public: void go() {} };\n'
                         'void run(Thing* t) { t->go(); }\n')
    assert _edges(b, 'run(Thing *)') == {'A::go()': 'unresolved', 'B::go()': 'unresolved'}
    assert {(c.call_type, c.receiver) for c in b.call_sites} == {('unresolved', 't')}


def test_constructor_and_callback_call_types_and_precedence(tmp_path):
    b = _build(tmp_path, 'class A { public: A() {} };\ntypedef void (*Fn)();\nvoid cb() {}\nvoid reg(Fn f) {}\n'
                         'void make() { A a; }\nvoid only_cb() { reg(cb); }\nvoid both() { reg(cb); cb(); }\n')
    assert _edges(b, 'make()') == {'A::A()': 'constructor'}
    assert _edges(b, 'only_cb()') == {'reg(Fn)': 'direct', 'cb()': 'callback'}
    assert _edges(b, 'both()')['cb()'] == 'direct'      # the strongest evidence wins


def test_implicit_this_call_to_itself_is_recursion(tmp_path):
    b = _build(tmp_path, 'class C { public: int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); } };\n')
    assert b._graph.has_edge('C::fact(int)', 'C::fact(int)')
    assert 'C::fact(int)' in b.get_function('C::fact(int)').calls


def test_receiver_of_own_class_is_recursion(tmp_path):
    b = _build(tmp_path, 'struct N { N* next; int depth() { return 1 + next->depth(); } };\n')
    assert b._graph.has_edge('N::depth()', 'N::depth()')


def test_out_of_line_namespace_function_recursion_by_bare_name(tmp_path):
    b = _build(tmp_path, 'namespace ns { int f(int n); }\nint ns::f(int n) { return n ? f(n - 1) : 0; }\n')
    assert b.get_function('ns::f(int)').class_name is None
    assert b._graph.has_edge('ns::f(int)', 'ns::f(int)')


def test_call_with_omitted_default_argument_keeps_its_edge(tmp_path):
    b = _build(tmp_path, 'void f(const char*, int) {}\nvoid f(int a, int b, int c = 0) {}\nvoid caller() { f(1, 2); }\n')
    assert b._graph.has_edge('caller()', 'f(int, int, int)')
    assert not b._graph.has_edge('caller()', 'f(const char *, int)')


def test_variadic_call_matches_only_the_variadic_overload(tmp_path):
    b = _build(tmp_path, 'void log(const char* fmt, ...) {}\nvoid log(int code) {}\n'
                         'void caller() { log("x", 1, 2); log("y"); }\n')
    assert _edges(b, 'caller()') == {'log(const char *, ...)': 'direct'}


def test_literals_that_all_mismatch_do_not_drop_the_only_candidate(tmp_path):
    b = _build(tmp_path, 'void f(int a, int b) {}\nvoid g() { f(1.5, 2.5); }\n')
    assert b._graph.has_edge('g()', 'f(int, int)')


# ── Fix round 1 ────────────────────────────────────────────────────────────────

def test_default_argument_declared_in_the_header_counts_for_out_of_line_definitions(tmp_path):
    (tmp_path / 'disp.h').write_text('class Disp { public: void show(float v, int p = 2); void show(const char* s); };\n')
    (tmp_path / 'disp.cpp').write_text('#include "disp.h"\nvoid Disp::show(float v, int p) {}\nvoid Disp::show(const char* s) {}\n')
    (tmp_path / 'run.cpp').write_text('#include "disp.h"\nvoid run(Disp& d) { d.show(1.0f); }\n')
    b = CallGraphBuilder(CodeParser().parse_directory(str(tmp_path)))
    b.build()
    assert _edges(b, 'run(Disp &)') == {'Disp::show(float, int)': 'direct'}


def test_nested_call_is_not_a_callback(tmp_path):
    b = _build(tmp_path, 'bool ok(int x) { return x > 0; }\nvoid take(bool b) {}\n'
                         'void run(int x) { if (ok(x)) {} }\nvoid nest(int x) { take(ok(x)); }\n')
    assert [(c.target, c.call_type) for c in b.call_sites if c.caller == 'run(int)'] == [('ok(int)', 'direct')]
    assert {(c.target, c.call_type) for c in b.call_sites if c.caller == 'nest(int)'} == {
        ('take(bool)', 'direct'), ('ok(int)', 'direct')}


def test_function_passed_by_name_is_still_a_callback_site():
    site = next(c for c in _sensors().call_sites if c.target == 'onTimerTick()')
    assert (site.caller, site.call_type) == ('main()', 'callback')
    assert site.line == _line_of('main.cpp', 'registerCallback(onTimerTick)')


_DERIVED = 'class Base { public: virtual void f() {} };\nclass Der : public Base { public: void f() {} };\n'


def test_value_receivers_bind_statically(tmp_path):
    src = (_DERIVED + 'void run() { Base b; b.f(); Der d; }\nvoid viaValue(Base v) { v.f(); }\n'
           'class V { Base b; public: void run() { b.f(); } };\n')
    b = _build(tmp_path, src)
    for caller in ('run()', 'viaValue(Base)', 'V::run()'):
        assert _edges(b, caller) == {'Base::f()': 'direct'}, caller
        assert [c.call_type for c in b.call_sites if c.caller == caller] == ['direct']


def test_pointer_reference_and_this_receivers_dispatch(tmp_path):
    src = (_DERIVED + 'void make() { Der d; }\n'
           'void viaPointer(Base* p) { p->f(); }\n'
           'void viaReference(const Base& r) { r.f(); }\n'
           'void viaLocalPointer() { Base* p = nullptr; p->f(); }\n'
           'void viaLocalReference(Base& src) { Base& r = src; r.f(); }\n'
           'class P { Base* b; public: void run() { b->f(); } };\n'
           'class R { Base& b; public: void run() { b.f(); } };\n'
           'class T { public: virtual void g() {} void run() { g(); this->g(); } };\n'
           'class U : public T { public: void g() {} };\nvoid makeU() { U u; }\n')
    b = _build(tmp_path, src)
    for caller in ('viaPointer(Base *)', 'viaReference(const Base &)', 'viaLocalPointer()',
                   'viaLocalReference(Base &)', 'P::run()', 'R::run()'):
        assert _edges(b, caller) == {'Base::f()': 'direct', 'Der::f()': 'virtual'}, caller
    assert _edges(b, 'T::run()') == {'T::g()': 'direct', 'U::g()': 'virtual'}


def test_member_call_on_an_expression_does_not_fan_out_to_free_functions(tmp_path):
    b = _build(tmp_path, 'class A { public: A& self(); void go(); };\nA& A::self() { return *this; }\nvoid A::go() {}\n'
                         'void go() {}\nvoid run(A* a) { a->self().go(); }\n')
    assert _edges(b, 'run(A *)') == {'A::self()': 'direct', 'A::go()': 'unresolved'}


def test_return_and_else_are_not_local_object_declarations(tmp_path):
    b = _build(tmp_path, 'class value { public: value() {} };\nint f(int x) { if (x) return x; else x = 1; return value_of(x); }\n'
                         'int value_of(int v) { return v; }\n')
    assert b.instantiated_classes == set()


def test_call_edges_from_alternative_definitions_carry_their_own_lines(tmp_path):
    b = _build(tmp_path, 'void a() {}\nvoid b() {}\n#ifdef X\nvoid f() {\n  a();\n}\n#else\nvoid f() {\n\n  b();\n}\n#endif\n')
    assert b.graph['f()']['a()']['lines'] == [5]
    assert b.graph['f()']['b()']['lines'] == [10]
