from pathlib import Path

import pytest

from rag_engine.core.parser import CodeParser

SENSORS = Path(__file__).parent / 'fixtures' / 'cpp' / 'sensors'


@pytest.fixture(scope='module')
def sensors():
    return CodeParser().parse_directory(str(SENSORS))


@pytest.fixture(scope='module')
def funcs(sensors):
    return {fn.sig_key: fn for r in sensors.values() for fn in r.functions}


@pytest.fixture(scope='module')
def classes(sensors):
    return {c.name: c for r in sensors.values() for c in r.class_defs}


def _parse_src(tmp_path, source, name='t.cpp'):
    path = tmp_path / name
    path.write_text(source)
    return CodeParser().parse_file(str(path))


def test_inline_methods_are_class_qualified(funcs):
    expected = {
        'TemperatureSensor::read()', 'PressureSensor::read()', 'AltitudeSensor::read()',
        'AltitudeSensor::selfTest()', 'Sensor::selfTest()', 'Sensor::~Sensor()',
        'Sensor::getValue()',
    }
    assert expected <= set(funcs)


def test_same_inline_name_in_two_classes_is_not_collapsed(tmp_path):
    src = 'class A { public: int read() { return 1; } };\nclass B { public: int read() { return 2; } };\n'
    keys = {fn.sig_key for fn in _parse_src(tmp_path, src).functions}
    assert keys == {'A::read()', 'B::read()'}


def test_inline_override_is_virtual_and_override(funcs):
    fn = funcs['AltitudeSensor::read()']
    assert fn.is_virtual and fn.is_override
    assert funcs['Sensor::selfTest()'].is_virtual
    assert not funcs['Sensor::selfTest()'].is_override
    assert not funcs['Sensor::getValue()'].is_virtual


def test_class_name_for_in_class_and_out_of_line(funcs):
    assert funcs['AltitudeSensor::read()'].class_name == 'AltitudeSensor'
    assert funcs['TemperatureSensor::isStale(unsigned long)'].class_name == 'TemperatureSensor'
    assert funcs['main()'].class_name is None


def test_namespace_prefix_for_in_class_and_out_of_line(tmp_path):
    src = ('namespace hw {\n'
           'class Sensor { public: virtual void process(); int id() { return 1; } };\n'
           'void Sensor::process() {}\n'
           'void helper() {}\n'
           '}\n')
    result = _parse_src(tmp_path, src)
    by_key = {fn.sig_key: fn for fn in result.functions}
    assert set(by_key) == {'hw::Sensor::process()', 'hw::Sensor::id()', 'hw::helper()'}
    assert by_key['hw::Sensor::process()'].class_name == 'hw::Sensor'
    assert by_key['hw::helper()'].class_name is None
    assert [c.name for c in result.class_defs] == ['hw::Sensor']
    assert result.classes == ['Sensor']
    assert result.class_defs[0].methods[0].sig_key == 'hw::Sensor::process()'


def test_class_def_bases(classes):
    assert classes['TemperatureSensor'].bases == ['Sensor']
    assert classes['Sensor'].bases == []
    assert not classes['Sensor'].is_struct


def test_struct_and_multiple_bases(tmp_path):
    result = _parse_src(tmp_path, 'struct S : public A, private virtual B { int n; };\n')
    cdef = result.class_defs[0]
    assert cdef.is_struct and cdef.bases == ['A', 'B']
    assert cdef.end_line_number >= cdef.line_number


def test_pure_virtual_method_decl(classes):
    methods = {m.name: m for m in classes['Sensor'].methods}
    read = methods['read']
    assert read.is_pure and read.is_virtual and not read.has_body
    assert read.sig_key == 'Sensor::read()'
    assert not methods['process'].is_pure and not methods['process'].has_body
    assert methods['selfTest'].has_body and methods['selfTest'].is_virtual
    assert methods['~Sensor'].is_virtual and methods['~Sensor'].has_body
    assert not methods['getValue'].is_virtual


def test_override_and_final_specifiers(classes, tmp_path):
    process = {m.name: m for m in classes['TemperatureSensor'].methods}['process']
    assert process.is_override and process.is_virtual and not process.is_final
    assert process.sig_key == 'TemperatureSensor::process()'
    result = _parse_src(tmp_path, 'class C : public B { void f() final; void g() override final; };\n')
    methods = {m.name: m for m in result.class_defs[0].methods}
    assert methods['f'].is_final and methods['f'].is_virtual and not methods['f'].is_override
    assert methods['g'].is_final and methods['g'].is_override


def test_constructor_and_destructor_declarations(tmp_path):
    result = _parse_src(tmp_path, 'class C { public: C(); ~C(); C(int a) : n(a) {} int n; };\n')
    sigs = {m.sig_key for m in result.class_defs[0].methods}
    assert sigs == {'C::C()', 'C::~C()', 'C::C(int)'}


def test_pointer_returning_pure_virtual_is_recorded(tmp_path):
    result = _parse_src(tmp_path, 'class C { public: virtual C* clone() const = 0; };\n')
    (m,) = result.class_defs[0].methods
    assert m.name == 'clone' and m.is_pure and m.sig_key == 'C::clone()'


def test_fields(classes):
    sensor_fields = {f.name: f for f in classes['Sensor'].fields}
    assert {'value', 'lastValue', 'lastUpdateMs'} == set(sensor_fields)
    assert sensor_fields['lastUpdateMs'].type_ == 'unsigned long'
    assert sensor_fields['value'].line_number == 15
    manager = {f.name: f for f in classes['SensorManager'].fields}
    assert manager['sensor'].type_ == 'TemperatureSensor'
    assert manager['sensors'].type_ == 'Sensor *'
    assert 'init' not in manager


def test_multi_declarator_and_static_fields(tmp_path):
    result = _parse_src(tmp_path, 'class C { int a, *b, c[3]; static int count; };\n')
    fields = {f.name: f for f in result.class_defs[0].fields}
    assert {n: f.type_ for n, f in fields.items()} == {
        'a': 'int', 'b': 'int *', 'c': 'int', 'count': 'int'}
    assert fields['count'].is_static and not fields['a'].is_static


def test_preproc_guard(funcs, tmp_path):
    assert funcs['disabledFeature()'].preproc_guard == '0'
    assert funcs['main()'].preproc_guard is None
    assert funcs['Sensor::getValue()'].preproc_guard is None  # include guard is ignored
    src = ('#ifdef FOO\nvoid a() {}\n#endif\n'
           '#if defined(A) && B\nvoid b() {}\n#elif C\nvoid c() {}\n#else\nvoid d() {}\n#endif\n'
           '#ifndef BAR\nvoid e() {}\n#endif\n')
    guards = {fn.name: fn.preproc_guard for fn in _parse_src(tmp_path, src).functions}
    assert guards == {'a': 'FOO', 'b': 'defined(A) && B', 'c': 'C', 'd': None, 'e': None}


def test_end_line_and_body_start_line(funcs):
    main = funcs['main()']
    assert main.end_line_number > main.line_number
    assert main.body_start_line == main.line_number
    one_liner = funcs['AltitudeSensor::read()']
    assert one_liner.end_line_number == one_liner.line_number == one_liner.body_start_line
    for fn in funcs.values():
        assert fn.end_line_number >= fn.line_number > 0


def test_data_access_update_temperature(funcs):
    fn = funcs['SensorManager::updateTemperature()']
    assert fn.data_writes == {'temperature'}
    assert fn.data_reads == {'sensor', 'temperature', 'validator', 'display'}


def test_data_access_sensor_process(funcs):
    fn = funcs['Sensor::process()']
    assert fn.data_writes == {'lastValue', 'value'}
    assert fn.data_reads == {'value'}


def test_data_access_excludes_locals_and_callees(funcs):
    update = funcs['SensorManager::update()']
    assert update.data_reads == {'sensors'}  # i and s are locals; process() callee is not a read
    assert update.data_writes == set()
    debounce = funcs['TemperatureSensor::debounce(float)']
    assert debounce.data_reads == {'v', 'lastValue', 'TEMP_DEBOUNCE_C'}  # std::fabs is a callee
    assert funcs['registerCallback(Callback)'].data_reads == set()


def test_data_access_write_forms(tmp_path):
    src = ('struct S { int x; int items[4]; };\n'
           'void S_run(int p, int* ptr, S obj, int idx) {\n'
           '    int local = 0;\n'
           '    for (int i = 0; i < p; i++) { local += i; }\n'
           '    cnt++;\n'
           '    --other;\n'
           '    total += p;\n'
           '    arr[idx] = 1;\n'
           '    *ptr = 2;\n'
           '    obj.x = 3;\n'
           '    use(local, gread);\n'
           '}\n'
           'class K { int x; int items[4]; void m(int v) { this->x = v; this->items[v] += 1; y = this->x; } };\n')
    fns = {fn.name: fn for fn in _parse_src(tmp_path, src).functions}
    run = fns['S_run']
    assert run.data_writes == {'cnt', 'other', 'total', 'arr', 'ptr', 'obj'}
    assert run.data_reads == {'p', 'cnt', 'other', 'total', 'idx', 'gread'}
    method = fns['K::m']
    assert method.data_writes == {'x', 'items', 'y'}
    assert method.data_reads == {'v', 'items', 'x'}


def test_data_access_catch_parameter_is_local(tmp_path):
    src = 'void f() { try { g_state = 1; } catch (const E& ex) { g_err = ex; } }\n'
    (fn,) = _parse_src(tmp_path, src).functions
    assert fn.data_writes == {'g_state', 'g_err'}
    assert fn.data_reads == set()


def test_default_arguments_do_not_split_declaration_and_definition_keys(tmp_path):
    src = ('class D { public:\n'
           '    void clear(int mode = 0);\n'
           '    void put(const Foo& f = Foo(1, 2), float* p = nullptr);\n'
           '    void skip(int = 5);\n'
           '};\n'
           'void D::clear(int mode) {}\n'
           'void D::put(const Foo& f, float* p) {}\n'
           'void D::skip(int) {}\n')
    result = _parse_src(tmp_path, src)
    declared = {m.name: m.sig_key for m in result.class_defs[0].methods}
    defined = {fn.name.rsplit('::', 1)[-1]: fn.sig_key for fn in result.functions}
    assert declared == defined == {
        'clear': 'D::clear(int)', 'put': 'D::put(const Foo &, float *)', 'skip': 'D::skip(int)'}


def test_default_value_text_is_stored_on_the_parameter(tmp_path):
    (fn,) = _parse_src(tmp_path, 'void f(int a, int mode = 0, const Foo& c = Foo(1, 2), int = 5) {}\n').functions
    assert [p.name for p in fn.parameters] == ['a', 'mode', 'c', '']
    assert [p.type_ for p in fn.parameters] == ['int', 'int', 'const Foo &', 'int']
    assert [p.default_value for p in fn.parameters] == [None, '0', 'Foo(1, 2)', '5']
    assert fn.sig_key == 'f(int, int, const Foo &, int)'


def test_variadic_parameters_are_part_of_the_key(tmp_path):
    src = ('class L { public: void log(const char* fmt, ...); };\n'
           'void L::log(const char* fmt, ...) {}\n'
           'void v(...) {}\n'
           'template <typename... Args> void pack(Args... args, int n = 1) {}\n')
    result = _parse_src(tmp_path, src)
    keys = {fn.name: fn.sig_key for fn in result.functions}
    assert keys == {'L::log': 'L::log(const char *, ...)', 'v': 'v(...)', 'pack': 'pack(Args ..., int)'}
    assert result.class_defs[0].methods[0].sig_key == keys['L::log']


def test_const_and_non_const_overloads_with_one_sig_key_are_not_merged(tmp_path):
    src = 'class A { public: int get() const; int get(); };\nint A::get() const { return 1; }\nint A::get() { return 2; }\n'
    fns = [fn for fn in _parse_src(tmp_path, src).functions if fn.name == 'A::get']
    assert len(fns) == 1 and fns[0].preproc_guard is None and fns[0].extra_bodies == []


def test_return_type_keeps_the_declarator_of_a_member_function(tmp_path):
    src = 'struct S {};\nclass A { public: S* p(); S& r(); };\nS* A::p() { return 0; }\n'
    types = {fn.name: fn.return_type for fn in _parse_src(tmp_path, src).functions}
    assert types == {'A::p': 'S *'}


def test_constructor_initialiser_arguments_are_kept_without_the_member_names(tmp_path):
    src = 'class A { public: A(int size); int size; int n; };\nA::A(int size) : size(size), n(compute(size)) {}\n'
    fn = _parse_src(tmp_path, src).functions[0]
    assert [text for _, text in fn.extra_bodies] == ['(size)', '(compute(size))']
    assert [line for line, _ in fn.extra_bodies] == [2, 2]


def test_default_member_initialisers_and_file_scope_initialisers_are_recorded(tmp_path):
    src = ('class A { public: int a = f(); int b{g()}; int c = 3; static int d; };\n'
           'int A::d = h();\n'
           'static int g_x = k(1);\n'
           'namespace ns { static int g_y = m(); }\n'
           'int g_plain = 5;\n')
    result = _parse_src(tmp_path, src)
    assert [text for _, text in result.class_defs[0].field_inits] == ['f()', '{g()}', '3']
    assert result.static_inits == [(2, ' = h()', None), (3, ' = k(1)', None), (4, ' = m()', 'ns')]


def test_parameter_names_are_not_part_of_array_and_function_pointer_types(tmp_path):
    src = 'void f(unsigned char buf[], int m[4][4], void (*cb)(int value), int (&ref)[4]) {}\n'
    fn = _parse_src(tmp_path, src).functions[0]
    assert [(p.name, p.type_) for p in fn.parameters] == [
        ('buf', 'unsigned char []'), ('m', 'int [4][4]'), ('cb', 'void (*)(int)'),
        ('ref', 'int (&)[4]')]
    assert fn.sig_key == 'f(unsigned char [], int [] [], void (*)(int), int (&) [])'
