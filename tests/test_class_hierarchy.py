from pathlib import Path

import pytest

from rag_engine.core.class_hierarchy import ClassHierarchy
from rag_engine.core.parser import CodeParser

SENSORS = Path(__file__).parent / 'fixtures' / 'cpp' / 'sensors'


@pytest.fixture()
def parsed():
    return CodeParser().parse_directory(str(SENSORS))


@pytest.fixture()
def hierarchy(parsed):
    return ClassHierarchy(parsed)


def _hierarchy_of(tmp_path, source):
    (tmp_path / 't.cpp').write_text(source)
    results = CodeParser().parse_directory(str(tmp_path))
    h = ClassHierarchy(results)
    return h, {fn.sig_key: fn for r in results.values() for fn in r.functions}


def test_classes_are_merged_by_qualified_name(hierarchy):
    assert set(hierarchy.classes) == {
        'Sensor', 'TemperatureSensor', 'PressureSensor', 'AltitudeSensor',
        'SensorManager', 'Validator', 'Display',
    }


def test_same_class_in_two_files_is_merged(tmp_path):
    (tmp_path / 'a.h').write_text('class A { public: void f(); int x; };\n')
    (tmp_path / 'b.h').write_text('class A { public: void f(); void g(); int y; };\n')
    h = ClassHierarchy(CodeParser().parse_directory(str(tmp_path)))
    assert [m.sig_key for m in h.classes['A'].methods] == ['A::f()', 'A::g()']
    assert [f.name for f in h.classes['A'].fields] == ['x', 'y']


def test_bases_derived_descendants_ancestors(hierarchy):
    assert hierarchy.bases('TemperatureSensor') == ['Sensor']
    assert hierarchy.bases('Sensor') == []
    assert hierarchy.derived('Sensor') == ['AltitudeSensor', 'PressureSensor', 'TemperatureSensor']
    assert hierarchy.descendants('Sensor') == ['AltitudeSensor', 'PressureSensor', 'TemperatureSensor']
    assert hierarchy.ancestors('PressureSensor') == ['Sensor']
    assert hierarchy.descendants('Display') == [] and hierarchy.ancestors('Display') == []
    assert hierarchy.bases('NoSuchClass') == []


def test_multilevel_and_multiple_inheritance_order(tmp_path):
    h, _ = _hierarchy_of(tmp_path, 'class A {}; class B : public A {}; class C : public B {};\n'
                                   'class M1 {}; class M2 {}; class D : public M2, public B, private M1 {};\n')
    assert h.ancestors('C') == ['B', 'A']                 # nearest first
    assert h.ancestors('D') == ['M2', 'B', 'M1', 'A']
    assert h.descendants('A') == ['B', 'C', 'D']
    assert h.bases('D') == ['M2', 'B', 'M1']


def test_bases_resolve_through_enclosing_namespace_and_drop_unparsed_ones(tmp_path):
    h, _ = _hierarchy_of(tmp_path, 'namespace hw { class Base {}; class D : public Base, public std::exception {}; }\n')
    assert h.bases('hw::D') == ['hw::Base']
    assert h.derived('hw::Base') == ['hw::D']


def test_inheritance_cycle_does_not_hang(tmp_path):
    h, _ = _hierarchy_of(tmp_path, 'class A : public B {}; class B : public A {};\n')
    assert h.descendants('A') == ['B'] and h.ancestors('A') == ['B']
    assert h.inheritance_tree('A') == 'A\n└── B'


def test_inheritance_tree_matches_the_report_example(hierarchy):
    assert hierarchy.inheritance_tree('Sensor') == (
        'Sensor\n'
        '├── AltitudeSensor\n'
        '├── PressureSensor\n'
        '└── TemperatureSensor'
    )
    assert hierarchy.inheritance_tree('Display') == 'Display'


def test_inheritance_tree_nested_levels(tmp_path):
    h, _ = _hierarchy_of(tmp_path, 'class A {}; class B : public A {}; class C : public B {};\n'
                                   'class D : public A {}; class E : public B {};\n')
    assert h.inheritance_tree('A') == (
        'A\n'
        '├── B\n'
        '│   ├── C\n'
        '│   └── E\n'
        '└── D'
    )


def test_find_method_is_a_static_lookup(hierarchy):
    assert hierarchy.find_method('TemperatureSensor', 'process') == 'TemperatureSensor::process()'
    assert hierarchy.find_method('TemperatureSensor', 'selfTest') == 'Sensor::selfTest()'   # nearest ancestor
    assert hierarchy.find_method('AltitudeSensor', 'selfTest') == 'AltitudeSensor::selfTest()'
    assert hierarchy.find_method('Sensor', 'read') == 'Sensor::read()'                       # pure, no definition
    assert hierarchy.find_method('Sensor', 'isStale') is None                                # not up the tree
    assert hierarchy.find_method('Display', 'update', ['float']) == 'Display::update(float)'
    assert hierarchy.find_method('Display', 'update', ['int']) is None


def test_find_methods_returns_overloads_of_the_nearest_declaring_class(tmp_path):
    h, _ = _hierarchy_of(tmp_path, 'class B { public: void f(); void f(int); void g(); };\n'
                                   'class D : public B { public: void f(char); };\n')
    assert h.find_methods('D', 'f') == ['D::f(char)']                    # D hides B::f
    assert h.find_methods('D', 'g') == ['B::g()']
    assert h.find_methods('B', 'f') == ['B::f()', 'B::f(int)']


def test_virtual_methods_overrides_and_roots(hierarchy):
    assert hierarchy.is_virtual_method('Sensor::process()')
    assert hierarchy.is_virtual_method('TemperatureSensor::process()')
    assert not hierarchy.is_virtual_method('Sensor::getValue()')
    assert not hierarchy.is_virtual_method('TemperatureSensor::isStale(unsigned long)')
    assert not hierarchy.is_virtual_method('No::such()')
    assert hierarchy.overrides_of('Sensor::read()') == [
        'AltitudeSensor::read()', 'PressureSensor::read()', 'TemperatureSensor::read()']
    assert hierarchy.overrides_of('Sensor::selfTest()') == ['AltitudeSensor::selfTest()']
    assert hierarchy.overrides_of('TemperatureSensor::read()') == []
    assert hierarchy.overrides_of('Sensor::getValue()') == []
    assert hierarchy.root_virtual_methods() == [
        'Sensor::~Sensor()', 'Sensor::process()', 'Sensor::read()', 'Sensor::selfTest()']


def test_virtual_destructor_makes_derived_destructors_virtual(tmp_path):
    h, funcs = _hierarchy_of(tmp_path, 'class B { public: virtual ~B() {} };\n'
                                       'class D : public B { public: ~D() {} };\n')
    assert funcs['D::~D()'].is_virtual
    assert h.overrides_of('B::~B()') == ['D::~D()']
    assert h.root_virtual_methods() == ['B::~B()']


def test_overloads_override_by_parameter_types(tmp_path):
    h, funcs = _hierarchy_of(tmp_path, 'class B { public: virtual void f(int); virtual void f(char); };\n'
                                       'class D : public B { public: void f(char) {} void f(long) {} };\n')
    assert h.overrides_of('B::f(char)') == ['D::f(char)']
    assert h.overrides_of('B::f(int)') == []
    assert funcs['D::f(char)'].is_virtual and not funcs['D::f(long)'].is_virtual


def test_field_type_strips_qualifiers_and_searches_ancestors(hierarchy):
    assert hierarchy.field_type('SensorManager', 'sensor') == 'TemperatureSensor'
    assert hierarchy.field_type('SensorManager', 'sensors') == 'Sensor'          # 'Sensor *'
    assert hierarchy.field_type('SensorManager', 'temperature') == 'float'
    assert hierarchy.field_type('TemperatureSensor', 'value') == 'float'          # declared in Sensor
    assert hierarchy.field_type('SensorManager', 'nope') is None
    assert hierarchy.field_type('Unknown', 'x') is None


def test_field_type_resolves_to_the_qualified_class(tmp_path):
    h, _ = _hierarchy_of(tmp_path, 'namespace hw { class Sensor {}; class Mgr { const Sensor& ref; Sensor* ptr; }; }\n')
    assert h.field_type('hw::Mgr', 'ref') == 'hw::Sensor'
    assert h.field_type('hw::Mgr', 'ptr') == 'hw::Sensor'


def test_out_of_line_definition_of_a_virtual_method_is_virtual(parsed):
    ClassHierarchy(parsed)
    funcs = {fn.sig_key: fn for r in parsed.values() for fn in r.functions}
    assert funcs['Sensor::process()'].is_virtual            # 'virtual' only on the header declaration
    assert not funcs['Sensor::process()'].is_override
    assert funcs['TemperatureSensor::process()'].is_virtual and funcs['TemperatureSensor::process()'].is_override
    assert funcs['PressureSensor::read()'].is_virtual and funcs['PressureSensor::read()'].is_override
    assert not funcs['TemperatureSensor::isStale(unsigned long)'].is_virtual
    assert not funcs['SensorManager::update()'].is_virtual


def test_override_without_keyword_is_virtual_and_override(tmp_path):
    h, funcs = _hierarchy_of(tmp_path, 'class B { public: virtual void f(); virtual void g(); };\n'
                                       'class M : public B { public: void f(); };\n'
                                       'class D : public M { public: void f() {} void g() {} };\n')
    assert funcs['D::f()'].is_virtual and funcs['D::f()'].is_override
    assert funcs['D::g()'].is_virtual and funcs['D::g()'].is_override            # grandparent's virtual
    assert h.overrides_of('B::f()') == ['M::f()', 'D::f()']
    assert h.root_virtual_methods() == ['B::f()', 'B::g()']


def test_non_virtual_method_with_same_signature_does_not_override(tmp_path):
    h, funcs = _hierarchy_of(tmp_path, 'class B { public: void f(); };\nclass D : public B { public: void f() {} };\n')
    assert not funcs['D::f()'].is_virtual and not funcs['D::f()'].is_override
    assert h.root_virtual_methods() == []


def test_post_pass_is_idempotent(parsed):
    ClassHierarchy(parsed)
    first = {fn.sig_key: (fn.class_name, fn.is_virtual, fn.is_override) for r in parsed.values() for fn in r.functions}
    ClassHierarchy(parsed)
    again = {fn.sig_key: (fn.class_name, fn.is_virtual, fn.is_override) for r in parsed.values() for fn in r.functions}
    assert first == again


def test_namespace_qualifier_is_not_a_class(tmp_path):
    src = ('namespace util { int clamp(int v); }\n'
           'int util::clamp(int v) { return v; }\n'
           'class C { public: void m(); };\n'
           'void C::m() {}\n')
    result = CodeParser().parse_file(str(_write(tmp_path, src)))
    fns = {fn.sig_key: fn for fn in result.functions}
    assert fns['util::clamp(int)'].class_name == 'util'              # as the parser sees it
    ClassHierarchy({result.file_path: result})
    assert fns['util::clamp(int)'].class_name is None
    assert fns['C::m()'].class_name == 'C'


def _write(tmp_path, source):
    path = tmp_path / 't.cpp'
    path.write_text(source)
    return path


def test_method_declarations_keep_their_default_arguments(tmp_path):
    h, _ = _hierarchy_of(tmp_path, 'class D { public: void show(float v, int p = 2); void log(const char* f, ...); };\n')
    show = h.method('D::show(float, int)')
    assert [p.default_value for p in show.parameters] == [None, '2']
    assert h.method('D::log(const char *, ...)').param_types == ['const char *', '...']
