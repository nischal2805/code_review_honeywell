from pathlib import Path
from rag_engine.config import Config
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.core.parser import CodeParser
from rag_engine.features.virtual_analysis import VirtualAnalyzer
from rag_engine.reporting.report_generator import ReportGenerator

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
    assert 'Vehicle::brake' in added


def test_detects_modified_virtual():
    result = _analyzer().analyze()
    modified = {c.function.name for c in result.changes if c.change_type == 'modified'}
    assert 'Vehicle::start' in modified


def test_unchanged_virtual_present():
    result = _analyzer().analyze()
    unchanged = {c.function.name for c in result.changes if c.change_type == 'unchanged'}
    assert 'Vehicle::getSpeed' in unchanged


def test_added_is_category2():
    result = _analyzer().analyze()
    for c in result.changes:
        if c.change_type == 'added':
            assert c.do178c_category == 'Category 2'


def test_modified_param_change_is_category2():
    result = _analyzer().analyze()
    for c in result.changes:
        if c.change_type == 'modified' and c.function.name == 'Vehicle::start':
            assert c.do178c_category == 'Category 2'


def test_summary_counts():
    s = _analyzer().analyze().summary
    assert s['added'] >= 1
    assert s['modified'] >= 1
    assert s['unchanged'] >= 1


def test_non_virtual_not_in_changes():
    result = _analyzer().analyze()
    names = {c.function.name for c in result.changes}
    assert 'Vehicle::stop' not in names
    assert 'utilityFunc' not in names


# ── R1 virtual function inventory ──────────────────────────────────────────────

SENSORS = FIXTURES / 'sensors'


def _sensor_result(entry_points=None):
    results = CodeParser().parse_directory(str(SENSORS))
    return VirtualAnalyzer(results, results, entry_points=entry_points).analyze()


def _inventory(result):
    return {v.function: v for v in result.inventory}


def _line_of(filename, text):
    lines = (SENSORS / filename).read_text().splitlines()
    return next(i for i, line in enumerate(lines, 1) if text in line)


def test_inventory_lists_the_root_virtual_functions():
    result = _sensor_result()
    assert result.entry_points == ['main']
    inv = _inventory(result)
    assert {'Sensor::process()', 'Sensor::read()', 'Sensor::selfTest()'} <= set(inv)
    assert not any(k.startswith(('TemperatureSensor::', 'PressureSensor::', 'AltitudeSensor::')) for k in inv)
    # the destructor is run by the compiler when an instantiated subclass is destroyed: an implicit entry point
    assert inv['Sensor::~Sensor()'].reachable
    assert not _inventory(_sensor_result(entry_points=['nothing_here']))['Sensor::process()'].reachable


def test_process_is_overridden_reachable_and_called_from_update():
    process = _inventory(_sensor_result())['Sensor::process()']
    assert (process.class_name, process.short_name) == ('Sensor', 'process')
    assert process.file_path.endswith('sensor.h') and process.line_number == _line_of('sensor.h', 'virtual void process();')
    assert not process.is_pure and process.has_definition and process.reachable
    assert process.requirement_trace == 'SRS-SNS-001'           # from the comment above the declaration
    assert {o.function for o in process.overridden_by} == {'TemperatureSensor::process()', 'PressureSensor::process()'}
    assert all(o.class_instantiated and o.reachable for o in process.overridden_by)
    assert [(c.caller, c.line) for c in process.call_sites] == [
        ('SensorManager::update()', _line_of('sensor_manager.cpp', 's->process();'))]


def test_pure_virtual_read_with_a_never_instantiated_override():
    read = _inventory(_sensor_result())['Sensor::read()']
    assert read.is_pure and not read.has_definition and not read.reachable
    overrides = {o.function: o for o in read.overridden_by}
    assert set(overrides) == {'TemperatureSensor::read()', 'PressureSensor::read()', 'AltitudeSensor::read()'}
    altitude = overrides['AltitudeSensor::read()']
    assert not altitude.class_instantiated and not altitude.reachable
    assert overrides['TemperatureSensor::read()'].class_instantiated and overrides['TemperatureSensor::read()'].reachable
    # calls whose static target is the override itself
    assert [c.caller for c in overrides['TemperatureSensor::read()'].call_sites] == [
        'SensorManager::updateTemperature()', 'TemperatureSensor::process()']
    assert [c.caller for c in read.call_sites] == ['Sensor::process()']


def test_self_test_override_is_never_called():
    self_test = _inventory(_sensor_result())['Sensor::selfTest()']
    assert self_test.call_sites == [] and not self_test.reachable
    assert [o.function for o in self_test.overridden_by] == ['AltitudeSensor::selfTest()']
    assert not self_test.overridden_by[0].reachable


def test_hierarchy_trees_only_for_roots_with_subclasses():
    trees = _sensor_result().hierarchy_trees
    assert list(trees) == ['Sensor']
    assert trees['Sensor'] == ('Sensor\n'
                               '├── AltitudeSensor\n'
                               '├── PressureSensor\n'
                               '└── TemperatureSensor')
    assert _inventory(_sensor_result())['Sensor::process()'].inheritance_tree == trees['Sensor']


def test_reachability_depends_on_the_entry_points():
    inv = _inventory(_sensor_result(entry_points=['SensorManager::updateTemperature']))
    assert not inv['Sensor::process()'].reachable
    assert all(not o.reachable for o in inv['Sensor::process()'].overridden_by)
    temperature_read = next(o for o in inv['Sensor::read()'].overridden_by if o.function == 'TemperatureSensor::read()')
    assert temperature_read.reachable
    bare = _inventory(_sensor_result(entry_points=['updateTemperature']))   # a bare method name resolves too
    assert next(o for o in bare['Sensor::read()'].overridden_by if o.function == 'TemperatureSensor::read()').reachable
    assert not _inventory(_sensor_result(entry_points=['nothing_here']))['Sensor::process()'].reachable


def test_a_built_graph_builder_can_be_passed_in():
    results = CodeParser().parse_directory(str(SENSORS))
    gb = CallGraphBuilder(results)
    gb.build()
    result = VirtualAnalyzer(results, results, graph_builder=gb).analyze()
    assert _inventory(result)['Sensor::process()'].call_sites == [
        s for s in gb.call_sites if s.target == 'Sensor::process()']


def test_virtuals_reached_only_from_an_isr_use_the_entry_point_patterns(tmp_path):
    (tmp_path / 'a.cpp').write_text(
        'class Base { public: virtual void f() {} };\nclass Derived : public Base { public: void f() override {} };\n'
        'void TIMER_IRQHandler() { Derived d; Base* b = &d; b->f(); }\nint main() { return 0; }\n')
    results = CodeParser().parse_directory(str(tmp_path))
    inv = _inventory(VirtualAnalyzer(results, results).analyze())
    assert inv['Base::f()'].reachable and [o.reachable for o in inv['Base::f()'].overridden_by] == [True]
    inv = _inventory(VirtualAnalyzer(results, results, entry_point_patterns=[]).analyze())
    assert not inv['Base::f()'].reachable and [o.reachable for o in inv['Base::f()'].overridden_by] == [False]


def test_out_of_line_virtuals_take_part_in_the_change_classification():
    summary = _sensor_result().summary
    assert summary['unchanged'] >= 8 and summary['added'] == summary['removed'] == summary['modified'] == 0


def test_report_renders_the_inventory(tmp_path):
    result = _sensor_result()
    path = ReportGenerator(Config(output_dir=str(tmp_path))).generate_virtual_analysis(result, format='markdown')
    text = Path(path).read_text(encoding='utf-8')
    assert 'Virtual Function: Sensor::process()' in text
    assert 'Overridden by' in text and 'TemperatureSensor::process()' in text
    assert 'SensorManager::update()' in text and 'Requirement Trace: SRS-SNS-001' in text
    assert '└── TemperatureSensor' in text
    unreachable = text.split('## Overrides not reachable from entry points')[1]
    assert 'AltitudeSensor::read()' in unreachable and 'class never instantiated' in unreachable
    assert 'TemperatureSensor::read()' not in unreachable


def test_report_renders_without_virtual_functions(tmp_path):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'a.cpp').write_text('int main() { return 0; }\n')
    results = CodeParser().parse_directory(str(tmp_path / 'src'))
    result = VirtualAnalyzer(results, results).analyze()
    path = ReportGenerator(Config(output_dir=str(tmp_path / 'out'))).generate_virtual_analysis(result, format='markdown')
    assert 'No virtual functions were found' in Path(path).read_text(encoding='utf-8')


# ── §12 change classification, keyed by sig_key ────────────────────────────────

def _changes(tmp_path, base_src, current_src):
    (tmp_path / 'base').mkdir()
    (tmp_path / 'current').mkdir()
    (tmp_path / 'base' / 'a.cpp').write_text(base_src)
    (tmp_path / 'current' / 'a.cpp').write_text(current_src)
    base = CodeParser().parse_directory(str(tmp_path / 'base'))
    current = CodeParser().parse_directory(str(tmp_path / 'current'))
    return {(c.function.sig_key, c.change_type): c for c in VirtualAnalyzer(base, current).analyze().changes}


def test_new_overload_is_an_addition_not_a_modification(tmp_path):
    changes = _changes(tmp_path,
                       'class B { public: virtual void f() {} };\n',
                       'class B { public: virtual void f() {} virtual void f(int) {} };\n')
    assert set(changes) == {('B::f()', 'unchanged'), ('B::f(int)', 'added')}


def test_parameter_type_change_is_category_2_even_with_an_identical_body(tmp_path):
    changes = _changes(tmp_path,
                       'class B { public: virtual void f(int a) { g(); } };\n',
                       'class B { public: virtual void f(long a) { g(); } };\n')
    assert set(changes) == {('B::f(long)', 'modified')}
    assert changes[('B::f(long)', 'modified')].do178c_category == 'Category 2'


def test_removed_virtual_is_reported_by_sig_key(tmp_path):
    changes = _changes(tmp_path,
                       'class B { public: virtual void f() {} virtual void g() {} };\n',
                       'class B { public: virtual void f() {} };\n')
    assert ('B::g()', 'removed') in changes and changes[('B::g()', 'removed')].do178c_category == 'Category 2'


def test_report_lists_the_call_sites_of_each_override(tmp_path):
    result = _sensor_result()
    path = ReportGenerator(Config(output_dir=str(tmp_path))).generate_virtual_analysis(result, format='markdown')
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('  - TemperatureSensor::read()'))
    block = []
    for line in lines[start + 1:]:
        if not line.startswith('    - '):
            break
        block.append(line)
    assert any('called from SensorManager::updateTemperature()' in line and 'sensor_manager.cpp:' in line for line in block)
    assert any('called from TemperatureSensor::process()' in line for line in block)
