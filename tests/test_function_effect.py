from pathlib import Path

import pytest

from rag_engine.config import Config
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.core.parser import CodeParser
from rag_engine.features.function_effect import (
    FunctionEffectAnalyzer, file_scope_variables, format_effect,
)
from rag_engine.reporting.report_generator import ReportGenerator

SENSORS = Path(__file__).parent / 'fixtures' / 'cpp' / 'sensors'


def _analyzer_for(directory):
    results = CodeParser().parse_directory(str(directory))
    return FunctionEffectAnalyzer(results, CallGraphBuilder(results))


@pytest.fixture(scope='module')
def analyzer():
    return _analyzer_for(SENSORS)


@pytest.fixture(scope='module')
def update_temperature(analyzer):
    effects = analyzer.analyze('updateTemperature')
    assert len(effects) == 1
    return effects[0]


def _names(accesses):
    return [v.name for v in accesses]


def _components(effect, coupling):
    return {c.name: c for c in effect.coupled_components if c.coupling == coupling}


# ── Lookup ──────────────────────────────────────────────────────────────────────

def test_short_qualified_and_sig_key_queries_find_the_same_function(analyzer):
    for query in ('updateTemperature', 'SensorManager::updateTemperature',
                  'SensorManager::updateTemperature()', 'updateTemperature()'):
        assert [e.function for e in analyzer.analyze(query)] == ['SensorManager::updateTemperature()'], query


def test_overloads_and_same_name_in_several_classes_are_all_returned(analyzer):
    assert [e.function for e in analyzer.analyze('process')] == [
        'PressureSensor::process()', 'Sensor::process()', 'TemperatureSensor::process()']
    assert [e.function for e in analyzer.analyze('check(float)')] == ['Validator::check(float)']


def test_unknown_method_gives_no_effects(analyzer):
    assert analyzer.analyze('nope') == []
    assert analyzer.analyze('pdateTemperature') == []      # a suffix of the name is not a match


def test_an_unbuilt_graph_builder_is_built():
    results = CodeParser().parse_directory(str(SENSORS))
    gb = CallGraphBuilder(results)
    assert not gb.built
    FunctionEffectAnalyzer(results, gb)
    assert gb.built


# ── updateTemperature(): the fixture's R2 example ───────────────────────────────

def test_identity_and_location(update_temperature):
    e = update_temperature
    assert (e.function, e.class_name, e.return_type, e.parameters) == (
        'SensorManager::updateTemperature()', 'SensorManager', 'void', [])
    assert Path(e.file_path).name == 'sensor_manager.cpp' and (e.start_line, e.end_line) == (15, 19)


def test_writes_the_member_temperature(update_temperature):
    (w,) = update_temperature.writes
    assert (w.name, w.kind, w.type_, w.owner) == ('temperature', 'member', 'float', 'SensorManager')


def test_reads_members_in_source_order(update_temperature):
    reads = update_temperature.reads
    assert _names(reads) == ['temperature', 'sensor', 'validator', 'display']
    assert {v.kind for v in reads} == {'member'}
    assert {v.name: v.type_ for v in reads}['sensor'] == 'TemperatureSensor'


def test_calls_in_line_order_with_receiver_and_class(update_temperature):
    calls = update_temperature.calls
    assert [(c.callee, c.callee_class, c.receiver, c.line) for c in calls] == [
        ('TemperatureSensor::read()', 'TemperatureSensor', 'sensor', 16),
        ('Validator::check(float)', 'Validator', 'validator', 17),
        ('Display::update(float)', 'Display', 'display', 18),
    ]
    assert {c.call_type for c in calls} == {'direct'}


def test_callers(update_temperature):
    assert update_temperature.callers == ['main()']
    assert update_temperature.transitive_callers == ['main()']


def test_control_coupled_components_with_reasons(update_temperature):
    control = _components(update_temperature, 'control')
    assert set(control) == {'TemperatureSensor', 'Sensor', 'Validator', 'Display'}
    assert control['TemperatureSensor'].reason == "calls TemperatureSensor::read() via member 'sensor'"
    assert control['Validator'].reason == "calls Validator::check(float) via member 'validator'"
    assert control['Sensor'].reason == 'base class of TemperatureSensor'


def test_own_class_is_not_a_coupled_component(update_temperature):
    assert 'SensorManager' not in {c.name for c in update_temperature.coupled_components}
    assert update_temperature.data_coupled_functions == []


def test_affected_components(update_temperature):
    assert update_temperature.affected_components == ['SensorManager', 'main']


def test_generated_description(update_temperature):
    assert update_temperature.description == (
        'Writes temperature; reads sensor, validator, display; calls TemperatureSensor::read(), '
        'Validator::check(float), Display::update(float).')


def test_no_numeric_score_anywhere(update_temperature):
    assert not any(isinstance(v, float) for v in vars(update_temperature).values())
    assert 'score' not in format_effect(update_temperature).lower()


# ── Global variables ────────────────────────────────────────────────────────────

def test_file_scope_variables_include_guarded_header_constants_and_statics():
    found = file_scope_variables(CodeParser().parse_directory(str(SENSORS)))
    assert {'cycleCount', 'TEMP_STALE_MS', 'TEMP_MIN_C', 'TEMP_MAX_C', 'TEMP_DEBOUNCE_C'} <= set(found)
    assert found['cycleCount'].type_ == 'int' and found['cycleCount'].line_number == 3
    assert found['TEMP_MIN_C'].type_ == 'const float'
    assert 'Callback' not in found and 'registerCallback' not in found    # typedef and function


def test_file_scope_variables_in_namespaces_skip_prototypes_and_class_members(tmp_path):
    (tmp_path / 'a.cpp').write_text(
        'namespace ns { static int a = 1, *b; extern int c; int proto(int); int *proto2(int);\n'
        '  void (*fp)(int); struct S { int member; }; }\n'
        'extern "C" { int d; }\n'
        'int arr[3];\n'
        'int f() { static int local = 0; return local; }\n')
    found = file_scope_variables(CodeParser().parse_directory(str(tmp_path)))
    assert set(found) == {'a', 'b', 'c', 'fp', 'd', 'arr'}
    assert found['b'].type_ == 'int *'


def test_global_written_by_a_callback_and_read_by_main(analyzer):
    (e,) = analyzer.analyze('onTimerTick')
    (w,) = e.writes
    assert (w.name, w.kind, w.type_, w.owner) == ('cycleCount', 'global', 'int', None)
    assert 'main()' in e.data_coupled_functions
    assert e.callers == ['main()']
    assert 'main' in e.affected_components


def test_main_reads_the_global_and_is_data_coupled_to_its_writers(analyzer):
    (e,) = analyzer.analyze('main')
    reads = {v.name: v.kind for v in e.reads}
    assert reads['cycleCount'] == 'global'
    assert 'onTimerTick' not in reads                  # a function used as a callback is not a variable
    assert {'onTimerTick()', 'TIMER0_IRQHandler()', 'legacyCalibrate()'} <= set(e.data_coupled_functions)
    assert e.callers == [] and e.transitive_callers == []


# ── Inheritance and virtual dispatch ────────────────────────────────────────────

def test_inherited_member_is_owned_by_the_declaring_class(analyzer):
    effect = next(e for e in analyzer.analyze('process') if e.class_name == 'TemperatureSensor')
    writes = {v.name: v for v in effect.writes}
    assert (writes['value'].kind, writes['value'].owner, writes['value'].type_) == ('member', 'Sensor', 'float')
    assert writes['lastUpdateMs'].type_ == 'unsigned long'


def test_member_data_coupling_follows_the_class_hierarchy_not_siblings(analyzer):
    effect = next(e for e in analyzer.analyze('process') if e.class_name == 'TemperatureSensor')
    assert 'Sensor::process()' in effect.data_coupled_functions and 'Sensor::getValue()' in effect.data_coupled_functions
    assert 'PressureSensor::process()' not in effect.data_coupled_functions
    data = _components(effect, 'data')
    assert set(data) == {'Sensor'} and "member 'value'" in data['Sensor'].reason


def test_virtual_call_reaches_every_instantiated_override(analyzer):
    (update,) = analyzer.analyze('SensorManager::update')
    by_callee = {c.callee: c for c in update.calls}
    assert by_callee['Sensor::process()'].call_type == 'virtual'
    assert by_callee['TemperatureSensor::process()'].call_type == 'virtual'
    assert by_callee['TemperatureSensor::process()'].receiver == 's'
    assert by_callee['TemperatureSensor::process()'].line == by_callee['Sensor::process()'].line == 11
    assert {'Sensor', 'TemperatureSensor', 'PressureSensor'} <= set(_components(update, 'control'))
    temperature_process = next(e for e in analyzer.analyze('process') if e.class_name == 'TemperatureSensor')
    assert temperature_process.callers == ['SensorManager::update()']
    assert temperature_process.transitive_callers == ['SensorManager::update()', 'main()']
    assert {'SensorManager', 'main', 'TemperatureSensor'} <= set(temperature_process.affected_components)


# ── Classification and descriptions on small sources ────────────────────────────

def _effect(tmp_path, source, query):
    (tmp_path / 't.cpp').write_text(source)
    return _analyzer_for(tmp_path).analyze(query)


def test_parameters_globals_and_unknowns(tmp_path):
    (e,) = _effect(tmp_path, 'int g = 0;\nint f(int p) { g = p + MAGIC; return g; }\n', 'f')
    kinds = {v.name: (v.kind, v.type_) for v in e.reads}
    assert kinds == {'p': ('parameter', 'int'), 'MAGIC': ('unknown', ''), 'g': ('global', 'int')}
    assert [(v.name, v.kind) for v in e.writes] == [('g', 'global')]


def test_member_beats_parameter_when_a_setter_shares_the_name(tmp_path):
    (e,) = _effect(tmp_path, 'class C { public: void set(int x); int x; };\n'
                             'void C::set(int x) { this->x = x; }\n', 'set')
    assert [(v.name, v.kind, v.owner) for v in e.writes] == [('x', 'member', 'C')]


def test_free_functions_use_the_file_stem_as_component(tmp_path):
    (tmp_path / 'util.cpp').write_text('int shared = 0;\nvoid bump() { shared++; }\n')
    (tmp_path / 'app.cpp').write_text('extern int shared;\nint peek() { return shared; }\nvoid run() { bump(); }\n')
    analyzer = _analyzer_for(tmp_path)
    (run,) = analyzer.analyze('run')
    assert [c.name for c in run.coupled_components] == ['util'] and run.affected_components == ['app']
    (bump,) = analyzer.analyze('bump')
    assert bump.callers == ['run()'] and bump.affected_components == ['app', 'util']
    assert bump.data_coupled_functions == ['peek()']
    assert [(c.name, c.coupling) for c in bump.coupled_components] == [('app', 'data')]


def test_docstring_first_line_is_the_description(tmp_path):
    source = ('// Computes the checksum.\nint a() { return 1; }\n'
              '/**\n * Reads the bus.\n * More detail.\n */\nint b() { return 2; }\n'
              '// DO-178C-REQ: SRS-1\nint c() { return 3; }\n')
    assert _effect(tmp_path, source, 'a')[0].description == 'Computes the checksum.'
    assert _effect(tmp_path, source, 'b')[0].description == 'Reads the bus.'
    assert _effect(tmp_path, source, 'c')[0].description == 'No variable access or calls.'


# ── Text block and report ───────────────────────────────────────────────────────

def test_format_effect_has_every_r2_section(update_temperature):
    text = format_effect(update_temperature)
    assert FunctionEffectAnalyzer.format_effect(update_temperature) == text
    for heading in ('Method: SensorManager::updateTemperature()', 'Reads:', 'Writes:', 'Calls:', 'Called by:',
                    'Coupled components:', 'Affected if changed:'):
        assert heading in text
    assert "TemperatureSensor (control): calls TemperatureSensor::read() via member 'sensor'" in text
    assert 'Affected if changed: SensorManager, main' in text


def test_report_markdown(tmp_path, analyzer):
    effects = analyzer.analyze('process')
    path = ReportGenerator(Config(output_dir=str(tmp_path))).generate_function_effect(effects, 'markdown')
    assert Path(path).name == 'function_effect_process.md'
    text = Path(path).read_text(encoding='utf-8')
    assert text.count('Coupled components') >= 3
    assert 'TemperatureSensor::process()' in text and 'PressureSensor::process()' in text


def test_report_docx_and_odd_names(tmp_path, analyzer):
    generator = ReportGenerator(Config(output_dir=str(tmp_path)))
    assert Path(generator.generate_function_effect(analyzer.analyze('updateTemperature'))).name == \
        'function_effect_updateTemperature.docx'
    path = generator.generate_function_effect([], 'markdown', query='operator<')
    assert Path(path).name == 'function_effect_operator_.md'
    assert 'No function matches' in Path(path).read_text(encoding='utf-8')
