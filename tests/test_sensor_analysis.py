from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pytest

from rag_engine.config import Config
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.core.parser import CodeParser
from rag_engine.features.sensor_analysis import SensorAnalyzer
from rag_engine.features.sensor_template import SensorTemplate, load_sensor_templates
from rag_engine.reporting.report_generator import ReportGenerator

FIXTURES = Path(__file__).parent / 'fixtures'
SENSORS = FIXTURES / 'cpp' / 'sensors'


def _analyze(parsed, templates):
    gb = CallGraphBuilder(parsed)
    gb.build()
    return SensorAnalyzer(parsed, gb, templates).analyze()


@pytest.fixture(scope='module')
def analysis():
    parsed = CodeParser().parse_directory(str(SENSORS))
    return _analyze(parsed, load_sensor_templates(str(FIXTURES / 'sensor_templates.yaml')))


@pytest.fixture(scope='module')
def findings(analysis):
    return {f.template.name: f for f in analysis.findings}


def _keys(entries: List[str]) -> List[str]:
    """``'A::f() (file.cpp:3)'`` -> ``'A::f()'``."""
    return [e.rsplit(' (', 1)[0] for e in entries]


def _probe(tmp_path, source: str, **template) -> object:
    (tmp_path / 'probe.cpp').write_text(source)
    parsed = CodeParser().parse_directory(str(tmp_path))
    template.setdefault('name', 'Probe')
    return _analyze(parsed, [SensorTemplate(**template)]).findings[0]


# ── Fixture: TemperatureSensor ──────────────────────────────────────────────────

def test_one_finding_per_template_in_order(analysis):
    assert [f.template.name for f in analysis.findings] == [
        'TemperatureSensor', 'PressureSensor', 'AltitudeSensor', 'SpeedSensor']


def test_temperature_class_data_and_hierarchy(findings):
    f = findings['TemperatureSensor']
    assert f.status == 'found' and f.class_name == 'TemperatureSensor'
    assert f.file_path.endswith('temperature_sensor.h') and f.line_number == 10
    assert f.instantiated is True
    assert f.data_members == ['value', 'lastValue', 'lastUpdateMs']
    assert f.hierarchy == ['Temperature Parameter', 'Temperature I/O Group', 'TemperatureSensor',
                           'data: value, lastValue, lastUpdateMs']


def test_temperature_checks_are_all_implemented(findings):
    checks = findings['TemperatureSensor'].checks
    assert list(checks) == ['corrupt', 'stale', 'debounce']
    assert checks['corrupt'].status == 'implemented'
    assert checks['corrupt'].expected == 'range [-54, 54]'
    assert {-54, 54} <= set(checks['corrupt'].found)
    assert checks['stale'].status == 'implemented' and checks['stale'].expected == 'threshold 500 ms'
    assert checks['stale'].found == [500]
    assert checks['debounce'].status == 'implemented' and checks['debounce'].expected == 'threshold 2'
    assert checks['debounce'].found == [2.0]


def test_negative_literal_keeps_its_sign_and_validator_scope_is_searched(findings):
    corrupt = findings['TemperatureSensor'].checks['corrupt']
    assert {-100, 100} <= set(corrupt.found)          # Validator::check: v > -100.0f && v < 100.0f
    assert any('Validator::check(float)' in e and 'v > -100.0f' in e for e in corrupt.evidence)


def test_evidence_names_file_line_function_and_comparison(findings):
    evidence = findings['TemperatureSensor'].checks['corrupt'].evidence
    assert any(re.fullmatch(r'.*temperature_sensor\.cpp:13 TemperatureSensor::inRange\(float\): v >= TEMP_MIN_C',
                            e) for e in evidence), evidence
    stale = findings['TemperatureSensor'].checks['stale'].evidence
    assert len(stale) == 1 and stale[0].endswith(
        'TemperatureSensor::isStale(unsigned long): (nowMs - lastUpdateMs) > TEMP_STALE_MS')


def test_temperature_data_flow(findings):
    f = findings['TemperatureSensor']
    assert 'TemperatureSensor::read()' in _keys(f.sources)
    assert {'TemperatureSensor::process()', 'SensorManager::updateTemperature()'} <= set(_keys(f.processors))
    assert {'TemperatureSensor::inRange(float)', 'TemperatureSensor::isStale(unsigned long)',
            'TemperatureSensor::debounce(float)', 'Validator::check(float)'} <= set(_keys(f.validators))
    assert 'Display::update(float)' in _keys(f.consumers)


def test_data_flow_entries_carry_file_and_line(findings):
    entry = next(e for e in findings['TemperatureSensor'].sources if e.startswith('TemperatureSensor::read()'))
    assert re.fullmatch(r'TemperatureSensor::read\(\) \(.*temperature_sensor\.cpp:4\)', entry)


def test_other_sensors_are_not_consumers_of_temperature_data(findings):
    f = findings['TemperatureSensor']
    # Sensor::process() reaches read() by dispatch, but TemperatureSensor overrides process().
    assert 'Sensor::process()' not in _keys(f.processors)
    assert 'PressureSensor::read()' not in _keys(f.consumers)
    assert not any(k.startswith(('Sensor::', 'TemperatureSensor::', 'PressureSensor::', 'AltitudeSensor::'))
                   for k in _keys(f.consumers))
    assert set(_keys(f.consumers)).isdisjoint(_keys(f.sources) + _keys(f.validators))


# ── Fixture: PressureSensor / AltitudeSensor / SpeedSensor ──────────────────────

def test_pressure_range_mismatch_and_missing_checks(findings):
    f = findings['PressureSensor']
    assert f.status == 'found' and f.instantiated is True
    corrupt = f.checks['corrupt']
    assert corrupt.status == 'mismatch' and corrupt.expected == 'range [0, 110000]'
    assert {0, 120000} <= set(corrupt.found)
    assert any('PressureSensor::process()' in e and 'raw > 120000.0f' in e for e in corrupt.evidence)
    for name in ('stale', 'debounce'):
        assert f.checks[name].status == 'missing' and f.checks[name].found == [] and f.checks[name].evidence == []
    assert 'PressureSensor::read()' in _keys(f.sources)
    assert 'Display::update(float)' not in _keys(f.consumers)


def test_altitude_is_found_but_not_instantiated_and_unchecked(findings):
    f = findings['AltitudeSensor']
    assert f.status == 'found' and f.class_name == 'AltitudeSensor' and f.instantiated is False
    assert {n: c.status for n, c in f.checks.items()} == {
        'corrupt': 'missing', 'stale': 'missing', 'debounce': 'missing'}
    assert 'AltitudeSensor::read()' in _keys(f.sources)
    assert f.processors == []


def test_speed_sensor_is_not_found(findings):
    f = findings['SpeedSensor']
    assert f.status == 'not_found' and f.class_name is None and f.instantiated is False
    assert f.sources == f.processors == f.validators == f.consumers == f.data_members == []
    assert f.checks == {}


# ── Unmapped sensor classes ─────────────────────────────────────────────────────

def test_fully_mapped_fixture_has_no_unmapped_sensor_classes(analysis):
    assert analysis.unmapped_sensor_classes == []


def test_sibling_sensor_classes_without_a_template_are_unmapped():
    parsed = CodeParser().parse_directory(str(SENSORS))
    templates = [t for t in load_sensor_templates(str(FIXTURES / 'sensor_templates.yaml'))
                 if t.name == 'TemperatureSensor']
    # The abstract base `Sensor` is an ancestor of the mapped sensor, not an unmapped sensor.
    assert _analyze(parsed, templates).unmapped_sensor_classes == ['AltitudeSensor', 'PressureSensor']


def test_class_named_sensor_without_template_is_unmapped(tmp_path):
    (tmp_path / 'a.h').write_text('class FuelSensor { public: float read(); };\nclass Gauge { };\n')
    parsed = CodeParser().parse_directory(str(tmp_path))
    result = _analyze(parsed, [SensorTemplate(name='Other')])
    assert result.unmapped_sensor_classes == ['FuelSensor']
    assert result.findings[0].status == 'not_found'


def test_template_class_name_overrides_name():
    parsed = CodeParser().parse_directory(str(SENSORS))
    result = _analyze(parsed, [SensorTemplate(name='Thermo', class_name='TemperatureSensor')])
    assert result.findings[0].class_name == 'TemperatureSensor'
    assert result.unmapped_sensor_classes == ['AltitudeSensor', 'PressureSensor']


# ── Check evaluation on small sources ───────────────────────────────────────────

PROBE = '''
#define STALE_LIMIT_MS 250UL
#define LOW (-5.5f)
const float HI = 10.0F;
const unsigned long HEX_LIMIT = 0x1F;
class Probe {
public:
    float read() { return 1.0f; }
    bool ok(float v, unsigned long now) {
        return v >= LOW && v <= HI && (now - lastMs) > STALE_LIMIT_MS;
    }
    float smooth(float v) { return fabs(v - prev) <= 0.5 ? prev : v; }
private:
    float prev = 0;
    unsigned long lastMs = 0;
};
'''


def test_defines_consts_suffixes_and_parenthesised_negatives_resolve(tmp_path):
    f = _probe(tmp_path, PROBE, valid_min=-5.5, valid_max=10.0, stale_threshold_ms=250, debounce_threshold=0.5)
    assert {n: c.status for n, c in f.checks.items()} == {
        'corrupt': 'implemented', 'stale': 'implemented', 'debounce': 'implemented'}
    assert f.checks['corrupt'].found == [-5.5, 10.0]
    assert f.checks['stale'].found == [250] and f.checks['debounce'].found == [0.5]


def test_other_constants_make_a_mismatch(tmp_path):
    f = _probe(tmp_path, PROBE, valid_min=-5.5, valid_max=12.0, stale_threshold_ms=300, debounce_threshold=0.25)
    assert {n: c.status for n, c in f.checks.items()} == {
        'corrupt': 'mismatch', 'stale': 'mismatch', 'debounce': 'mismatch'}
    assert f.checks['stale'].expected == 'threshold 300 ms'


def test_hex_literal_constant(tmp_path):
    f = _probe(tmp_path, PROBE.replace('> STALE_LIMIT_MS', '> HEX_LIMIT'), stale_threshold_ms=31)
    assert f.checks['stale'].status == 'implemented' and f.checks['stale'].found == [31]


def test_abs_of_a_difference_is_debounce_even_without_telling_names(tmp_path):
    src = 'class Probe { public: float read() { return 0; } bool same(float a, float b) { return fabs(a - b) < 0.25f; } };'
    f = _probe(tmp_path, src, debounce_threshold=0.25, valid_min=0.0, valid_max=1.0)
    assert f.checks['debounce'].status == 'implemented' and f.checks['debounce'].found == [0.25]
    assert f.checks['corrupt'].status == 'missing'


def test_float_equality_uses_a_relative_tolerance(tmp_path):
    src = 'class Probe { public: float read() { return 0; } bool hi(double v) { return v < 0.3; } };'
    f = _probe(tmp_path, src, valid_min=None, valid_max=0.1 + 0.2)
    assert f.checks['corrupt'].status == 'implemented'


def test_comparison_with_a_non_constant_is_not_evidence(tmp_path):
    src = 'class Probe { public: float read() { return 0; } bool lt(float a, float b) { return a < b; } };'
    f = _probe(tmp_path, src, valid_min=0.0, valid_max=1.0)
    assert f.checks['corrupt'].status == 'missing'


def test_disabled_checks_are_not_required_and_unset_values_are_not_configured(tmp_path):
    f = _probe(tmp_path, PROBE, valid_min=-5.5, valid_max=10.0, corrupt_range_check=False,
               debounce_enabled=False, debounce_threshold=0.5)
    assert f.checks['corrupt'].status == 'not_required'
    assert f.checks['debounce'].status == 'not_required'
    assert f.checks['stale'].status == 'not_configured'    # no stale threshold in the template
    assert all(c.found == [] and c.evidence == [] for c in f.checks.values())


def test_missing_template_values_are_not_configured(tmp_path):
    f = _probe(tmp_path, PROBE)       # no range, no stale threshold, debounce enabled without a threshold
    assert {n: c.status for n, c in f.checks.items()} == {
        'corrupt': 'not_configured', 'stale': 'not_configured', 'debounce': 'not_configured'}
    assert f.checks['corrupt'].expected == 'no valid_range configured'
    assert f.checks['debounce'].expected == 'no debounce threshold configured'


def test_one_sided_range_is_checked_against_the_configured_bound(tmp_path):
    f = _probe(tmp_path, PROBE, valid_max=10.0)
    assert f.checks['corrupt'].status == 'implemented' and f.checks['corrupt'].expected == 'range [-inf, 10]'


def test_checks_only_see_the_sensors_own_scope(tmp_path):
    src = PROBE + '\nbool unrelated(float v) { return v > 777.0f; }\n'
    f = _probe(tmp_path, src, valid_min=-5.5, valid_max=10.0)
    assert 777.0 not in f.checks['corrupt'].found


def test_class_without_fields_has_no_data_element_in_the_hierarchy(tmp_path):
    f = _probe(tmp_path, 'class Probe { public: float read() { return 0; } };', parameter_group='PG')
    assert f.hierarchy == ['PG', 'Probe'] and f.data_members == []


# ── Report ──────────────────────────────────────────────────────────────────────

def test_report_renders_hierarchy_flow_checks_and_unmapped(tmp_path, analysis):
    path = ReportGenerator(Config(output_dir=str(tmp_path))).generate_sensor_analysis(analysis, format='markdown')
    assert path.endswith('sensor_analysis.md')
    text = Path(path).read_text(encoding='utf-8')
    assert '## Sensor: TemperatureSensor' in text
    assert 'Temperature Parameter → Temperature I/O Group → TemperatureSensor → data: value, lastValue, lastUpdateMs' in text
    assert '| Check | Status | Expected | Found | Evidence |' in text
    assert re.search(r'\| corrupt \| implemented \| range \[-54, 54\] \|', text)
    assert re.search(r'\| corrupt \| mismatch \| range \[0, 110000\] \|', text)
    assert 'Validator::check(float)' in text and 'Display::update(float)' in text
    assert re.search(r'## Sensor: SpeedSensor\n(?:.*\n)*?.*not found', text)
    assert '## Unmapped Sensor Classes' in text


def test_report_lists_unmapped_classes_and_escapes_pipes(tmp_path):
    (tmp_path / 'a.cpp').write_text(
        'class FuelSensor { public: float read() { return 0; } };\n'
        'class Probe { public: float read() { return 0; } bool c(int v) { return (v | 1) > 5; } };\n')
    parsed = CodeParser().parse_directory(str(tmp_path))
    result = _analyze(parsed, [SensorTemplate(name='Probe', valid_min=0, valid_max=9)])
    path = ReportGenerator(Config(output_dir=str(tmp_path / 'out'))).generate_sensor_analysis(result, format='markdown')
    text = Path(path).read_text(encoding='utf-8')
    assert '- FuelSensor' in text.split('## Unmapped Sensor Classes')[1]
    assert r'(v \| 1) > 5' in text


def test_report_docx(tmp_path, analysis):
    path = ReportGenerator(Config(output_dir=str(tmp_path))).generate_sensor_analysis(analysis)
    assert path.endswith('sensor_analysis.docx') and Path(path).stat().st_size > 0


# ── Fix round 1: siblings, classification, family, getters, narrative ────────────

def _two_sensors(tmp_path, source: str, *templates: SensorTemplate):
    (tmp_path / 'probe.cpp').write_text(source)
    return _analyze(CodeParser().parse_directory(str(tmp_path)), list(templates))


SIBLINGS = '''
class Sensor {
public:
    virtual ~Sensor() {}
    void process() { float v = read(); validate(v); }
    virtual float read() = 0;
    virtual bool validate(float v) = 0;
};
class TempA : public Sensor {
public:
    float read() override { return 1.0f; }
    bool validate(float v) override { return v >= -54.0f && v <= 54.0f; }
};
class TempB : public Sensor {
public:
    float read() override { return 2.0f; }
    bool validate(float v) override { return true; }
};
class Mgr {
public:
    void run() { a.process(); b.process(); }
private:
    TempA a;
    TempB b;
};
'''


def test_a_siblings_validator_does_not_count_for_the_other_sensor(tmp_path):
    result = _two_sensors(tmp_path, SIBLINGS,
                          SensorTemplate(name='TempA', valid_min=-54, valid_max=54),
                          SensorTemplate(name='TempB', valid_min=-54, valid_max=54))
    a, b = result.findings
    assert a.checks['corrupt'].status == 'implemented'
    assert _keys(a.validators) == ['TempA::validate(float)']
    assert b.checks['corrupt'].status == 'missing' and b.checks['corrupt'].evidence == []
    assert _keys(b.validators) == ['TempB::validate(float)']
    assert 'Sensor::process()' in _keys(b.processors)         # inherited, not overridden: still a processor
    assert not any(k.startswith('TempA::') for k in _keys(b.processors + b.validators + b.consumers))


OVERRIDE = '''
class Sensor {
public:
    virtual float read() = 0;
    virtual void process() { float v = read(); if (v > 999.0f) { } }
};
class Probe : public Sensor {
public:
    float read() override { return 1.0f; }
    void process() override { float v = read(); }
};
class Mgr { public: void run() { p.process(); } private: Probe p; };
'''


def test_an_ancestors_overridden_method_is_neither_processor_nor_scope(tmp_path):
    (f,) = _two_sensors(tmp_path, OVERRIDE, SensorTemplate(name='Probe', valid_min=0, valid_max=999)).findings
    assert 'Sensor::process()' not in _keys(f.processors)
    assert 'Probe::process()' in _keys(f.processors)
    assert f.checks['corrupt'].status == 'missing' and 999.0 not in f.checks['corrupt'].found


def test_tokenised_names_rpms_is_a_range_check_not_stale(tmp_path):
    src = ('class RpmSensor { public: float read() { return rpms; }'
           ' bool ok() { return rpms >= 0 && rpms <= 8000; } private: float rpms = 0; };')
    f = _probe(tmp_path, src, name='RpmSensor', valid_min=0, valid_max=8000, stale_threshold_ms=8000)
    assert f.checks['corrupt'].status == 'implemented' and f.checks['corrupt'].found == [0, 8000]
    assert f.checks['stale'].status == 'missing'


def test_items_and_joystick_are_not_time_like(tmp_path):
    src = ('class Probe { public: float read() { return 0; }'
           ' bool a() { return items_ > 3; } bool b(float joystick) { return joystick > 3; }'
           ' private: int items_ = 0; };')
    f = _probe(tmp_path, src, stale_threshold_ms=3, valid_min=0, valid_max=3)
    assert f.checks['stale'].status == 'missing'
    assert f.checks['corrupt'].status == 'mismatch'         # joystick is a parameter: sensor data, but only 3


@pytest.mark.parametrize('ident', ['ticks', 'elapsed_us', 'ageMs', 'timestamp', 'nowUs', 'TIMEOUT_MS', 'msec'])
def test_time_like_identifiers_are_tokenised_on_underscores_and_camel_case(tmp_path, ident):
    src = f'class Probe {{ public: float read() {{ return 0; }} bool late(unsigned {ident}) {{ return {ident} > 50; }} }};'
    f = _probe(tmp_path, src, stale_threshold_ms=50)
    assert f.checks['stale'].status == 'implemented', ident


@pytest.mark.parametrize('ident', ['previous', 'prevValue', 'delta_v', 'lastGood', 'diff', 'stable_count'])
def test_debounce_identifiers_are_tokenised(tmp_path, ident):
    src = f'class Probe {{ public: float read() {{ return 0; }} bool same(float {ident}) {{ return {ident} < 0.5; }} }};'
    f = _probe(tmp_path, src, debounce_threshold=0.5)
    assert f.checks['debounce'].status == 'implemented', ident


def test_loop_condition_is_not_a_range_check(tmp_path):
    src = ('class Probe { public: float read() { return 0; } };\n'
           'class Mgr { public: void run() { float v = p.read();'
           ' for (int i = 0; i < 4; i++) { total += v; } if (v > 9.0f) { } }'
           ' private: Probe p; float total = 0; };')
    f = _probe(tmp_path, src, valid_min=0, valid_max=4)
    assert 'Mgr::run()' in _keys(f.processors)
    assert f.checks['corrupt'].status == 'mismatch' and f.checks['corrupt'].found == [9.0]


def test_range_comparison_needs_a_sensor_data_operand(tmp_path):
    src = ('class Probe { public: float read() { return 0; } int count() { int n = other(); return n > 7; }'
           ' int other(); float reading() { float raw = read(); return raw > 5.0f; } };')
    f = _probe(tmp_path, src, valid_min=0, valid_max=7)
    assert f.checks['corrupt'].found == [5.0]       # raw comes from read(); n does not


def test_variable_assigned_from_the_source_call_is_sensor_data(tmp_path):
    src = ('class Probe { public: float read() { return 0; } };\n'
           'class Mgr { public: void run() { reading = p.read(); if (reading > 9.0f) { } }'
           ' private: Probe p; float reading = 0; };')
    f = _probe(tmp_path, src, valid_max=9.0)
    assert f.checks['corrupt'].status == 'implemented'


FAMILY = '''
class Component { public: virtual ~Component() {} };
class Sensor : public Component { public: virtual float read() = 0; };
class Probe : public Sensor { public: float read() override { return 1.0f; } };
class Display : public Component { public: void update(float v) { shown = v; } private: float shown = 0; };
class Mgr {
public:
    void tick() { float t = probe.read(); display.update(t); }
private:
    Probe probe;
    Display display;
};
'''


def test_family_root_is_the_top_declarer_of_the_source_method(tmp_path):
    (f,) = _two_sensors(tmp_path, FAMILY, SensorTemplate(name='Probe')).findings
    assert _keys(f.consumers) == ['Display::update(float)']       # a Component, but not a Sensor


def test_concrete_intermediate_sensor_class_is_reported_when_unmapped(tmp_path):
    src = ('class Sensor { public: virtual float read() = 0; };\n'
           'class TemperatureSensor : public Sensor { public: float read() override { return 1; } };\n'
           'class RedundantTemperatureSensor : public TemperatureSensor { public: float read() override { return 2; } };\n')
    result = _two_sensors(tmp_path, src, SensorTemplate(name='RedundantTemperatureSensor'))
    assert result.unmapped_sensor_classes == ['TemperatureSensor']   # abstract Sensor is not a sensor to map


def test_family_members_outside_the_sensor_naming_are_unmapped(tmp_path):
    src = ('class Sensor { public: virtual float read() = 0; };\n'
           'class Probe : public Sensor { public: float read() override { return 1; } };\n'
           'class Gyro : public Sensor { public: float read() override { return 2; } };\n')
    assert _two_sensors(tmp_path, src, SensorTemplate(name='Probe')).unmapped_sensor_classes == ['Gyro']


LOGGER = '''
class Logger {
public:
    int getId() const { return id; }
    void log(int level) { if (level > 3) { } }
private:
    int id = 0;
};
class App { public: void run() { int n = logger.getId(); logger.log(n); } private: Logger logger; };
'''


def test_an_unrelated_getter_is_not_a_source(tmp_path):
    f = _probe(tmp_path, LOGGER, name='Logger')
    assert f.sources == [] and f.processors == [] and f.consumers == []


def test_getter_is_a_source_when_there_is_no_reader_and_the_type_matches(tmp_path):
    src = ('class Probe { public: float getTemp() const { return t; } int getId() const { return 1; } private: float t = 0; };\n'
           'class App { public: void run() { float x = probe.getTemp(); log(x); } void log(float v) { } private: Probe probe; };')
    f = _probe(tmp_path, src, data_type='float')
    assert _keys(f.sources) == ['Probe::getTemp()'] and _keys(f.processors) == ['App::run()']
    g = _probe(tmp_path, src, data_type='int')
    assert _keys(g.sources) == ['Probe::getId()']


def test_getters_are_ignored_when_the_class_has_a_reader(tmp_path):
    src = 'class Probe { public: float read() { return 0; } float getTemp() const { return 1; } };'
    assert _keys(_probe(tmp_path, src).sources) == ['Probe::read()']


def test_fixture_getter_is_no_longer_a_source(findings):
    assert not any('getValue' in s for f in findings.values() for s in f.sources)


# ── Fix round 1: report ─────────────────────────────────────────────────────────

def test_report_renders_the_optional_narrative(tmp_path, analysis):
    generator = ReportGenerator(Config(output_dir=str(tmp_path)))
    plain = Path(generator.generate_sensor_analysis(analysis, format='markdown')).read_text(encoding='utf-8')
    assert 'AI Certification Assessment' not in plain
    path = generator.generate_sensor_analysis(analysis, format='markdown', llm_narrative='Looks fine.')
    assert '## AI Certification Assessment\n\nLooks fine.' in Path(path).read_text(encoding='utf-8')


def test_report_distinguishes_not_configured_from_not_required(tmp_path):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'p.cpp').write_text(PROBE)
    parsed = CodeParser().parse_directory(str(tmp_path / 'src'))
    result = _analyze(parsed, [SensorTemplate(name='Probe', valid_min=-5.5, valid_max=10.0,
                                              debounce_enabled=False)])
    path = ReportGenerator(Config(output_dir=str(tmp_path / 'out'))).generate_sensor_analysis(result, format='markdown')
    text = Path(path).read_text(encoding='utf-8')
    assert re.search(r'\| stale \| not_configured \| no stale threshold configured \|', text)
    assert re.search(r'\| debounce \| not_required \| debounce disabled \|', text)
    assert 'Not configured in the template: stale' in text
