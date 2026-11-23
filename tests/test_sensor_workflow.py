"""Configuration-driven sensor workflow, end to end: YAML -> discovery and mapping -> generic engine ->
C++ verification (incl. inconclusive) -> CSV replay -> report -> question answering.

All limits are synthetic, illustrative values for tests, not aerospace limits."""
import textwrap

import pytest

from rag_engine.config import Config
from rag_engine.features.sensor_data_checker import replay_samples
from rag_engine.features.sensor_template import SensorTemplate, load_sensor_templates
from rag_engine.features.sensor_validation import Debouncer, check_requirements, is_corrupt, is_frozen, is_stale
from rag_engine.llm.offline_qa import answer_question
from rag_engine.pipeline import InputError, run_pipeline

CODE = {
    'checks.h': """
        #ifndef CHECKS_H
        #define CHECKS_H
        #include <cmath>
        // Shared generic validators: written once, every sensor passes its own limits.
        inline bool inRange(float v, float lo, float hi) { return v >= lo && v <= hi; }
        inline bool isStale(unsigned long nowMs, unsigned long lastMs, unsigned long thresholdMs) {
            return (nowMs - lastMs) > thresholdMs;
        }
        inline bool jumpTooLarge(float v, float prev, float threshold) { return std::fabs(v - prev) > threshold; }
        #endif
    """,
    'sensors.h': """
        #include "checks.h"
        class SensorBase {
        public:
            virtual float read() = 0;
        protected:
            float value = 0.0f;
            float lastValue = 0.0f;
            unsigned long lastUpdateMs = 0;
        };
        const float ENG_MIN = -40.0f;   const float ENG_MAX = 150.0f;
        const unsigned long ENG_STALE_MS = 1000;   const float ENG_JUMP = 2.0f;
        const float CAB_MIN = -20.0f;   const float CAB_MAX = 80.0f;
        const unsigned long CAB_STALE_MS = 1500;   // YAML says 2000: mismatch
        class EngineTemperatureSensor : public SensorBase {
        public:
            float read() override { return 20.0f; }
            void process(unsigned long nowMs);
        };
        class CabinTemperatureSensor : public SensorBase {
        public:
            float read() override { return 21.0f; }
            void process(unsigned long nowMs);
        };
        // Limits only known at run time: cannot be verified statically.
        class OilPressureSensor : public SensorBase {
        public:
            float read() override { return 3.0f; }
            void process();
            float maxLimit = 0.0f;
        };
        // Not named *Sensor: discovered through its base class.
        class FuelProbe : public SensorBase {
        public:
            float read() override { return 1.0f; }
        };
        // No sensor base, no *Sensor name: discovered through its numeric read().
        class Thermocouple {
        public:
            double sample() { return 0.0; }
        };
        class Display { public: void show(float v); };
    """,
    'sensors.cpp': """
        #include "sensors.h"
        Display display;
        void Display::show(float v) { value_ = v; }
        void EngineTemperatureSensor::process(unsigned long nowMs) {
            float raw = read();
            if (!inRange(raw, ENG_MIN, ENG_MAX)) { return; }
            if (isStale(nowMs, lastUpdateMs, ENG_STALE_MS)) { return; }
            if (jumpTooLarge(raw, lastValue, ENG_JUMP)) { return; }
            lastValue = raw;
            display.show(raw);
        }
        void CabinTemperatureSensor::process(unsigned long nowMs) {
            float raw = read();
            if (!inRange(raw, CAB_MIN, CAB_MAX)) { return; }
            if (isStale(nowMs, lastUpdateMs, CAB_STALE_MS)) { return; }
            display.show(raw);
        }
        void OilPressureSensor::process() {
            float raw = read();
            if (raw > maxLimit) { return; }
            display.show(raw);
        }
        int main() {
            EngineTemperatureSensor e; CabinTemperatureSensor c; OilPressureSensor o;
            e.process(0); c.process(0); o.process();
            return 0;
        }
    """,
}

YAML = """
defaults:
  debounce: {count: 3}
sensors:
  - name: EngineTemperature
    category: temperature
    class_name: EngineTemperatureSensor
    unit: Celsius
    valid_range: {min: -40, max: 150}
    stale: {threshold_ms: 1000}
    debounce: {enabled: true, threshold: 2}
  - name: CabinTemperature            # same category, different limits
    category: temperature
    class_name: CabinTemperatureSensor
    valid_range: {min: -20, max: 80}
    stale: {threshold_ms: 2000}
    debounce: {enabled: false}
  - name: OilPressure
    category: pressure
    class_name: OilPressureSensor
    valid_range: {min: 0, max: 10}
    debounce: {enabled: false}
  - name: HydraulicPressure           # no such class
    category: pressure
    valid_range: {min: 0, max: 300}
"""

SAMPLES = """sensor,timestamp_ms,value
EngineTemperature,0,20
EngineTemperature,100,200
EngineTemperature,200,20
EngineTemperature,2000,20
CabinTemperatureSensor,0,25
Unknown,0,1
"""


@pytest.fixture(scope='module')
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp('sensor_project')
    (root / 'src').mkdir()
    for name, text in CODE.items():
        (root / 'src' / name).write_text(textwrap.dedent(text))
    (root / 'sensors.yaml').write_text(YAML)
    (root / 'samples.csv').write_text(SAMPLES)
    return root


@pytest.fixture(scope='module')
def results(project):
    cfg = Config(sensor_templates_file=str(project / 'sensors.yaml'), sensor_samples_file=str(project / 'samples.csv'),
                 output_dir=str(project / 'out'), cache_dir=str(project / 'cache'))
    return run_pipeline(cfg, str(project / 'src'), output_format='markdown', use_semantic_search=False)


def _f(results, name):
    return next(f for f in results.sensors.findings if f.template.name == name)


# ── 1-3: YAML ─────────────────────────────────────────────────────────────────

def test_yaml_category_and_same_category_with_different_limits(project):
    t = {x.name: x for x in load_sensor_templates(project / 'sensors.yaml')}
    assert t['EngineTemperature'].category == t['CabinTemperature'].category == 'temperature'
    assert (t['EngineTemperature'].valid_min, t['EngineTemperature'].valid_max) == (-40, 150)
    assert (t['CabinTemperature'].valid_min, t['CabinTemperature'].valid_max) == (-20, 80)


@pytest.mark.parametrize('entry, message', [
    ("{name: A, category: temperature, type: pressure}", 'disagree'),
    ("{name: A, attributes: [engine]}", 'attributes must be a mapping'),
    ("{name: A, valid_range: {min: 5, max: 1}}", 'must be less than max'),
    ("{name: A, stale: {threshold_ms: -1}}", 'must not be negative'),
    ("{name: A, debounce: {count: 0}}", 'debounce.count must be >= 1'),
    ("{name: A, valid_rnage: {min: 0, max: 1}}", "unknown key 'valid_rnage'"),
])
def test_invalid_yaml_is_rejected_with_a_message(tmp_path, entry, message):
    path = tmp_path / 's.yaml'
    path.write_text(f"sensors:\n  - {entry}\n")
    with pytest.raises(ValueError, match=message):
        load_sensor_templates(path)


def test_attributes_are_kept_for_later_phases(tmp_path):
    path = tmp_path / 's.yaml'
    path.write_text("sensors:\n  - {name: A, category: Temperature, attributes: {location: engine}}\n")
    tpl = load_sensor_templates(path)[0]
    assert tpl.category == 'temperature' and tpl.attributes == {'location': 'engine'}


# ── Generic engine ───────────────────────────────────────────────────────────

def test_generic_rules_use_each_sensors_parameters():
    a = SensorTemplate('A', valid_min=0, valid_max=10, stale_threshold_ms=100, debounce_threshold=1, debounce_count=2)
    b = SensorTemplate('B', valid_min=0, valid_max=100, stale_threshold_ms=1000, debounce_enabled=False)
    assert is_corrupt(a, 50) and not is_corrupt(b, 50)
    assert is_stale(a, 300, 100) and not is_stale(b, 300, 100)
    assert is_frozen(a, 300, 100) and not is_frozen(SensorTemplate('C', stale_threshold_ms=100, stale_detect_frozen=False), 300, 100)
    assert check_requirements(b)['debounce'].state == 'disabled'
    d = Debouncer(a)
    assert [d.update(v) for v in (1, 5, 5, 5)] == [False, True, False, False] and d.accepted == 5


# ── 4-8: discovery and mapping ───────────────────────────────────────────────

def test_discovery_does_not_rely_on_sensor_names(results):
    s = results.sensors
    assert {'FuelProbe', 'Thermocouple'} <= set(s.unmapped_sensor_classes)      # base class / numeric reader
    assert 'Display' not in s.discovered_sensor_classes and 'SensorBase' not in s.discovered_sensor_classes
    assert _f(results, 'EngineTemperature').class_name == 'EngineTemperatureSensor'


def test_unused_configuration_is_reported(results):
    assert _f(results, 'HydraulicPressure').status == 'not_found'
    assert results.sensors.unused_configs == ['HydraulicPressure']


def test_two_configurations_for_one_class_are_ambiguous(project, tmp_path):
    yml = tmp_path / 's.yaml'
    yml.write_text("sensors:\n  - {name: A, class_name: CabinTemperatureSensor}\n"
                   "  - {name: B, class_name: CabinTemperatureSensor}\n")
    r = run_pipeline(Config(sensor_templates_file=str(yml), cache_dir=str(tmp_path / 'c')), str(project / 'src'),
                     use_semantic_search=False, write_reports=False)
    assert [f.status for f in r.sensors.findings] == ['ambiguous', 'ambiguous']
    assert all(not f.checks for f in r.sensors.findings)              # no configuration silently chosen


def test_same_short_name_in_two_namespaces_is_ambiguous(tmp_path):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'a.cpp').write_text('namespace a { class Probe { public: float read(); }; }\n'
                                            'namespace b { class Probe { public: float read(); }; }\n')
    yml = tmp_path / 's.yaml'
    yml.write_text("sensors:\n  - {name: Probe}\n")
    r = run_pipeline(Config(sensor_templates_file=str(yml), cache_dir=str(tmp_path / 'c')), str(tmp_path / 'src'),
                     use_semantic_search=False, write_reports=False)
    f = r.sensors.findings[0]
    assert f.status == 'ambiguous' and f.candidates == ['a::Probe', 'b::Probe']


# ── 9-15, 19: verification against the YAML ─────────────────────────────────

def test_shared_validators_with_matching_limits_are_implemented(results):
    checks = _f(results, 'EngineTemperature').checks
    assert {n: c.status for n, c in checks.items()} == {'corrupt': 'implemented', 'stale': 'implemented',
                                                         'debounce': 'implemented'}
    assert checks['corrupt'].confidence == 'high' and checks['corrupt'].found == [-40, 150]


def test_wrong_stale_threshold_is_a_mismatch_and_disabled_debounce_is_not_a_failure(results):
    checks = _f(results, 'CabinTemperature').checks
    assert checks['corrupt'].status == 'implemented'
    assert checks['stale'].status == 'mismatch' and checks['stale'].found == [1500]
    assert 'Configured 2000 not used' in checks['stale'].note and checks['stale'].recommendation
    assert checks['debounce'].status == 'disabled'


def test_run_time_limit_is_inconclusive_and_absent_check_is_missing(results):
    checks = _f(results, 'OilPressure').checks
    assert checks['corrupt'].status == 'inconclusive' and checks['corrupt'].confidence == 'low'
    assert 'not compile-time constants' in checks['corrupt'].note
    assert checks['stale'].status == 'not_configured'


def test_no_acquisition_function_is_inconclusive(tmp_path):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'a.cpp').write_text('class Gauge { public: void update(); float level; };\n')
    yml = tmp_path / 's.yaml'
    yml.write_text("sensors:\n  - {name: Gauge, valid_range: {min: 0, max: 1}}\n")
    r = run_pipeline(Config(sensor_templates_file=str(yml), cache_dir=str(tmp_path / 'c')), str(tmp_path / 'src'),
                     use_semantic_search=False, write_reports=False)
    corrupt = r.sensors.findings[0].checks['corrupt']
    assert corrupt.status == 'inconclusive' and 'could not be traced' in corrupt.note
    assert any('Gauge' in l for l in r.sensors.limitations)


def test_data_flow_is_traced(results):
    f = _f(results, 'EngineTemperature')
    keys = lambda items: [i.split(' (')[0] for i in items]   # noqa: E731
    assert keys(f.sources) == ['EngineTemperatureSensor::read()']
    assert 'EngineTemperatureSensor::process(unsigned long)' in keys(f.processors)
    assert {'inRange(float, float, float)', 'isStale(unsigned long, unsigned long, unsigned long)'} <= set(keys(f.validators))
    assert 'Display::show(float)' in keys(f.consumers)


# ── 16: CSV replay through the same engine ───────────────────────────────────

def test_csv_replay(results):
    replay = results.sensor_replay
    eng = replay.sensors['EngineTemperature']
    # t=2000: 1800 ms since the last update (stale) and since the last change of value (frozen)
    assert [s.flags for s in eng.results] == [[], ['corrupt'], [], ['stale', 'frozen']]
    assert 'CabinTemperature' in replay.sensors                                   # matched by class_name
    assert replay.unknown_sensors == ['Unknown']
    assert set(replay.without_samples) == {'OilPressure', 'HydraulicPressure'}


def test_bad_samples_csv_is_an_input_error(project, tmp_path):
    bad = tmp_path / 'bad.csv'
    bad.write_text('sensor,value\nA,1\n')
    cfg = Config(sensor_templates_file=str(project / 'sensors.yaml'), sensor_samples_file=str(bad))
    with pytest.raises(InputError, match='header'):
        run_pipeline(cfg, str(project / 'src'), use_semantic_search=False, write_reports=False)


def test_unchanged_value_with_fresh_updates_is_frozen_not_stale():
    tpl = SensorTemplate('A', stale_threshold_ms=100)
    r = replay_samples([tpl], {'A': [(t, 5.0) for t in range(0, 400, 50)]}).sensors['A']
    assert r.counts['stale'] == 0 and r.counts['frozen'] > 0


# ── 17: report ───────────────────────────────────────────────────────────────

def test_report_sections_and_checklist(project, results):
    text = (project / 'out' / 'sensor_analysis.md').read_text()
    for section in ('## A. Overall Summary', '## B. Sensor Inventory', '## C. Validation Results',
                    '## D. Issues and Recommendations', '## CSV Replay', '## E. Analysis Limitations'):
        assert section in text
    assert '| CabinTemperature | stale | enabled: threshold 2000 ms | detected in' in text
    assert '| CabinTemperature | debounce | disabled | not verified | disabled | - |' in text
    assert '**OilPressure — corrupt inconclusive.**' in text
    assert '**Thermocouple — no YAML configuration.**' in text
    assert 'do not demonstrate DO-178C' in text
    sen03 = next(i for i in results.checklist.items if i.item_id == 'SEN-03')
    assert sen03.status == 'OPEN' and 'OilPressure.corrupt' in sen03.evidence


# ── 18: question answering from the structured results ───────────────────────

@pytest.mark.parametrize('question, expected', [
    ('Which sensors were discovered?', ['EngineTemperature', 'Thermocouple', 'FuelProbe']),
    ('What is the category of CabinTemperature?', ['category temperature']),
    ('Which YAML configuration is mapped to CabinTemperatureSensor?', ['CabinTemperature', 'mapped to C++ class CabinTemperatureSensor']),
    ('What is the valid range of OilPressure?', ['valid range [0, 10]']),
    ('Where is EngineTemperature acquired?', ['EngineTemperatureSensor::read()']),
    ('Is the stale check implemented for CabinTemperature?', ['stale check: mismatch', '1500']),
    ('Does EngineTemperature match the YAML?', ['matches the YAML for every enabled check']),
    ('Why is OilPressure inconclusive?', ['not compile-time constants']),
    ('Which sensors have missing validation checks?', ['Missing sensor checks']),
    ('Which sensors have configuration mismatches?', ['CabinTemperature.stale: mismatch']),
    ('What is the data flow of EngineTemperature?', ['Acquisition', 'Display::show(float)']),
    ('What improvements are needed for CabinTemperature?', ['Recommended changes for CabinTemperature']),
    ('What does the CSV replay show for EngineTemperature?', ['corrupt 1', 'stale 1', 'not proof']),
    ('Is HydraulicPressure debounce implemented?', ['could not be established']),
])
def test_sensor_questions(results, question, expected):
    answer = answer_question(question, results)
    for text in expected:
        assert text in answer, (question, answer)
