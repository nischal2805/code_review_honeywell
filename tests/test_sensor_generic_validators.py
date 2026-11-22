"""A sensor added only through YAML is checked by shared, generic C++ validators.

The validators are written once (``sensor_checks.h``); each sensor passes its own limits at the call site.
The analyzer binds those constant arguments to the validator's parameters, so no sensor-specific check code
is needed for a new sensor to be mapped, checked against its YAML parameters, reported and queryable."""
import shutil
from pathlib import Path

import pytest

from rag_engine.config import Config
from rag_engine.llm.offline_qa import answer_question
from rag_engine.pipeline import run_pipeline

FIXTURE = Path(__file__).parent / 'fixtures' / 'cpp' / 'sensors' / 'sensor.h'

CHECKS_H = """#ifndef SENSOR_CHECKS_H
#define SENSOR_CHECKS_H
#include <cmath>
inline bool inRange(float v, float lo, float hi) { return v >= lo && v <= hi; }
inline bool isStale(unsigned long nowMs, unsigned long lastMs, unsigned long thresholdMs) {
    return (nowMs - lastMs) > thresholdMs;
}
inline bool debounceExceeded(float v, float prev, float threshold) { return std::fabs(v - prev) > threshold; }
#endif
"""

SENSOR = """#include "sensor.h"
#include "sensor_checks.h"
const float {P}_MIN = {lo};
const float {P}_MAX = {hi};
const unsigned long {P}_STALE_MS = {stale};
const float {P}_DEBOUNCE = {deb};
class {name} : public Sensor {{
public:
    void process() override;
    float read() override {{ return 1.0f; }}
}};
void {name}::process() {{
    float raw = read();
    if (!inRange(raw, {lo_arg}, {P}_MAX)) {{
        return;
    }}
    if (isStale(lastUpdateMs + 10, lastUpdateMs, {P}_STALE_MS)) {{
        return;
    }}
    if (debounceExceeded(raw, lastValue, {P}_DEBOUNCE)) {{
        return;
    }}
    value = raw;
}}
"""

MAIN = """#include "airspeed.h"
#include "oil_temp.h"
#include "fuel_flow.h"
int main() {{ AirspeedSensor a; OilTempSensor o; FuelFlowSensor f; a.process(); o.process(); f.process(); return 0; }}
"""


def _sensor(name, prefix, lo, hi, stale, deb, lo_arg=None):
    return SENSOR.format(name=name, P=prefix, lo=lo, hi=hi, stale=stale, deb=deb, lo_arg=lo_arg or f'{prefix}_MIN')


@pytest.fixture
def project(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    shutil.copy(FIXTURE, src / 'sensor.h')
    (src / 'sensor_checks.h').write_text(CHECKS_H)
    (src / 'airspeed.h').write_text(_sensor('AirspeedSensor', 'AS', '0.0f', '450.0f', 250, '5.0f'))
    (src / 'oil_temp.h').write_text(_sensor('OilTempSensor', 'OT', '-40.0f', '150.0f', 1000, '2.0f'))
    # limits passed as a variable, not a constant: nothing can be verified against the YAML
    (src / 'fuel_flow.h').write_text(_sensor('FuelFlowSensor', 'FF', '0.0f', '90.0f', 100, '1.0f', lo_arg='raw'))
    (src / 'main.cpp').write_text(MAIN.format())
    yml = tmp_path / 'sensors.yaml'
    yml.write_text("""sensors:
  - name: AirspeedSensor
    valid_range: {min: 0, max: 450}
    stale: {threshold_ms: 250}
    debounce: {threshold: 5.0}
  - name: OilTempSensor
    valid_range: {min: -40, max: 200}      # code limits the range to 150: mismatch
    debounce: {enabled: false}             # disabled: not required
  - name: FuelFlowSensor
    valid_range: {min: 0, max: 90}
    stale: {threshold_ms: 100}
    debounce: {threshold: 1.0}
""")
    return src, yml, tmp_path / 'out'


def _run(project, write_reports=False):
    src, yml, out = project
    cfg = Config(sensor_templates_file=str(yml), output_dir=str(out), cache_dir=str(out / 'cache'))
    return run_pipeline(cfg, str(src), use_semantic_search=False, write_reports=write_reports)


def _checks(results):
    return {f.template.name: {k: c for k, c in f.checks.items()} for f in results.sensors.findings}


def test_new_sensor_checked_through_shared_validators(project):
    checks = _checks(_run(project))['AirspeedSensor']
    assert {k: c.status for k, c in checks.items()} == {'corrupt': 'implemented', 'stale': 'implemented',
                                                         'debounce': 'implemented'}
    assert checks['corrupt'].found == [0.0, 450.0]
    assert checks['stale'].found == [250.0] and checks['debounce'].found == [5.0]
    assert 'inRange(raw, AS_MIN, AS_MAX)' in checks['corrupt'].evidence[0]


def test_each_sensor_uses_its_own_yaml_parameters(project):
    checks = _checks(_run(project))['OilTempSensor']
    assert checks['corrupt'].status == 'mismatch' and checks['corrupt'].found == [-40.0, 150.0]
    assert checks['stale'].status == 'not_configured'       # no stale threshold in its YAML entry
    assert checks['debounce'].status == 'not_required'      # disabled in its YAML entry


def test_non_constant_limit_is_not_taken_as_verified(project):
    corrupt = _checks(_run(project))['FuelFlowSensor']['corrupt']
    assert corrupt.status == 'mismatch' and corrupt.found == [90.0]   # only the upper limit is a constant


def test_new_sensor_reaches_report_and_assistant(project):
    results = _run(project, write_reports=True)
    report = (project[2] / 'sensor_analysis.md').read_text()
    assert 'AirspeedSensor' in report and 'OilTempSensor' in report
    answer = answer_question('what validation checks are applied to AirspeedSensor', results)
    assert 'AirspeedSensor' in answer and 'implemented' in answer
