from __future__ import annotations

from pathlib import Path

import pytest

from rag_engine.config import load_config
from rag_engine.features.sensor_template import SensorTemplate, load_sensor_templates

FIXTURES = Path(__file__).parent / 'fixtures'
REPO_ROOT = Path(__file__).parent.parent


def _load(tmp_path, text: str):
    path = tmp_path / 'templates.yaml'
    path.write_text(text, encoding='utf-8')
    return load_sensor_templates(str(path))


def test_fixture_list_form():
    templates = {t.name: t for t in load_sensor_templates(str(FIXTURES / 'sensor_templates.yaml'))}
    assert list(templates) == ['TemperatureSensor', 'PressureSensor', 'AltitudeSensor', 'SpeedSensor']
    temp = templates['TemperatureSensor']
    assert (temp.type, temp.unit, temp.data_type) == ('temperature', 'Celsius', 'float')
    assert (temp.parameter_group, temp.io_group) == ('Temperature Parameter', 'Temperature I/O Group')
    assert (temp.valid_min, temp.valid_max) == (-54, 54)
    assert temp.sampling_interval_ms == 100
    assert temp.stale_threshold_ms == 500 and temp.stale_detect_frozen is True
    assert temp.debounce_enabled is True and temp.debounce_threshold == 2.0 and temp.debounce_count == 3
    assert temp.corrupt_range_check is True
    assert temp.class_name == ''


def test_defaults_merge_does_not_clobber_sensor_keys(tmp_path):
    (t,) = _load(tmp_path, '''
defaults:
  debounce: {enabled: true, count: 5}
sensors:
  - name: A
    debounce: {threshold: 1.5}
''')
    assert (t.debounce_enabled, t.debounce_count, t.debounce_threshold) == (True, 5, 1.5)


def test_sensor_overrides_default(tmp_path):
    a, b = _load(tmp_path, '''
defaults:
  debounce: {enabled: true, count: 5}
sensors:
  - name: A
  - name: B
    debounce: {enabled: false, count: 2}
''')
    assert a.debounce_count == 5
    assert (b.debounce_enabled, b.debounce_count) == (False, 2)


def test_single_sensor_mapping_form(tmp_path):
    (t,) = _load(tmp_path, '''
sensor:
  name: TemperatureSensor
  type: temperature
  data_type: float
  unit: Celsius
  valid_range: {min: -54, max: 54}
  sampling_interval_ms: 100
  stale: {threshold_ms: 500, detect_frozen: false}
  debounce: {enabled: true, threshold: 2.0}
  corrupt: {range_check: true}
''')
    assert isinstance(t, SensorTemplate)
    assert (t.valid_min, t.valid_max) == (-54, 54)
    assert t.stale_detect_frozen is False


def test_class_name_defaults_to_empty_and_can_be_set(tmp_path):
    (t,) = _load(tmp_path, 'sensors:\n  - {name: A, class_name: Thermo}\n')
    assert t.class_name == 'Thermo'


@pytest.mark.parametrize('text, needle', [
    ('sensors:\n  - {type: temperature}\n', 'name'),
    ('sensors:\n  - {name: A, valid_range: {min: 5, max: 5}}\n', 'valid_range'),
    ('sensors:\n  - {name: A, valid_range: {min: 9, max: 1}}\n', 'valid_range'),
    ('sensors:\n  - {name: A, stale: {threshold_ms: -1}}\n', 'stale'),
    ('sensors:\n  - {name: A, debounce: {threshold: -0.5}}\n', 'debounce'),
    ('sensors:\n  - {name: A, debounce: {count: 0}}\n', 'count'),
    ('sensors:\n  - {name: A, sampling_interval_ms: -10}\n', 'sampling_interval_ms'),
])
def test_validation_errors_name_sensor_and_field(tmp_path, text, needle):
    with pytest.raises(ValueError) as exc:
        _load(tmp_path, text)
    assert needle in str(exc.value)


def test_validation_error_names_the_sensor(tmp_path):
    with pytest.raises(ValueError, match='BadSensor'):
        _load(tmp_path, 'sensors:\n  - {name: BadSensor, debounce: {count: 0}}\n')


def test_no_sensors_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        _load(tmp_path, 'defaults: {unit: x}\n')


def test_repo_root_example_loads_the_four_sensors():
    templates = load_sensor_templates(str(REPO_ROOT / 'sensor_templates.yaml'))
    assert [t.name for t in templates] == ['TemperatureSensor', 'PressureSensor', 'AltitudeSensor', 'SpeedSensor']


@pytest.mark.parametrize('text, sensor, key', [
    ('sensors:\n  - {name: A, stale: {treshold_ms: 500}}\n', 'A', 'treshold_ms'),
    ('sensors:\n  - {name: A, valid_range: {minimum: 0, max: 5}}\n', 'A', 'minimum'),
    ('sensors:\n  - {name: A, debounce: {thresh: 1.0}}\n', 'A', 'thresh'),
    ('sensors:\n  - {name: A, corrupt: {rangecheck: true}}\n', 'A', 'rangecheck'),
    ('sensors:\n  - {name: A, valid_rnage: {min: 0, max: 5}}\n', 'A', 'valid_rnage'),
    ('sensors:\n  - {name: A}\n  - {name: B, unitt: x}\n', 'B', 'unitt'),
    ('sensor: {name: A, stale: {treshold_ms: 500}}\n', 'A', 'treshold_ms'),
    ('defaults: {debounce: {cnt: 3}}\nsensors:\n  - {name: A}\n', 'defaults', 'cnt'),
    ('defaults: {untis: x}\nsensors:\n  - {name: A}\n', 'defaults', 'untis'),
    ('default: {unit: x}\nsensors:\n  - {name: A}\n', 'default', 'default'),
])
def test_unknown_keys_raise_naming_sensor_and_key(tmp_path, text, sensor, key):
    with pytest.raises(ValueError) as exc:
        _load(tmp_path, text)
    assert sensor in str(exc.value) and key in str(exc.value)


def test_config_sensor_templates_file_key(tmp_path):
    cfg_path = tmp_path / 'config.yaml'
    cfg_path.write_text("sensor_templates_file: 'x.yaml'\n", encoding='utf-8')
    assert load_config(str(cfg_path)).sensor_templates_file == str((tmp_path / 'x.yaml').resolve())
    assert load_config(str(tmp_path / 'missing.yaml')).sensor_templates_file == ''


def test_config_sensor_templates_file_absolute_path_unchanged(tmp_path):
    absolute = (tmp_path / 'abs.yaml').resolve()
    cfg_path = tmp_path / 'config.yaml'
    cfg_path.write_text(f"sensor_templates_file: '{absolute.as_posix()}'\n", encoding='utf-8')
    assert load_config(str(cfg_path)).sensor_templates_file == str(absolute)


def test_config_empty_sensor_templates_file_stays_empty(tmp_path):
    cfg_path = tmp_path / 'config.yaml'
    cfg_path.write_text("sensor_templates_file: ''\n", encoding='utf-8')
    assert load_config(str(cfg_path)).sensor_templates_file == ''
