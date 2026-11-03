from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from rag_engine.features.sensor_data_checker import (
    check_samples, load_samples_csv, parse_inline_samples, summarize,
)
from rag_engine.features.sensor_template import load_sensor_templates

FIXTURES = Path(__file__).parent / 'fixtures'

# (t, value, accepted, flags) for TemperatureSensor: range [-54, 54], stale 500 ms, debounce 2.0 x3
VECTOR = [
    (0, 25.0, 25.0, []),
    (100, 25.2, 25.2, []),
    (200, 80.0, 25.2, ['corrupt']),
    (300, 25.2, 25.2, []),
    (400, 25.2, 25.2, []),
    (500, 25.2, 25.2, []),
    (600, 25.2, 25.2, []),
    (700, 25.2, 25.2, ['stale']),
    (800, 25.2, 25.2, ['stale']),
    (900, 30.0, 25.2, ['debounce']),
    (1000, 30.1, 25.2, ['debounce']),
    (1100, 30.0, 30.0, []),
    (1800, 30.0, 30.0, ['stale']),
]


@pytest.fixture(scope='module')
def templates():
    return {t.name: t for t in load_sensor_templates(str(FIXTURES / 'sensor_templates.yaml'))}


@pytest.fixture(scope='module')
def samples():
    return load_samples_csv(str(FIXTURES / 'sensor_samples.csv'))


def test_exact_vector(templates, samples):
    results = check_samples(templates['TemperatureSensor'], samples['TemperatureSensor'])
    assert [(r.timestamp_ms, r.value, r.accepted_value, r.flags) for r in results] == VECTOR


def test_csv_loader(samples):
    assert set(samples) == {'TemperatureSensor', 'PressureSensor'}
    assert len(samples['TemperatureSensor']) == 13
    assert samples['PressureSensor'] == [(0.0, 101325.0), (100.0, 150000.0)]


def test_pressure_second_row_corrupt(templates, samples):
    results = check_samples(templates['PressureSensor'], samples['PressureSensor'])
    assert [r.flags for r in results] == [[], ['corrupt']]


def test_summarize(templates, samples):
    results = check_samples(templates['TemperatureSensor'], samples['TemperatureSensor'])
    assert summarize(results) == {'corrupt': 1, 'stale': 3, 'debounce': 2, 'total': 13}


def test_samples_sorted_by_timestamp(templates):
    results = check_samples(templates['TemperatureSensor'], [(100.0, 25.2), (0.0, 25.0)])
    assert [r.timestamp_ms for r in results] == [0.0, 100.0]


def test_nan_and_inf_are_corrupt_even_without_range_check(templates):
    tpl = dataclasses.replace(templates['TemperatureSensor'], corrupt_range_check=False)
    results = check_samples(tpl, [(0, 25.0), (100, math.nan), (200, math.inf)])
    assert [r.flags for r in results] == [[], ['corrupt'], ['corrupt']]
    assert results[1].accepted_value == 25.0


def test_range_check_disabled_accepts_out_of_range(templates):
    tpl = dataclasses.replace(templates['TemperatureSensor'], corrupt_range_check=False)
    assert check_samples(tpl, [(0, 80.0)])[0].flags == []


def test_corrupt_sample_after_gap_also_stale(templates):
    results = check_samples(templates['TemperatureSensor'], [(0, 25.0), (600, 80.0)])
    assert results[1].flags == ['corrupt', 'stale']


def test_debounce_disabled_accepts_every_valid_sample(templates):
    tpl = dataclasses.replace(templates['TemperatureSensor'], debounce_enabled=False)
    results = check_samples(tpl, [(0, 25.0), (100, 40.0)])
    assert [r.flags for r in results] == [[], []]
    assert results[1].accepted_value == 40.0


def test_stale_disabled_when_threshold_none(templates):
    tpl = dataclasses.replace(templates['TemperatureSensor'], stale_threshold_ms=None)
    results = check_samples(tpl, [(0, 25.0), (100000, 25.0)])
    assert [r.flags for r in results] == [[], []]


def test_frozen_detection_can_be_turned_off(templates):
    tpl = dataclasses.replace(templates['TemperatureSensor'], stale_detect_frozen=False)
    results = check_samples(tpl, [(t, 25.0) for t in range(0, 1000, 100)])
    assert all(r.flags == [] for r in results)


def test_gap_stale_still_flagged_when_frozen_off(templates):
    tpl = dataclasses.replace(templates['TemperatureSensor'], stale_detect_frozen=False)
    results = check_samples(tpl, [(0, 25.0), (600, 25.1)])
    assert results[1].flags == ['stale']


# ── CSV loader: malformed files raise ValueError with the line number ─────────

def _csv(tmp_path, text):
    path = tmp_path / 'samples.csv'
    path.write_text(text, encoding='utf-8')
    return path


def test_csv_columns_may_come_in_any_order(tmp_path):
    path = _csv(tmp_path, 'timestamp_ms,value,sensor\n0,25.0,TemperatureSensor\n100,25.5,TemperatureSensor\n')
    assert load_samples_csv(path) == {'TemperatureSensor': [(0.0, 25.0), (100.0, 25.5)]}


def test_csv_short_row_is_a_value_error_with_its_line(tmp_path):
    # sensor is the last column, so a short row leaves it None (the loader used to die with AttributeError)
    path = _csv(tmp_path, 'timestamp_ms,value,sensor\n0,25.0,TemperatureSensor\n100,25.5\n')
    with pytest.raises(ValueError, match=r'samples\.csv:3: bad sample row'):
        load_samples_csv(path)


@pytest.mark.parametrize('body, line', [
    ('TemperatureSensor,0\n', 2),                      # value column missing
    ('TemperatureSensor,0,25\nTemperatureSensor,100,\n', 3),   # empty value
    ('TemperatureSensor,abc,25\n', 2),                 # non-numeric timestamp
    (',0,25\n', 2),                                    # blank sensor name
])
def test_csv_bad_rows_name_their_line(tmp_path, body, line):
    path = _csv(tmp_path, 'sensor,timestamp_ms,value\n' + body)
    with pytest.raises(ValueError, match=rf'samples\.csv:{line}: bad sample row'):
        load_samples_csv(path)


def test_csv_field_over_the_csv_module_limit_is_a_value_error(tmp_path):
    path = _csv(tmp_path, 'sensor,timestamp_ms,value\n' + 'A' * 200_000 + ',0,1\n')
    with pytest.raises(ValueError, match=r'samples\.csv:2: .*field larger than field limit'):
        load_samples_csv(path)


def test_csv_missing_column_in_header(tmp_path):
    with pytest.raises(ValueError, match='header must contain'):
        load_samples_csv(_csv(tmp_path, 'sensor,value\nTemperatureSensor,25\n'))


def test_csv_with_a_utf8_bom_is_read(tmp_path):         # Excel's "CSV UTF-8" starts with a BOM
    path = tmp_path / 'excel.csv'
    path.write_bytes(b'\xef\xbb\xbfsensor,timestamp_ms,value\nTemperatureSensor,0,25.0\n')
    assert load_samples_csv(path) == {'TemperatureSensor': [(0.0, 25.0)]}


def test_csv_that_is_not_utf8_is_a_value_error_naming_the_file(tmp_path):
    path = tmp_path / 'latin.csv'
    path.write_bytes('sensor,timestamp_ms,value\nTemp\xe9,0,25.0\n'.encode('latin-1'))
    with pytest.raises(ValueError, match=r'latin\.csv: .*UTF-8'):
        load_samples_csv(path)


# ── Inline samples ──────────────────────────────────────────────────────────

def test_parse_inline_samples():
    assert parse_inline_samples('0:25.0, 100:80.0') == [(0.0, 25.0), (100.0, 80.0)]
    assert parse_inline_samples(' 5:-1.5e1 ,, 10:3, ') == [(5.0, -15.0), (10.0, 3.0)]   # blanks and trailing comma


@pytest.mark.parametrize('text', ['', ' , ', 'abc', '0:25.0, 100', '0:x', ':5', '1:2:3'])
def test_parse_inline_samples_rejects_bad_text(text):
    with pytest.raises(ValueError):
        parse_inline_samples(text)
