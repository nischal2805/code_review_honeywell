from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from rag_engine.features.sensor_data_checker import check_samples, load_samples_csv, summarize
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
