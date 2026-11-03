from copy import deepcopy
from pathlib import Path

import pytest

from rag_engine.config import Config
from rag_engine.pipeline import run_pipeline
from rag_engine.reporting.checklist_filler import ChecklistFiller
from rag_engine.reporting.report_generator import ReportGenerator

ROOT = Path(__file__).resolve().parent.parent
SENSORS = ROOT / 'tests' / 'fixtures' / 'cpp' / 'sensors'
TEMPLATES = ROOT / 'tests' / 'fixtures' / 'sensor_templates.yaml'


@pytest.fixture(scope='module')
def results(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('checklist')
    cfg = Config(output_dir=str(tmp / 'out'), cache_dir=str(tmp / 'cache'), sensor_templates_file=str(TEMPLATES))
    return run_pipeline(cfg, str(SENSORS), use_semantic_search=False, write_reports=False)


def _fill(results, tmp_path, dead=None, sensors='from_results'):
    cfg = Config(output_dir=str(tmp_path / 'out'))
    sensors = results.sensors if sensors == 'from_results' else sensors
    cl = ChecklistFiller(cfg).fill_sqa_checklist(results.virtual, dead or results.dead, results.standards,
                                                 sensors=sensors)
    return {i.item_id: i for i in cl.items}


def test_sensor_items_report_unmapped_templates_and_check_gaps(results, tmp_path):
    items = _fill(results, tmp_path)
    assert items['SEN-01'].status == 'OPEN' and 'unmapped: SpeedSensor' in items['SEN-01'].evidence
    assert items['SEN-02'].status == 'FAIL'
    for gap in ('PressureSensor.corrupt (mismatch)', 'PressureSensor.stale (missing)', 'AltitudeSensor.debounce (missing)'):
        assert gap in items['SEN-02'].evidence
    assert 'TemperatureSensor' not in items['SEN-02'].evidence


def test_sensor_items_pass_when_everything_is_mapped_and_implemented(results, tmp_path):
    only_temperature = deepcopy(results.sensors)
    only_temperature.findings = [f for f in only_temperature.findings if f.template.name == 'TemperatureSensor']
    items = _fill(results, tmp_path, sensors=only_temperature)
    assert items['SEN-01'].status == 'PASS' and items['SEN-02'].status == 'PASS'


def test_without_sensors_there_are_no_sensor_items(results, tmp_path):
    assert not [i for i in _fill(results, tmp_path, sensors=None) if i.startswith('SEN-')]


def test_va03_lists_unreachable_overrides(results, tmp_path):
    item = _fill(results, tmp_path)['VA-03']
    assert item.status == 'OPEN'
    assert 'AltitudeSensor::read()' in item.evidence and 'AltitudeSensor::selfTest()' in item.evidence
    assert 'TemperatureSensor' not in item.evidence


def test_va03_passes_once_the_unreachable_overrides_are_justified(results, tmp_path):
    dead = deepcopy(results.dead)
    for i in dead.items:
        if i.name in ('AltitudeSensor::read()', 'AltitudeSensor::selfTest()'):
            i.review_status, i.reviewer_justification = 'false_positive', 'Altitude sensor is not fitted on this LRU'
    assert _fill(results, tmp_path, dead=dead)['VA-03'].status == 'PASS'


@pytest.mark.parametrize('status', ['justified', 'open', 'typo'])
def test_va03_ignores_statuses_that_are_not_a_review_outcome(results, tmp_path, status):
    dead = deepcopy(results.dead)
    for i in dead.items:
        if i.name in ('AltitudeSensor::read()', 'AltitudeSensor::selfTest()'):
            i.review_status, i.reviewer_justification = status, 'Altitude sensor is not fitted on this LRU'
    assert _fill(results, tmp_path, dead=dead)['VA-03'].status == 'OPEN'


def test_generate_all_writes_the_sensor_report_only_when_given_sensors(results, tmp_path):
    gen = ReportGenerator(Config(output_dir=str(tmp_path / 'out')))
    args = (results.virtual, results.coupling, results.dead, results.standards)
    assert len(gen.generate_all(*args, format='markdown')) == 4
    paths = gen.generate_all(*args, format='markdown', sensors=results.sensors, llm_narratives={'sensors': 'NARR'})
    assert Path(paths[-1]).name == 'sensor_analysis.md' and 'NARR' in Path(paths[-1]).read_text(encoding='utf-8')
