from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import main

ROOT = Path(__file__).resolve().parent.parent
SENSORS = str(ROOT / 'tests' / 'fixtures' / 'cpp' / 'sensors')
TEMPLATES = str(ROOT / 'tests' / 'fixtures' / 'sensor_templates.yaml')
SAMPLES = str(ROOT / 'tests' / 'fixtures' / 'sensor_samples.csv')

runner = CliRunner()


@pytest.fixture
def cfg_file(tmp_path):
    """A config that writes into tmp_path and never talks to Ollama."""
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({
        'output_dir': str(tmp_path / 'out'), 'cache_dir': str(tmp_path / 'cache'), 'ollama_enabled': False,
    }), encoding='utf-8')
    return str(path)


def _invoke(*args):
    assert Path(main.__file__).resolve().parent == ROOT   # the worktree's main.py, not another checkout's
    return runner.invoke(main.app, [str(a) for a in args])


def test_analyze_end_to_end_writes_every_report(cfg_file, tmp_path):
    result = _invoke('analyze', SENSORS, '--sensor-templates', TEMPLATES, '--config', cfg_file,
                     '--output-format', 'markdown', '--no-semantic-search')
    # The fixture has no CRITICAL violation, so analyze exits 0 (it exits 1 only when there is one).
    assert result.exit_code == 0, result.output
    out = tmp_path / 'out'
    for name in ('virtual_analysis.md', 'coupling_analysis.md', 'dead_code_report.md', 'dead_code_findings.json',
                 'standards_report.md', 'sensor_analysis.md', 'sqa_checklist.md'):
        assert (out / name).is_file(), name
    assert 'sensor_analysis.md' in result.stdout and 'sqa_checklist.md' in result.stdout
    checklist = (out / 'sqa_checklist.md').read_text(encoding='utf-8')
    assert '| SEN-02 |' in checklist and '| VA-03 |' in checklist


def test_analyze_exits_1_on_critical_violations(cfg_file, tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.cpp').write_text('int main() { goto end; end: return 0; }\n', encoding='utf-8')
    result = _invoke('analyze', src, '--config', cfg_file, '--output-format', 'markdown', '--no-semantic-search')
    assert result.exit_code == 1 and 'CRITICAL violation' in result.stderr
    assert (tmp_path / 'out' / 'standards_report.md').is_file()   # reports are written before the exit code


def test_analyze_rejects_a_missing_templates_file_and_an_empty_codebase(cfg_file, tmp_path):
    result = _invoke('analyze', SENSORS, '--sensor-templates', tmp_path / 'nope.yaml', '--config', cfg_file)
    assert result.exit_code == 1 and 'not found' in result.stderr
    (tmp_path / 'empty').mkdir()
    result = _invoke('analyze', tmp_path / 'empty', '--config', cfg_file, '--no-semantic-search')
    assert result.exit_code == 1 and 'No C++ files found' in result.stderr


def test_invalid_sensor_templates_in_the_config_are_a_clean_error(cfg_file, tmp_path):
    bad = tmp_path / 'bad.yaml'
    bad.write_text('sensors:\n  - name: S\n    valid_range: {min: 5, max: 1}\n', encoding='utf-8')
    cfg = yaml.safe_load(Path(cfg_file).read_text(encoding='utf-8'))
    cfg['sensor_templates_file'] = str(bad)
    Path(cfg_file).write_text(yaml.safe_dump(cfg), encoding='utf-8')
    for command in (['sensors', SENSORS], ['ask', SENSORS, 'hello'], ['analyze', SENSORS, '--no-semantic-search']):
        result = _invoke(*command, '--config', cfg_file)
        assert result.exit_code == 1 and 'must be less than max' in result.stderr
        assert 'Traceback' not in result.output


def test_a_bug_in_an_analyzer_is_not_hidden_as_a_user_error(cfg_file, monkeypatch):
    def boom(self):
        raise ValueError('analyzer bug')

    monkeypatch.setattr('rag_engine.pipeline.VirtualAnalyzer.analyze', boom)
    result = _invoke('virtuals', SENSORS, '--config', cfg_file)
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError) and 'Error: analyzer bug' not in result.output


def test_virtuals_prints_inventory_and_trees(cfg_file):
    result = _invoke('virtuals', SENSORS, '--config', cfg_file)
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert 'Virtual Function: Sensor::process()' in out
    assert 'Overridden by: PressureSensor::process(), TemperatureSensor::process()' in out
    assert 'Called from SensorManager::update()' in out
    assert '├── AltitudeSensor' in out and '└── TemperatureSensor' in out
    assert 'Class hierarchies:' in out


def test_effect_prints_the_block_and_writes_the_report(cfg_file, tmp_path):
    result = _invoke('effect', SENSORS, 'updateTemperature', '--config', cfg_file)
    assert result.exit_code == 0, result.output
    for text in ('Method: SensorManager::updateTemperature()', 'Reads:', 'Writes:', 'Calls:',
                 'Validator', 'Display', 'Affected if changed:'):
        assert text in result.stdout, text
    assert (tmp_path / 'out' / 'function_effect_updateTemperature.md').is_file()


def test_effect_for_an_unknown_method_lists_close_names(cfg_file):
    result = _invoke('effect', SENSORS, 'updateTemprature', '--config', cfg_file)
    assert result.exit_code == 1 and 'Close names: updateTemperature' in result.stderr


def test_sensors_prints_the_check_table(cfg_file):
    result = _invoke('sensors', SENSORS, '--templates', TEMPLATES, '--config', cfg_file)
    assert result.exit_code == 0, result.output
    rows = {line.split()[0]: line.split() for line in result.stdout.splitlines() if line and not line.startswith(' ')}
    assert rows['TemperatureSensor'][2:] == ['implemented'] * 3
    assert rows['PressureSensor'][2:] == ['mismatch', 'missing', 'missing']
    assert rows['SpeedSensor'][1:3] == ['(not', 'found)']
    assert 'PressureSensor.corrupt: mismatch (expected range [0, 110000]; found 0, 120000)' in result.stdout


def test_sensors_needs_templates(cfg_file):
    result = _invoke('sensors', SENSORS, '--config', cfg_file)
    assert result.exit_code == 1 and 'No sensor templates' in result.stderr


def test_sensor_check_flags_each_sample_and_summarizes(cfg_file):
    result = _invoke('sensor-check', SAMPLES, '--templates', TEMPLATES, '--config', cfg_file)
    assert result.exit_code == 0, result.output
    assert 'TemperatureSensor' in result.stdout and 'PressureSensor' in result.stdout
    assert 't=200 ms  value=80  accepted=25.2  corrupt' in result.stdout
    assert 'Summary: 13 samples, corrupt 1, stale 3, debounce 2' in result.stdout


def test_sensor_check_rejects_unknown_sensors_and_bad_csv(cfg_file, tmp_path):
    unknown = tmp_path / 'unknown.csv'
    unknown.write_text('sensor,timestamp_ms,value\nNoSuchSensor,0,1.0\n', encoding='utf-8')
    result = _invoke('sensor-check', unknown, '--templates', TEMPLATES, '--config', cfg_file)
    assert result.exit_code == 1 and 'NoSuchSensor' in result.stderr
    bad = tmp_path / 'bad.csv'
    bad.write_text('a,b\n1,2\n', encoding='utf-8')
    result = _invoke('sensor-check', bad, '--templates', TEMPLATES, '--config', cfg_file)
    assert result.exit_code == 1 and 'CSV header' in result.stderr


@pytest.mark.parametrize('question, expected', [
    ('What are the virtual functions implemented by TemperatureSensor?', ['TemperatureSensor::process()']),
    ('What components are affected if updateTemperature() changes?', ['Validator', 'Display']),
    ('What validation checks are applied to TemperatureSensor?',
     ['corrupt: implemented', 'stale: implemented', 'debounce: implemented']),
])
def test_ask_answers_the_r4_questions_offline(cfg_file, question, expected):
    result = _invoke('ask', SENSORS, question, '--sensor-templates', TEMPLATES, '--config', cfg_file)
    assert result.exit_code == 0, result.output
    for text in expected:
        assert text in result.stdout, text
    assert 'Explanation (local LLM)' not in result.stdout


def test_ask_adds_the_llm_explanation_when_ollama_is_reachable(cfg_file, monkeypatch):
    class FakeLLM:
        def __init__(self, base_url, model):
            pass

        def is_available(self):
            return True

        def generate(self, prompt, max_tokens=512):
            assert 'TemperatureSensor::process()' in prompt   # the structured facts are in the prompt
            assert 'FAISS' not in prompt and 'STATIC-ANALYSIS FACTS' in prompt   # no semantic index in `ask`
            return 'Both overrides are exercised.'

    monkeypatch.setattr(main, 'OllamaClient', FakeLLM)
    cfg = yaml.safe_load(Path(cfg_file).read_text(encoding='utf-8'))
    cfg['ollama_enabled'] = True
    Path(cfg_file).write_text(yaml.safe_dump(cfg), encoding='utf-8')
    result = _invoke('ask', SENSORS, 'What are the virtual functions implemented by TemperatureSensor?',
                     '--config', cfg_file)
    assert result.exit_code == 0, result.output
    assert 'TemperatureSensor::process()' in result.stdout and 'Both overrides are exercised.' in result.stdout


def test_validate_config_still_works(cfg_file):
    assert 'Config OK' in _invoke('validate-config', '--config', cfg_file).stdout


def test_malformed_yaml_is_reported_as_an_error_not_a_traceback(tmp_path):
    cfg = tmp_path / 'broken.yaml'
    cfg.write_text('entry_points: [unclosed\n', encoding='utf-8')
    result = _invoke('validate-config', '--config', cfg)
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)   # the _fail() exit, not an uncaught ValueError
    assert 'Error:' in result.stderr and 'broken.yaml' in result.stderr
    assert 'Traceback' not in result.output
