from pathlib import Path

import pytest

from rag_engine.config import Config
from rag_engine.llm.offline_qa import answer_question, offline_answer, results_summary
from rag_engine.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent
SENSORS = ROOT / 'tests' / 'fixtures' / 'cpp' / 'sensors'
TEMPLATES = ROOT / 'tests' / 'fixtures' / 'sensor_templates.yaml'


@pytest.fixture(scope='module')
def results(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('qa')
    cfg = Config(output_dir=str(tmp / 'out'), cache_dir=str(tmp / 'cache'), sensor_templates_file=str(TEMPLATES))
    return run_pipeline(cfg, str(SENSORS), use_semantic_search=False, write_reports=False)


# ── The three R4 questions ──────────────────────────────────────────────────

def test_virtual_functions_implemented_by_a_class(results):
    answer = offline_answer('What are the virtual functions implemented by TemperatureSensor?', results)
    assert 'TemperatureSensor::process()' in answer and 'TemperatureSensor::read()' in answer
    assert 'overrides Sensor::process()' in answer and 'overrides Sensor::read()' in answer
    assert 'PressureSensor' not in answer and 'isStale' not in answer


def test_components_affected_if_a_method_changes(results):
    answer = offline_answer('What components are affected if updateTemperature() changes?', results)
    assert 'SensorManager::updateTemperature()' in answer
    assert 'Validator' in answer and 'Display' in answer and 'TemperatureSensor' in answer
    assert 'Affected if changed:' in answer


def test_validation_checks_applied_to_a_sensor(results):
    answer = offline_answer('What validation checks are applied to TemperatureSensor?', results)
    for check in ('corrupt', 'stale', 'debounce'):
        assert f'{check}: implemented' in answer
    assert 'Validator::check(float)' in answer and 'Display::update(float)' in answer   # data flow


def test_sensor_checks_show_missing_and_mismatch(results):
    answer = offline_answer('Which validation checks apply to PressureSensor?', results)
    assert 'corrupt: mismatch' in answer and 'stale: missing' in answer and 'debounce: missing' in answer


# ── Other phrasings and the remaining handlers ──────────────────────────────

@pytest.mark.parametrize('question', [
    'Which virtual functions does TemperatureSensor override?',
    'virtual methods overridden by the class TemperatureSensor',
])
def test_virtual_functions_phrasings(results, question):
    assert 'TemperatureSensor::process() — overrides Sensor::process()' in offline_answer(question, results)


def test_virtual_functions_of_a_base_class_list_what_it_introduces(results):
    answer = offline_answer('virtual functions implemented by Sensor', results)
    assert 'Sensor::read() — introduces pure virtual function; overridden by:' in answer
    assert 'AltitudeSensor::read()' in answer and 'PressureSensor::read()' in answer


@pytest.mark.parametrize('question', ['effect of updateTemperature', 'What is the coupling of SensorManager::updateTemperature'])
def test_effect_phrasings(results, question):
    assert 'Method: SensorManager::updateTemperature()' in offline_answer(question, results)


def test_inheritance_of_a_class(results):
    answer = offline_answer('What is the inheritance hierarchy of TemperatureSensor?', results)
    assert 'Bases (nearest first): Sensor' in answer
    assert '├── AltitudeSensor' in answer and '└── TemperatureSensor' in answer
    base = offline_answer('hierarchy of Sensor', results)
    assert 'Derived: AltitudeSensor, PressureSensor, TemperatureSensor' in base


@pytest.mark.parametrize('question, expected', [
    ('virtual functions implemented by TemperatureSensr', "Class 'TemperatureSensr' not found. Close names: TemperatureSensor"),
    ('What is affected if updateTemprature changes?', "Function 'updateTemprature' not found. Close names: updateTemperature"),
    ('validation checks applied to TemperatureSensr', "Sensor 'TemperatureSensr' not found. Close names: TemperatureSensor"),
    ('inheritance of Nothing', "Class 'Nothing' not found. Known: "),
])
def test_unknown_names_get_a_not_found_message_with_close_names(results, question, expected):
    assert expected in offline_answer(question, results)


@pytest.mark.parametrize('question, expected', [
    ('What is the impact of the critical violations?', 'CRITICAL violation(s)'),
    ('What is the coupling of the LRUs', 'LRU coupling summary'),
    ('What is the effect of dead code on coverage?', 'dead function(s) to remove'),
    ('Is the code ready for certification? check the critical violations by severity', 'CRITICAL violation(s)'),
    ('Which virtual functions are overridden by derived classes?', 'Virtual functions: 4 declared, 6 override(s)'),
])
def test_summary_questions_are_not_hijacked_by_a_structural_not_found(results, question, expected):
    answer = offline_answer(question, results)
    assert expected in answer and 'not found' not in answer


def test_a_generic_word_that_no_summary_answers_still_ends_in_not_found(results):
    assert "Class 'Nothing' not found" in offline_answer('inheritance of Nothing', results)


def test_sensor_question_without_templates_says_so(results):
    from dataclasses import replace
    no_sensors = replace(results, sensors=None)
    assert 'no sensor templates' in offline_answer('validation checks applied to TemperatureSensor', no_sensors)


def test_existing_summary_answers_still_work(results):
    assert 'CRITICAL violation(s)' in offline_answer('how many critical violations are there', results)
    assert 'Compliance score' in offline_answer('what is the compliance score', results)
    assert offline_answer('what is the meaning of life', results).startswith('I can answer questions about')


# ── LLM path ────────────────────────────────────────────────────────────────

class FakeNarrator:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def answer_question(self, question, analysis_context, **kw):
        self.calls.append((question, analysis_context))
        return self.reply


def test_llm_receives_the_structured_facts_and_only_explains_them(results):
    narrator = FakeNarrator('It overrides two base functions.')
    question = 'What are the virtual functions implemented by TemperatureSensor?'
    answer = answer_question(question, results, narrator)
    assert narrator.calls == [(question, offline_answer(question, results))]
    assert answer.startswith(offline_answer(question, results)) and answer.endswith('It overrides two base functions.')


def test_without_an_llm_or_an_explanation_the_facts_are_returned_as_is(results):
    question = 'What validation checks are applied to TemperatureSensor?'
    assert answer_question(question, results) == offline_answer(question, results)
    assert answer_question(question, results, FakeNarrator(None)) == offline_answer(question, results)


def test_llm_gets_the_results_summary_and_a_label_when_no_fact_matches(results):
    narrator = FakeNarrator('General answer.')
    answer = answer_question('explain MC/DC coverage', results, narrator)
    assert answer == ("Explanation (local LLM — no specific static-analysis facts matched; summary facts used)\n"
                      "General answer.")
    assert narrator.calls == [('explain MC/DC coverage', results_summary(results))]


def test_results_summary_has_a_line_per_analysis(results):
    lines = results_summary(results).splitlines()
    assert lines[0] == 'Analysis summary:'
    assert [l.split(':')[0].strip() for l in lines[1:]] == [
        'Standards', 'Virtual functions', 'Dead code', 'Coupling risk by LRU', 'Sensors']
    assert 'compliance' in lines[1] and 'CRITICAL=0' in lines[1]
    assert 'dead=5, deactivated=1' in lines[3]
    assert 'checks: implemented=3, mismatch=1, missing=5' in lines[5] and '3/4 templates mapped' in lines[5]
