"""TUI pilot tests: App.run_test() driven by asyncio.run, on the sensors fixture without the embedding model."""
from __future__ import annotations

import asyncio
import dataclasses
import threading
from pathlib import Path

import pytest
from loguru import logger
from textual.app import App
from textual.widgets import (
    Checkbox, DataTable, Input, Label, ProgressBar, RichLog, Select, Static, TabbedContent, TabPane, Tree,
)

import rag_engine.tui.app as tui_app
from rag_engine.config import Config
from rag_engine.features.sensor_data_checker import SampleResult
from rag_engine.pipeline import AnalysisResults, InputError, run_pipeline
from rag_engine.tui.app import (
    AnalysisScreen, ChatScreen, ConfigScreen, RAGEngineApp, ResultsScreen, stage_progress, visible_samples,
)

ROOT = Path(__file__).resolve().parent.parent
SENSORS = ROOT / 'tests' / 'fixtures' / 'cpp' / 'sensors'
TEMPLATES = ROOT / 'tests' / 'fixtures' / 'sensor_templates.yaml'
SAMPLES_CSV = ROOT / 'tests' / 'fixtures' / 'sensor_samples.csv'
SIZE = (140, 50)
PROGRESS_BARS = ['pb-parse', 'pb-virtual', 'pb-dead', 'pb-standards', 'pb-coupling', 'pb-sensors']


def _cfg(tmp_path: Path, templates: bool = True) -> Config:
    return Config(output_dir=str(tmp_path / 'out'), cache_dir=str(tmp_path / 'cache'), ollama_enabled=False,
                  sensor_templates_file=str(TEMPLATES) if templates else '')


@pytest.fixture(scope='module')
def results(tmp_path_factory) -> AnalysisResults:
    return run_pipeline(_cfg(tmp_path_factory.mktemp('tui')), str(SENSORS),
                        use_semantic_search=False, write_reports=False)


@pytest.fixture(scope='module')
def results_without_templates(tmp_path_factory) -> AnalysisResults:
    return run_pipeline(_cfg(tmp_path_factory.mktemp('tui-plain'), templates=False), str(SENSORS),
                        use_semantic_search=False, write_reports=False)


def _drive(screen, scenario, size=SIZE):
    """Host *screen* in a headless app and return the result of the async scenario(app, screen, pilot)."""
    class Host(App):
        def on_mount(self) -> None:
            self.push_screen(screen)

    async def main():
        app = Host()
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            return await scenario(app, screen, pilot)
    return asyncio.run(main())


def _nodes(node):
    yield node
    for child in node.children:
        yield from _nodes(child)


def _rows(table: DataTable):
    return [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]


def _columns(table: DataTable):
    return [str(column.label) for column in table.columns.values()]


def _log_text(log: RichLog) -> str:
    return ''.join(line.text for line in log.lines)   # wrapped lines rejoin: a long path breaks mid-word


async def _settle(app, pilot) -> None:
    """Let a pushed screen mount, its thread workers finish and the follow-up screen show."""
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _wait_until(pilot, condition, tries: int = 200) -> None:
    for _ in range(tries):
        if condition():
            return
        await pilot.pause(0.05)
    raise AssertionError('condition not reached')


@pytest.fixture
def logged():
    """Loguru ERROR records (what logger.exception writes) emitted during the test."""
    records = []
    sink = logger.add(lambda message: records.append(message.record), level='ERROR')
    yield records
    logger.remove(sink)


async def _open_tab(screen, pilot, pane_id: str) -> None:
    screen.query_one(TabbedContent).active = pane_id
    await pilot.pause()
    await pilot.wait_for_scheduled_animations()   # a click before the pane is laid out misses its button
    await pilot.pause()


# ── Results tabs ────────────────────────────────────────────────────────────

def test_results_screen_has_all_tabs(results):
    async def scenario(app, screen, pilot):
        ids = {pane.id for pane in screen.query(TabPane)}
        assert ids >= {'tab-standards', 'tab-virtual', 'tab-dead', 'tab-effect', 'tab-sensors',
                       'tab-coupling', 'tab-checklist'}
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)


def test_virtual_tree_lists_functions_overrides_and_call_sites(results):
    async def scenario(app, screen, pilot):
        tree = screen.query_one('#virtual-tree', Tree)
        labels = [str(n.label) for n in _nodes(tree.root)]
        text = '\n'.join(labels)
        assert any('Sensor::process()' in label for label in labels)
        assert 'TemperatureSensor::process()' in text and 'PressureSensor::process()' in text
        # Reachability and instantiation are marked per override.
        override = next(label for label in labels if label.startswith('TemperatureSensor::process()'))
        assert 'reachable ✓' in override and 'instantiated ✓' in override
        unused = next(label for label in labels if label.startswith('AltitudeSensor::read()'))
        assert 'reachable ✗' in unused and 'instantiated ✗' in unused
        # Call sites are leaves.
        site = next(n for n in _nodes(tree.root) if 'SensorManager::update()' in str(n.label))
        assert not site.children and site.parent is not None and 'Sensor::process()' in str(site.parent.label)
        # The class is the root level and the existing change table stays.
        assert 'Sensor' in [str(n.label) for n in tree.root.children]
        assert _columns(screen.query_one('#dt-virtual', DataTable))[:2] == ['Change', 'Function']
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)


def test_dead_code_table_and_notes(results):
    async def scenario(app, screen, pilot):
        table = screen.query_one('#dt-dead', DataTable)
        assert _columns(table) == ['ID', 'File', 'Symbol', 'Lines', 'Type', 'Confidence', 'Reason']
        assert table.row_count == len(results.dead.items)
        row = next(r for r in _rows(table) if r[2] == 'legacyCalibrate()')
        assert row[0].startswith('DC-FUNC-') and row[1] == 'main.cpp' and row[3] == '20-22'
        assert row[4].startswith('dead_code') and row[5] == 'high' and 'entry points' in row[6]
        info = str(screen.query_one('#dead-info', Static).content)
        assert 'main()' in info and 'TIMER0_IRQHandler()' in info and 'Entry points' in info
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)


def test_function_effect_tab_analyzes_a_typed_method(results):
    async def scenario(app, screen, pilot):
        await _open_tab(screen, pilot, 'tab-effect')
        box = screen.query_one('#effect-input', Input)
        box.focus()
        await pilot.press(*'updateTemperature')   # the typed 'r's reach the Input, not the Re-run binding
        assert box.value == 'updateTemperature'
        await pilot.click('#btn-effect')
        await _settle(app, pilot)
        out = str(screen.query_one('#effect-output', Static).content)
        assert 'SensorManager::updateTemperature()' in out and 'Validator' in out
        assert app.screen is screen
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)


def test_function_effect_unknown_and_empty_input(results):
    async def scenario(app, screen, pilot):
        await _open_tab(screen, pilot, 'tab-effect')
        box = screen.query_one('#effect-input', Input)
        out = screen.query_one('#effect-output', Static)
        box.value = 'updateTemprature'
        box.focus()
        await pilot.press('enter')                   # Enter submits like the button
        await _settle(app, pilot)
        assert "not found" in str(out.content) and 'updateTemperature' in str(out.content)
        box.value = ''
        await pilot.click('#btn-effect')
        await pilot.pause()
        assert 'Enter a method name' in str(out.content)
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)


def test_sensors_table_shows_check_statuses(results):
    async def scenario(app, screen, pilot):
        table = screen.query_one('#dt-sensors', DataTable)
        assert _columns(table) == ['Sensor', 'Class', 'Corrupt', 'Stale', 'Debounce', 'Instantiated']
        rows = {r[0]: r for r in _rows(table)}
        assert rows['TemperatureSensor'][1:] == ['TemperatureSensor', 'implemented', 'implemented',
                                                 'implemented', 'yes']
        assert rows['PressureSensor'][2:5] == ['mismatch', 'missing', 'missing']
        assert rows['AltitudeSensor'][5] == 'no'
        assert rows['SpeedSensor'][1:] == ['not found', '-', '-', '-', '-']
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)


def _check_inline(results, template: str, samples: str):
    async def scenario(app, screen, pilot):
        await _open_tab(screen, pilot, 'tab-sensors')
        screen.query_one('#sensor-select', Select).value = template
        screen.query_one('#sample-input', Input).value = samples
        await pilot.pause()
        await pilot.click('#btn-sensor-run')
        await _settle(app, pilot)
        return (_columns(screen.query_one('#dt-samples', DataTable)),
                _rows(screen.query_one('#dt-samples', DataTable)),
                str(screen.query_one('#sample-summary', Static).content))
    return _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)


def test_sample_tester_flags_an_out_of_range_inline_sample(results):
    columns, rows, summary = _check_inline(results, 'TemperatureSensor', '0:25.0, 100:80.0')
    assert columns == ['t (ms)', 'Value', 'Accepted', 'Flags']
    assert rows == [['0', '25', '25', 'ok'], ['100', '80', '25', 'corrupt']]
    assert summary == ('2 samples: corrupt 1, stale 0, debounce 0\n'
                       'showing 2 of 2 samples (all flagged rows included)')


def test_sample_tester_reads_a_csv_for_the_selected_sensor(results):
    _, rows, summary = _check_inline(results, 'TemperatureSensor', str(SAMPLES_CSV))
    assert len(rows) == 13 and rows[2][3] == 'corrupt' and rows[9][3] == 'debounce'
    assert summary == ('13 samples: corrupt 1, stale 3, debounce 2\n'
                       'showing 13 of 13 samples (all flagged rows included)')


@pytest.mark.parametrize('template, samples, message', [
    ('TemperatureSensor', 'not samples', "Cannot read samples 'not samples'"),
    ('TemperatureSensor', '0:25.0, 100', "Cannot read samples"),
    ('TemperatureSensor', '', 'Enter samples'),
    ('AltitudeSensor', str(SAMPLES_CSV), 'No samples for AltitudeSensor'),
])
def test_sample_tester_reports_bad_input_without_rows(results, template, samples, message):
    _, rows, summary = _check_inline(results, template, samples)
    assert rows == [] and message in summary


def test_sample_tester_is_disabled_without_templates(results_without_templates):
    async def scenario(app, screen, pilot):
        assert screen.query_one('#dt-sensors', DataTable).row_count == 0
        for widget_id in ('#sensor-select', '#sample-input', '#btn-sensor-run'):
            assert screen.query_one(widget_id).disabled, widget_id
        assert 'sensor templates' in str(screen.query_one('#sensor-hint', Static).content).lower()
    _drive(ResultsScreen(results_without_templates, 'out', _cfg(Path('.'), templates=False)), scenario)


def test_sensor_select_has_a_visible_label(results):
    async def scenario(app, screen, pilot):
        await _open_tab(screen, pilot, 'tab-sensors')
        label = screen.query_one('#sensor-select-label', Label)
        assert str(label.content) == 'Sensor template'
        # it sits in the same row, directly before the Select
        row = list(screen.query_one('#sample-row').children)
        assert row.index(label) + 1 == row.index(screen.query_one('#sensor-select'))
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)


def test_sample_tester_shows_every_flagged_row_and_the_first_200(results, tmp_path):
    # alternating values stay inside the debounce threshold and never freeze, so only the 2 injected spikes are flagged
    lines = ['sensor,timestamp_ms,value']
    for i in range(1000):
        value = 80.0 if i in (700, 900) else 25.0 + (i % 2) * 0.5
        lines.append(f'TemperatureSensor,{i * 100},{value}')
    csv_path = tmp_path / 'long.csv'
    csv_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    _, rows, summary = _check_inline(results, 'TemperatureSensor', str(csv_path))
    assert len(rows) == 202
    assert [r[0] for r in rows[-2:]] == ['70000', '90000'] and [r[3] for r in rows[-2:]] == ['corrupt', 'corrupt']
    assert all(r[3] == 'ok' for r in rows[:200])
    assert summary == ('1000 samples: corrupt 2, stale 0, debounce 0\n'
                       'showing 202 of 1000 samples (all flagged rows included)')


def test_many_flagged_rows_arrive_in_chunks_without_loss_or_reordering(results, tmp_path, monkeypatch):
    monkeypatch.setattr(tui_app, 'SAMPLE_CHUNK', 100)   # 450 rows -> 5 UI-thread calls
    csv_path = tmp_path / 'all_bad.csv'
    csv_path.write_text('sensor,timestamp_ms,value\n' + ''.join(f'TemperatureSensor,{i * 100},80\n' for i in range(450)),
                        encoding='utf-8')

    _, rows, summary = _check_inline(results, 'TemperatureSensor', str(csv_path))
    assert [r[0] for r in rows] == [f'{i * 100:g}' for i in range(450)] and all(r[3] == 'corrupt' for r in rows)
    assert summary.endswith('showing 450 of 450 samples (all flagged rows included)')


def test_visible_samples_keeps_the_first_n_and_every_flagged_one_in_order():
    res = [SampleResult(t, 0.0, 0.0, ['corrupt'] if t in (1, 7, 9) else []) for t in range(12)]
    assert [r.timestamp_ms for r in visible_samples(res, first=3)] == [0, 1, 2, 7, 9]
    assert visible_samples([], first=3) == []


def test_sample_check_runs_on_a_worker_thread_and_shows_running(results, monkeypatch):
    entered, release, threads = threading.Event(), threading.Event(), []
    real = tui_app.check_samples

    def slow(template, samples):
        threads.append(threading.current_thread())
        entered.set()
        assert release.wait(20)
        return real(template, samples)
    monkeypatch.setattr(tui_app, 'check_samples', slow)

    async def scenario(app, screen, pilot):
        await _open_tab(screen, pilot, 'tab-sensors')
        screen.query_one('#sensor-select', Select).value = 'TemperatureSensor'
        screen.query_one('#sample-input', Input).value = '0:25.0, 100:80.0'
        await pilot.pause()
        await pilot.click('#btn-sensor-run')
        await _wait_until(pilot, entered.is_set)
        await pilot.pause()                                 # the app answers while the check is still blocked
        assert 'Running' in str(screen.query_one('#sample-summary', Static).content)
        assert screen.query_one('#dt-samples', DataTable).row_count == 0
        release.set()
        await _settle(app, pilot)
        assert screen.query_one('#dt-samples', DataTable).row_count == 2
        assert 'Running' not in str(screen.query_one('#sample-summary', Static).content)
    try:
        _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)
    finally:
        release.set()
    assert threads and threads[0] is not threading.main_thread()   # the pilot's event loop runs on the main thread


def test_effect_analyzer_is_built_once_on_a_worker_thread(results, monkeypatch):
    entered, release, built = threading.Event(), threading.Event(), []

    class Spy(tui_app.FunctionEffectAnalyzer):
        def __init__(self, *args, **kwargs):
            built.append(threading.current_thread())
            entered.set()
            assert release.wait(20)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(tui_app, 'FunctionEffectAnalyzer', Spy)

    async def scenario(app, screen, pilot):
        await _open_tab(screen, pilot, 'tab-effect')
        box, out = screen.query_one('#effect-input', Input), screen.query_one('#effect-output', Static)
        box.value = 'updateTemperature'
        await pilot.pause()
        await pilot.click('#btn-effect')
        await _wait_until(pilot, entered.is_set)
        await pilot.pause()
        assert 'Running' in str(out.content)
        release.set()
        await _settle(app, pilot)
        assert 'Validator' in str(out.content)
        box.value = 'process'                                # a second lookup reuses the analyzer
        box.focus()
        await pilot.press('enter')                           # not a 2nd click: a Button ignores clicks while -active
        await _settle(app, pilot)
        assert 'Sensor::process()' in str(out.content)
    try:
        _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)
    finally:
        release.set()
    assert len(built) == 1 and built[0] is not threading.main_thread()


def test_malformed_csv_is_reported_and_the_app_keeps_running(results, tmp_path):
    # sensor is the last column here: the short row used to raise AttributeError and kill the TUI
    bad = tmp_path / 'short.csv'
    bad.write_text('timestamp_ms,value,sensor\n0,25.0,TemperatureSensor\n100,25.5\n', encoding='utf-8')

    async def scenario(app, screen, pilot):
        await _open_tab(screen, pilot, 'tab-sensors')
        screen.query_one('#sensor-select', Select).value = 'TemperatureSensor'
        screen.query_one('#sample-input', Input).value = str(bad)
        await pilot.pause()
        await pilot.click('#btn-sensor-run')
        await _settle(app, pilot)
        assert app.screen is screen
        assert 'short.csv:3: bad sample row' in str(screen.query_one('#sample-summary', Static).content)
        assert screen.query_one('#dt-samples', DataTable).row_count == 0
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)


def test_unexpected_sample_check_error_is_shown_with_its_type_and_logged(results, monkeypatch, logged):
    def boom(template, samples):
        raise RuntimeError('boom [i]')
    monkeypatch.setattr(tui_app, 'check_samples', boom)

    async def scenario(app, screen, pilot):
        await _open_tab(screen, pilot, 'tab-sensors')
        screen.query_one('#sensor-select', Select).value = 'TemperatureSensor'
        screen.query_one('#sample-input', Input).value = '0:25.0'
        await pilot.pause()
        await pilot.click('#btn-sensor-run')
        await _settle(app, pilot)
        assert app.screen is screen
        assert str(screen.query_one('#sample-summary', Static).content) == 'RuntimeError: boom [i]'
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)
    assert any(r['message'] == 'TUI sample check failed' and r['exception'].type is RuntimeError for r in logged)


def test_unexpected_effect_error_is_shown_with_its_type_and_logged(results, monkeypatch, logged):
    class Boom(tui_app.FunctionEffectAnalyzer):
        def analyze(self, method):
            raise RuntimeError('boom [i]')
    monkeypatch.setattr(tui_app, 'FunctionEffectAnalyzer', Boom)

    async def scenario(app, screen, pilot):
        await _open_tab(screen, pilot, 'tab-effect')
        screen.query_one('#effect-input', Input).value = 'updateTemperature'
        await pilot.pause()
        await pilot.click('#btn-effect')
        await _settle(app, pilot)
        assert app.screen is screen
        assert str(screen.query_one('#effect-output', Static).content) == 'RuntimeError: boom [i]'
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)
    assert any(r['message'] == 'TUI function effect failed' and r['exception'].type is RuntimeError for r in logged)


def test_virtual_tree_shows_the_inheritance_under_each_class(results):
    async def scenario(app, screen, pilot):
        tree = screen.query_one('#virtual-tree', Tree)
        sensor = next(n for n in tree.root.children if str(n.label) == 'Sensor')
        inheritance = next(n for n in sensor.children if str(n.label) == 'Inheritance')
        assert [str(n.label) for n in inheritance.children] == ['AltitudeSensor', 'PressureSensor', 'TemperatureSensor']
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)


def test_inheritance_nodes_nest_and_stop_at_a_cycle():
    class FakeHierarchy:
        kids = {'A': ['B', 'C'], 'B': ['D'], 'D': ['A']}   # D -> A closes a cycle that must not recurse forever

        def derived(self, cls):
            return self.kids.get(cls, [])
    tree = Tree('root')
    ResultsScreen._add_derived(tree.root, FakeHierarchy(), 'A', {'A'})

    def shape(node):
        return {str(n.label): shape(n) for n in node.children}
    assert shape(tree.root) == {'B': {'D': {}}, 'C': {}}


def test_dead_code_warnings_beyond_five_are_counted(results):
    many = dataclasses.replace(results.dead, warnings=[f'warning {i}' for i in range(8)])
    capped = dataclasses.replace(results, dead=many)

    async def scenario(app, screen, pilot):
        info = str(screen.query_one('#dead-info', Static).content)
        assert 'warning 4' in info and 'warning 5' not in info and '(+3 more)' in info
    _drive(ResultsScreen(capped, 'out', _cfg(Path('.'))), scenario)


def test_every_input_has_a_placeholder(results):
    async def scenario(app, screen, pilot):
        inputs = list(screen.query(Input))
        assert {i.id for i in inputs} >= {'effect-input', 'sample-input'}
        assert all(i.placeholder for i in inputs)
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)

    async def config_scenario(app, screen, pilot):
        inputs = list(screen.query(Input))
        assert 'sensor-templates' in {i.id for i in inputs}
        assert all(i.placeholder for i in inputs)
    _drive(ConfigScreen(), config_scenario)


# ── Bindings ────────────────────────────────────────────────────────────────

def test_results_bindings_chat_back_and_quit(results, monkeypatch):
    async def scenario(app, screen, pilot):
        await pilot.press('c')
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        await pilot.press('escape')
        await pilot.pause()
        assert app.screen is screen
        quits = []
        monkeypatch.setattr(app, 'exit', lambda *a, **k: quits.append(True))
        await pilot.press('q')
        assert quits
        await pilot.press('escape')
        await pilot.pause()
        assert app.screen is not screen
    _drive(ResultsScreen(results, 'out', _cfg(Path('.'))), scenario)


# ── Chat ────────────────────────────────────────────────────────────────────

def _chat_cfg() -> Config:
    return Config(ollama_enabled=False)


def test_chat_lists_the_three_example_questions(results):
    async def scenario(app, screen, pilot):
        text = _log_text(screen.query_one('#chat-log', RichLog))
        for question in ('What are the virtual functions implemented by TemperatureSensor?',
                         'What components are affected if updateTemperature() changes?',
                         'What validation checks are applied to TemperatureSensor?'):
            assert question in text
        assert 'offline' in str(screen.query_one('#chat-status', Static).content).lower()
    _drive(ChatScreen(results, _chat_cfg()), scenario)


def test_chat_answers_offline_and_shows_ranges_literally(results):
    async def scenario(app, screen, pilot):
        screen.query_one('#chat-input', Input).focus()
        screen.query_one('#chat-input', Input).value = 'What validation checks are applied to TemperatureSensor?'
        await pilot.press('enter')
        await _settle(app, pilot)
        text = _log_text(screen.query_one('#chat-log', RichLog))
        assert 'You: What validation checks are applied to TemperatureSensor?' in text
        assert 'range [-54, 54]' in text and 'Assistant:' in text
    _drive(ChatScreen(results, _chat_cfg()), scenario)


def test_chat_answers_arrive_in_question_order_and_thinking_lasts_until_the_last(results):
    first_started, release_first = threading.Event(), threading.Event()
    asked = []

    def fake_ask(question):
        asked.append(question)
        if question == 'first?':
            first_started.set()
            assert release_first.wait(20)
        return f'answer to {question}'

    async def scenario(app, screen, pilot):
        screen._ask = fake_ask
        chat, status = screen.query_one('#chat-input', Input), screen.query_one('#chat-status', Static)
        chat.focus()
        for question in ('first?', 'second?'):
            chat.value = question
            await pilot.press('enter')
            await pilot.pause()
        await _wait_until(pilot, first_started.is_set)
        assert asked == ['first?']                          # the second waits for the first
        assert 'Thinking' in str(status.content)
        release_first.set()
        await _wait_until(pilot, lambda: 'answer to second?' in _log_text(screen.query_one('#chat-log', RichLog)))
        text = _log_text(screen.query_one('#chat-log', RichLog))
        order = [text.index(m) for m in ('You: first?', 'You: second?', 'Assistant: answer to first?',
                                         'Assistant: answer to second?')]
        assert order == sorted(order) and asked == ['first?', 'second?']
        assert 'Thinking' not in str(status.content)
    try:
        _drive(ChatScreen(results, _chat_cfg()), scenario)
    finally:
        release_first.set()


def test_chat_status_stays_thinking_between_two_answers(results):
    first_started, release_first = threading.Event(), threading.Event()
    statuses = []

    def fake_ask(question):
        if question == 'first?':
            first_started.set()
            assert release_first.wait(20)
        return f'answer to {question}'

    async def scenario(app, screen, pilot):
        screen._ask = fake_ask
        status = screen.query_one('#chat-status', Static)
        real_update = screen._update_status
        screen._update_status = lambda thinking=False: (statuses.append((thinking, str(status.content))), real_update(thinking))
        chat = screen.query_one('#chat-input', Input)
        chat.focus()
        for question in ('first?', 'second?'):
            chat.value = question
            await pilot.press('enter')
            await pilot.pause()
        await _wait_until(pilot, first_started.is_set)
        release_first.set()
        await _wait_until(pilot, lambda: 'answer to second?' in _log_text(screen.query_one('#chat-log', RichLog)))
        await pilot.pause()
        assert 'Thinking' not in str(status.content)
        # the status was never reset to idle between the two answers: only the final answer calls it with thinking=False
        assert [t for t, _ in statuses].count(False) == 1 and statuses[-1][0] is False
    try:
        _drive(ChatScreen(results, _chat_cfg()), scenario)
    finally:
        release_first.set()


def test_chat_answer_is_not_parsed_as_markup(results):
    async def scenario(app, screen, pilot):
        screen._show_answer('a[i] = [bold]x[/bold] in range [-54, 54] [red]')
        await pilot.pause()
        assert 'Assistant: a[i] = [bold]x[/bold] in range [-54, 54] [red]' in _log_text(
            screen.query_one('#chat-log', RichLog))
    _drive(ChatScreen(results, _chat_cfg()), scenario)


def test_chat_question_is_not_parsed_as_markup(results):
    async def scenario(app, screen, pilot):
        screen.query_one('#chat-input', Input).value = 'what about [bold]x[/bold]?'
        screen.query_one('#chat-input', Input).focus()
        await pilot.press('enter')
        await _settle(app, pilot)
        assert 'You: what about [bold]x[/bold]?' in _log_text(screen.query_one('#chat-log', RichLog))
    _drive(ChatScreen(results, _chat_cfg()), scenario)


# ── Config -> Analysis -> Results ───────────────────────────────────────────

@pytest.fixture
def tui_config(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, templates=False)
    monkeypatch.setattr('rag_engine.tui.app.load_config', lambda *a, **k: cfg)
    monkeypatch.setattr('rag_engine.pipeline._semantic_search', lambda cfg, functions: None)   # no model
    return cfg


def _run_app(scenario):
    async def main():
        app = RAGEngineApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            return await scenario(app, pilot)
    return asyncio.run(main())


def test_analysis_flow_passes_templates_through_and_reruns(tui_config):
    async def scenario(app, pilot):
        config = app.screen
        assert isinstance(config, ConfigScreen)
        config.query_one('#codebase', Input).value = str(SENSORS)
        config.query_one('#sensor-templates', Input).value = str(TEMPLATES)
        config.query_one('#output-format', Select).value = 'markdown'
        await pilot.click('#btn-run')
        await _settle(app, pilot)
        assert isinstance(app.screen, ResultsScreen)
        analysis = app.screen_stack[-2]
        assert isinstance(analysis, AnalysisScreen)
        assert [analysis.query_one(f'#{i}', ProgressBar).progress for i in PROGRESS_BARS] == [100] * 6
        assert analysis._results.error is None and analysis._results.sensors is not None
        log = _log_text(analysis.query_one('#log-box', RichLog))
        assert 'Sensor analysis' in log and 'Done' in log
        assert 'keyword matching' in log        # semantic search was unavailable: said so
        assert (Path(tui_config.output_dir) / 'sensor_analysis.md').is_file()
        assert app.screen.query_one('#dt-sensors', DataTable).row_count == 4
        await pilot.press('r')
        await pilot.pause()
        assert isinstance(app.screen, ConfigScreen)
    _run_app(scenario)


@pytest.mark.parametrize('step, expected', [
    ('Parsing C++ files', [50, 0, 0, 0, 0, 0]),
    ('Building call graph', [50, 0, 0, 0, 0, 0]),
    ('Virtual function analysis', [100, 50, 0, 0, 0, 0]),
    ('Dead code detection', [100, 100, 50, 0, 0, 0]),
    ('Standards validation', [100, 100, 100, 50, 0, 0]),
    ('Coupling analysis', [100, 100, 100, 100, 50, 0]),
    ('Sensor analysis', [100, 100, 100, 100, 100, 50]),
    ('LLM enrichment via llama3.1:8b', None),
    ('Writing reports and checklist', None),
    ('Done', None),
])
def test_stage_progress_maps_pipeline_steps_to_rows(step, expected):
    assert stage_progress(step) == expected


def test_steps_without_a_row_leave_the_bars_unchanged_until_the_pipeline_returns(monkeypatch):
    release = threading.Event()

    def blocked(*args, **kwargs):
        assert release.wait(20)
        raise InputError('stop here')
    monkeypatch.setattr(tui_app, 'run_pipeline', blocked)
    screen = AnalysisScreen(str(SENSORS), None, None, None, 'markdown', 'B', cfg=_cfg(Path('.')))

    async def scenario(app, screen, pilot):
        def bars():
            return [screen.query_one(f'#{i}', ProgressBar).progress for i in PROGRESS_BARS]
        screen._show_progress('Dead code detection')
        assert bars() == [100, 100, 50, 0, 0, 0]
        for step in ('LLM enrichment via llama3.1:8b', 'Writing reports and checklist', 'Done'):
            screen._show_progress(step)
            assert bars() == [100, 100, 50, 0, 0, 0], step
        release.set()
        await _settle(app, pilot)
    try:
        _drive(screen, scenario)
    finally:
        release.set()


@pytest.fixture
def pipeline_calls(monkeypatch):
    """Spy on run_pipeline: records each call's keyword arguments and runs the real pipeline."""
    calls = []
    real = tui_app.run_pipeline

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)
    monkeypatch.setattr('rag_engine.tui.app.run_pipeline', spy)
    return calls


def test_semantic_search_checkbox_is_labelled_and_on_by_default():
    async def scenario(app, screen, pilot):
        box = screen.query_one('#semantic-search', Checkbox)
        assert box.value is True
        assert 'semantic search' in str(box.label).lower() and 'sentence-transformers' in str(box.label)
    _drive(ConfigScreen(), scenario)


def test_semantic_search_is_requested_by_default(tui_config, pipeline_calls):
    async def scenario(app, pilot):
        app.screen.query_one('#codebase', Input).value = str(SENSORS)
        await pilot.click('#btn-run')
        await _settle(app, pilot)
        assert [c['use_semantic_search'] for c in pipeline_calls] == [True]
        assert 'keyword matching' in _log_text(app.screen_stack[-2].query_one('#log-box', RichLog))
    _run_app(scenario)


def test_unchecking_semantic_search_runs_the_pipeline_without_it(tui_config, pipeline_calls):
    async def scenario(app, pilot):
        app.screen.query_one('#codebase', Input).value = str(SENSORS)
        await pilot.click('#semantic-search')
        assert app.screen.query_one('#semantic-search', Checkbox).value is False
        await pilot.click('#btn-run')
        await _settle(app, pilot)
        assert [c['use_semantic_search'] for c in pipeline_calls] == [False]
        assert isinstance(app.screen, ResultsScreen)
        log = _log_text(app.screen_stack[-2].query_one('#log-box', RichLog))
        assert 'Semantic search is off' in log and 'keyword matching' in log
        assert 'unavailable' not in log
    _run_app(scenario)


def test_analysis_without_templates_skips_the_sensor_row(tui_config):
    async def scenario(app, pilot):
        app.screen.query_one('#codebase', Input).value = str(SENSORS)
        await pilot.click('#btn-run')
        await _settle(app, pilot)
        analysis = app.screen_stack[-2]
        bars = {i: analysis.query_one(f'#{i}', ProgressBar).progress for i in PROGRESS_BARS}
        assert bars.pop('pb-sensors') == 0 and set(bars.values()) == {100}
        assert app.screen._results.sensors is None
    _run_app(scenario)


def test_config_rejects_missing_templates_file(tui_config):
    async def scenario(app, pilot):
        app.screen.query_one('#codebase', Input).value = str(SENSORS)
        app.screen.query_one('#sensor-templates', Input).value = str(SENSORS / 'nope.yaml')
        await pilot.click('#btn-run')
        await pilot.pause()
        assert isinstance(app.screen, ConfigScreen)
        assert any('nope.yaml' in n.message for n in app._notifications)
    _run_app(scenario)


def test_input_error_is_shown_plainly_and_recorded(tui_config, tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()

    async def scenario(app, pilot):
        app.screen.query_one('#codebase', Input).value = str(empty)
        await pilot.click('#btn-run')
        await _settle(app, pilot)
        screen = app.screen
        assert isinstance(screen, AnalysisScreen)
        message = f'No C++ files found in {empty}'
        assert screen._results.error == message
        assert message in _log_text(screen.query_one('#log-box', RichLog))
        assert any(message in n.message and n.severity == 'error' for n in app._notifications)
        await pilot.press('escape')
        await pilot.pause()
        assert isinstance(app.screen, ConfigScreen)
    _run_app(scenario)


def test_unexpected_error_names_its_type_and_is_logged(tui_config, monkeypatch, logged):
    def boom(*args, **kwargs):
        raise RuntimeError('boom [i]')
    monkeypatch.setattr('rag_engine.tui.app.run_pipeline', boom)

    async def scenario(app, pilot):
        app.screen.query_one('#codebase', Input).value = str(SENSORS)
        await pilot.click('#btn-run')
        await _settle(app, pilot)
        screen = app.screen
        assert screen._results.error == 'RuntimeError: boom [i]'
        assert 'RuntimeError: boom [i]' in _log_text(screen.query_one('#log-box', RichLog))
        assert any('RuntimeError: boom [i]' in n.message for n in app._notifications)
    _run_app(scenario)
    assert any(r['message'] == 'TUI analysis failed' and r['exception'].type is RuntimeError for r in logged)
