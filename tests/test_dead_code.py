import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rag_engine.config import Config, load_config
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.core.parser import CodeParser
from rag_engine.features.dead_code_detector import DeadCodeDetector
from rag_engine.reporting.checklist_filler import ChecklistFiller
from rag_engine.reporting.report_generator import ReportGenerator

FIXTURES = Path(__file__).parent / 'fixtures' / 'cpp'
SENSORS = FIXTURES / 'sensors'


def _build(src_dir):
    results = CodeParser().parse_directory(str(src_dir))
    gb = CallGraphBuilder(results)
    gb.build()
    return gb, results


def _detector(src_dir=FIXTURES / 'base_build', entry_points=None, **kw):
    gb, results = _build(src_dir)
    return DeadCodeDetector(gb, results, entry_points, **kw)


def _sensors(entry_points=('main',), **kw):
    return _detector(SENSORS, list(entry_points), **kw).analyze()


def _by_category(report, category):
    return {i.name: i for i in report.items if i.category == category}


def _source_tree(tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return tmp_path


# ── Base build (sig_key names) ─────────────────────────────────────────────────

def test_analyze_returns_report():
    assert _detector().analyze().items is not None


def test_orphan_func_detected():
    report = _detector().analyze()
    assert 'orphanFunc()' in {i.name for i in report.items}


def test_orphan_is_dead_code():
    item_map = {i.name: i for i in _detector().analyze().items}
    assert item_map['orphanFunc()'].category == 'dead_code'
    assert item_map['orphanFunc()'].do178c_disposition == 'Remove'


def test_coverage_impact_populated():
    for item in _detector().analyze().items:
        assert item.coverage_impact


def test_entry_points_not_in_dead():
    assert 'main()' not in {i.name for i in _detector().analyze().items}


def test_virtual_methods_of_uninstantiated_class_are_reported_not_hidden():
    items = {i.name: i for i in _detector().analyze().items}
    assert items['Vehicle::start()'].category == 'unused_export'
    assert items['Vehicle::stop()'].category == 'dead_code' and items['Vehicle::stop()'].symbol_type == 'method'


# ── Sensors fixture: classification (brief 3.4) ────────────────────────────────

def test_dead_code_functions_are_removable_with_high_confidence():
    dead = _by_category(_sensors(), 'dead_code')
    assert set(dead) == {'orphanHelper()', 'legacyCalibrate()', 'SensorManager::unusedDiagnostic()',
                         'Display::clear(int)', 'Sensor::getValue()'}
    assert {i.do178c_disposition for i in dead.values()} == {'Remove'}
    assert {i.confidence for i in dead.values()} == {'high'}
    assert dead['orphanHelper()'].symbol_type == 'function'
    assert dead['Display::clear(int)'].symbol_type == 'method'
    assert 'no call path from entry points' in dead['orphanHelper()'].reason and 'main()' in dead['orphanHelper()'].reason


def test_unused_exports_are_investigated():
    exports = _by_category(_sensors(), 'unused_export')
    assert set(exports) == {'AltitudeSensor::read()', 'AltitudeSensor::selfTest()', 'Sensor::selfTest()'}
    assert {i.do178c_disposition for i in exports.values()} == {'Investigate'}
    assert {i.confidence for i in exports.values()} == {'medium'}
    assert 'never instantiated' in exports['AltitudeSensor::read()'].reason
    assert 'never instantiated' in exports['AltitudeSensor::selfTest()'].reason
    assert 'never called' in exports['Sensor::selfTest()'].reason


def test_preprocessor_excluded_function_is_deactivated_with_high_confidence():
    deactivated = _by_category(_sensors(), 'deactivated_code')
    assert set(deactivated) == {'disabledFeature()'}
    item = deactivated['disabledFeature()']
    assert (item.do178c_disposition, item.confidence) == ('Justify as Deactivated', 'high')
    assert '#if 0' in item.reason


def test_other_preprocessor_guard_is_deactivated_with_medium_confidence(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'int main() { return 0; }\n#ifdef DEBUG_BUILD\nvoid dbg() {}\n#endif\n'})
    item = _by_category(_detector(tmp_path, ['main']).analyze(), 'deactivated_code')['dbg()']
    assert item.confidence == 'medium' and 'DEBUG_BUILD' in item.reason


def test_test_support_names_are_deactivated_with_low_confidence(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'int main() { return 0; }\nvoid mockSensor() {}\nvoid stubClock() {}\n'})
    items = _by_category(_detector(tmp_path, ['main']).analyze(), 'deactivated_code')
    assert set(items) == {'mockSensor()', 'stubClock()'}
    assert {i.confidence for i in items.values()} == {'low'}


def test_callback_like_names_are_investigated_with_low_confidence(tmp_path):
    names = ['onTimerTick', 'sensorHandler', 'uartIsr', 'timerCallback', 'handler', 'timer_hook', 'initSystem']
    _source_tree(tmp_path, {'a.cpp': 'int main() { return 0; }\n' + ''.join(f'void {n}() {{}}\n' for n in names)
                                     + 'class C { public: void onEvent() {} };\n'})
    exports = _by_category(_detector(tmp_path, ['main']).analyze(), 'unused_export')
    assert set(exports) == {f'{n}()' for n in names} | {'C::onEvent()'}
    for item in exports.values():
        assert (item.do178c_disposition, item.confidence) == ('Investigate', 'low')
        assert 'entry_points' in item.reason


def test_names_without_a_callback_token_are_dead_code(tmp_path):
    # no capital after "on", no token boundary before "handler", "init" only as a whole word
    names = ['onionPeel', 'chandler', 'reinit', 'shook']
    _source_tree(tmp_path, {'a.cpp': 'int main() { return 0; }\n' + ''.join(f'void {n}() {{}}\n' for n in names)})
    assert set(_by_category(_detector(tmp_path, ['main']).analyze(), 'dead_code')) == {f'{n}()' for n in names}


def test_dead_fragments():
    frags = _by_category(_sensors(), 'dead_fragment')
    assert len(frags) == 2
    after_return = next(i for i in frags.values() if i.function_name == 'computeChecksum(int)')
    assert after_return.name == 'x = 0;' and (after_return.line_number, after_return.end_line) == (30, 30)
    assert after_return.symbol_type == 'fragment'
    if_zero = next(i for i in frags.values() if i.function_name == 'main()')
    assert (if_zero.line_number, if_zero.end_line) == (42, 44)
    assert {i.do178c_disposition for i in frags.values()} == {'Fix Fragment'}
    assert {i.confidence for i in frags.values()} == {'high'}
    assert all(i.file_path.endswith('main.cpp') and i.reason for i in frags.values())


def test_unused_variables_and_parameters():
    items = _by_category(_sensors(), 'unused_variable')
    got = {(i.name, i.function_name, i.symbol_type): i for i in items.values()}
    assert set(got) == {('unused', 'orphanHelper()', 'variable'),
                        ('code', 'SensorManager::unusedDiagnostic()', 'variable'),
                        ('mode', 'Display::clear(int)', 'parameter')}
    local = got[('unused', 'orphanHelper()', 'variable')]
    assert (local.do178c_disposition, local.confidence, local.line_number) == ('Remove', 'medium', 25)
    param = got[('mode', 'Display::clear(int)', 'parameter')]
    assert (param.do178c_disposition, param.confidence) == ('Investigate', 'low')


def test_everything_else_is_not_reported_as_a_function_finding():
    findings = {i.name for i in _sensors().items if i.symbol_type in ('function', 'method')}
    assert findings == {
        'orphanHelper()', 'legacyCalibrate()', 'SensorManager::unusedDiagnostic()', 'Display::clear(int)',
        'Sensor::getValue()', 'AltitudeSensor::read()', 'AltitudeSensor::selfTest()', 'Sensor::selfTest()',
        'disabledFeature()'}
    for reached in ('main()', 'TIMER0_IRQHandler()', 'onTimerTick()', 'registerCallback(Callback)',
                    'SensorManager::init()', 'SensorManager::update()', 'SensorManager::updateTemperature()',
                    'TemperatureSensor::process()', 'PressureSensor::read()', 'Validator::check(float)',
                    'Display::update(float)', 'Sensor::process()', 'Sensor::~Sensor()', 'computeChecksum(int)'):
        assert reached not in findings


def test_configured_unknown_entry_point_is_warned_about():
    report = _sensors(entry_points=('main', 'doesNotExist'))
    assert any("'doesNotExist' not found" in w for w in report.warnings)
    assert not any("'main'" in w for w in report.warnings)
    assert not any('not found' in w for w in _sensors().warnings)


def test_entry_point_patterns_make_interrupt_handlers_entry_points():
    assert 'TIMER0_IRQHandler()' in _sensors().entry_points_used
    assert 'main()' in _sensors().entry_points_used
    report = _sensors(entry_point_patterns=[])
    handler = _by_category(report, 'unused_export')['TIMER0_IRQHandler()']   # callback-like name, so low-confidence Investigate
    assert (handler.confidence, handler.do178c_disposition) == ('low', 'Investigate')
    assert 'TIMER0_IRQHandler()' not in report.entry_points_used


def test_report_totals():
    report = _sensors()
    assert report.counts_by_category == {'dead_code': 5, 'deactivated_code': 1, 'unused_export': 3,
                                         'dead_fragment': 2, 'unused_variable': 3}
    assert (report.dead_count, report.deactivated_count) == (5, 1)
    assert report.total_functions > 20
    assert report.false_positive_notes
    # 5 + 3 + 1 function findings and 2 fragments; the if (0) block lies inside main()
    assert report.structural_coverage_impact == '14 findings; ~24 uncoverable source lines'


def test_a_file_the_flow_analysis_cannot_handle_is_a_warning_not_a_crash(monkeypatch):
    import rag_engine.features.dead_code_detector as detector_module
    monkeypatch.setattr(detector_module, 'find_dead_fragments', lambda result: 1 / 0)
    report = _sensors()
    assert any('Control-flow analysis' in w and 'main.cpp' in w for w in report.warnings)
    assert 'orphanHelper()' in {i.name for i in report.items} and not _by_category(report, 'dead_fragment')


def test_warnings_for_a_graph_without_edges_and_mostly_unreachable_code(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'int main() { return 0; }\nvoid a() {}\nvoid b() {}\nvoid c() {}\n'})
    warnings = _detector(tmp_path, ['main']).analyze().warnings
    assert any('no call edges' in w for w in warnings)
    assert any('3 of 4 functions' in w and 'check entry_points configuration' in w for w in warnings)


# ── Entry point names ──────────────────────────────────────────────────────────

def test_entry_points_may_be_bare_qualified_or_sig_key_names():
    for name in ('updateTemperature', 'SensorManager::updateTemperature', 'SensorManager::updateTemperature()'):
        report = _sensors(entry_points=(name,))
        assert report.entry_points_used.count('SensorManager::updateTemperature()') == 1
        assert not any('not found' in w for w in report.warnings)
        assert 'Validator::check(float)' not in {i.name for i in report.items}
        assert 'SensorManager::update()' in {i.name for i in report.items}


def test_ambiguous_bare_entry_point_resolves_to_every_match_with_a_warning():
    report = _sensors(entry_points=('update',))
    assert {'SensorManager::update()', 'Display::update(float)'} <= set(report.entry_points_used)
    assert any("'update' is ambiguous" in w for w in report.warnings)


def test_graph_builder_resolves_entry_points():
    gb, _ = _build(SENSORS)
    nodes, warnings = gb.resolve_entry_points(['main', 'Sensor::process', 'read', 'nope'])
    assert {'main()', 'Sensor::process()', 'TemperatureSensor::read()', 'PressureSensor::read()',
            'AltitudeSensor::read()'} <= nodes
    assert len(warnings) == 2 and any("'read'" in w for w in warnings) and any("'nope'" in w for w in warnings)
    assert gb.find_reachable_from('updateTemperature') == {
        'TemperatureSensor::read()', 'Validator::check(float)', 'Display::update(float)'}


def test_implicit_entry_points_are_constructors_and_destructors_of_instantiated_classes_and_ancestors():
    gb, _ = _build(SENSORS)
    assert gb.implicit_entry_points() == {'Sensor::~Sensor()'}
    gb, _ = _build(FIXTURES / 'base_build')
    assert gb.implicit_entry_points() == set()


def test_implicit_entry_points_include_constructors(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'class B { public: B() {} virtual ~B() {} };\n'
                                     'class D : public B { public: D() {} ~D() {} void f() {} };\n'
                                     'class U { public: U() {} };\n'
                                     'int main() { D d; return 0; }\n'})
    gb, _ = _build(tmp_path)
    assert gb.implicit_entry_points() == {'B::B()', 'B::~B()', 'D::D()', 'D::~D()'}


# ── Dead branches in call extraction ───────────────────────────────────────────

def test_call_inside_if_zero_creates_no_edge():
    gb, _ = _build(SENSORS)
    assert gb.graph.in_degree('legacyCalibrate()') == 0


def test_constant_false_branches_are_blanked_but_else_branches_and_other_conditions_are_not(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'void a() {} void b() {} void c() {} void d() {} void e() {}\n'
                                     'int main() {\n'
                                     '  if (false) { a(); } else { b(); }\n'
                                     '  while (0) { if (x) { c(); } }\n'
                                     '  if (0x1) { d(); }\n'
                                     '  do { e(); } while (0);\n'
                                     '  return 0;\n}\n'})
    gb, _ = _build(tmp_path)
    assert set(gb.graph.successors('main()')) == {'b()', 'd()', 'e()'}


# ── Stable IDs ─────────────────────────────────────────────────────────────────

def test_finding_ids_have_the_documented_shape():
    items = _sensors().items
    ids = [i.finding_id for i in items]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r'DC-(FUNC|METH|FRAG|VAR|CFG)-[0-9A-F]{8}', i) for i in ids)
    kinds = {i.name: i.finding_id.split('-')[1] for i in items}
    assert kinds['orphanHelper()'] == 'FUNC' and kinds['Display::clear(int)'] == 'METH'
    assert kinds['disabledFeature()'] == 'CFG' and kinds['unused'] == 'VAR' and kinds['mode'] == 'VAR'
    assert kinds['x = 0;'] == 'FRAG'


def test_ids_survive_a_blank_line_at_the_top_of_a_file(tmp_path):
    copy = tmp_path / 'sensors'
    shutil.copytree(SENSORS, copy)
    before = {(i.name, i.function_name): i.finding_id for i in _detector(copy, ['main']).analyze().items}
    main_cpp = copy / 'main.cpp'
    main_cpp.write_text('\n' + main_cpp.read_text())
    after = {(i.name, i.function_name): i.finding_id for i in _detector(copy, ['main']).analyze().items}
    assert after == before and len(before) == 14


def test_identical_fragments_in_one_function_get_distinct_ids(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'int main() { return 0; x = 0; }\nint g(int x) { return 1; x = 0; }\n'
                                     'int h(int x) {\n  if (0) { x = 1; }\n  if (0) { x = 1; }\n  return x;\n}\n'})
    ids = [i.finding_id for i in _detector(tmp_path, ['main']).analyze().items if i.category == 'dead_fragment']
    assert len(ids) == 4 and len(set(ids)) == 4


# ── Evidence and removal impact ────────────────────────────────────────────────

def test_evidence_and_removal_impact(tmp_path):
    _source_tree(tmp_path, {
        'a.h': 'void helper();\nvoid used();\nextern int g_count;\n',
        'a.cpp': '#include "a.h"\nint g_count = 0;\nvoid used() {}\nvoid helper() { g_count++; }\n'
                 '// DO-178C-REQ: SRS-9\nvoid parent() { helper(); used(); }\n'
                 'int main() { used(); return 0; }\n'})
    items = {i.name: i for i in _detector(tmp_path, ['main']).analyze().items if i.category == 'dead_code'}
    assert set(items) == {'helper()', 'parent()'}
    assert 'callers: none' in items['parent()'].evidence
    assert any(e.startswith('only called by unreachable: parent()') for e in items['helper()'].evidence)
    assert any('direct' in e for e in items['helper()'].evidence)
    assert 'helper() would lose its only caller' in items['parent()'].removal_impact
    assert not any('used()' in r for r in items['parent()'].removal_impact)   # main() still calls it
    assert any('g_count' in r for r in items['helper()'].removal_impact)
    assert 'update trace matrix (SRS-9)' in items['parent()'].removal_impact
    assert 'remove declaration in a.h' in items['helper()'].removal_impact
    assert not any('declaration' in r for r in items['parent()'].removal_impact)


def test_method_declared_in_a_header_has_a_declaration_to_remove():
    item = _by_category(_sensors(), 'dead_code')['Display::clear(int)']
    assert 'remove declaration in display.h' in item.removal_impact
    assert item.evidence[0] == 'callers: none'
    assert not any('declaration' in r for r in _by_category(_sensors(), 'dead_code')['Sensor::getValue()'].removal_impact)


# ── Dispositions file ──────────────────────────────────────────────────────────

def _ids(report):
    return {i.name: i.finding_id for i in report.items}


def _write_dispositions(path, findings):
    path.write_text(yaml.safe_dump({'findings': findings}))
    return str(path)


def test_dispositions_are_applied(tmp_path):
    ids = _ids(_sensors())
    path = _write_dispositions(tmp_path / 'd.yaml', {
        ids['legacyCalibrate()']: {'status': 'accepted', 'justification': 'kept for rollback', 'reviewer': 'NR'}})
    report = _sensors(dispositions_file=path)
    item = _by_category(report, 'dead_code')['legacyCalibrate()']
    assert (item.review_status, item.reviewer_justification, item.reviewer) == ('accepted', 'kept for rollback', 'NR')
    assert _by_category(report, 'dead_code')['orphanHelper()'].review_status == 'open'
    assert not report.warnings


def test_disposition_without_justification_is_ignored_with_a_warning(tmp_path):
    ids = _ids(_sensors())
    path = _write_dispositions(tmp_path / 'd.yaml', {ids['legacyCalibrate()']: {'status': 'accepted'}})
    report = _sensors(dispositions_file=path)
    assert _by_category(report, 'dead_code')['legacyCalibrate()'].review_status == 'open'
    assert any(ids['legacyCalibrate()'] in w and 'justification' in w for w in report.warnings)


def test_disposition_for_an_unknown_id_is_a_stale_warning(tmp_path):
    path = _write_dispositions(tmp_path / 'd.yaml', {'DC-FUNC-00000000': {'status': 'accepted', 'justification': 'x'}})
    report = _sensors(dispositions_file=path)
    assert any('stale disposition' in w and 'DC-FUNC-00000000' in w for w in report.warnings)


def test_missing_dispositions_file_is_a_warning(tmp_path):
    report = _sensors(dispositions_file=str(tmp_path / 'nope.yaml'))
    assert any('nope.yaml' in w for w in report.warnings) and report.items


# ── Report outputs ─────────────────────────────────────────────────────────────

def test_report_and_findings_json(tmp_path):
    report = _sensors(entry_points=('main', 'doesNotExist'))
    gen = ReportGenerator(Config(output_dir=str(tmp_path)))
    text = Path(gen.generate_dead_code(report, format='markdown')).read_text(encoding='utf-8')
    for section in ('## Analysis Scope', '## Warnings', '## Summary', '## Findings', '## Finding Details'):
        assert section in text
    assert '| ID | File | Function/Symbol | Lines | Type | Reason | Confidence | Status |' in text
    assert 'TIMER0_IRQHandler()' in text and "Entry point 'doesNotExist' not found" in text
    orphan = next(i for i in report.items if i.name == 'orphanHelper()')
    assert f'| {orphan.finding_id} | main.cpp | orphanHelper() | 24-26 | function |' in text
    assert 'callers: none' in text and 'dead_code' in text

    data = json.loads((tmp_path / 'dead_code_findings.json').read_text(encoding='utf-8'))
    assert {i['finding_id'] for i in data['items']} == {i.finding_id for i in report.items}
    assert data['warnings'] == report.warnings and data['entry_points'] == report.entry_points_used
    first = next(i for i in data['items'] if i['name'] == 'orphanHelper()')
    assert first['category'] == 'dead_code' and first['confidence'] == 'high' and first['review_status'] == 'open'


def test_pipes_in_a_snippet_do_not_break_the_findings_table(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'int main() { return 0; }\nint g(int a, int b) { return 1; a = a || b; }\n'})
    report = _detector(tmp_path, ['main']).analyze()
    gen = ReportGenerator(Config(output_dir=str(tmp_path / 'out')))
    text = Path(gen.generate_dead_code(report, format='markdown')).read_text(encoding='utf-8')
    assert r'g(int, int) / a = a \|\| b;' in text
    assert Path(gen.generate_dead_code(report, format='docx')).exists()


def test_report_with_no_findings(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'int main() { return 0; }\n'})
    report = _detector(tmp_path, ['main']).analyze()
    text = Path(ReportGenerator(Config(output_dir=str(tmp_path / 'out'))).generate_dead_code(
        report, format='markdown')).read_text(encoding='utf-8')
    assert 'No dead or deactivated code findings' in text
    assert json.loads((tmp_path / 'out' / 'dead_code_findings.json').read_text())['items'] == []


# ── SQA checklist ──────────────────────────────────────────────────────────────

def _checklist(tmp_path, dead):
    virtual = SimpleNamespace(changes=[], summary={})
    standards = SimpleNamespace(violations_by_severity={}, compliance_score=100.0)
    cl = ChecklistFiller(Config(output_dir=str(tmp_path))).fill_sqa_checklist(virtual, dead, standards)
    return {i.item_id: i for i in cl.items}


def test_checklist_dc03_is_open_while_a_high_confidence_dead_function_is_open(tmp_path):
    report = _sensors()
    assert _checklist(tmp_path, report)['DC-03'].status == 'OPEN'
    ids = _ids(report)
    dispositions = {ids[n]: {'status': 'accepted', 'justification': 'ok'}
                    for n in _by_category(report, 'dead_code')}
    closed = _sensors(dispositions_file=_write_dispositions(tmp_path / 'd.yaml', dispositions))
    assert _checklist(tmp_path, closed)['DC-03'].status == 'PASS'


def test_checklist_dc03_ignores_other_categories(tmp_path):
    report = _sensors()
    report.items = [i for i in report.items if i.category != 'dead_code']
    assert _checklist(tmp_path, report)['DC-03'].status == 'PASS'


# ── Config ─────────────────────────────────────────────────────────────────────

def test_load_config_keeps_list_fields_from_yaml(tmp_path):
    path = tmp_path / 'c.yaml'
    path.write_text(yaml.safe_dump({
        'entry_points': ['main', 'Vehicle::start'], 'lru_names': ['ADS', 'FMS'],
        'entry_point_patterns': ['*_Callback'], 'dead_code_dispositions': 'dispositions.yaml',
        'not_a_field': 1}))
    cfg = load_config(str(path))
    assert cfg.entry_points == ['main', 'Vehicle::start']
    assert cfg.lru_names == ['ADS', 'FMS']
    assert cfg.entry_point_patterns == ['*_Callback']
    assert Path(cfg.dead_code_dispositions) == (tmp_path / 'dispositions.yaml').resolve()   # relative to the config file


def test_absolute_and_empty_dispositions_paths_are_left_alone(tmp_path):
    absolute = str((tmp_path / 'abs.yaml').resolve())
    path = tmp_path / 'c.yaml'
    path.write_text(yaml.safe_dump({'dead_code_dispositions': absolute}))
    assert load_config(str(path)).dead_code_dispositions == absolute
    path.write_text(yaml.safe_dump({'dead_code_dispositions': ''}))
    assert load_config(str(path)).dead_code_dispositions == ''


def test_config_defaults():
    cfg = Config()
    assert cfg.entry_point_patterns == ['*_IRQHandler', '*_ISR', 'ISR_*', '*Task']
    assert cfg.dead_code_dispositions == ''


# ── Fix round 1: probes ────────────────────────────────────────────────────────

def _functions(report):
    return {i.name: i for i in report.items if i.symbol_type in ('function', 'method')}


def test_functions_under_if_zero_are_never_entry_points(tmp_path):
    _source_tree(tmp_path, {'a.cpp': (
        'int main() { K k; return 0; }\nvoid helper() {}\n'
        'class K {\npublic:\n#if 0\n  ~K() { helper(); }\n#endif\n};\n'
        '#if 0\nvoid UART0_IRQHandler() { helper(); }\nvoid setup() {}\n#endif\n')})
    report = _detector(tmp_path, ['main', 'setup']).analyze()
    assert not {'UART0_IRQHandler()', 'setup()', 'K::~K()'} & set(report.entry_points_used)
    found = _functions(report)
    assert found['UART0_IRQHandler()'].category == 'deactivated_code'
    assert found['helper()'].category == 'dead_code'            # only reachable from #if 0 code
    assert any("'setup'" in w and '#if 0' in w for w in report.warnings)


_REFERENCED = '''typedef void (*Fn)();
struct Ops { Fn run; };
void cmdReset() {}
void cmdStatus() {}
void doRun() {}
void cmdA() {}
void orphanTarget() {}
class C { public: void tick() {} };
template <typename T> T clampv(T v) { return v; }
void logImpl(int) {}
#define LOG(x) logImpl(x)
static const Fn table[] = {cmdReset, cmdStatus};
static const Ops ops = { doRun };
extern "C" void api_init() {}
extern "C" { void api_stop() {} }
struct P { int v; bool operator==(const P& o) const { return v == o.v; } };
int main() {
    Fn f = cmdA;
    auto m = &C::tick;
    clampv<int>(3);
    LOG(1);
    return 0;
}
'''


def test_functions_referenced_by_name_outside_the_call_graph_are_investigated(tmp_path):
    _source_tree(tmp_path, {'a.cpp': _REFERENCED})
    found = _functions(_detector(tmp_path, ['main']).analyze())
    for name in ('cmdReset()', 'cmdStatus()', 'doRun()', 'cmdA()', 'C::tick()', 'clampv(T)', 'logImpl(int)'):
        item = found[name]
        assert (item.category, item.do178c_disposition, item.confidence) == ('unused_export', 'Investigate', 'low'), name
        assert item.reason == ('referenced by name outside the call graph '
                               '(function pointer, table, macro or template use)')
        assert 1 <= len(item.evidence) <= 3 and all(e.startswith('a.cpp:') for e in item.evidence)
    assert found['logImpl(int)'].evidence == ['a.cpp:11']       # the #define line
    assert found['cmdReset()'].evidence == ['a.cpp:12']
    assert found['orphanTarget()'].category == 'dead_code' and found['orphanTarget()'].confidence == 'high'


def test_extern_c_and_operator_functions_are_investigated_with_their_own_reasons(tmp_path):
    _source_tree(tmp_path, {'a.cpp': _REFERENCED})
    found = _functions(_detector(tmp_path, ['main']).analyze())
    for name in ('api_init()', 'api_stop()'):
        assert (found[name].category, found[name].do178c_disposition, found[name].confidence) == (
            'unused_export', 'Investigate', 'low')
        assert 'extern "C"' in found[name].reason
    op = found['P::operator==(const P &)']
    assert (op.category, op.do178c_disposition, op.confidence) == ('unused_export', 'Investigate', 'low')
    assert 'operator' in op.reason


def test_a_function_only_called_by_dead_functions_is_still_dead_code(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'void leaf() {}\nvoid mid() { leaf(); }\nint main() { return 0; }\n'})
    found = _functions(_detector(tmp_path, ['main']).analyze())
    assert {n: (i.category, i.confidence) for n, i in found.items()} == {
        'leaf()': ('dead_code', 'high'), 'mid()': ('dead_code', 'high')}


def test_file_scope_objects_are_instantiated_so_their_constructors_run_and_dispatch_sees_them(tmp_path):
    _source_tree(tmp_path, {'a.cpp': (
        'class Base { public: virtual ~Base() {} virtual void run() {} };\n'
        'class Mgr : public Base {\npublic:\n  Mgr() { init(); }\n  ~Mgr() {}\n  void run() override { step(); }\n'
        '  void init() {}\n  void step() {}\n};\n'
        'static Mgr g_mgr;\n'
        'namespace hw { class Dev { public: Dev() {} ~Dev() {} }; Dev g_dev; Dev g_arr[2]; }\n'
        'class Ptr { public: Ptr() {} };\nPtr* g_ptr = nullptr;\nPtr& g_ref = *g_ptr;\n'
        'int main() { Base* b = &g_mgr; b->run(); return 0; }\n')})
    gb, _ = _build(tmp_path)
    assert gb.instantiated_classes == {'Mgr', 'hw::Dev'}
    assert gb.graph.has_edge('Base::run()', 'Mgr::run()') or gb.graph.has_edge('main()', 'Mgr::run()')
    found = _functions(_detector(tmp_path, ['main']).analyze())
    assert set(found) == {'Ptr::Ptr()'}                        # a pointer or reference does not construct anything
    assert found['Ptr::Ptr()'].category == 'dead_code'


def test_every_finding_has_evidence_and_removal_impact():
    for item in _sensors().items:
        assert item.evidence and item.removal_impact, item.name


def test_fragment_evidence_and_removal_impact():
    frags = {i.function_name: i for i in _by_category(_sensors(), 'dead_fragment').values()}
    after_return = frags['computeChecksum(int)']
    assert after_return.evidence == ['follows unconditional `return` at line 29']
    assert after_return.removal_impact == ['no behaviour change expected: statements are unreachable']
    if_zero = frags['main()']
    assert if_zero.evidence == ['condition `0` is constant false']
    assert if_zero.removal_impact == ['contains calls to legacyCalibrate (never executed); '
                                      'removing the fragment removes these call sites']


def test_unused_local_removal_impact_names_initializer_side_effects(tmp_path):
    _source_tree(tmp_path, {'a.cpp': (
        'volatile int status_reg;\nint compute();\n'
        'void f(int* p) {\n  int plain = 1;\n  int r = compute();\n  int v = *p;\n  int s = status_reg;\n'
        '  int q = p->x;\n}\n')})
    items = _by_category(_detector(tmp_path, ['f']).analyze(), 'unused_variable')
    assert items['plain'].evidence == ['declared at line 4; no later reference']
    assert items['plain'].removal_impact == ['none']
    prefix, suffix = 'initializer has side effects (', ') — keep the expression when removing the variable'
    assert items['r'].removal_impact == [prefix + 'compute()' + suffix]
    assert items['v'].removal_impact == [prefix + '*p' + suffix]
    assert items['s'].removal_impact == [prefix + 'volatile access to status_reg' + suffix]
    assert items['q'].removal_impact == [prefix + 'p->x' + suffix]


def test_unused_parameter_lists_the_callers_a_signature_change_would_affect(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'void cb(int a, int b) { b++; }\nint main() { cb(1, 2); return 0; }\n'
                                     'void lone(int z) {}\n'})
    items = _by_category(_detector(tmp_path, ['main']).analyze(), 'unused_variable')
    assert items['a'].evidence == ['declared at line 1; no reference in the function']
    assert items['a'].removal_impact == ['signature change affects callers: main()']
    assert items['z'].removal_impact == ['signature change affects callers: none known']


def test_unresolved_calls_are_listed_in_the_report_and_json(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.cpp').write_text('struct A { void f() {} };\nstruct B { void f() {} };\n'
                               'void use(X* x) {\n  x->f();\n}\nint main() { return 0; }\n')
    report = _detector(src, ['main']).analyze()
    path = str(src / 'a.cpp')
    assert report.unresolved_calls == [f"use(X *) → A::f() ({path}:4)", f"use(X *) → B::f() ({path}:4)"]
    gen = ReportGenerator(Config(output_dir=str(tmp_path / 'out')))
    text = Path(gen.generate_dead_code(report, format='markdown')).read_text(encoding='utf-8')
    assert '## Unresolved calls (conservative edges)' in text and f"use(X *) → A::f() ({path}:4)" in text
    data = json.loads((tmp_path / 'out' / 'dead_code_findings.json').read_text(encoding='utf-8'))
    assert data['unresolved_calls'] == report.unresolved_calls
    assert not _sensors().unresolved_calls


def test_entry_point_patterns_apply_to_free_functions_unless_they_contain_a_scope(tmp_path):
    _source_tree(tmp_path, {'a.cpp': ('class Sched { public: void addTask() { helper(); } };\nvoid helper() {}\n'
                                      'void BlinkTask() { other(); }\nvoid other() {}\nint main() { return 0; }\n')})
    report = _detector(tmp_path, ['main']).analyze()
    assert 'BlinkTask()' in report.entry_points_used and 'Sched::addTask()' not in report.entry_points_used
    assert set(_functions(report)) == {'Sched::addTask()', 'helper()'}
    report = _detector(tmp_path, ['main'], entry_point_patterns=['Sched::*']).analyze()
    assert 'Sched::addTask()' in report.entry_points_used and 'BlinkTask()' not in report.entry_points_used


def test_a_bare_entry_point_name_ignores_the_parameter_list(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'void helper() {}\nint main(int argc, char** argv) { helper(); return 0; }\n'})
    report = _detector(tmp_path, ['main']).analyze()
    assert report.entry_points_used == ['main(int, char * *)']
    assert not any('not found' in w for w in report.warnings)
    assert 'helper()' not in _functions(report)


# ── Fix round 2: probes ────────────────────────────────────────────────────────

def test_initializers_that_write_state_are_side_effects(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'int cnt;\nvoid f() {\n  int old = cnt++;\n  int a = (cnt = 3);\n  int c = cnt += 2;\n}\n'})
    items = _by_category(_detector(tmp_path, ['f']).analyze(), 'unused_variable')
    suffix = ') — keep the expression when removing the variable'
    assert items['old'].removal_impact == ['initializer has side effects (cnt++' + suffix]
    assert items['a'].removal_impact == ['initializer has side effects (cnt = 3' + suffix]
    assert items['c'].removal_impact == ['initializer has side effects (cnt += 2' + suffix]


_MEMBERS = '''class Dev {
public:
    int config() { return 1; }
    bool ready() { return true; }
    int state() { return 2; }
    int count() { return 3; }
    int idle() { return 4; }
    void tick() {}
    static void make() {}
    static void make2();
};
void Dev::make2() {}
struct Config {};
struct Stats { int total, count; };
enum State { idle, busy };
void setup(const Config& config) {}
void setup2(Config&& config) {}
bool check(bool ready) { if (ready) return ready; return false; }
int use(int state) { return state; }
typedef void (*Fn)();
Fn keep = Dev::make2;
auto m = &Dev::tick;
int main() { return 0; }
'''


def test_names_that_only_look_like_a_member_function_do_not_hide_it(tmp_path):
    _source_tree(tmp_path, {'a.cpp': _MEMBERS})
    found = _functions(_detector(tmp_path, ['main']).analyze())
    # reference parameters, locals, fields and enumerators of the same name: a non-static member function
    # can only be named as &C::f, f<T> or inside a #define
    for name in ('Dev::config()', 'Dev::ready()', 'Dev::state()', 'Dev::count()', 'Dev::idle()', 'Dev::make()'):
        assert (found[name].category, found[name].do178c_disposition, found[name].confidence) == (
            'dead_code', 'Remove', 'high'), name
    assert found['Dev::tick()'].category == 'unused_export'        # &Dev::tick
    assert found['Dev::make2()'].category == 'unused_export'       # static, out-of-line definition: Dev::make2 is a value


_FREE = '''int count() { return 3; }
int level() { return 1; }
int bits() { return 2; }
struct Stats { int total, count; };
enum class State { idle, level };
void k(int a) { int x, bits; }
int m(int a) { return a & bits; }
int main() { return 0; }
'''


def test_declarations_and_binary_operators_do_not_hide_free_functions(tmp_path):
    _source_tree(tmp_path, {'a.cpp': _FREE})
    found = _functions(_detector(tmp_path, ['main']).analyze())
    for name in ('count()', 'level()', 'bits()'):
        assert (found[name].category, found[name].confidence) == ('dead_code', 'high'), name


def test_file_scope_objects_and_references_under_if_zero_do_not_count(tmp_path):
    _source_tree(tmp_path, {'a.cpp': (
        'class Mgr { public: Mgr() {} };\nclass Dev { public: Dev() {} };\nclass Aux { public: Aux() {} };\n'
        'typedef void (*Fn)();\nvoid cmdA() {}\nvoid cmdB() {}\n'
        '#if 0\nstatic Mgr g_mgr;\nstatic Fn table[] = {cmdA};\nvoid f() { Aux a; }\n#else\nstatic Dev g_dev;\n'
        'static Fn live[] = {cmdB};\n#endif\n'
        'int main() { return 0; }\n')})
    gb, _ = _build(tmp_path)
    assert gb.instantiated_classes == {'Dev'}
    found = _functions(_detector(tmp_path, ['main']).analyze())
    assert found['Mgr::Mgr()'].category == 'dead_code' and found['Aux::Aux()'].category == 'dead_code'
    assert 'Dev::Dev()' not in found
    assert found['cmdA()'].category == 'dead_code'
    assert found['cmdB()'].category == 'unused_export'              # the #else branch of #if 0 is live
    assert found['f()'].category == 'deactivated_code'


def test_references_are_indexed_once_not_once_per_function(tmp_path, monkeypatch):
    calls = []
    real = DeadCodeDetector._build_reference_index
    monkeypatch.setattr(DeadCodeDetector, '_build_reference_index', lambda self: calls.append(1) or real(self))
    (tmp_path / 'a.cpp').write_text('int main() { return 0; }\n' + ''.join(f'void dead{i}() {{}}\n' for i in range(500)))
    (tmp_path / 'b.cpp').write_text('static void (*table[])() = {dead7, dead9};\n'
                                    + ''.join(f'int filler{i} = {i};\n' for i in range(20000)))
    found = _functions(_detector(tmp_path, ['main']).analyze())
    assert len(found) == 500 and found['dead7()'].category == 'unused_export' and found['dead8()'].category == 'dead_code'
    assert len(calls) == 1


def test_evidence_paths_tell_same_named_files_apart(tmp_path):
    for sub in ('sub', 'other'):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / 'a.cpp').write_text('static void (*tbl[])() = {cmdX};\n')
    (tmp_path / 'main.cpp').write_text('void cmdX() {}\nint main() { return 0; }\n')
    item = _functions(_detector(tmp_path, ['main']).analyze())['cmdX()']
    assert item.evidence == ['other/a.cpp:1', 'sub/a.cpp:1']


def test_the_warning_for_an_entry_point_under_if_false_names_the_guard(tmp_path):
    _source_tree(tmp_path, {'a.cpp': 'int main() { return 0; }\n#if false\nvoid setup() {}\n#endif\n'})
    warnings = _detector(tmp_path, ['main', 'setup']).analyze().warnings
    assert any("'setup'" in w and '`#if false`' in w for w in warnings)


def test_address_of_after_a_keyword_or_a_cast_is_a_reference(tmp_path):
    _source_tree(tmp_path, {'a.cpp': (
        'void helperB() {}\nvoid cmdA() {}\nvoid cmdB() { helperB(); }\nvoid cmdC() {}\nvoid cmdD() {}\n'
        'void* pick(int n) {\n  switch (n) {\n    case 1: return &cmdB;\n    case 2: throw &cmdD;\n'
        '    default: return (&cmdC);\n  }\n}\n'
        'void* vec[] = { (void*)&cmdA };\nint main() { return 0; }\n')})
    found = _functions(_detector(tmp_path, ['main']).analyze())
    for name in ('cmdA()', 'cmdB()', 'cmdC()', 'cmdD()'):
        assert (found[name].category, found[name].confidence) == ('unused_export', 'low'), name
    assert found['helperB()'].category == 'unused_export'            # called by a possibly-live function only
    assert (found['pick(int)'].category, found['pick(int)'].confidence) == ('dead_code', 'high')


def test_callbacks_nested_in_control_statements_keep_their_edges(tmp_path):
    _source_tree(tmp_path, {'a.cpp': (
        'typedef void (*Fn)();\nbool registerCb(Fn cb) { return cb != 0; }\n'
        'void helperF() {}\nvoid processFrame() { helperF(); }\nvoid helperO() {}\nvoid processOther() { helperO(); }\n'
        'int main() {\n  if (registerCb(processFrame)) {}\n  while (!registerCb(processOther)) {}\n  return 0;\n}\n')})
    gb, _ = _build(tmp_path)
    assert gb.graph.edges['main()', 'processFrame()']['call_type'] == 'callback'
    assert gb.graph.edges['main()', 'processOther()']['call_type'] == 'callback'
    assert not _functions(_detector(tmp_path, ['main']).analyze())


def test_functions_reachable_only_from_a_possibly_live_function_are_investigated_not_removed(tmp_path):
    _source_tree(tmp_path, {'a.cpp': (
        'typedef void (*Fn)();\nvoid helper2() {}\nvoid helper() { helper2(); }\nvoid cmdReset() { helper(); }\n'
        'void cmdStatus() { helper(); }\nstatic const Fn table[] = {cmdReset, cmdStatus};\n'
        'class Alt { public: virtual void run() { calibrate(); } };\nvoid calibrate() {}\n'
        'void leaf() {}\nvoid mid() { leaf(); }\nint main() { return 0; }\n')})
    found = _functions(_detector(tmp_path, ['main']).analyze())
    helper = found['helper()']
    assert (helper.category, helper.do178c_disposition, helper.confidence) == ('unused_export', 'Investigate', 'low')
    assert helper.reason == 'only reachable from possibly-live cmdReset(), cmdStatus()'
    assert helper.evidence == ['cmdReset() → helper()']
    assert found['helper2()'].evidence == ['cmdReset() → helper() → helper2()']
    assert found['calibrate()'].reason == 'only reachable from possibly-live Alt::run()'
    # reachable only from genuinely dead code: still removable
    assert (found['mid()'].category, found['mid()'].confidence) == ('dead_code', 'high')
    assert (found['leaf()'].category, found['leaf()'].confidence) == ('dead_code', 'high')


def test_possibly_live_names_are_capped_at_three(tmp_path):
    _source_tree(tmp_path, {'a.cpp': (
        'typedef void (*Fn)();\nvoid shared() {}\n' + ''.join(f'void cmd{c}() {{ shared(); }}\n' for c in 'ABCDE')
        + 'static const Fn table[] = {cmdA, cmdB, cmdC, cmdD, cmdE};\nint main() { return 0; }\n')})
    reason = _functions(_detector(tmp_path, ['main']).analyze())['shared()'].reason
    assert reason == 'only reachable from possibly-live cmdA(), cmdB(), cmdC()'


def test_reference_lookups_do_not_scan_every_function_per_reference(tmp_path):
    n = 1500
    (tmp_path / 'a.cpp').write_text('int main() { return 0; }\n' + ''.join(f'void dead{i}(int p{i}) {{ int l{i} = p{i}; }}\n' for i in range(n)))
    (tmp_path / 'b.cpp').write_text('static void (*table[])(int) = {' + ', '.join(f'dead{i}' for i in range(n)) + '};\n')
    results = CodeParser().parse_directory(str(tmp_path))
    gb = CallGraphBuilder(results)
    gb.build()

    class CountingList(list):
        scans = 0

        def __iter__(self):
            CountingList.scans += 1
            return super().__iter__()

    for result in results.values():
        result.functions = CountingList(result.functions)
    found = _functions(DeadCodeDetector(gb, results, ['main']).analyze())
    assert len(found) == n and {i.category for i in found.values()} == {'unused_export'}
    assert CountingList.scans < 30                                 # a few passes per file, not one per reference


def test_keywords_and_locals_are_not_callbacks(tmp_path):
    _source_tree(tmp_path, {'a.cpp': (
        'int status() { return 1; }\nint limit() { return 2; }\nvoid use(int) {}\nvoid onTimerTick() {}\n'
        'void registerCallback(void (*cb)()) { cb(); }\n'
        'void f(int status) { if (status) use(status); int limit = 3; while (limit < 10) use(limit); }\n'
        'int main() { registerCallback(onTimerTick); f(1); return 0; }\n')})
    found = _functions(_detector(tmp_path, ['main']).analyze())
    assert {n: (i.category, i.confidence) for n, i in found.items()} == {
        'status()': ('dead_code', 'high'), 'limit()': ('dead_code', 'high')}      # onTimerTick has its callback edge
