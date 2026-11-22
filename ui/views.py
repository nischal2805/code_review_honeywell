"""Page renderers. They only read ``AnalysisResults`` produced by the existing backend; no analysis logic lives here."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import yaml

from rag_engine.config import Config, load_config
from rag_engine.features.dead_code_detector import DISPOSITION_STATUSES
from rag_engine.llm.offline_qa import answer_question, results_summary
from rag_engine.llm.ollama_client import OllamaClient
from rag_engine.llm.rag_narrator import RAGNarrativeGenerator
from rag_engine.pipeline import InputError, _load_templates
from ui import components as c
from ui.runner import WORKSPACE, RunState, save_upload, start_run

SEVERITIES = ('CRITICAL', 'MAJOR', 'MEDIUM', 'MINOR')
CHECKS = ('corrupt', 'stale', 'debounce')


# ---------------------------------------------------------------- state helpers
def conf() -> Dict[str, Any]:
    return st.session_state.conf


def run_state() -> RunState:
    return st.session_state.run


def results():
    return run_state().results


def go(page: str) -> None:
    st.session_state.nav = page


def build_config() -> Config:
    """Config from config.yaml plus the UI overrides. Raises ValueError with a readable message."""
    k = conf()
    cfg = load_config(k['config_path'] or 'config.yaml')
    cfg.dal_level = k['dal']
    for attr, key in (('standards_file', 'standards_path'), ('sensor_templates_file', 'sensor_path'),
                      ('dead_code_dispositions', 'disp_path')):
        if k[key]:
            setattr(cfg, attr, str(Path(k[key]).expanduser().resolve()))
    return cfg


def _path_input(label: str, key: str, help: str = '', must_be_dir: bool = False) -> None:
    k = conf()
    k[key] = st.text_input(label, value=k[key], help=help)
    if k[key]:
        p = Path(k[key]).expanduser()
        if not p.exists():
            st.caption(f':red[Path not found: {k[key]}]')
        elif must_be_dir and not p.is_dir():
            st.caption(f':red[Not a directory: {k[key]}]')


def _upload(label: str, types: List[str], key: str, target: str) -> None:
    """Save a newly uploaded file under the workspace and point conf[target] at it (once per upload)."""
    up = st.file_uploader(label, type=types, key=key)
    if up is not None and st.session_state.uploaded.get(key) != up.file_id:
        st.session_state.uploaded[key] = up.file_id
        conf()[target] = save_upload(up)


# ---------------------------------------------------------------- dashboard
def dashboard() -> None:
    r, rs = results(), run_state()
    if r is None:
        c.empty_state('No analysis has been run yet',
                      'Set the source directory under Project Configuration, then start the analysis.')
        st.write('')
        st.button('Open Project Configuration', on_click=go, args=('Project Configuration',))
        _history()
        return
    funcs = sum(len(p.functions) for p in r.parse_results.values())
    cards: List[tuple] = [('Files analyzed', len(r.parse_results)), ('Functions', funcs)]
    if r.virtual:
        s = r.virtual.summary
        cards.append(('Virtual changes', s.get('added', 0) + s.get('removed', 0) + s.get('modified', 0),
                      f"{r.virtual.current_virtual_count} virtual functions"))
    if r.dead:
        cards.append(('Dead code', r.dead.dead_count, f"{r.dead.deactivated_count} deactivated",
                      'err' if r.dead.dead_count else 'ok'))
    c.metric_row(cards[:4])
    cards = []
    if r.standards:
        v = r.standards.violations_by_severity
        cards.append(('Compliance score', f"{r.standards.compliance_score:.1f}%",
                      f"{len(r.standards.violations)} violations",
                      'err' if v.get('CRITICAL') else 'warn' if v.get('MAJOR') else 'ok'))
    if r.coupling:
        risks = [x.risk_level for x in r.coupling.lru_impacts.values()]
        cards.append(('Coupling risk', 'high' if 'high' in risks else 'medium' if 'medium' in risks else 'low',
                      f"{len(risks)} LRUs", 'err' if 'high' in risks else ''))
    if r.checklist:
        done = sum(i.status == 'PASS' for i in r.checklist.items)
        cards.append(('Checklist PASS', f"{done}/{len(r.checklist.items)}", '', ''))
    cards.append(_sensor_card(r))
    c.metric_row(cards)

    left, right = st.columns([3, 2])
    with left:
        c.section('Important findings')
        _key_findings(r)
    with right:
        c.section('Project')
        k = conf()
        st.markdown(f"**Source:** `{k['codebase']}`  \n**Status:** {c.badge(rs.status)}  \n"
                    f"**DAL:** {rs.cfg.dal_level if rs.cfg else k['dal']}", unsafe_allow_html=True)
        st.write('')
        for label, page in (('Standards', 'Standards & Compliance'), ('Sensors', 'Sensor Analysis'),
                            ('AI Assistant', 'AI Assistant'), ('Reports', 'Reports')):
            st.button(label, on_click=go, args=(page,), key=f'q_{page}')
    _history()


def _sensor_card(r) -> tuple:
    if r.sensors is None:
        return ('Sensors', 'n/a', 'no sensor YAML configured')
    found = [f for f in r.sensors.findings if f.status == 'found']
    bad = [f for f in found if any(ch.status in ('missing', 'mismatch') for ch in f.checks.values())]
    return ('Sensors mapped', f"{len(found)}/{len(r.sensors.findings)}", f"{len(bad)} with check gaps",
            'warn' if bad or len(found) < len(r.sensors.findings) else 'ok')


def _key_findings(r) -> None:
    lines: List[str] = []
    if r.standards:
        for sev in ('CRITICAL', 'MAJOR'):
            n = r.standards.violations_by_severity.get(sev, 0)
            if n:
                lines.append(f"{c.badge(sev)} {n} {sev.lower()} standards violation(s)")
    if r.dead and r.dead.items:
        open_n = sum(1 for i in r.dead.items if i.review_status == 'open')
        lines.append(f"{c.badge('open')} {open_n} dead-code finding(s) awaiting disposition")
    if r.sensors is not None:
        for f in r.sensors.findings:
            if f.status == 'not_found':
                lines.append(f"{c.badge('not_found')} sensor {f.template.name}: no matching C++ class")
        if r.sensors.unmapped_sensor_classes:
            lines.append(f"{c.badge('warning')} {len(r.sensors.unmapped_sensor_classes)} sensor-like class(es) "
                         "not covered by the YAML")
    if r.virtual:
        mod = r.virtual.summary.get('modified', 0) + r.virtual.summary.get('removed', 0)
        if mod:
            lines.append(f"{c.badge('modified')} {mod} virtual function(s) modified or removed (re-verification)")
    if lines:
        st.markdown('<br>'.join(lines), unsafe_allow_html=True)
    else:
        st.success('No critical findings were reported by the analysis.')


def _history() -> None:
    hist = st.session_state.history
    if hist:
        c.section('Recent analysis activity')
        c.data_table(list(reversed(hist))[:10])


# ---------------------------------------------------------------- configuration
def configuration() -> None:
    k = conf()
    c.section('Source code')
    a, b = st.columns(2)
    with a:
        _path_input('C++ source directory (current build)', 'codebase', must_be_dir=True)
        _path_input('Base build directory (optional, for virtual-function diff)', 'base_build', must_be_dir=True)
    with b:
        _path_input('LRU documents directory (optional PDF/DOCX)', 'lru_docs', must_be_dir=True)
        _path_input('Project config file', 'config_path', 'config.yaml: DAL, entry points, thresholds, Ollama, ...')

    c.section('Analysis settings')
    a, b, d = st.columns(3)
    k['dal'] = a.selectbox('DAL level', ['A', 'B', 'C', 'D'], index='ABCD'.index(k['dal']))
    k['out_fmt'] = b.selectbox('Report format', ['markdown', 'docx'], index=['markdown', 'docx'].index(k['out_fmt']))
    k['semantic'] = d.checkbox('Semantic search (needs sentence-transformers)', value=k['semantic'])
    _upload('Upload a standards file (optional)', ['yaml', 'yml', 'json', 'pdf', 'docx', 'txt'], 'up_std', 'standards_path')
    _path_input('Coding standards file path (YAML/JSON/PDF/DOCX/TXT, optional)', 'standards_path')
    _path_input('Dead-code dispositions file (optional YAML)', 'disp_path',
                'Where reviewer decisions are read from and saved to.')

    c.section('Sensor configuration (YAML)')
    _upload('Upload a sensor YAML', ['yaml', 'yml'], 'up_sensor', 'sensor_path')
    _path_input('Sensor templates file path', 'sensor_path', 'Enables the sensor analysis stage of the main run.')
    fallback = '' if k['sensor_path'] else _from_config_yaml('sensor_templates_file')
    if fallback:
        st.caption(f'Using `sensor_templates_file` from the config file: `{fallback}`')
    _sensor_yaml_view(k['sensor_path'] or fallback)

    c.section('Validate configuration')
    if st.button('Validate configuration', type='primary'):
        try:
            cfg = build_config()
            st.success(f"Config OK — DAL {cfg.dal_level}, output: {cfg.output_dir}")
        except ValueError as exc:
            st.error(str(exc))
        if not k['codebase'] or not Path(k['codebase']).expanduser().is_dir():
            st.warning('The C++ source directory is missing or not a directory.')


def _from_config_yaml(attr: str) -> str:
    """*attr* as the selected config file sets it ('' when unset or the file is invalid)."""
    try:
        return getattr(load_config(conf()['config_path'] or 'config.yaml'), attr) or ''
    except ValueError:
        return ''


def _sensor_yaml_view(path: str) -> None:
    if not path:
        st.caption('No sensor YAML selected: sensor analysis will be skipped.')
        return
    try:
        templates = _load_templates(str(Path(path).expanduser()))
    except InputError as exc:
        st.error(f'Invalid sensor configuration: {exc}')
        return
    st.success(f'Sensor YAML valid — {len(templates)} sensor(s) in `{Path(path).name}`')
    rows = [{'Sensor': t.name, 'Type': t.type, 'Unit': t.unit, 'C++ class': t.class_name or t.name,
             'Range': _range(t), 'Stale ms': t.stale_threshold_ms, 'Frozen detect': t.stale_detect_frozen,
             'Debounce': (f'on, thr {t.debounce_threshold}, n={t.debounce_count}' if t.debounce_enabled else 'off'),
             'Range check': 'on' if t.corrupt_range_check else 'off'} for t in templates]
    c.data_table(rows)
    with st.expander('Raw YAML'):
        st.code(Path(path).expanduser().read_text(encoding='utf-8', errors='replace'), language='yaml')


def _range(t) -> str:
    return '—' if t.valid_min is None and t.valid_max is None else f'[{t.valid_min}, {t.valid_max}]'


# ---------------------------------------------------------------- execution
def execution() -> None:
    rs = run_state()
    k = conf()
    cols = st.columns([1, 3])
    codebase_ok = bool(k['codebase']) and Path(k['codebase']).expanduser().is_dir()
    if cols[0].button('Start analysis', type='primary', disabled=rs.status == 'running' or not codebase_ok):
        try:
            cfg = build_config()
            if k['sensor_path']:
                _load_templates(str(Path(k['sensor_path']).expanduser()))   # same check run_pipeline makes
        except (ValueError, InputError) as exc:
            st.error(str(exc))
        else:
            start_run(rs, cfg, str(Path(k['codebase']).expanduser()),
                      str(Path(k['base_build']).expanduser()) if k['base_build'] else None,
                      str(Path(k['lru_docs']).expanduser()) if k['lru_docs'] else None,
                      k['out_fmt'], k['semantic'])
            st.session_state.recorded = False
            st.rerun()
    if not codebase_ok:
        cols[1].warning('Set a valid C++ source directory under Project Configuration first.')
    st.fragment(_progress, run_every=1.0 if rs.status == 'running' else None)()


def _progress() -> None:
    rs = run_state()
    if rs.status == 'idle':
        c.empty_state('Analysis has not been started')
        return
    with rs.lock:
        status, step, pct, logs, err = rs.status, rs.step, rs.percent, list(rs.logs), rs.error
    st.markdown(f'**Status:** {c.badge(status)} &nbsp; **Stage:** {step}', unsafe_allow_html=True)
    st.progress(min(max(pct, 0), 100) / 100)
    c.log_console(logs)
    if status == 'failed':
        st.error(err or 'Analysis failed')
    if status in ('completed', 'failed') and not st.session_state.get('recorded'):
        st.session_state.recorded = True
        st.session_state.history.append({
            'Time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rs.finished)), 'Status': status,
            'Source': conf()['codebase'], 'Seconds': f'{rs.finished - rs.started:.1f}'})
        st.rerun()
    if status == 'completed':
        st.success('Analysis completed.')
        _summary(rs.results)


def _summary(r) -> None:
    if r is None:
        return
    st.markdown(f"Files **{len(r.parse_results)}** · reports **{len(r.report_paths)}** · "
                f"sensors **{'n/a' if r.sensors is None else len(r.sensors.findings)}** · "
                f"semantic search **{'on' if r.search is not None else 'off (keyword matching)'}**")


# ---------------------------------------------------------------- virtual functions
def virtual() -> None:
    r = results()
    if not c.need_results(r, 'virtual'):
        return
    v = r.virtual
    s = v.summary
    c.metric_row([('Added', s.get('added', 0)), ('Removed', s.get('removed', 0)), ('Modified', s.get('modified', 0)),
                  ('Unchanged', s.get('unchanged', 0)),
                  ('Virtual functions', v.current_virtual_count, f'base: {v.base_virtual_count}')])
    t_changes, t_inv, t_tree = st.tabs(['Changes', 'Inventory', 'Class hierarchy'])
    with t_changes:
        kinds = st.multiselect('Change type', ['added', 'removed', 'modified', 'unchanged'],
                               default=['added', 'removed', 'modified'])
        rows = [{'Change': ch.change_type, 'Function': ch.function.name, 'File': Path(ch.function.file_path).name,
                 'Line': ch.function.line_number, 'Category': ch.do178c_category, 'Re-verification': ch.reverification_scope}
                for ch in v.changes if ch.change_type in kinds]
        c.data_table(rows, 'No changes of the selected types. (Without a base build every function is unchanged.)')
    with t_inv:
        rows = [{'Function': i.function, 'Declared': f'{Path(i.file_path).name}:{i.line_number}',
                 'Pure': i.is_pure, 'Reachable': i.reachable, 'Overrides': len(i.overridden_by),
                 'Call sites': len(i.call_sites), 'Trace': i.requirement_trace} for i in v.inventory]
        c.data_table(rows, 'No virtual functions found.')
        for i in v.inventory:
            if i.overridden_by or i.call_sites:
                with st.expander(i.function):
                    c.data_table([{'Override': o.function, 'Declared': f'{Path(o.file_path).name}:{o.line_number}',
                                   'Instantiated': o.class_instantiated, 'Reachable': o.reachable,
                                   'Call sites': len(o.call_sites)} for o in i.overridden_by], 'No overrides.')
                    c.evidence('Call sites of the base function',
                               [f'{x.file_path}:{x.line} {x.caller} -> {x.target} ({x.call_type})' for x in i.call_sites])
    with t_tree:
        if not v.hierarchy_trees:
            st.caption('No class has subclasses.')
        for root, tree in v.hierarchy_trees.items():
            with st.expander(root, expanded=True):
                st.code(tree, language=None)


# ---------------------------------------------------------------- dead code
def dead_code() -> None:
    r = results()
    if not c.need_results(r, 'dead'):
        return
    d = r.dead
    c.metric_row([('Functions analyzed', d.total_functions), ('Dead code', d.dead_count, '', 'err' if d.dead_count else 'ok'),
                  ('Deactivated', d.deactivated_count), ('Findings', len(d.items), d.structural_coverage_impact)])
    for w in d.warnings:
        st.warning(w)
    cats = sorted({i.category for i in d.items})
    f1, f2, f3 = st.columns(3)
    cat = f1.multiselect('Category', cats, default=cats)
    stat = f2.multiselect('Review status', list(DISPOSITION_STATUSES), default=list(DISPOSITION_STATUSES))
    conf_f = f3.multiselect('Confidence', ['high', 'medium', 'low'], default=['high', 'medium', 'low'])
    items = [i for i in d.items if i.category in cat and i.review_status in stat and i.confidence in conf_f]
    c.data_table([{'ID': i.finding_id, 'Category': i.category, 'Name': i.name, 'File': Path(i.file_path).name,
                   'Line': i.line_number, 'Disposition': i.do178c_disposition, 'Confidence': i.confidence,
                   'Status': i.review_status} for i in items], 'No findings match the filters.')
    for i in items[:100]:
        with st.expander(f'{i.finding_id} — {i.name}'):
            st.markdown(f"{c.badge(i.review_status)} {c.badge(i.confidence)} · `{i.file_path}:{i.line_number}`",
                        unsafe_allow_html=True)
            st.write(i.reason or 'No reason recorded.')
            c.evidence('Evidence', i.evidence)
            c.evidence('Removal impact', i.removal_impact)
            if i.reviewer_justification:
                st.info(f'Reviewer {i.reviewer or "—"}: {i.reviewer_justification}')
    if len(items) > 100:
        st.caption(f'Showing the first 100 of {len(items)} expandable details; the table lists all.')
    if d.false_positive_notes:
        with st.expander('False-positive notes'):
            for n in d.false_positive_notes:
                st.write(f'- {n}')
    if d.unresolved_calls:
        c.evidence('Unresolved (conservative) call edges', d.unresolved_calls)
    _disposition_panel(d)


def _disposition_panel(d) -> None:
    c.section('Disposition review')
    open_items = d.items
    if not open_items:
        st.caption('No findings to review.')
        return
    rs = run_state()
    path = (rs.cfg.dead_code_dispositions if rs.cfg else '') or str(WORKSPACE / 'dead_code_dispositions.yaml')
    st.caption(f'Decisions are written to `{path}` in the existing `findings:` format. '
               'Re-run the analysis to apply them to the findings, checklist and reports.')
    with st.form('disposition'):
        fid = st.selectbox('Finding', [i.finding_id for i in open_items],
                           format_func=lambda x: f"{x} — {next(i.name for i in open_items if i.finding_id == x)}")
        status = st.selectbox('Status', list(DISPOSITION_STATUSES))
        why = st.text_area('Justification (required)')
        who = st.text_input('Reviewer')
        if st.form_submit_button('Save disposition', type='primary'):
            if not why.strip():
                st.error('A justification is required; the backend ignores entries without one.')
            else:
                _save_disposition(path, fid, status, why.strip(), who.strip())
                st.session_state.conf['disp_path'] = st.session_state.conf['disp_path'] or path
                st.success(f'Saved {fid} as {status}. Re-run the analysis to apply it.')


def _save_disposition(path: str, fid: str, status: str, why: str, who: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(p.read_text(encoding='utf-8')) if p.exists() else {}
    data = data if isinstance(data, dict) else {}
    findings = data.setdefault('findings', {})
    findings[fid] = {'status': status, 'justification': why, **({'reviewer': who} if who else {})}
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


# ---------------------------------------------------------------- standards
def standards() -> None:
    r = results()
    if not c.need_results(r, 'standards'):
        return
    s = r.standards
    sev = s.violations_by_severity
    c.metric_row([('Compliance score', f'{s.compliance_score:.1f}%', f'{s.total_functions_checked} functions checked'),
                  ('Violations', len(s.violations)),
                  ('Critical', sev.get('CRITICAL', 0), '', 'err' if sev.get('CRITICAL') else 'ok'),
                  ('Major', sev.get('MAJOR', 0), '', 'warn' if sev.get('MAJOR') else 'ok'),
                  ('Medium / Minor', f"{sev.get('MEDIUM', 0)} / {sev.get('MINOR', 0)}")])
    st.caption(f'Standards source: {s.standards_source}. This is a static-analysis result, not a certification verdict.')
    chosen = st.multiselect('Severity', SEVERITIES, default=list(SEVERITIES))
    vs = [v for v in s.violations if v.severity in chosen]
    c.data_table([{'Severity': v.severity, 'Rule': v.rule, 'File': Path(v.file).name,
                   'Function line': v.function_line, 'Line': v.line, 'Element': v.element, 'Message': v.message,
                   'MISRA': v.misra_ref} for v in vs], 'No violations match.')
    for v in [x for x in vs if x.severity in ('CRITICAL', 'MAJOR')][:50]:
        with st.expander(f'{v.severity} · {v.rule} · {v.element}'):
            st.markdown(f"{c.badge(v.severity)} `{v.file}:{v.line}`", unsafe_allow_html=True)
            st.write(v.message)
            fix = s.corrections.get(v.rule)
            if fix:
                st.info(f'Suggested correction: {fix}')


# ---------------------------------------------------------------- coupling
def coupling() -> None:
    r = results()
    if not c.need_results(r, 'coupling'):
        return
    cp = r.coupling
    rows = [{'LRU': n, 'Risk': x.risk_level, 'Control': len(x.control_coupling), 'Data': len(x.data_coupling),
             'Shared globals': len(x.shared_globals), 'Timing': len(x.timing_dependencies)}
            for n, x in cp.lru_impacts.items()]
    high = sum(1 for x in cp.lru_impacts.values() if x.risk_level == 'high')
    c.metric_row([('LRUs', len(rows)), ('High risk', high, '', 'err' if high else 'ok'),
                  ('Control-coupled functions', len(cp.control_coupling_map)),
                  ('Data-coupled functions', len(cp.data_coupling_map)), ('Global variables', len(cp.global_variables))])
    c.data_table(rows, 'No LRU coupling found.')
    if cp.lru_impacts:
        name = st.selectbox('LRU detail', list(cp.lru_impacts))
        x = cp.lru_impacts[name]
        st.markdown(f'Risk: {c.badge(x.risk_level)}', unsafe_allow_html=True)
        c.evidence('Control coupling', x.control_coupling, True)
        c.evidence('Data coupling', x.data_coupling)
        c.evidence('Shared globals', x.shared_globals)
        c.evidence('Timing dependencies', x.timing_dependencies)
    with st.expander('Coupling maps (function → coupled items)'):
        t1, t2 = st.tabs(['Control', 'Data'])
        t1.json(cp.control_coupling_map)
        t2.json(cp.data_coupling_map)
    c.evidence('Global variables', cp.global_variables)


# ---------------------------------------------------------------- checklist
def checklist() -> None:
    r = results()
    if not c.need_results(r, 'checklist'):
        return
    items = r.checklist.items
    counts = {s: sum(i.status == s for i in items) for s in ('PASS', 'FAIL', 'OPEN', 'N/A')}
    c.metric_row([('PASS', counts['PASS'], '', 'ok'), ('FAIL', counts['FAIL'], '', 'err' if counts['FAIL'] else 'ok'),
                  ('OPEN', counts['OPEN'], '', 'warn' if counts['OPEN'] else ''), ('N/A', counts['N/A'])])
    st.caption(f'Generated {r.checklist.generated_date} for DAL {r.checklist.dal_level}.')
    chosen = st.multiselect('Status', list(counts), default=list(counts))
    shown = [i for i in items if i.status in chosen]
    c.data_table([{'Category': i.category, 'ID': i.item_id, 'Status': i.status, 'Description': i.description}
                  for i in shown])
    for i in shown:
        if i.evidence:
            with st.expander(f'{i.item_id} — {i.status}'):
                st.write(i.description)
                st.code(i.evidence, language=None)


# ---------------------------------------------------------------- sensors
def sensors() -> None:
    r = results()
    if r is None:
        c.need_results(r)
        return
    if r.sensors is None:
        c.empty_state('Sensor analysis was not run',
                      'Select a sensor YAML under Project Configuration and re-run the analysis.')
        _sensor_yaml_view(conf()['sensor_path'])
        return
    fs = r.sensors.findings
    found = [f for f in fs if f.status == 'found']
    gaps = [f for f in found if any(ch.status in ('missing', 'mismatch') for ch in f.checks.values())]
    c.metric_row([('Configured', len(fs)), ('Mapped to C++', len(found), '', 'ok' if len(found) == len(fs) else 'warn'),
                  ('Unmapped', len(fs) - len(found), '', 'err' if len(fs) > len(found) else 'ok'),
                  ('Check gaps', len(gaps), 'missing or mismatch', 'warn' if gaps else 'ok'),
                  ('Extra sensor classes', len(r.sensors.unmapped_sensor_classes), 'not in the YAML')])
    t_inv, t_det, t_flow, t_cfg = st.tabs(['Inventory', 'Sensor details', 'Data flow', 'YAML configuration'])
    with t_inv:
        c.data_table([_sensor_row(f) for f in fs], 'No sensors configured.')
        if r.sensors.unmapped_sensor_classes:
            c.evidence('Sensor-like classes not covered by the YAML', r.sensors.unmapped_sensor_classes)
    with t_det:
        if not fs:
            st.caption('No sensors configured.')
        else:
            f = next(x for x in fs if x.template.name == st.selectbox('Sensor', [x.template.name for x in fs]))
            _sensor_detail(f)
    with t_flow:
        for f in found:
            with st.expander(f.template.name):
                _sensor_flow(f)
        if not found:
            st.caption('No sensor was mapped to a class, so there is no data flow to show.')
    with t_cfg:
        used = run_state().cfg.sensor_templates_file if run_state().cfg else ''
        st.caption(f'Configuration used by the last run: `{used}`')
        _sensor_yaml_view(used)


def _sensor_row(f) -> Dict[str, Any]:
    row = {'Sensor': f.template.name, 'C++ class': f.class_name or '—', 'Mapping': f.status,
           'Instantiated': 'yes' if f.instantiated else 'no'}
    for ch in CHECKS:
        row[ch.title()] = f.checks[ch].status if ch in f.checks else '—'
    return row


def _sensor_detail(f) -> None:
    t = f.template
    st.markdown(f"{c.badge(f.status)} class `{f.class_name or '—'}` · `{f.file_path}:{f.line_number}`" if f.file_path
                else f"{c.badge(f.status)} class `{f.class_name or '—'}`", unsafe_allow_html=True)
    st.json({k: v for k, v in asdict(t).items() if v not in ('', None)}, expanded=False)
    if f.status == 'not_found':
        st.warning('No C++ class matched this template, so no checks could be evaluated.')
        return
    for name, ch in f.checks.items():
        st.markdown(f"**{name.title()}** {c.badge(ch.status)} — expected: `{ch.expected}`"
                    + (f" · found constants: `{ch.found}`" if ch.found else ''), unsafe_allow_html=True)
        c.evidence(f'{name.title()} evidence', ch.evidence)


def _sensor_flow(f) -> None:
    cols = st.columns(4)
    for col, (title, items) in zip(cols, (('Acquisition', f.sources), ('Processing', f.processors),
                                          ('Validation', f.validators), ('Consumers', f.consumers))):
        col.markdown(f'**{title}**')
        col.code('\n'.join(items) if items else 'none found', language=None)
    if f.hierarchy:
        st.caption('Hierarchy: ' + ' → '.join(f.hierarchy))
    if f.data_members:
        st.caption('Data members: ' + ', '.join(f.data_members))


# ---------------------------------------------------------------- assistant
def assistant() -> None:
    r = results()
    if r is None:
        c.need_results(r)
        return
    cfg = run_state().cfg
    llm = None
    if cfg is not None and cfg.ollama_enabled:
        client = OllamaClient(base_url=cfg.ollama_url, model=cfg.ollama_model)
        llm = client if client.is_available() else None
    st.caption('Answers come from the analysis results.' +
               (f' Local LLM ({cfg.ollama_model}) adds explanations.' if llm else
                ' The local LLM is unavailable or disabled, so only deterministic answers are given.'))
    with st.expander('Examples'):
        st.markdown('- what validation checks are applied to TemperatureSensor\n'
                    '- what is the inheritance of Sensor\n'
                    '- what are the virtual functions implemented by TemperatureSensor\n'
                    '- what is the compliance score\n- how many critical violations\n'
                    '- which functions need reverification\n- which LRU has highest risk')
    with st.expander('Analysis summary'):
        st.code(results_summary(r), language=None)
    chat = st.session_state.chat
    for role, text in chat:
        with st.chat_message(role):
            st.markdown(text) if role == 'user' else st.code(text, language=None)
    q = st.chat_input('Ask about the analysis…')
    if q:
        chat.append(('user', q))
        with st.spinner('Thinking…'):
            try:
                narrator = RAGNarrativeGenerator(llm, r.search) if llm else None
                ans = answer_question(q, r, narrator)
            except Exception as exc:
                ans = f'The assistant failed: {type(exc).__name__}: {exc}'
        chat.append(('assistant', ans))
        st.rerun()
    if chat and st.button('Clear conversation'):
        chat.clear()
        st.rerun()


# ---------------------------------------------------------------- reports
def reports() -> None:
    rs = run_state()
    r = rs.results
    if r is None:
        c.need_results(r)
        return
    paths = [Path(p) for p in r.report_paths]
    rows = [{'Report': p.stem.replace('_', ' ').title(), 'Type': p.suffix.lstrip('.') or '—',
             'Status': 'ready' if p.exists() else 'missing', 'Size KB': f'{p.stat().st_size / 1024:.1f}' if p.exists() else '—',
             'Path': str(p)} for p in paths]
    c.data_table(rows, 'This run produced no report files.')
    st.caption('The backend writes one report per module plus the SQA checklist; it has no single combined report.')
    for p in paths:
        if not p.exists():
            continue
        with st.expander(p.name):
            if p.suffix == '.md':
                st.markdown(p.read_text(encoding='utf-8', errors='replace'), unsafe_allow_html=False)
            else:
                st.caption('Preview is available for Markdown reports only.')
            st.download_button(f'Download {p.name}', p.read_bytes(), file_name=p.name, key=f'dl_{p.name}')
