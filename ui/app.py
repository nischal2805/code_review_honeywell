"""Streamlit entry point: ``python run_ui.py`` (or ``streamlit run ui/app.py``)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from ui import components as c
from ui import views
from ui.runner import RunState

PAGES = {
    'Overview': {'Dashboard': views.dashboard},
    'Setup': {'Project Configuration': views.configuration, 'Analysis Execution': views.execution},
    'Analysis': {'Virtual Functions': views.virtual, 'Dead Code': views.dead_code,
                 'Standards & Compliance': views.standards, 'Coupling': views.coupling,
                 'Checklist': views.checklist, 'Sensor Analysis': views.sensors},
    'Output': {'AI Assistant': views.assistant, 'Reports': views.reports},
}
FLAT = {name: fn for group in PAGES.values() for name, fn in group.items()}

DEFAULTS = {'codebase': '', 'base_build': '', 'lru_docs': '', 'config_path': 'config.yaml', 'dal': 'B',
            'out_fmt': 'markdown', 'semantic': False, 'standards_path': '', 'sensor_path': '', 'samples_path': '', 'disp_path': ''}


def init_state() -> None:
    ss = st.session_state
    ss.setdefault('conf', dict(DEFAULTS))
    ss.setdefault('run', RunState())
    ss.setdefault('history', [])
    ss.setdefault('chat', [])
    ss.setdefault('uploaded', {})
    ss.setdefault('nav', 'Dashboard')


def main() -> None:
    st.set_page_config(page_title='DO-178C RAG Engine', page_icon='🛩️', layout='wide')
    c.inject_css()
    init_state()
    with st.sidebar:
        st.markdown('### DO-178C RAG Engine')
        page = st.radio('Navigation', list(FLAT), key='nav', label_visibility='collapsed')
        st.divider()
        st.caption('Fully offline analysis. Findings support, but do not replace, DO-178C review.')
    rs = st.session_state.run
    k = st.session_state.conf
    project = Path(k['codebase']).name if k['codebase'] else 'No project selected'
    c.header(project, rs.cfg.dal_level if rs.cfg else k['dal'], rs.status)
    st.subheader(page)
    try:
        FLAT[page]()
    except Exception as exc:   # a page bug must not take the app down
        st.error(f'This page failed to render: {type(exc).__name__}: {exc}')
        st.exception(exc)


main()
