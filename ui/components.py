"""Reusable HTML/Streamlit rendering helpers. Every dynamic string is HTML-escaped."""
from __future__ import annotations

from html import escape
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import streamlit as st

from ui.styles import CSS

_BADGE = {
    'ok': ('b-ok', ('implemented', 'pass', 'found', 'completed', 'done', 'low', 'closed', 'confirmed_dead', 'false_positive',
                    'deactivated', 'unchanged', 'ready', 'yes')),
    'warn': ('b-warn', ('mismatch', 'medium', 'minor', 'open', 'inconclusive', 'modified', 'major', 'running', 'warning',
                        'investigate')),
    'err': ('b-err', ('missing', 'fail', 'not_found', 'failed', 'high', 'critical', 'error', 'removed', 'no')),
    'info': ('b-info', ('added', 'info', 'idle')),
}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(text: Any, kind: Optional[str] = None) -> str:
    """HTML status badge; the colour is derived from the text unless *kind* is given."""
    label = str(text if text is not None else '—')
    key = label.strip().lower().replace(' ', '_')
    if kind is None:
        cls = next((c for c, words in _BADGE.values() if key in words), 'b-mute')
    else:
        cls = {'ok': 'b-ok', 'warn': 'b-warn', 'err': 'b-err', 'info': 'b-info'}.get(kind, 'b-mute')
    return f'<span class="badge {cls}">{escape(label.replace("_", " "))}</span>'


def header(project: str, dal: Optional[str], status: str) -> None:
    dal_html = f'DAL {escape(str(dal))}' if dal else 'DAL —'
    st.markdown(
        f'<div class="app-header"><div><h1>DO-178C RAG Engine</h1>'
        f'<div class="sub">Codebase Analysis &amp; Certification Tool</div></div>'
        f'<div class="right">{escape(project)}<br>{dal_html} &nbsp;·&nbsp; {badge(status)}</div></div>',
        unsafe_allow_html=True)


def metric_card(label: str, value: Any, note: str = '', tone: str = '') -> str:
    return (f'<div class="card {tone}"><div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(str(value))}</div><div class="note">{escape(note)}</div></div>')


def metric_row(cards: Sequence[tuple]) -> None:
    """cards: (label, value[, note[, tone]]) tuples laid out in one row."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(metric_card(*card), unsafe_allow_html=True)
    st.write('')


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)


def empty_state(message: str, hint: str = '') -> None:
    extra = f'<div style="margin-top:6px;font-size:.85rem">{escape(hint)}</div>' if hint else ''
    st.markdown(f'<div class="empty"><b>{escape(message)}</b>{extra}</div>', unsafe_allow_html=True)


def log_console(lines: Iterable[str], limit: int = 300) -> None:
    body = escape('\n'.join(list(lines)[-limit:])) or 'No output yet.'
    st.markdown(f'<div class="console">{body}</div>', unsafe_allow_html=True)


def data_table(rows: List[Dict[str, Any]], empty: str = 'No rows to show.', height: Optional[int] = None) -> None:
    """Render dict rows as a dataframe; None values show as '—'."""
    if not rows:
        st.caption(empty)
        return
    df = pd.DataFrame(rows).fillna('—').astype(str)
    st.dataframe(df, width="stretch", hide_index=True, **({"height": height} if height else {}))


def evidence(title: str, lines: Iterable[Any], expanded: bool = False) -> None:
    items = [str(x) for x in (lines or [])]
    with st.expander(title, expanded=expanded):
        if items:
            st.code('\n'.join(items), language=None)
        else:
            st.caption('No supporting evidence recorded.')


def need_results(results: Any, attr: Optional[str] = None) -> bool:
    """True when the page can render; otherwise shows the right empty state."""
    if results is None:
        empty_state('No analysis results yet', 'Configure the project, then run the analysis from "Analysis Execution".')
        return False
    if attr is not None and getattr(results, attr, None) is None:
        empty_state(f'This module produced no results ({attr})',
                    'It was not run, or the last run failed before reaching it.')
        return False
    return True
