"""Background execution of the existing ``run_pipeline`` so the Streamlit page never blocks."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger

from rag_engine.config import Config
from rag_engine.pipeline import AnalysisResults, InputError, run_pipeline

WORKSPACE = Path(__file__).resolve().parent.parent / '.ui_workspace'


def save_upload(uploaded: Any, subdir: str = 'uploads') -> str:
    """Persist a Streamlit UploadedFile under the workspace and return its absolute path (stays valid across reruns)."""
    target = WORKSPACE / subdir
    target.mkdir(parents=True, exist_ok=True)
    path = target / Path(uploaded.name).name
    path.write_bytes(uploaded.getvalue())
    return str(path)


@dataclass
class RunState:
    """Shared between the worker thread and the page; mutations go through the lock."""
    status: str = 'idle'                  # idle | running | completed | failed
    step: str = ''
    percent: int = 0
    logs: List[str] = field(default_factory=list)
    results: Optional[AnalysisResults] = None
    cfg: Optional[Config] = None
    error: Optional[str] = None
    started: float = 0.0
    finished: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def log(self, msg: str) -> None:
        with self.lock:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def start_run(state: RunState, cfg: Config, codebase: str, base_build: Optional[str], lru_docs: Optional[str],
              output_format: str, semantic: bool) -> None:
    """Launch the pipeline on a daemon thread; *state* is updated as it progresses."""
    with state.lock:
        state.status, state.step, state.percent = 'running', 'Starting', 0
        state.logs, state.results, state.error, state.cfg = [], None, None, cfg
        state.started, state.finished = time.time(), 0.0

    def progress(step: str, pct: int) -> None:
        with state.lock:
            state.step, state.percent = step, pct
        state.log(step)

    def work() -> None:
        sink = logger.add(lambda m: state.log(m.record['message']), level='WARNING',
                          filter=lambda r: r['thread'].id == threading.get_ident())
        try:
            results = run_pipeline(cfg, codebase, base_build or None, lru_docs or None, output_format,
                                   progress=progress, use_semantic_search=semantic)
            with state.lock:
                state.results, state.status = results, 'completed'
        except InputError as exc:
            with state.lock:
                state.error, state.status = str(exc), 'failed'
        except Exception as exc:   # surfaced in the UI, never a crashed page
            logger.exception('UI analysis failed')
            with state.lock:
                state.error, state.status = f'{type(exc).__name__}: {exc}', 'failed'
        finally:
            logger.remove(sink)
            with state.lock:
                state.finished = time.time()
            state.log(f"Run {state.status}" + (f": {state.error}" if state.error else ''))

    threading.Thread(target=work, daemon=True, name='ui-analysis').start()
