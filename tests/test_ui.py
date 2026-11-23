"""Smoke tests for the Streamlit UI: every page renders, before and after a real analysis run."""
import time
from pathlib import Path

import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / 'ui' / 'app.py')
PAGES = ['Dashboard', 'Project Configuration', 'Analysis Execution', 'Virtual Functions', 'Dead Code',
         'Standards & Compliance', 'Coupling', 'Checklist', 'Sensor Analysis', 'AI Assistant', 'Reports']


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


@pytest.mark.parametrize('page', PAGES)
def test_page_renders_without_results(page):
    at = _app()
    at.session_state['nav'] = page
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_full_run_then_every_page(tmp_path):
    at = _app()
    at.session_state['conf'] = {**at.session_state['conf'], 'codebase': str(ROOT / 'sample_codebase'),
                                'sensor_path': str(ROOT / 'sensor_templates.yaml'), 'semantic': False}
    # point the output at a scratch dir via the shared config: run through the runner directly
    from rag_engine.config import load_config
    from ui.runner import start_run
    cfg = load_config(str(ROOT / 'config.yaml'))
    cfg.output_dir, cfg.cache_dir = str(tmp_path / 'out'), str(tmp_path / 'cache')
    cfg.sensor_templates_file = str(ROOT / 'tests' / 'fixtures' / 'sensor_templates.yaml')
    cfg.sensor_samples_file = str(ROOT / 'tests' / 'fixtures' / 'sensor_samples.csv')
    state = at.session_state['run']
    start_run(state, cfg, str(ROOT / 'tests' / 'fixtures' / 'cpp'), None, None, 'markdown', False)
    for _ in range(120):
        if state.status != 'running':
            break
        time.sleep(0.5)
    assert state.status == 'completed', state.error
    for page in PAGES:
        at.session_state['nav'] = page
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
