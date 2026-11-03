import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from loguru import logger

from rag_engine.config import Config
from rag_engine.pipeline import AnalysisResults, InputError, run_pipeline

ROOT = Path(__file__).resolve().parent.parent
SENSORS = ROOT / 'tests' / 'fixtures' / 'cpp' / 'sensors'
TEMPLATES = ROOT / 'tests' / 'fixtures' / 'sensor_templates.yaml'


def _cfg(tmp_path, templates=True):
    return Config(output_dir=str(tmp_path / 'out'), cache_dir=str(tmp_path / 'cache'),
                  sensor_templates_file=str(TEMPLATES) if templates else '', ollama_enabled=False)


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    cfg = _cfg(tmp_path_factory.mktemp('pipeline'))
    steps = []
    results = run_pipeline(cfg, str(SENSORS), progress=lambda s, p: steps.append((s, p)),
                           use_semantic_search=False)
    return cfg, results, steps


def test_results_hold_every_analysis_and_the_code_model(run):
    _, r, _ = run
    assert isinstance(r, AnalysisResults)
    assert r.virtual.inventory and r.dead and r.standards and r.coupling is not None
    assert [f.template.name for f in r.sensors.findings] == [
        'TemperatureSensor', 'PressureSensor', 'AltitudeSensor', 'SpeedSensor']
    assert r.graph.built and r.parse_results
    assert r.search is None and r.checklist is not None and r.error is None


def test_writes_all_reports_and_checklist(run):
    cfg, r, _ = run
    out = Path(cfg.output_dir)
    for name in ('virtual_analysis.md', 'coupling_analysis.md', 'dead_code_report.md', 'standards_report.md',
                 'sensor_analysis.md', 'dead_code_findings.json', 'sqa_checklist.md'):
        assert (out / name).is_file(), name
    assert json.loads((out / 'dead_code_findings.json').read_text(encoding='utf-8'))['items']
    assert {Path(p).name for p in r.report_paths} >= {'virtual_analysis.md', 'sensor_analysis.md', 'sqa_checklist.md'}
    assert all(Path(p).is_file() for p in r.report_paths)


def test_progress_reports_every_step_and_ends_at_100(run):
    _, _, steps = run
    pcts = [p for _, p in steps]
    assert pcts == sorted(pcts) and pcts[-1] == 100
    text = ' '.join(s.lower() for s, _ in steps)
    for word in ('pars', 'call graph', 'virtual', 'dead', 'standards', 'coupling', 'sensor', 'report'):
        assert word in text, word


def test_write_reports_false_writes_nothing(tmp_path):
    cfg = _cfg(tmp_path)
    r = run_pipeline(cfg, str(SENSORS), use_semantic_search=False, write_reports=False)
    assert not Path(cfg.output_dir).exists()
    assert r.report_paths == [] and r.checklist is None and r.virtual is not None


def test_without_sensor_templates_the_sensor_step_is_skipped(tmp_path):
    cfg = _cfg(tmp_path, templates=False)
    r = run_pipeline(cfg, str(SENSORS), use_semantic_search=False)
    assert r.sensors is None
    assert not (Path(cfg.output_dir) / 'sensor_analysis.md').exists()
    assert 'SEN-01' not in (Path(cfg.output_dir) / 'sqa_checklist.md').read_text(encoding='utf-8')


def test_docx_format(tmp_path):
    cfg = _cfg(tmp_path)
    run_pipeline(cfg, str(SENSORS), output_format='docx', use_semantic_search=False)
    assert (Path(cfg.output_dir) / 'sensor_analysis.docx').is_file()


def test_empty_codebase_raises(tmp_path):
    (tmp_path / 'empty').mkdir()
    with pytest.raises(InputError, match='No C\\+\\+ files'):
        run_pipeline(_cfg(tmp_path), str(tmp_path / 'empty'), use_semantic_search=False, write_reports=False)


def test_semantic_search_unavailable_continues_with_one_warning(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'rag_engine.core.embeddings', None)   # import raises ImportError
    warnings = []
    sink = logger.add(lambda m: warnings.append(m), level='WARNING', format='{message}')
    try:
        r = run_pipeline(_cfg(tmp_path), str(SENSORS), write_reports=False)
    finally:
        logger.remove(sink)
    assert r.search is None and r.coupling is not None
    assert len([w for w in warnings if 'Semantic search' in w]) == 1


def test_semantic_search_is_used_when_it_loads(tmp_path, monkeypatch):
    class FakeSearch:
        def __init__(self, model_name):
            self.indexed = []

        def index_functions(self, functions):
            self.indexed = functions

        def find_related(self, query, k=10):
            return []

    fake = types.ModuleType('rag_engine.core.embeddings')
    fake.SemanticSearch = FakeSearch
    monkeypatch.setitem(sys.modules, 'rag_engine.core.embeddings', fake)
    r = run_pipeline(_cfg(tmp_path), str(SENSORS), write_reports=False)
    assert isinstance(r.search, FakeSearch) and r.search.indexed


def test_llm_narratives_reach_every_report_including_sensors(tmp_path, monkeypatch):
    class FakeLLM:
        def __init__(self, base_url, model):
            pass

        def is_available(self):
            return True

        def generate(self, prompt, max_tokens=512):
            return 'NARRATIVE-FOR: ' + prompt.split('=== ANALYSIS RESULTS ===')[1].split('\n')[1][:20]

    monkeypatch.setattr('rag_engine.pipeline.OllamaClient', FakeLLM)
    cfg = _cfg(tmp_path)
    cfg.ollama_enabled = True
    run_pipeline(cfg, str(SENSORS), use_semantic_search=False)
    sensor_md = (Path(cfg.output_dir) / 'sensor_analysis.md').read_text(encoding='utf-8')
    assert 'AI Certification Assessment' in sensor_md and 'NARRATIVE-FOR: Sensor analysis' in sensor_md
    assert 'NARRATIVE-FOR' in (Path(cfg.output_dir) / 'virtual_analysis.md').read_text(encoding='utf-8')


def test_importing_the_cli_and_running_without_semantic_search_never_loads_sentence_transformers(tmp_path):
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import main, rag_engine.pipeline, rag_engine.features.coupling_analysis;"
        "import rag_engine.knowledge_base.index_manager;"
        "from rag_engine.config import Config;"
        "cfg = Config(output_dir=sys.argv[2] + '/out', cache_dir=sys.argv[2] + '/cache', sensor_templates_file=sys.argv[4]);"
        "r = rag_engine.pipeline.run_pipeline(cfg, sys.argv[3], use_semantic_search=False);"
        "assert r.sensors is not None and r.sensors.findings;"   # the sensor branch ran too
        "bad = [m for m in ('sentence_transformers', 'faiss', 'rag_engine.core.embeddings') if m in sys.modules];"
        "assert not bad, bad"
    )
    proc = subprocess.run([sys.executable, '-c', code, str(ROOT), str(tmp_path), str(SENSORS), str(TEMPLATES)],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_bad_sensor_templates_are_reported_as_input_errors_before_any_parsing(tmp_path, monkeypatch):
    class NoParsing:
        def __init__(self, *a, **kw):
            raise AssertionError('the codebase was parsed before the templates were validated')

    monkeypatch.setattr('rag_engine.pipeline.CodeReader', NoParsing)
    bad = tmp_path / 'bad.yaml'
    bad.write_text('sensors:\n  - name: S\n    valid_range: {min: 5, max: 1}\n', encoding='utf-8')
    for path, message in ((bad, 'must be less than max'), (tmp_path / 'missing.yaml', 'missing.yaml')):
        cfg = _cfg(tmp_path)
        cfg.sensor_templates_file = str(path)
        with pytest.raises(InputError, match=message) as excinfo:
            run_pipeline(cfg, str(SENSORS), use_semantic_search=False, write_reports=False)
        assert str(path) in str(excinfo.value)


def test_template_file_with_a_yaml_syntax_error_is_an_input_error_naming_the_file_once(tmp_path):
    bad = tmp_path / 'broken.yaml'
    bad.write_text('sensors:\n  - name: S\n    valid_range: {min: 1\n', encoding='utf-8')
    cfg = _cfg(tmp_path)
    cfg.sensor_templates_file = str(bad)
    with pytest.raises(InputError) as excinfo:
        run_pipeline(cfg, str(SENSORS), use_semantic_search=False, write_reports=False)
    assert str(excinfo.value).count(str(bad)) == 1 and 'YAML' in str(excinfo.value)


def test_template_error_messages_do_not_repeat_the_path(tmp_path):
    empty = tmp_path / 'empty.yaml'
    empty.write_text('sensors: []\n', encoding='utf-8')
    cfg = _cfg(tmp_path)
    cfg.sensor_templates_file = str(empty)
    with pytest.raises(InputError, match='no sensors defined') as excinfo:
        run_pipeline(cfg, str(SENSORS), use_semantic_search=False, write_reports=False)
    assert str(excinfo.value).count(str(empty)) == 1
