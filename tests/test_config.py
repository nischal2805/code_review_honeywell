from pathlib import Path

import pytest
from loguru import logger

from rag_engine.config import Config, load_config
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.core.parser import CodeParser
from rag_engine.features.dead_code_detector import DeadCodeDetector


def _load(tmp_path, text):
    path = tmp_path / 'config.yaml'
    path.write_text(text, encoding='utf-8')
    return load_config(str(path))


@pytest.fixture
def warnings():
    messages = []
    sink = logger.add(lambda m: messages.append(m.record['message']), level='WARNING')
    yield messages
    logger.remove(sink)


def test_a_string_for_a_list_field_is_a_one_item_list_with_a_warning(tmp_path, warnings):
    cfg = _load(tmp_path, "entry_points: main\nentry_point_patterns: '*Task'\nlru_names: ADS\n")
    assert (cfg.entry_points, cfg.entry_point_patterns, cfg.lru_names) == (['main'], ['*Task'], ['ADS'])
    assert len(warnings) == 3
    assert all("expects a list" in w for w in warnings)
    assert any("'entry_point_patterns'" in w and "'*Task'" in w for w in warnings)


def test_a_string_pattern_does_not_make_every_function_an_entry_point(tmp_path):
    (tmp_path / 'a.cpp').write_text('int main() { return 0; }\nvoid deadOne() {}\nvoid deadTwo() {}\n')
    cfg = _load(tmp_path, "entry_point_patterns: '*Task'\n")
    results = CodeParser().parse_directory(str(tmp_path))
    gb = CallGraphBuilder(results)
    gb.build()
    report = DeadCodeDetector(gb, results, cfg.entry_points, cfg.entry_point_patterns).analyze()
    assert {i.name for i in report.items} == {'deadOne()', 'deadTwo()'}


@pytest.mark.parametrize('key, value', [
    ('entry_points', '[main, 3]'),
    ('entry_point_patterns', '[[a], b]'),
    ('lru_names', '[ADS, {x: 1}]'),
    ('entry_points', '{main: 1}'),
    ('entry_point_patterns', '5'),
])
def test_a_list_field_with_a_non_string_item_names_the_key(tmp_path, key, value):
    with pytest.raises(ValueError, match=f"'{key}'"):
        _load(tmp_path, f'{key}: {value}\n')


@pytest.mark.parametrize('key, value', [
    ('max_workers', 'four'), ('max_workers', '2.5'), ('max_workers', 'true'),
    ('cyclomatic_complexity_max', '"10"'), ('function_length_max', '[50]'),
    ('nesting_depth_max', 'deep'), ('param_count_max', 'true'),
    ('faiss_similarity_threshold', 'high'), ('faiss_similarity_threshold', 'false'),
])
def test_numeric_fields_are_type_checked(tmp_path, key, value):
    with pytest.raises(ValueError, match=f"'{key}'"):
        _load(tmp_path, f'{key}: {value}\n')


def test_numeric_fields_accept_numbers(tmp_path):
    cfg = _load(tmp_path, 'max_workers: 2\nfaiss_similarity_threshold: 1\nparam_count_max: 9\n')
    assert (cfg.max_workers, cfg.faiss_similarity_threshold, cfg.param_count_max) == (2, 1, 9)
    assert _load(tmp_path, 'faiss_similarity_threshold: 0.75\n').faiss_similarity_threshold == 0.75


def test_a_blank_value_keeps_the_default(tmp_path):
    cfg = _load(tmp_path, 'entry_points:\nmax_workers:\n')
    assert cfg.entry_points == ['main'] and cfg.max_workers == 4


def test_malformed_yaml_is_a_value_error_naming_the_file_and_line(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('dal_level: B\nentry_points: [main\n  - x\nmax_workers: 4\n', encoding='utf-8')
    with pytest.raises(ValueError) as exc:
        load_config(str(path))
    assert 'config.yaml' in str(exc.value) and 'invalid YAML at line 4' in str(exc.value)   # where the parser gave up


def test_a_config_that_is_not_a_mapping_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match='mapping'):
        _load(tmp_path, '- main\n- other\n')


def test_the_repository_config_loads():
    cfg = load_config(str(Path(__file__).parent.parent / 'config.yaml'))
    assert cfg.entry_points == ['main'] and '*Task' in cfg.entry_point_patterns


def test_a_missing_file_gives_the_defaults(tmp_path):
    assert load_config(str(tmp_path / 'nope.yaml')) == Config()
