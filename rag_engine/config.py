from __future__ import annotations
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Literal
import yaml
from loguru import logger

from rag_engine.knowledge_base.standards_profile import extract_standards_profile_sections


# fnmatch patterns on the short function name; interrupt handlers and RTOS tasks are never called from code.
DEFAULT_ENTRY_POINT_PATTERNS = ['*_IRQHandler', '*_ISR', 'ISR_*', '*Task']


@dataclass
class Config:
    dal_level: Literal['A', 'B', 'C', 'D'] = 'B'
    output_dir: str = 'output'
    cache_dir: str = '.rag_cache'
    embedding_model: str = 'all-MiniLM-L6-v2'
    ollama_url: str = 'http://localhost:11434'
    ollama_model: str = 'llama3.1:8b'
    ollama_enabled: bool = False
    max_workers: int = 4
    entry_points: List[str] = field(default_factory=lambda: ['main'])
    entry_point_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_ENTRY_POINT_PATTERNS))
    dead_code_dispositions: str = ''
    cyclomatic_complexity_max: int = 10
    function_length_max: int = 50
    nesting_depth_max: int = 5
    param_count_max: int = 7
    faiss_similarity_threshold: float = 0.6
    standards_file: str = ''
    sensor_templates_file: str = ''
    standards_profile: Dict[str, Any] = field(default_factory=dict)
    lru_names: List[str] = field(default_factory=lambda: [
        'ADS', 'AGMCAL', 'AGM', 'APM', 'BCU', 'CLOCK',
        'FADEC', 'FCS', 'FECU', 'FMS', 'GGF', 'MWS', 'TACTICAL',
    ])


_LIST_FIELDS = ('entry_points', 'entry_point_patterns', 'lru_names')
_INT_FIELDS = ('max_workers', 'cyclomatic_complexity_max', 'function_length_max', 'nesting_depth_max', 'param_count_max')


def _check_types(data: Dict[str, Any]) -> None:
    """Normalise the list fields in place and reject a value of the wrong type, naming its key: a string
    iterated as a list of characters (``entry_point_patterns: '*Task'`` -> ``*``) silently changes the analysis."""
    for key in _LIST_FIELDS:
        value = data.get(key)
        if isinstance(value, str):
            logger.warning(f"config key '{key}' expects a list, got the string {value!r}; using [{value!r}]")
            value = data[key] = [value]
        if key in data and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            raise ValueError(f"config key '{key}' must be a list of strings, got {value!r}")
    for key in _INT_FIELDS + ('faiss_similarity_threshold',):
        value = data.get(key)
        kinds = int if key in _INT_FIELDS else (int, float)
        if key in data and (isinstance(value, bool) or not isinstance(value, kinds)):
            raise ValueError(f"config key '{key}' must be {'an integer' if key in _INT_FIELDS else 'a number'}, got {value!r}")


def load_config(path: str = 'config.yaml') -> Config:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open(encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return Config()
    except yaml.YAMLError as exc:
        mark = getattr(exc, 'problem_mark', None)
        where = f" at line {mark.line + 1}" if mark is not None else ''
        raise ValueError(f"{config_path}: invalid YAML{where}: {getattr(exc, 'problem', None) or exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: the configuration must be a mapping of keys to values")
    # hasattr(Config, k) is False for default_factory fields, which silently dropped them; a blank value keeps the default
    known = {f.name for f in fields(Config)}
    valid = {k: v for k, v in data.items() if k in known and k != 'standards_profile' and v is not None}
    _check_types(valid)
    cfg = Config(**valid)
    for attr in ('standards_file', 'sensor_templates_file', 'dead_code_dispositions'):
        if getattr(cfg, attr):
            file_path = Path(getattr(cfg, attr))
            if not file_path.is_absolute():
                file_path = (config_path.parent / file_path).resolve()
            setattr(cfg, attr, str(file_path))
    cfg.standards_profile = extract_standards_profile_sections(data)
    return cfg
