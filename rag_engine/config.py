from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal
import yaml

from rag_engine.knowledge_base.standards_profile import extract_standards_profile_sections


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
    cyclomatic_complexity_max: int = 10
    function_length_max: int = 50
    nesting_depth_max: int = 5
    param_count_max: int = 7
    faiss_similarity_threshold: float = 0.6
    standards_file: str = ''
    standards_profile: Dict[str, Any] = field(default_factory=dict)
    lru_names: List[str] = field(default_factory=lambda: [
        'ADS', 'AGMCAL', 'AGM', 'APM', 'BCU', 'CLOCK',
        'FADEC', 'FCS', 'FECU', 'FMS', 'GGF', 'MWS', 'TACTICAL',
    ])


def load_config(path: str = 'config.yaml') -> Config:
    try:
        config_path = Path(path).expanduser().resolve()
        with config_path.open() as f:
            data = yaml.safe_load(f) or {}
        valid = {k: v for k, v in data.items() if hasattr(Config, k) and k != 'standards_profile'}
        cfg = Config(**valid)
        if cfg.standards_file:
            standards_path = Path(cfg.standards_file)
            if not standards_path.is_absolute():
                standards_path = (config_path.parent / standards_path).resolve()
            cfg.standards_file = str(standards_path)
        cfg.standards_profile = extract_standards_profile_sections(data)
        return cfg
    except FileNotFoundError:
        return Config()
