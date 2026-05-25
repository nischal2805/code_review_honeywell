from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal
import yaml


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
    lru_names: List[str] = field(default_factory=lambda: [
        'ADS', 'AGMCAL', 'AGM', 'APM', 'BCU', 'CLOCK',
        'FADEC', 'FCS', 'FECU', 'FMS', 'GGF', 'MWS', 'TACTICAL',
    ])


def load_config(path: str = 'config.yaml') -> Config:
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {k: v for k, v in data.items() if hasattr(Config, k)}
        return Config(**valid)
    except FileNotFoundError:
        return Config()
