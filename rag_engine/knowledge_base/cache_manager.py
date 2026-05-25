from __future__ import annotations
import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

from rag_engine.models import ParseResult


class AnalysisCache:
    def __init__(self, cache_dir: str = '.rag_cache') -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._dir / 'hashes.json'
        self._hashes: dict = self._load_hashes()

    def check_cache(self, file_path: str) -> Optional[ParseResult]:
        current_hash = self._hash_file(file_path)
        if self._hashes.get(file_path) != current_hash:
            return None
        cache_file = self._cache_path(file_path)
        if not cache_file.exists():
            return None
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    def cache_result(self, file_path: str, result: ParseResult) -> None:
        self._hashes[file_path] = self._hash_file(file_path)
        cache_file = self._cache_path(file_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        self._save_hashes()

    @staticmethod
    def _hash_file(file_path: str) -> str:
        return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()

    def _cache_path(self, file_path: str) -> Path:
        safe = file_path.replace('/', '_').replace('\\', '_').replace(':', '_')
        return self._dir / f"{safe}.pkl"

    def _load_hashes(self) -> dict:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text())
        return {}

    def _save_hashes(self) -> None:
        self._meta_path.write_text(json.dumps(self._hashes, indent=2))
