from __future__ import annotations
import hashlib
import json
import os
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from rag_engine.models import ParseResult

# Bump whenever the pickled ParseResult/FunctionDef shape changes: it is part of the stored
# hash, so entries written by an older shape miss instead of loading without new attributes.
CACHE_VERSION = 6


class AnalysisCache:
    def __init__(self, cache_dir: str = '.rag_cache') -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._dir / 'hashes.json'
        self._hashes: dict = self._load_hashes()
        self._batching = 0
        self._dirty = False

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
        self._dirty = True
        if not self._batching:
            self.flush()

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Write hashes.json once, when the block ends, instead of after every cache_result().

        A crash inside the block loses the hashes of the files cached in it (they are parsed again
        next run; the pickles are never read without a matching hash).
        """
        self._batching += 1
        try:
            yield
        finally:
            self._batching -= 1
            if not self._batching:
                self.flush()

    def flush(self) -> None:
        """Write hashes.json if anything changed since the last write."""
        if self._dirty:
            self._save_hashes()
            self._dirty = False

    @staticmethod
    def _hash_file(file_path: str) -> str:
        return f"v{CACHE_VERSION}:{hashlib.sha256(Path(file_path).read_bytes()).hexdigest()}"

    def _cache_path(self, file_path: str) -> Path:
        safe = file_path.replace('/', '_').replace('\\', '_').replace(':', '_')
        return self._dir / f"{safe}.pkl"

    def _load_hashes(self) -> dict:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text())
        return {}

    def _save_hashes(self) -> None:
        tmp = self._meta_path.with_suffix('.json.tmp')   # replaced atomically: a crash never leaves a torn file
        tmp.write_text(json.dumps(self._hashes, indent=2))
        os.replace(tmp, self._meta_path)
