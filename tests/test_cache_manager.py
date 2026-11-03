import hashlib
import json

from rag_engine.knowledge_base.cache_manager import CACHE_VERSION, AnalysisCache
from rag_engine.models import ParseResult


def test_cache_miss_on_new_file(tmp_path):
    cache = AnalysisCache(cache_dir=str(tmp_path / 'cache'))
    f = tmp_path / 'test.cpp'
    f.write_text('int main() {}')
    assert cache.check_cache(str(f)) is None


def test_cache_hit_after_store(tmp_path):
    cache = AnalysisCache(cache_dir=str(tmp_path / 'cache'))
    f = tmp_path / 'test.cpp'
    f.write_text('int main() {}')
    pr = ParseResult(file_path=str(f), functions=[], classes=[], includes=[], raw_source=b'int main() {}')
    cache.cache_result(str(f), pr)
    result = cache.check_cache(str(f))
    assert result is not None
    assert result.file_path == str(f)


def test_cache_miss_after_file_change(tmp_path):
    cache = AnalysisCache(cache_dir=str(tmp_path / 'cache'))
    f = tmp_path / 'test.cpp'
    f.write_text('int main() {}')
    pr = ParseResult(file_path=str(f), functions=[], classes=[], includes=[], raw_source=b'int main() {}')
    cache.cache_result(str(f), pr)
    f.write_text('int main() { return 1; }')
    assert cache.check_cache(str(f)) is None


def test_stored_hash_carries_cache_version(tmp_path):
    cache = AnalysisCache(cache_dir=str(tmp_path / 'cache'))
    f = tmp_path / 'test.cpp'
    f.write_text('int main() {}')
    pr = ParseResult(file_path=str(f), functions=[], classes=[], includes=[], raw_source=b'int main() {}')
    cache.cache_result(str(f), pr)
    stored = json.loads((tmp_path / 'cache' / 'hashes.json').read_text())[str(f)]
    assert stored.startswith(f'v{CACHE_VERSION}:')


def test_cache_entry_from_older_version_misses(tmp_path):
    cache_dir = tmp_path / 'cache'
    cache = AnalysisCache(cache_dir=str(cache_dir))
    f = tmp_path / 'test.cpp'
    f.write_text('int main() {}')
    pr = ParseResult(file_path=str(f), functions=[], classes=[], includes=[], raw_source=b'int main() {}')
    cache.cache_result(str(f), pr)
    hashes_path = cache_dir / 'hashes.json'
    hashes = json.loads(hashes_path.read_text())
    hashes[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()  # pre-versioning format
    hashes_path.write_text(json.dumps(hashes))
    assert AnalysisCache(cache_dir=str(cache_dir)).check_cache(str(f)) is None
