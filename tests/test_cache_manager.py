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


def _stored(tmp_path):
    return json.loads((tmp_path / 'cache' / 'hashes.json').read_text())


def _pr(f):
    return ParseResult(file_path=str(f), functions=[], classes=[], includes=[], raw_source=b'int main() {}')


def test_batch_writes_hashes_once_at_the_end(tmp_path):
    cache = AnalysisCache(cache_dir=str(tmp_path / 'cache'))
    files = []
    for i in range(3):
        f = tmp_path / f'f{i}.cpp'
        f.write_text(f'int f{i}() {{}}')
        files.append(f)
    with cache.batch():
        for f in files:
            cache.cache_result(str(f), _pr(f))
        assert not (tmp_path / 'cache' / 'hashes.json').exists()        # deferred
        assert cache.check_cache(str(files[0])) is not None             # the in-memory hashes already count
    assert set(_stored(tmp_path)) == {str(f) for f in files}
    assert AnalysisCache(cache_dir=str(tmp_path / 'cache')).check_cache(str(files[2])) is not None


def test_batch_flushes_even_when_the_block_raises(tmp_path):
    cache = AnalysisCache(cache_dir=str(tmp_path / 'cache'))
    f = tmp_path / 'a.cpp'
    f.write_text('int a() {}')
    try:
        with cache.batch():
            cache.cache_result(str(f), _pr(f))
            raise RuntimeError('boom')
    except RuntimeError:
        pass
    assert str(f) in _stored(tmp_path)


def test_without_batch_every_cache_result_is_written_at_once(tmp_path):
    cache = AnalysisCache(cache_dir=str(tmp_path / 'cache'))
    f = tmp_path / 'a.cpp'
    f.write_text('int a() {}')
    cache.cache_result(str(f), _pr(f))
    assert str(f) in _stored(tmp_path)
    cache.flush()   # nothing pending: harmless
    assert not list((tmp_path / 'cache').glob('*.tmp'))


def test_sig_key_is_rebuilt_after_a_pickle_round_trip(tmp_path):
    from rag_engine.core.parser import CodeParser
    f = tmp_path / 'a.cpp'
    f.write_text('int f(const Frame & x) { return 1; }')
    cache = AnalysisCache(cache_dir=str(tmp_path / 'cache'))
    result = CodeParser().parse_file(str(f))
    assert result.functions[0].sig_key == 'f(const Frame &)'      # filled the in-memory sig_key cache
    cache.cache_result(str(f), result)
    loaded = cache.check_cache(str(f))
    assert '_sig_cache' not in loaded.functions[0].__dict__
    assert loaded.functions[0].sig_key == 'f(const Frame &)'
