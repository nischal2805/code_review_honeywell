from rag_engine.knowledge_base.cache_manager import AnalysisCache
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
