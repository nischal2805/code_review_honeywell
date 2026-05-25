import pytest
from pathlib import Path
from rag_engine.core.parser import CodeParser
from rag_engine.core.embeddings import SemanticSearch
from rag_engine.models import FunctionDef

FIXTURES = Path(__file__).parent / 'fixtures' / 'cpp'


@pytest.fixture(scope='module')
def indexed_search():
    search = SemanticSearch()
    results = CodeParser().parse_directory(str(FIXTURES / 'base_build'))
    functions = [fn for r in results.values() for fn in r.functions]
    search.index_functions(functions)
    return search


def test_find_related_returns_list(indexed_search):
    assert isinstance(indexed_search.find_related('get vehicle speed', k=3), list)


def test_find_related_returns_function_defs(indexed_search):
    results = indexed_search.find_related('start engine speed', k=2)
    for r in results:
        assert isinstance(r, FunctionDef)


def test_find_related_relevance(indexed_search):
    results = indexed_search.find_related('getSpeed vehicle', k=5)
    names = [r.name for r in results]
    assert 'getSpeed' in names


def test_save_and_load(indexed_search, tmp_path):
    indexed_search.save(str(tmp_path / 'idx'))
    loaded = SemanticSearch()
    loaded.load(str(tmp_path / 'idx'))
    assert len(loaded.find_related('speed', k=2)) >= 1
