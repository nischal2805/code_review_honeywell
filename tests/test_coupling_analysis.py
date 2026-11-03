from pathlib import Path

from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.core.parser import CodeParser
from rag_engine.features.coupling_analysis import CouplingAnalyzer

SENSORS = Path(__file__).parent / 'fixtures' / 'cpp' / 'sensors'


def _analyze(docs):
    results = CodeParser().parse_directory(str(SENSORS))
    gb = CallGraphBuilder(results)
    gb.build()   # fills FunctionDef.calls / is_virtual, as the pipeline does
    return CouplingAnalyzer(results, gb, None, docs).analyze()


def test_without_semantic_search_lru_code_is_found_by_keyword_overlap():
    out = _analyze({'TEMP': 'The LRU samples the temperature sensor and validates every reading.'})
    coupling = out.lru_impacts['TEMP']
    assert any('TemperatureSensor::read' in c for c in coupling.control_coupling)
    ranked = [c.split(' (')[0] for c in coupling.control_coupling]   # best match first
    assert all('Temperature' in name for name in ranked[:3])


def test_without_semantic_search_unrelated_document_matches_nothing():
    out = _analyze({'ZZZ': 'Quarterly catering budget and parking arrangements.'})
    assert out.lru_impacts['ZZZ'].control_coupling == []
    assert out.lru_impacts['ZZZ'].risk_level == 'low'


def test_generic_verbs_in_a_document_do_not_couple_it_to_every_function():
    # "read", "update", "check", "data", "process" occur in the name of half the fixture's functions.
    out = _analyze({'BATTERY': 'The battery module shall read the cell voltage, then update and check the data '
                               'and process it every cycle.'})
    assert out.lru_impacts['BATTERY'].control_coupling == []


def test_functions_are_ranked_by_distinct_document_terms_and_capped_at_15():
    out = _analyze({'PT': 'pressure temperature sensor sensor sensor'})
    ranked = [c.split(' (')[0] for c in out.lru_impacts['PT'].control_coupling]
    assert 0 < len(ranked) <= 15
    # Two distinct terms (pressure or temperature, plus sensor) outrank the single term "sensor".
    two_terms = [n for n in ranked if 'Pressure' in n or 'Temperature' in n]
    assert two_terms and ranked[:len(two_terms)] == two_terms
