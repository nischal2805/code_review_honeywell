"""Any uploaded coding standard drives the standards check, and is searchable through the RAG pipeline.

Documents and limits here are synthetic test material."""
import textwrap
from pathlib import Path

import numpy as np
import pytest
from docx import Document
from reportlab.pdfgen import canvas

from rag_engine.config import Config
from rag_engine.features.standards_validator import StandardsValidator
from rag_engine.core.parser import CodeParser
from rag_engine.knowledge_base.standards_document import extract_rules, load_standards_document
from rag_engine.knowledge_base.standards_profile import load_raw_standards_profile
from rag_engine.llm.offline_qa import answer_question
from rag_engine.pipeline import InputError, run_pipeline
from rag_engine.retrieval import HybridRetriever, chunk_blocks, read_blocks, sentences
from rag_engine.retrieval.chunking import Block

ROOT = Path(__file__).resolve().parent.parent

STANDARD_MD = textwrap.dedent("""\
    # Example Flight Software Coding Standard

    ## 3 Complexity
    Cyclomatic complexity of any function shall not exceed 12.
    For DAL A software, cyclomatic complexity shall be no more than 8.
    Functions shall not exceed 25 executable lines.
    Control flow nesting depth shall be limited to 4 levels.
    A function shall take fewer than 6 parameters.

    ## 4 Language subset
    The goto statement shall not be used.
    Dynamic memory allocation (malloc, new) is prohibited after initialisation.
    Recursion shall not be used unless a bound on depth is proven and justified.
    C++ exceptions should be avoided.
    Run-time type information (dynamic_cast, typeid) may be used.

    ## 5 Naming
    Function names shall be written in camelCase, and class names shall use PascalCase.

    ## 6 Documentation
    Every function shall have a header comment describing inputs and outputs.
    All interrupts shall be disabled during flash writes.
""")

CODE = textwrap.dedent("""\
    #include <typeinfo>
    class base_shape { public: virtual ~base_shape() {} };
    // Computes the area.
    int computeArea(int w, int h) {
        int a = w * h;
        a = a + 1;
        a = a + 2;
        return a;
    }
    // A long function.
    int longFunction(int x) {
        int y = 0;
    """ + ''.join(f"    y = y + {i};\n" for i in range(30)) + """\
        return y;
    }
    // Uses RTTI, which the standard allows.
    bool isBase(base_shape* s) { return typeid(*s) == typeid(base_shape); }
""")


@pytest.fixture
def standard(tmp_path):
    path = tmp_path / 'standard.md'
    path.write_text(STANDARD_MD)
    return path


@pytest.fixture
def code(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.cpp').write_text(CODE)
    return src


# ── Ingestion and chunking ───────────────────────────────────────────────────

def _pdf(path, pages):
    c = canvas.Canvas(str(path))
    for lines in pages:
        y = 800
        for line in lines:
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return path


def test_pdf_pages_and_sections_are_kept(tmp_path):
    pdf = _pdf(tmp_path / 's.pdf', [['1 Scope', 'This standard applies to flight code.'],
                                    ['7 Functions', 'Functions shall not exceed 40 lines.']])
    items = list(sentences(read_blocks(pdf)))
    assert ('Functions shall not exceed 40 lines.', 2, '7 Functions') in items


def test_docx_headings_and_tables(tmp_path):
    doc = Document()
    doc.add_heading('Memory', level=1)
    doc.add_paragraph('Heap allocation is prohibited.')
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = 'Max nesting depth', '<= 3'
    path = tmp_path / 's.docx'
    doc.save(path)
    items = list(sentences(read_blocks(path)))
    assert ('Heap allocation is prohibited.', None, 'Memory') in items
    assert any('Max nesting depth | <= 3' in s for s, _, _ in items)


def test_unsupported_format_is_a_clear_error(tmp_path):
    bad = tmp_path / 's.xls'
    bad.write_text('x')
    with pytest.raises(ValueError, match='Unsupported document format'):
        load_standards_document(str(bad))


def test_chunks_overlap_stay_in_their_section_and_have_stable_ids():
    blocks = [Block('2 Rules')] + [Block(f'Sentence number {i} says something about rule {i}.') for i in range(40)] + \
             [Block(''), Block('3 Other'), Block('Another section starts here.')]
    chunks = chunk_blocks(blocks, 'doc.md', max_words=40, overlap_words=10)
    assert len(chunks) > 2 and chunks[-1].section == '3 Other'
    assert all(c.section == '2 Rules' for c in chunks[:-1])
    first_tail = chunks[0].text.split('. ')[-1]
    assert first_tail.rstrip('.') in chunks[1].text                       # overlap carries the tail sentence
    assert [c.id for c in chunk_blocks(blocks, 'doc.md', 40, 10)] == [c.id for c in chunks]


# ── Retrieval: BM25, fusion with dense vectors, reranking, re-embedding cache ─

class FakeEmbedder:
    """Deterministic 'semantic' vectors: words of one concept share a dimension."""
    name = 'fake'
    CONCEPTS = [('routine', 'function', 'functions', 'lines', 'long'), ('heap', 'malloc', 'memory', 'allocation'),
                ('exception', 'exceptions', 'try', 'catch', 'error', 'errors')]

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += len(texts)
        out = np.zeros((len(texts), len(self.CONCEPTS) + 1), dtype='float32')
        for i, t in enumerate(texts):
            words = t.lower().replace('.', ' ').split()
            for d, concept in enumerate(self.CONCEPTS):
                out[i, d] = sum(w in concept for w in words)
            out[i, -1] = 0.01
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def _blocks():
    return [Block('4 Size'), Block('Functions shall not exceed 25 lines.'), Block(''),
            Block('5 Memory'), Block('Heap allocation is prohibited.'), Block(''),
            Block('6 Errors'), Block('Errors are reported by return codes; C++ exception handling is not used.')]


def test_lexical_retrieval_expands_domain_terms():
    r = HybridRetriever(chunk_blocks(_blocks(), 'std.md'))
    assert r.mode.startswith('BM25 only')
    assert r.search('can I call malloc?')[0].chunk.section == '5 Memory'
    assert r.search('is try/catch ok')[0].chunk.section == '6 Errors'


def test_dense_side_finds_what_lexical_misses_and_cache_reembeds_only_changes(tmp_path):
    emb = FakeEmbedder()
    r = HybridRetriever(chunk_blocks(_blocks(), 'std.md'), emb, cache_dir=str(tmp_path), collection='s')
    assert r.mode.startswith('BM25 + dense') and r.embedded_now == 3
    hit = r.search('how long may a routine be', k=1)[0]
    assert hit.chunk.section == '4 Size' and hit.lexical_rank is None and hit.dense_rank == 0
    again = HybridRetriever(chunk_blocks(_blocks(), 'std.md'), emb, cache_dir=str(tmp_path), collection='s')
    assert again.embedded_now == 0                                        # every vector from the cache
    changed = _blocks()[:-1] + [Block('Exceptions are prohibited.')]
    edited = HybridRetriever(chunk_blocks(changed, 'std.md'), emb, cache_dir=str(tmp_path), collection='s')
    assert edited.embedded_now == 1                                       # only the edited chunk


def test_a_flat_reranker_cannot_overturn_agreeing_retrievers():
    class Flat:
        name = 'flat'

        def score(self, query, texts):
            return [-11.0 - 0.001 * i for i in range(len(texts))][::-1]   # nearly flat, favouring the last
    r = HybridRetriever(chunk_blocks(_blocks(), 'std.md'), FakeEmbedder(), Flat())
    assert r.search('try catch exception handling', k=1)[0].chunk.section == '6 Errors'


# ── Rule extraction from free text ───────────────────────────────────────────

@pytest.mark.parametrize('sentence, rule, value', [
    ('Cyclomatic complexity shall not exceed 15.', 'complexity.cyclomatic_complexity_max', 15),
    ('McCabe complexity must be at most 9 for every routine.', 'complexity.cyclomatic_complexity_max', 9),
    ('No function body shall be longer than 80 lines.', 'complexity.function_length_max', 80),
    ('Subprograms shall have fewer than 5 arguments.', 'complexity.parameter_count_max', 4),
    ('Nesting depth: maximum 3.', 'complexity.nesting_depth_max', 3),
    ('Use of goto is prohibited.', 'prohibited.goto', 'FORBIDDEN'),
    ('Heap memory shall not be allocated after start-up.', 'prohibited.dynamic_memory', 'FORBIDDEN'),
    ('Recursion is not permitted without a documented deviation.', 'prohibited.recursion', 'RESTRICTED'),
    ('Exceptions should not be thrown across module boundaries.', 'prohibited.exceptions', 'RESTRICTED'),
    ('dynamic_cast may be used for interface queries.', 'prohibited.rtti', 'ALLOWED'),
    ('Class names shall be in PascalCase.', 'naming.class', 'PascalCase'),
    ('Function identifiers must use snake_case.', 'naming.function', 'snake_case'),
])
def test_clause_to_rule(sentence, rule, value):
    sources, _ = extract_rules([(sentence, 3, '2 Rules')], 'std.pdf')
    assert sources[rule][0].value == value and sources[rule][0].citation == 'std.pdf, p.3, § 2 Rules'


def test_dal_scope_from_sentence_and_section():
    sources, _ = extract_rules([('For DAL A and B, complexity shall not exceed 6.', None, ''),
                                ('Functions shall not exceed 30 lines.', None, 'Level C requirements')], 'x.md')
    assert sources['complexity.cyclomatic_complexity_max'][0].dals == ('A', 'B')
    assert sources['complexity.function_length_max'][0].dals == ('C',)


def test_free_text_standard_extracts_rules_and_lists_unchecked_clauses(standard):
    doc = load_standards_document(str(standard))
    assert doc.profile['dal_specific_standards']['A']['complexity']['cyclomatic_complexity_max'] == 8
    assert doc.profile['dal_specific_standards']['B']['complexity']['cyclomatic_complexity_max'] == 12
    assert doc.profile['naming_conventions']['function']['convention'] == 'camelCase'
    assert [t for t, _ in doc.unchecked] == ['All interrupts shall be disabled during flash writes.']


def test_table_format_gives_the_same_limits_as_before_with_citations():
    pdf = ROOT / 'samples' / 'coding_standards_strict.pdf'
    old = load_raw_standards_profile(pdf)['dal_specific_standards']
    doc = load_standards_document(str(pdf))
    for dal in 'ABC':
        assert doc.profile['dal_specific_standards'][dal]['complexity'] == old[dal]['complexity']
    assert doc.citation_for('complexity.function_length_max', 'B').clause.startswith('SCS-M-02')


# ── The validator enforces the uploaded standard ─────────────────────────────

def _validate(code, standard, dal='B'):
    cfg = Config(standards_file=str(standard) if standard else '', dal_level=dal)
    return StandardsValidator(cfg, CodeParser().parse_directory(str(code))).analyze()


def test_uploaded_standard_changes_what_is_flagged_and_cites_it(code, standard):
    without = {v.rule for v in _validate(code, None).violations}
    report = _validate(code, standard)
    rules = {v.rule for v in report.violations}
    assert 'DO178-FUNC-LEN' not in without and 'DO178-FUNC-LEN' in rules          # 25-line limit from the document
    length = next(v for v in report.violations if v.rule == 'DO178-FUNC-LEN')
    assert length.standard_ref == 'standard.md, § 3 Complexity: "Functions shall not exceed 25 executable lines."'
    assert 'DO178-RTTI' in without and 'DO178-RTTI' not in rules                  # the document allows RTTI
    assert 'DO178-NAMING' in without and 'DO178-NAMING' not in rules              # camelCase functions are fine
    assert any(v.rule == 'DO178-CLASS-NAMING' and v.element == 'base_shape' for v in report.violations)
    table = {r['rule']: r for r in report.rule_table}
    assert table['complexity.function_length_max']['origin'] == 'document'
    assert table['complexity.function_length_max']['value'] == 25


def test_rules_the_document_does_not_set_are_labelled_defaults(code, tmp_path):
    std = tmp_path / 'tiny.txt'
    std.write_text('Functions shall not exceed 25 lines.\n')
    report = _validate(code, std)
    table = {r['rule']: r for r in report.rule_table}
    assert table['prohibited.rtti']['origin'] == 'default'
    rtti = next(v for v in report.violations if v.rule == 'DO178-RTTI')
    assert rtti.standard_ref == 'built-in default (not set by the standards document)'


# ── Pipeline, report and assistant ───────────────────────────────────────────

def test_unreadable_standard_stops_the_run(code, tmp_path):
    bad = tmp_path / 'std.xlsx'
    bad.write_text('x')
    with pytest.raises(InputError, match='Coding standards'):
        run_pipeline(Config(standards_file=str(bad), cache_dir=str(tmp_path / 'c')), str(code),
                     use_semantic_search=False, write_reports=False)


@pytest.fixture
def analysed(code, standard, tmp_path):
    cfg = Config(standards_file=str(standard), output_dir=str(tmp_path / 'out'), cache_dir=str(tmp_path / 'c'), dal_level='A')
    return run_pipeline(cfg, str(code), output_format='markdown', use_semantic_search=False), tmp_path / 'out'


def test_report_shows_rules_in_force_and_clause_references(analysed):
    text = (analysed[1] / 'standards_report.md').read_text()
    assert '## Rules in Force (DAL A)' in text
    assert '| Maximum cyclomatic complexity | 8 | document | yes | standard.md, § 3 Complexity: “For DAL A software' in text
    assert '## Normative Clauses Not Checked Automatically' in text and 'interrupts shall be disabled' in text
    assert '| Standard reference |' in text


@pytest.mark.parametrize('question, expected', [
    ('What is the maximum function length?', ['25', 'Functions shall not exceed 25 executable lines.']),
    ('What is the cyclomatic complexity limit?', ['8', 'For DAL A software']),
    ('Is recursion allowed?', ['RESTRICTED', 'unless a bound on depth']),
    ('Which rules come from the document?', ['set by the document', 'built-in default']),
    ('Which clauses are not checked automatically?', ['interrupts shall be disabled']),
    ('What does the standard say about flash writes?', ['standard.md, § 6 Documentation', 'interrupts']),
])
def test_assistant_answers_from_the_uploaded_standard(analysed, question, expected):
    answer = answer_question(question, analysed[0])
    for text in expected:
        assert text in answer, answer


def test_violation_questions_are_not_taken_over(analysed):
    assert 'violation' in answer_question('How many critical violations are there?', analysed[0]).lower()


# ── The real neural models (skipped when torch / sentence-transformers / the models are unavailable) ──

def test_real_embedder_and_cross_encoder(tmp_path):
    pytest.importorskip('sentence_transformers')
    from rag_engine.retrieval.models import load_embedder, load_reranker
    emb = load_embedder('all-MiniLM-L6-v2')
    if emb is None:
        pytest.skip('embedding model not available offline')
    r = HybridRetriever(chunk_blocks(_blocks(), 'std.md'), emb, load_reranker('cross-encoder/ms-marco-MiniLM-L-6-v2'),
                        cache_dir=str(tmp_path), collection='real')
    assert r.search('how long may a routine be', k=1)[0].chunk.section == '4 Size'
    assert r.search('can I use malloc', k=1)[0].chunk.section == '5 Memory'
