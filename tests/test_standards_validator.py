from pathlib import Path
import yaml
from docx import Document
from reportlab.pdfgen import canvas
from rag_engine.core.parser import CodeParser
from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.config import Config
from rag_engine.features.standards_validator import StandardsValidator
from rag_engine.knowledge_base.standards_profile import load_raw_standards_profile

FIXTURES = Path(__file__).parent / 'fixtures' / 'cpp'


def _validator():
    return StandardsValidator(Config(), CodeParser().parse_directory(str(FIXTURES / 'base_build')))


def test_analyze_returns_report():
    assert _validator().analyze() is not None


def test_violations_are_violation_objects():
    from rag_engine.models import Violation
    for v in _validator().analyze().violations:
        assert isinstance(v, Violation)


def test_compliance_score_range():
    score = _validator().analyze().compliance_score
    assert 0 <= score <= 100


def test_goto_flagged(tmp_path):
    cpp = tmp_path / 'bad.cpp'
    cpp.write_text('void badFunc() {\n    goto end;\n    end: return;\n}\n')
    v = StandardsValidator(Config(), CodeParser().parse_directory(str(tmp_path)))
    rules = {x.rule for x in v.analyze().violations}
    assert 'DO178-GOTO' in rules


def test_dynamic_memory_flagged(tmp_path):
    cpp = tmp_path / 'dynmem.cpp'
    cpp.write_text('void allocFunc() {\n    int* p = new int(5);\n    delete p;\n}\n')
    v = StandardsValidator(Config(), CodeParser().parse_directory(str(tmp_path)))
    rules = {x.rule for x in v.analyze().violations}
    assert 'DO178-DYNAMIC-MEM' in rules


def test_high_complexity_flagged(tmp_path):
    # CC = 26 exceeds every DAL profile's limit (A 10, B 15, C 20), so the default Config() DAL does not matter.
    conditions = '\n'.join(f'    if (x == {i}) {{ x++; }}' for i in range(25))
    cpp = tmp_path / 'complex.cpp'
    cpp.write_text(f'void complexFunc(int x) {{\n{conditions}\n}}\n')
    v = StandardsValidator(Config(), CodeParser().parse_directory(str(tmp_path)))
    rules = {x.rule for x in v.analyze().violations}
    assert 'DO178-CC' in rules


def _with_call_graph(path):
    results = CodeParser().parse_directory(str(path))
    CallGraphBuilder(results).build()  # fills fn.calls, as the pipeline does before validation
    return results


def test_recursion_flagged(tmp_path):
    cpp = tmp_path / 'rec.cpp'
    cpp.write_text('int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }\n')
    v = StandardsValidator(Config(), _with_call_graph(tmp_path))
    flagged = {x.element for x in v.analyze().violations if x.rule == 'DO178-RECURSION'}
    assert flagged == {'fact'}


def test_non_recursive_call_not_flagged_as_recursion(tmp_path):
    cpp = tmp_path / 'norec.cpp'
    cpp.write_text('int Twice(int n) { return n * 2; }\nint Quad(int n) { return Twice(Twice(n)); }\n')
    v = StandardsValidator(Config(), _with_call_graph(tmp_path))
    assert 'DO178-RECURSION' not in {x.rule for x in v.analyze().violations}


def test_naming_checks_short_name_and_skips_special_members(tmp_path):
    cpp = tmp_path / 'naming.cpp'
    cpp.write_text(
        'class Ctl {\n'
        'public:\n'
        '    Ctl() {}\n'
        '    ~Ctl() {}\n'
        '    bool operator==(const Ctl& o) const { return true; }\n'
        '    void Run() {}\n'
        '    void run() {}\n'
        '};\n'
    )
    v = StandardsValidator(Config(), CodeParser().parse_directory(str(tmp_path)))
    flagged = {x.element for x in v.analyze().violations if x.rule == 'DO178-NAMING'}
    assert flagged == {'Ctl::run'}


def test_custom_dal_profile_overrides_defaults(tmp_path):
    standards = {
        'universal_standards': {
            'prohibited_constructs': {'goto': 'FORBIDDEN'},
            'documentation': {'missing_docstring': 'CRITICAL'},
        },
        'dal_specific_standards': {
            'B': {
                'complexity': {
                    'cyclomatic_complexity_max': 1,
                    'function_length_max': 1,
                    'nesting_depth_max': 1,
                    'parameter_count_max': 0,
                },
                'severities': {
                    'prohibited_constructs': {
                        'dynamic_memory': 'RESTRICTED',
                        'exceptions': 'RESTRICTED',
                        'recursion': 'FORBIDDEN',
                        'rtti': 'FORBIDDEN',
                    },
                },
            },
        },
    }
    profile = tmp_path / 'custom_standards.yaml'
    profile.write_text(yaml.safe_dump(standards))
    cpp = tmp_path / 'bad.cpp'
    cpp.write_text(
        'void BadFunc(int value) {\n'
        '    int* p = new int(value);\n'
        '    if (value > 0) { value++; }\n'
        '    if (value > 1) { value++; }\n'
        '}\n'
    )
    cfg = Config(dal_level='B', standards_file=str(profile))
    v = StandardsValidator(cfg, CodeParser().parse_directory(str(tmp_path)))
    rules = {x.rule for x in v.analyze().violations}
    assert 'DO178-DYNAMIC-MEM' in rules
    assert 'DO178-NO-DOCSTRING' in rules
    assert 'DO178-PARAM-COUNT' in rules


def test_pdf_and_docx_standards_are_ingested(tmp_path):
    lines = [
        'SCS-L-01 Dynamic Memory new delete malloc free FORBIDDEN FORBIDDEN Restricted',
        'SCS-L-02 goto Statement FORBIDDEN FORBIDDEN FORBIDDEN',
        'SCS-L-03 Exceptions try catch throw FORBIDDEN FORBIDDEN Restricted',
        'SCS-L-04 Recursion FORBIDDEN Restricted Allowed',
        'SCS-L-05 RTTI dynamic_cast typeid FORBIDDEN FORBIDDEN Restricted',
        'SCS-M-01 Cyclomatic Complexity (McCabe) ≤ 10 ≤ 15 ≤ 20',
        'SCS-M-02 Function Length (executable lines of code) ≤ 50 ≤ 75 ≤ 100',
        'SCS-M-03 Control Flow Nesting Depth ≤ 3 ≤ 4 ≤ 5',
        'SCS-M-04 Number of Function Parameters ≤ 4 ≤ 5 ≤ 6',
    ]

    docx_path = tmp_path / 'standards.docx'
    doc = Document()
    for line in lines:
        doc.add_paragraph(line)
    doc.save(str(docx_path))

    pdf_path = tmp_path / 'standards.pdf'
    pdf = canvas.Canvas(str(pdf_path))
    y = 800
    for line in lines:
        pdf.drawString(40, y, line)
        y -= 16
    pdf.save()

    for path in (docx_path, pdf_path):
        profile = load_raw_standards_profile(path)
        assert profile['dal_specific_standards']['A']['complexity']['cyclomatic_complexity_max'] == 10
        assert profile['dal_specific_standards']['B']['complexity']['cyclomatic_complexity_max'] == 15
        assert profile['dal_specific_standards']['C']['complexity']['cyclomatic_complexity_max'] == 20
        assert profile['dal_specific_standards']['B']['severities']['prohibited_constructs']['recursion'] == 'RESTRICTED'
        assert profile['naming_conventions']['function']['convention'] == 'PascalCase'
