from pathlib import Path

from rag_engine.core.parser import CodeParser
from rag_engine.features.control_flow_analyzer import find_dead_fragments, find_unused_variables

SENSORS = Path(__file__).parent / 'fixtures' / 'cpp' / 'sensors'


def _parse(tmp_path, source):
    path = tmp_path / 't.cpp'
    path.write_text(source)
    return CodeParser().parse_file(str(path))


def _fragments(tmp_path, source):
    return [(f.function, f.start_line, f.end_line, f.snippet) for f in find_dead_fragments(_parse(tmp_path, source))]


def _unused(tmp_path, source):
    return [(u.function, u.name, u.line, u.kind) for u in find_unused_variables(_parse(tmp_path, source))]


# ── dead fragments ─────────────────────────────────────────────────────────────

def test_fixture_fragments():
    result = CodeParser().parse_file(str(SENSORS / 'main.cpp'))
    frags = find_dead_fragments(result)
    assert [(f.function, f.start_line, f.end_line, f.snippet) for f in frags] == [
        ('computeChecksum(int)', 30, 30, 'x = 0;'),
        ('main()', 42, 44, 'if (0) {')]
    assert all(f.file_path.endswith('main.cpp') and f.reason for f in frags)
    assert 'return' in frags[0].reason and '0' in frags[1].reason


def test_statements_after_return_throw_continue_and_goto(tmp_path):
    src = ('int a() {\n  return 1;\n  int dead = 2;\n  dead++;\n}\n'
           'void b() {\n  throw 1;\n  b();\n}\n'
           'void c(int n) {\n  while (n) {\n    continue;\n    n--;\n  }\n}\n'
           'void d() {\n  goto end;\n  d();\nend:\n  d();\n}\n')
    assert _fragments(tmp_path, src) == [
        ('a()', 3, 4, 'int dead = 2;'),
        ('b()', 8, 8, 'b();'),
        ('c(int)', 13, 13, 'n--;'),
        ('d()', 18, 18, 'd();')]


def test_code_after_break_in_a_loop(tmp_path):
    src = 'void f(int n) {\n  for (int i = 0; i < n; i++) {\n    if (i == 3) { n = 0; }\n    break;\n    n++;\n  }\n}\n'
    assert _fragments(tmp_path, src) == [('f(int)', 5, 5, 'n++;')]


def test_a_case_label_ends_the_dead_region(tmp_path):
    src = ('int f(int a) {\n  switch (a) {\n    case 1:\n      return 1;\n      a++;\n'
           '    case 2:\n      a--;\n      break;\n      a = 9;\n    default:\n      a = 3;\n  }\n  return a;\n}\n')
    assert _fragments(tmp_path, src) == [('f(int)', 5, 5, 'a++;'), ('f(int)', 9, 9, 'a = 9;')]


def test_constant_false_conditions(tmp_path):
    src = ('void f(int x) {\n  if (false) {\n    x = 1;\n  } else {\n    x = 2;\n  }\n'
           '  while (0) {\n    x = 3;\n  }\n  if (x) { x = 4; }\n}\n')
    assert _fragments(tmp_path, src) == [('f(int)', 2, 4, 'if (false) {'), ('f(int)', 7, 9, 'while (0) {')]


def test_nested_dead_code_is_reported_once(tmp_path):
    src = 'void f(int x) {\n  return;\n  if (0) {\n    x = 1;\n  }\n}\n'
    assert _fragments(tmp_path, src) == [('f(int)', 3, 5, 'if (0) {')]


def test_functions_excluded_by_if_zero_have_no_fragments(tmp_path):
    src = 'void f() {\n  return;\n  f();\n}\n#if 0\nvoid g() {\n  return;\n  g();\n}\n#endif\n'
    assert _fragments(tmp_path, src) == [('f()', 3, 3, 'f();')]


def test_method_fragments_use_the_qualified_signature(tmp_path):
    src = 'class C {\npublic:\n  int m(int a) {\n    return a;\n    a = 1;\n  }\n};\n'
    assert _fragments(tmp_path, src) == [('C::m(int)', 5, 5, 'a = 1;')]


# ── unused variables ───────────────────────────────────────────────────────────

def test_fixture_unused_variables():
    got = []
    for name in ('main.cpp', 'sensor_manager.cpp', 'display.cpp'):
        got += [(u.function, u.name, u.line, u.kind)
                for u in find_unused_variables(CodeParser().parse_file(str(SENSORS / name)))]
    assert sorted(got) == [('Display::clear(int)', 'mode', 7, 'parameter'),
                           ('SensorManager::unusedDiagnostic()', 'code', 22, 'local'),
                           ('orphanHelper()', 'unused', 25, 'local')]


def test_void_cast_counts_as_a_use(tmp_path):
    src = 'void f(int a) {\n  int x = 1;\n  int y = 2;\n  (void)x;\n  (void)a;\n}\n'
    assert _unused(tmp_path, src) == [('f(int)', 'y', 3, 'local')]


def test_volatile_and_maybe_unused_are_skipped(tmp_path):
    src = ('void f([[maybe_unused]] int a, int b) {\n  volatile int v = 0;\n  [[maybe_unused]] int m = 1;\n'
           '  int plain = 2;\n}\n')
    assert [(name, line, kind) for _, name, line, kind in _unused(tmp_path, src)] == [
        ('b', 1, 'parameter'), ('plain', 4, 'local')]


def test_unnamed_parameters_and_bodyless_functions_are_skipped(tmp_path):
    src = 'void f(int, int named) {\n}\nclass C {\npublic:\n  C(int a) = default;\n  void g(int z) = delete;\n};\n'
    assert _unused(tmp_path, src) == [('f(int, int)', 'named', 1, 'parameter')]


def test_constructor_initializers_use_parameters(tmp_path):
    src = 'class C {\npublic:\n  C(int a, int b) : x(a) {}\n  int x;\n};\n'
    assert _unused(tmp_path, src) == [('C::C(int, int)', 'b', 3, 'parameter')]


def test_a_variable_used_later_or_in_a_for_header_is_used(tmp_path):
    src = ('int f(int n) {\n  int total = 0;\n  for (int i = 0; i < n; i++) {\n    total += i;\n  }\n'
           '  int unusedLoop = 0;\n  for (int j = 0; j < 2; j++) { }\n  return total;\n}\n')
    assert _unused(tmp_path, src) == [('f(int)', 'unusedLoop', 6, 'local')]


def test_class_typed_locals_are_skipped_because_their_constructor_may_matter(tmp_path):
    src = 'void f() {\n  Lock guard(mtx);\n  Widget w;\n  Widget* p = nullptr;\n  int n = 0;\n}\n'
    assert _unused(tmp_path, src) == [('f()', 'p', 4, 'local'), ('f()', 'n', 5, 'local')]


def test_functions_excluded_by_if_zero_have_no_unused_variables(tmp_path):
    src = '#if 0\nvoid g(int a) {\n  int x = 1;\n}\n#endif\n'
    assert _unused(tmp_path, src) == []


# ── fix round 1 ────────────────────────────────────────────────────────────────

def test_variables_declared_in_a_condition_are_used_by_the_condition(tmp_path):
    src = ('int* get();\nint next();\nvoid f() {\n  if (int* p = get()) { *p = 1; }\n'
           '  while (int n = next()) { n--; }\n  if (int q = next()) { }\n}\n')
    assert _unused(tmp_path, src) == []


def test_two_functions_on_one_line_keep_their_own_parameters(tmp_path):
    assert _unused(tmp_path, 'void a(int x) {} void b(int y) { y++; }\n') == [('a(int)', 'x', 1, 'parameter')]
    assert _unused(tmp_path, 'void a(int x) {} void b(int y) {}\n') == [
        ('a(int)', 'x', 1, 'parameter'), ('b(int)', 'y', 1, 'parameter')]
    assert _fragments(tmp_path, 'int a() { return 1; a(); } int b() { return 2; b(); }\n') == [
        ('a()', 1, 1, 'a();'), ('b()', 1, 1, 'b();')]


def test_fragments_and_unused_variables_carry_their_evidence(tmp_path):
    result = _parse(tmp_path, 'void f(int a) {\n  int x = g();\n  if (0) { m(); }\n  return;\n  h(); k();\n}\n')
    if_zero, after_return = find_dead_fragments(result)
    assert (after_return.jump, after_return.jump_line, after_return.calls) == ('return', 4, ['h', 'k'])
    assert (if_zero.jump, if_zero.condition, if_zero.calls) == ('', '0', ['m'])
    local = next(u for u in find_unused_variables(result) if u.name == 'x')
    assert local.side_effects == ['g()']
