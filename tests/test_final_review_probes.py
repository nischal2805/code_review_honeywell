"""Regression tests for the whole-branch review probes (synthetic C++ that reproduced each finding)."""
from __future__ import annotations

import os
import time

import pytest
from loguru import logger

from rag_engine.core.graph_builder import CallGraphBuilder
from rag_engine.core.parser import CodeParser
from rag_engine.features.dead_code_detector import DeadCodeDetector
from rag_engine.features.function_effect import FunctionEffectAnalyzer, file_scope_variables
from rag_engine.models import _normalise_type


def _build(tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    results = CodeParser().parse_directory(str(tmp_path))
    gb = CallGraphBuilder(results)
    gb.build()
    return gb, results


def _callees(gb, caller):
    return set(gb.graph.successors(caller))


# ── 1. calls on anything but a plain variable, and ::f() ─────────────────────────

P1_H = """\
#ifndef APP_H
#define APP_H
class Display {
public:
    void update(float v);
    void clear();
};
class Mgr {
public:
    static Mgr& instance();
    void compute();
    Display& display();
    int value() const;
private:
    Display disp_;
};
class Holder {
public:
    Holder();
    int x_;
    int y_ = initialY();
    static int initialY();
};
int seedValue();
int computeX();
int tableHelper();
#endif
"""

P1_CPP = """\
#include "app.h"
static Mgr g_mgr;
static int g_seed = seedValue();
int seedValue() { return 42; }
int computeX() { return 7; }
int Holder::initialY() { return 3; }
Holder::Holder() : x_(computeX()) {}
Mgr& Mgr::instance() { return g_mgr; }
void Mgr::compute() { disp_.clear(); }
Display& Mgr::display() { return disp_; }
int Mgr::value() const { return 1; }
void Display::update(float v) { (void)v; }
void Display::clear() {}
int main() {
    Holder h;
    Mgr::instance().compute();
    Mgr::instance().display().update(1.0f);
    int v = Mgr::instance().value();
    return v + h.x_;
}
"""


def test_calls_on_call_results_get_edges(tmp_path):
    gb, _ = _build(tmp_path, {'app.h': P1_H, 'app.cpp': P1_CPP})
    assert _callees(gb, 'main()') >= {'Mgr::instance()', 'Mgr::compute()', 'Mgr::display()',
                                      'Display::update(float)', 'Mgr::value()'}
    assert gb.graph['main()']['Mgr::compute()']['call_type'] == 'unresolved'


def test_p1_nothing_reachable_is_reported_dead(tmp_path):
    gb, results = _build(tmp_path, {'app.h': P1_H, 'app.cpp': P1_CPP})
    flagged = {i.name for i in DeadCodeDetector(gb, results).analyze().items if i.symbol_type in ('function', 'method')}
    assert flagged == set()


CHAINS = """\
class Leaf { public: void ping(); void pong(); void pang(); void pung(); void peng(); };
class Node { public: Leaf* leaf(); Leaf& ref(); Leaf grid[2][2]; };
Node* getNode();
void Leaf::ping() {}
void Leaf::pong() {}
void Leaf::pang() {}
void Leaf::pung() {}
void Leaf::peng() {}
void forms(Node* p, Node** arr, void* raw) {
    getNode()->leaf()->ping();
    (*p).ref().pong();
    arr[0]->leaf()->pang();
    static_cast<Leaf*>(raw)->pung();
    p->grid[1][0].peng();
}
"""


def test_every_member_call_form_fans_out_when_the_receiver_is_not_a_plain_variable(tmp_path):
    gb, _ = _build(tmp_path, {'c.cpp': CHAINS})
    assert _callees(gb, 'forms(Node *, Node * *, void *)') == {
        'Leaf::ping()', 'Leaf::pong()', 'Leaf::pang()', 'Leaf::pung()', 'Leaf::peng()'}


GLOBAL_SCOPE = """\
void globalHelper() {}
void run() {}
namespace hw { void wrapped() { ::globalHelper(); } }
class Port { public: void run(); void go(); };
void Port::run() {}
void Port::go() { ::run(); }
int main() { hw::wrapped(); Port p; p.go(); return 0; }
"""


def test_global_scope_call_resolves_to_the_free_function(tmp_path):
    gb, _ = _build(tmp_path, {'g.cpp': GLOBAL_SCOPE})
    assert 'globalHelper()' in _callees(gb, 'hw::wrapped()')
    assert _callees(gb, 'Port::go()') == {'run()'}      # not the member Port::run


def test_chained_calls_on_a_member_object_reach_the_final_method(tmp_path):
    src = """\
class Port { public: void write(int v); Port& self(); };
void Port::write(int v) { (void)v; }
Port& Port::self() { return *this; }
struct Board { Port port; };
static Board g_board;
void tick() { Board* b = &g_board; b->port.self().write(3); }
int main() { tick(); return 0; }
"""
    gb, _ = _build(tmp_path, {'g.cpp': src})
    assert _callees(gb, 'tick()') >= {'Port::self()', 'Port::write(int)'}


# ── 2. file-scope object built from named constants ──────────────────────────────

UART = """\
#define UART1_BASE 0x4000
class Uart {
public:
    explicit Uart(unsigned base);
    ~Uart();
    void send(int x);
    void flush();
private:
    unsigned base_;
};
Uart::Uart(unsigned base) : base_(base) {}
Uart::~Uart() { flush(); }
void Uart::send(int x) { (void)x; }
void Uart::flush() {}
static Uart g_uart(UART1_BASE);
static int g_count = 0;
void tick() { g_uart.send(1); g_count++; }
int main() { tick(); return g_count; }
"""


def test_file_scope_object_with_named_constant_arguments_is_instantiated(tmp_path):
    gb, results = _build(tmp_path, {'u.cpp': UART})
    assert gb.instantiated_classes == {'Uart'}
    assert {'Uart::Uart(unsigned)', 'Uart::~Uart()'} <= gb.implicit_entry_points()
    flagged = {i.name for i in DeadCodeDetector(gb, results).analyze().items if i.symbol_type in ('function', 'method')}
    assert flagged == set()


def test_file_scope_object_with_named_constant_arguments_is_a_global_variable(tmp_path):
    gb, results = _build(tmp_path, {'u.cpp': UART})
    assert file_scope_variables(results)['g_uart'].type_ == 'Uart'
    effect = FunctionEffectAnalyzer(results, gb).analyze('tick')[0]
    assert {v.name: v.kind for v in effect.reads}['g_uart'] == 'global'


def test_prototype_returning_a_class_is_not_mistaken_for_an_object(tmp_path):
    src = """\
class Config {};
class Uart {};
Uart make(Config);
Uart make2(unsigned);
Uart build(UART_BASE_THAT_IS_NOT_A_TYPE);
"""
    (tmp_path / 'p.cpp').write_text(src)
    results = CodeParser().parse_directory(str(tmp_path))
    names = {n for n in file_scope_variables(results)}
    assert 'make' not in names and 'make2' not in names
    types = [t for r in results.values() for t, _ in r.file_scope_objects]
    assert types == ['Uart']        # only `build(...)` reads as an object


# ── 3. declaration / definition sig_key ──────────────────────────────────────────

P2_H = """\
class Frame {};
class Base {
public:
    virtual ~Base() {}
    virtual void handle(const Frame &f) = 0;
    virtual void fill(unsigned char buf[], int n) = 0;
};
class Impl : public Base {
public:
    void handle(const Frame &f) override;
    void fill(unsigned char buf[], int n) override;
    void helper(char const *msg);
    void add(void (*cb)(int value));
};
"""

P2_CPP = """\
#include "a.h"
void Impl::handle(Frame const &frame) { helper("x"); }
void Impl::fill(unsigned char data[], int n) { (void)data; (void)n; }
void Impl::helper(const char *m) { (void)m; }
void Impl::add(void (*handler)(int)) { (void)handler; }
int main() {
    Impl impl;
    Base* b = &impl;
    Frame fr;
    unsigned char buf[4];
    b->handle(fr);
    b->fill(buf, 4);
    return 0;
}
"""


def test_declaration_and_definition_share_one_sig_key(tmp_path):
    gb, results = _build(tmp_path, {'a.h': P2_H, 'a.cpp': P2_CPP})
    defined = {fn.sig_key for r in results.values() for fn in r.functions}
    declared = {m.sig_key for r in results.values() for c in r.class_defs if c.name == 'Impl' for m in c.methods}
    assert declared <= defined | {'Impl::Impl()'}
    assert {'Impl::handle(const Frame &)', 'Impl::fill(unsigned char [], int)',
            'Impl::helper(const char *)', 'Impl::add(void (*)(int))'} <= defined


def test_virtual_dispatch_reaches_overrides_declared_with_different_spellings(tmp_path):
    gb, _ = _build(tmp_path, {'a.h': P2_H, 'a.cpp': P2_CPP})
    assert _callees(gb, 'main()') >= {'Impl::handle(const Frame &)', 'Impl::fill(unsigned char [], int)'}
    assert 'Impl::helper(const char *)' in _callees(gb, 'Impl::handle(const Frame &)')


@pytest.mark.parametrize('written, canonical', [
    ('Frame const &', 'const Frame &'),
    ('char const *', 'const char *'),
    ('char const * const', 'const char * const'),
    ('unsigned char const', 'const unsigned char'),
    ('const char * const', 'const char * const'),
    ('Foo * const *', 'Foo * const *'),
    ('std::vector<Foo*> const &', 'const std::vector<Foo *> &'),
    ('unsigned char []', 'unsigned char []'),
    ('int [8]', 'int []'),
    ('void (*)(int)', 'void (*)(int)'),
])
def test_normalise_type(written, canonical):
    assert _normalise_type(written) == canonical


def test_sig_key_follows_parameters_set_after_construction():
    from rag_engine.models import FunctionDef, Parameter
    fn = FunctionDef('f', 'x.cpp', 1, 'void', [], False, False, False, '', None, 1, 1, 0)
    assert fn.sig_key == 'f()'
    fn.parameters = [Parameter('a', 'int')]
    assert fn.sig_key == 'f(int)'
    fn.parameters[0].type_ = 'long'
    assert fn.sig_key == 'f(long)'
    fn.name = 'g'
    assert fn.sig_key == 'g(long)'


def test_sig_key_survives_pickling():
    import pickle
    from rag_engine.models import FunctionDef, Parameter
    fn = FunctionDef('f', 'x.cpp', 1, 'void', [Parameter('a', 'int')], False, False, False, '', None, 1, 1, 0)
    assert fn.sig_key == 'f(int)'
    clone = pickle.loads(pickle.dumps(fn))
    assert clone.sig_key == 'f(int)'
    clone.parameters = []
    assert clone.sig_key == 'f()'


# ── 4. #ifdef / #else alternative definitions ────────────────────────────────────

P8 = """\
void hwStart() {}
void simStart() {}
#ifdef TARGET_HW
void boot() { hwStart(); }
#else
void boot() { simStart(); }
#endif
int main() { boot(); return 0; }
"""

P5 = """\
void legacySetup() {}
void currentSetup() {}
void featureHelper() {}
#ifdef FEATURE_X
void featureX() { featureHelper(); }
#endif
#if 0
void setup() { legacySetup(); }
#else
void setup() { currentSetup(); }
#endif
int main() { setup(); return 0; }
"""


def test_ifdef_else_alternatives_all_contribute_calls(tmp_path):
    gb, results = _build(tmp_path, {'h.cpp': P8})
    assert _callees(gb, 'boot()') == {'hwStart()', 'simStart()'}
    boots = [fn for r in results.values() for fn in r.functions if fn.name == 'boot']
    assert len(boots) == 1 and boots[0].preproc_guard == 'TARGET_HW | !TARGET_HW'
    flagged = {i.name for i in DeadCodeDetector(gb, results).analyze().items if i.symbol_type in ('function', 'method')}
    assert 'simStart()' not in flagged and 'hwStart()' not in flagged


def test_if_zero_else_keeps_the_compiled_variant(tmp_path):
    gb, results = _build(tmp_path, {'v.cpp': P5})
    assert _callees(gb, 'setup()') == {'currentSetup()'}
    init = gb.get_function('setup()')
    assert init.preproc_guard is None and not init.never_built
    flagged = {i.name: i for i in DeadCodeDetector(gb, results).analyze().items if i.symbol_type in ('function', 'method')}
    assert 'currentSetup()' not in flagged
    assert flagged['legacySetup()'].category == 'dead_code'
    assert flagged['featureX()'].category == 'deactivated_code'


def test_alternative_bodies_report_their_own_call_lines(tmp_path):
    gb, _ = _build(tmp_path, {'h.cpp': P8})
    assert gb.graph['boot()']['hwStart()']['lines'] == [4]
    assert gb.graph['boot()']['simStart()']['lines'] == [6]


# ── 5. calls from initialisers ───────────────────────────────────────────────────

def test_constructor_initialiser_list_calls_get_edges(tmp_path):
    gb, _ = _build(tmp_path, {'app.h': P1_H, 'app.cpp': P1_CPP})
    assert 'computeX()' in _callees(gb, 'Holder::Holder()')


def test_default_member_initialiser_calls_belong_to_every_constructor(tmp_path):
    src = """\
class H { public: H(); H(int a); int y_ = initY(); static int initY(); int z_{initZ()}; static int initZ(); };
int H::initY() { return 1; }
int H::initZ() { return 1; }
H::H() {}
H::H(int a) { (void)a; }
int main() { H h; return 0; }
"""
    gb, _ = _build(tmp_path, {'h.cpp': src})
    for ctor in ('H::H()', 'H::H(int)'):
        assert {'H::initY()', 'H::initZ()'} <= _callees(gb, ctor)


def test_default_member_initialiser_of_a_class_without_constructor_is_an_implicit_entry_point(tmp_path):
    src = """\
int makeA() { return 1; }
class Plain { public: int a = makeA(); };
class Unused { public: int b = makeB(); };
int makeB() { return 2; }
int main() { Plain p; return p.a; }
"""
    gb, _ = _build(tmp_path, {'p.cpp': src})
    assert 'makeA()' in gb.implicit_entry_points()
    assert 'makeB()' not in gb.implicit_entry_points()      # Unused is never constructed


def test_file_scope_initialiser_calls_are_implicit_entry_points(tmp_path):
    src = """\
int seedValue() { return 1; }
int tableA() { return 1; }
int tableB() { return 2; }
int notCalled() { return 3; }
static int g_seed = seedValue();
static int g_table[] = { tableA(), tableB() };
int main() { return 0; }
"""
    gb, results = _build(tmp_path, {'s.cpp': src})
    assert gb.implicit_entry_points() == {'seedValue()', 'tableA()', 'tableB()'}
    flagged = {i.name for i in DeadCodeDetector(gb, results).analyze().items if i.symbol_type in ('function', 'method')}
    assert flagged == {'notCalled()'}


def test_file_scope_initialiser_under_if_zero_does_not_count(tmp_path):
    src = """\
int seedValue() { return 1; }
#if 0
static int g_seed = seedValue();
#endif
int main() { return 0; }
"""
    gb, _ = _build(tmp_path, {'s.cpp': src})
    assert gb.implicit_entry_points() == set()


# ── 7. return types keep * and & ─────────────────────────────────────────────────

def test_return_type_keeps_pointer_and_reference(tmp_path):
    src = """\
struct S {};
S* makePtr() { return 0; }
S& makeRef() { static S s; return s; }
char** argvOf() { return 0; }
int plain() { return 0; }
"""
    (tmp_path / 'r.cpp').write_text(src)
    fns = {fn.name: fn.return_type for r in CodeParser().parse_directory(str(tmp_path)).values() for fn in r.functions}
    assert fns == {'makePtr': 'S *', 'makeRef': 'S &', 'argvOf': 'char **', 'plain': 'int'}


# ── 8. a class that cannot be parsed is a warning ────────────────────────────────

def test_class_that_fails_to_parse_is_logged_as_warning(tmp_path, monkeypatch):
    parser = CodeParser()

    def boom(*args, **kwargs):
        raise RuntimeError('boom')

    monkeypatch.setattr(parser, '_class_def', boom)
    (tmp_path / 'c.cpp').write_text('class A { public: void f() {} };\nint main() { return 0; }\n')
    messages = []
    handle = logger.add(lambda m: messages.append(m.record['message']), level='WARNING')
    try:
        result = parser.parse_file(str(tmp_path / 'c.cpp'))
    finally:
        logger.remove(handle)
    assert any('boom' in m for m in messages)
    assert {fn.name for fn in result.functions} == {'A::f', 'main'}


# ── 9. local declaration styles ──────────────────────────────────────────────────

@pytest.mark.parametrize('decl', [
    'Base *b = nullptr;', 'Base * b = nullptr;', 'Base* b = nullptr;', 'Base &b = g_base;',
    'Base const &b = g_base;', 'const Base &b = g_base;', 'Base const *b = nullptr;', 'Base&b = g_base;',
])
def test_pointer_and_reference_locals_are_typed(tmp_path, decl):
    src = f"""\
class Base {{ public: virtual void run(); }};
class Other {{ public: void run(); }};
void Base::run() {{}}
void Other::run() {{}}
static Base g_base;
void use() {{ {decl} b->run(); }}
"""
    if '&' in decl:
        src = src.replace('b->run()', 'b.run()')
    gb, _ = _build(tmp_path, {'l.cpp': src})
    assert _callees(gb, 'use()') == {'Base::run()'}


# ── 10. #if 0 functions are not callers, data-coupled or affected ────────────────

def test_functions_under_if_zero_are_not_affected_components(tmp_path):
    src = """\
int g_total = 0;
class Gone { public: void m(); };
class Live { public: void m(); };
void helper() { g_total = 1; }
void Live::m() { helper(); }
#if 0
void Gone::m() { helper(); g_total++; }
#endif
int main() { Live l; l.m(); return g_total; }
"""
    gb, results = _build(tmp_path, {'z.cpp': src})
    effect = FunctionEffectAnalyzer(results, gb).analyze('helper')[0]
    assert effect.callers == ['Live::m()']
    assert effect.transitive_callers == ['Live::m()', 'main()']
    assert 'Gone::m()' not in effect.data_coupled_functions
    assert 'Gone' not in effect.affected_components


# ── 11. callbacks must be free functions named as a value ────────────────────────

P7 = """\
struct Cfg { int status; int mode; };
int status() { return 1; }
void mode() {}
void logv(int v) { (void)v; }
int main() { Cfg c{0, 0}; Cfg* p = &c; logv(c.status); logv(p->mode); return 0; }
"""


def test_member_and_scoped_names_as_arguments_are_not_callbacks(tmp_path):
    gb, _ = _build(tmp_path, {'c.cpp': P7})
    assert _callees(gb, 'main()') == {'logv(int)'}


def test_enumerator_and_scoped_name_arguments_are_not_callbacks(tmp_path):
    src = """\
enum class Mode { status, idle };
int status() { return 1; }
void logv(Mode m) { (void)m; }
int main() { logv(Mode::status); return 0; }
"""
    gb, _ = _build(tmp_path, {'c.cpp': src})
    assert _callees(gb, 'main()') == {'logv(Mode)'}


def test_global_scope_function_names_are_callbacks(tmp_path):
    src = """\
typedef void (*Fn)();
void cmdA() {}
void cmdB() {}
void reg(Fn f) { (void)f; }
void boot() { reg(&::cmdA); reg(::cmdB); }
"""
    gb, _ = _build(tmp_path, {'c.cpp': src})
    assert _callees(gb, 'boot()') == {'reg(Fn)', 'cmdA()', 'cmdB()'}


# ── 6. scale ─────────────────────────────────────────────────────────────────────

def _generate(directory, modules):
    for i in range(modules):
        base = f" : public Comp{i - 1}" if i % 10 and i > 0 else ""
        hdr = [f"#ifndef MOD{i}_H", f"#define MOD{i}_H"]
        if base:
            hdr.append(f'#include "mod{i - 1}.h"')
        hdr.append(f"class Comp{i}{base} {{\npublic:")
        hdr += [f"    virtual int op{m}(int a, float b);" if m < 2 else f"    int op{m}(int a, float b);" for m in range(10)]
        hdr.append(f"    float value_{i};\n    int count_{i};\n    Comp{(i + 1) % modules}* peer;\n}};")
        hdr += [f"int helper{i}_{f}(int x);" for f in range(10)]
        hdr.append("#endif")
        (directory / f"mod{i}.h").write_text("\n".join(hdr) + "\n")
        src = [f'#include "mod{i}.h"', f"static int g_state{i} = 0;"]
        for m in range(10):
            callee = f"helper{i}_{m % 5}(a)" if m % 3 else f"peer->op{(m + 1) % 10}(a, b)"
            src.append(f"int Comp{i}::op{m}(int a, float b) {{\n    if (a > {m}) {{\n        value_{i} = b;\n"
                       f"        count_{i}++;\n    }}\n    for (int k = 0; k < a; ++k) {{ g_state{i} += k; }}\n"
                       f"    return {callee} + count_{i};\n}}")
        for f in range(10):
            nxt = f"helper{(i + 1) % modules}_{f}(x - 1)" if f < 3 else "x"
            src.append(f"int helper{i}_{f}(int x) {{\n    int unused{f} = 3;\n    return {nxt};\n}}")
        if i == 0:
            src.append("int main() {\n    Comp0 c;\n    return c.op0(1, 2.0f) + helper0_0(3);\n}")
        (directory / f"mod{i}.cpp").write_text("\n".join(src) + "\n")


def test_call_graph_and_effect_analysis_scale(tmp_path):
    """300 files / 3000 functions; analyze took 7.9 s here before the per-variable index (0.8 s after); build ~1 s."""
    if os.environ.get('RAG_SKIP_SCALE'):
        pytest.skip('RAG_SKIP_SCALE set')
    _generate(tmp_path, 150)
    results = CodeParser().parse_directory(str(tmp_path))
    assert sum(len(r.functions) for r in results.values()) >= 3000
    start = time.perf_counter()
    gb = CallGraphBuilder(results)
    gb.build()
    built = time.perf_counter() - start
    start = time.perf_counter()
    effects = FunctionEffectAnalyzer(results, gb).analyze('op3')
    analyzed = time.perf_counter() - start
    assert len(effects) == 150
    assert built < 10, f"build took {built:.1f}s"
    assert analyzed < 5, f"analyze took {analyzed:.1f}s"
