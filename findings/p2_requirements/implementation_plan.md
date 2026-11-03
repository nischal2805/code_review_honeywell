# Honeywell P2 — Implementation Plan

Source requirements: "Honeywell P2 – New Requirements & Proposed Approach" (PDF, 16 pages)
plus `findings/dead_code/dead_code_analysis_implementation_plan.md`.

## Spec (condensed from the PDF — binding)

R1 Virtual Function Analysis — identify: which functions are virtual; which classes
override them; base → derived relationships (inheritance tree); where virtual functions
are called (caller + location); whether overrides are actually reachable/used; produce a
report. Example output shape:

```text
Virtual Function: Sensor::process()
Overridden by:    TemperatureSensor::process(), PressureSensor::process()
Inheritance:      Sensor
                    ├── TemperatureSensor
                    └── PressureSensor
Usage:            Called from SensorManager::update()
```

R2 Function Effect & Coupling — input: a method name; output: what it does, functions it
calls, variables it reads, variables it modifies, classes/modules it interacts with, what
other components could be affected if it changes. NO numerical coupling score. Example:
`SensorManager::updateTemperature()` → Reads sensor; Writes temperature; Calls sensor.read(),
validator.check(), display.update(); Coupled components Sensor, Validator, Display.

R3 Sensor Data Analysis — hierarchy Parameter Group → I/O Group → Sensor → Sensor Data.
Generic, configurable sensor template (YAML: name, type, data_type, unit, valid_range
min/max, sampling_interval_ms, stale.threshold_ms, debounce.enabled/threshold,
corrupt.range_check). Same engine runs for every sensor; adding a sensor = adding config.
Three checks: Stale (no fresh update / value frozen beyond threshold), Corrupt (outside the
per-sensor valid range), Debounce (changes beyond threshold are debounced/flagged, small
fluctuations accepted). Architecture: AST → sensor classes → sensor variables → data flow
(call graph: where data is obtained, processed, validated, consumed) → match template →
load rules → stale/corrupt/debounce checks → report.

R4 RAG layer — deterministic analysis results feed Q&A: "What are the virtual functions
implemented by TemperatureSensor?", "What components are affected if updateTemperature()
changes?", "What validation checks are applied to TemperatureSensor?" Static analysis gives
the facts; the local LLM (Ollama, optional) only explains them.

User asks additionally: proper results, proper tests, a UI for testing, improved dead code
analysis (see the dead-code plan: unique identities, entry-point config + validation,
no silent exclusion of virtuals/callbacks, fragments, unused variables, finding IDs,
reasons, confidence, removal impact, JSON output, disposition file).

## Global Constraints

- Python 3.10+, fully offline. No new runtime dependencies (available: tree-sitter,
  tree-sitter-cpp, networkx, pyyaml, jinja2, python-docx, loguru, typer, textual).
- Tests must NOT load sentence-transformers / FAISS models. Test command (fast, ~5 s):
  `python -m pytest -q -p no:cacheprovider --no-cov --ignore=tests/test_embeddings.py`
- Existing public interfaces stay backward compatible; new constructor params are optional
  keyword args with defaults: `CodeParser.parse_file/parse_directory`,
  `CallGraphBuilder(parse_results).build()/find_reachable_from/find_unreachable/get_function`,
  `VirtualAnalyzer(base, current).analyze()`, `DeadCodeDetector(gb, results, entry_points).analyze()`,
  `StandardsValidator(cfg, results).analyze()`, `CouplingAnalyzer(...)`.
- Function identity everywhere is `FunctionDef.sig_key` (`Class::method(type, ...)`).
- Code style: `from __future__ import annotations`, dataclasses, type hints, loguru logger,
  comments only for non-obvious "why". Follow the surrounding file's idiom.
- Heuristics with known limits get a one-line `# shortcut: <limit>, <when to upgrade>` comment.
- Each task ends with ONE commit (or a few) whose message ends with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- Shared test fixture: `tests/fixtures/cpp/sensors/` (created in Task 1) is the contract for
  all later tasks. Its exact content is in Task 1. Do not change it in later tasks unless the
  brief says so.

---

## Task 1: Sensor fixture project + parser class model and qualified names

**Goal:** the parser emits correct, class-qualified identities, a class model (bases,
method declarations incl. pure virtuals, fields), override/virtual flags, end lines, and
AST-based data reads/writes. Fix the three baseline test failures and two latent bugs.

### 1.1 Create the fixture project — exact content

`tests/fixtures/cpp/sensors/sensor.h`
```cpp
#ifndef SENSOR_H
#define SENSOR_H

// Base class for every sensor.
class Sensor {
public:
    virtual ~Sensor() {}
    // DO-178C-REQ: SRS-SNS-001
    virtual void process();
    virtual float read() = 0;
    virtual bool selfTest() { return true; }
    float getValue() const { return value; }

protected:
    float value = 0.0f;
    float lastValue = 0.0f;
    unsigned long lastUpdateMs = 0;
};

#endif
```

`tests/fixtures/cpp/sensors/sensor.cpp`
```cpp
#include "sensor.h"

void Sensor::process() {
    lastValue = value;
    value = read();
}
```

`tests/fixtures/cpp/sensors/temperature_sensor.h`
```cpp
#ifndef TEMPERATURE_SENSOR_H
#define TEMPERATURE_SENSOR_H
#include "sensor.h"

const unsigned long TEMP_STALE_MS = 500;
const float TEMP_MIN_C = -54.0f;
const float TEMP_MAX_C = 54.0f;
const float TEMP_DEBOUNCE_C = 2.0f;

class TemperatureSensor : public Sensor {
public:
    void process() override;
    float read() override;
    bool isStale(unsigned long nowMs) const;
    bool inRange(float v) const;
    float debounce(float v);
};

#endif
```

`tests/fixtures/cpp/sensors/temperature_sensor.cpp`
```cpp
#include <cmath>
#include "temperature_sensor.h"

float TemperatureSensor::read() {
    return 25.0f;
}

bool TemperatureSensor::isStale(unsigned long nowMs) const {
    return (nowMs - lastUpdateMs) > TEMP_STALE_MS;
}

bool TemperatureSensor::inRange(float v) const {
    return v >= TEMP_MIN_C && v <= TEMP_MAX_C;
}

float TemperatureSensor::debounce(float v) {
    if (std::fabs(v - lastValue) < TEMP_DEBOUNCE_C) {
        return lastValue;
    }
    return v;
}

void TemperatureSensor::process() {
    float raw = read();
    if (!inRange(raw)) {
        return;
    }
    if (isStale(lastUpdateMs + 100)) {
        return;
    }
    lastValue = value;
    value = debounce(raw);
    lastUpdateMs = lastUpdateMs + 100;
}
```

`tests/fixtures/cpp/sensors/pressure_sensor.h`
```cpp
#ifndef PRESSURE_SENSOR_H
#define PRESSURE_SENSOR_H
#include "sensor.h"

class PressureSensor : public Sensor {
public:
    void process() override;
    float read() override;
};

#endif
```

`tests/fixtures/cpp/sensors/pressure_sensor.cpp`
```cpp
#include "pressure_sensor.h"

float PressureSensor::read() {
    return 101325.0f;
}

void PressureSensor::process() {
    float raw = read();
    if (raw < 0.0f || raw > 120000.0f) {
        return;
    }
    value = raw;
}
```

`tests/fixtures/cpp/sensors/altitude_sensor.h`
```cpp
#ifndef ALTITUDE_SENSOR_H
#define ALTITUDE_SENSOR_H
#include "sensor.h"

// Not instantiated anywhere in this build.
class AltitudeSensor : public Sensor {
public:
    float read() override { return 1500.0f; }
    bool selfTest() override { return false; }
};

#endif
```

`tests/fixtures/cpp/sensors/validator.h`
```cpp
#ifndef VALIDATOR_H
#define VALIDATOR_H

class Validator {
public:
    bool check(float v);
};

#endif
```

`tests/fixtures/cpp/sensors/validator.cpp`
```cpp
#include "validator.h"

bool Validator::check(float v) {
    return v > -100.0f && v < 100.0f;
}
```

`tests/fixtures/cpp/sensors/display.h`
```cpp
#ifndef DISPLAY_H
#define DISPLAY_H

class Display {
public:
    void update(float v);
    void clear(int mode);

private:
    float shown = 0.0f;
};

#endif
```

`tests/fixtures/cpp/sensors/display.cpp`
```cpp
#include "display.h"

void Display::update(float v) {
    shown = v;
}

void Display::clear(int mode) {
    shown = 0.0f;
}
```

`tests/fixtures/cpp/sensors/sensor_manager.h`
```cpp
#ifndef SENSOR_MANAGER_H
#define SENSOR_MANAGER_H
#include "temperature_sensor.h"
#include "pressure_sensor.h"
#include "validator.h"
#include "display.h"

class SensorManager {
public:
    void init();
    void update();
    void updateTemperature();
    void unusedDiagnostic();

private:
    TemperatureSensor sensor;
    PressureSensor pressureSensor;
    Validator validator;
    Display display;
    float temperature = 0.0f;
    Sensor* sensors[2];
};

#endif
```

`tests/fixtures/cpp/sensors/sensor_manager.cpp`
```cpp
#include "sensor_manager.h"

void SensorManager::init() {
    sensors[0] = &sensor;
    sensors[1] = &pressureSensor;
}

void SensorManager::update() {
    for (int i = 0; i < 2; i++) {
        Sensor* s = sensors[i];
        s->process();
    }
}

void SensorManager::updateTemperature() {
    temperature = sensor.read();
    validator.check(temperature);
    display.update(temperature);
}

void SensorManager::unusedDiagnostic() {
    int code = 0;
    display.update(-1.0f);
}
```

`tests/fixtures/cpp/sensors/main.cpp`
```cpp
#include "sensor_manager.h"

static int cycleCount = 0;

typedef void (*Callback)();

void registerCallback(Callback cb) {
    cb();
}

void onTimerTick() {
    cycleCount++;
}

// Interrupt handler: invoked by hardware, never called from code.
void TIMER0_IRQHandler() {
    cycleCount = 0;
}

void legacyCalibrate() {
    cycleCount = -1;
}

void orphanHelper() {
    int unused = 5;
}

int computeChecksum(int x) {
    return x * 31;
    x = 0;
}

int main() {
    SensorManager mgr;
    mgr.init();
    registerCallback(onTimerTick);
    while (cycleCount < 10) {
        mgr.update();
        mgr.updateTemperature();
        cycleCount++;
    }
    if (0) {
        legacyCalibrate();
    }
    return computeChecksum(cycleCount);
}

#if 0
void disabledFeature() {
    cycleCount = 99;
}
#endif
```

### 1.2 Model changes (`rag_engine/models.py`)

Add (all new FunctionDef fields have defaults so existing constructors keep working):

```python
@dataclass
class FieldDef:
    name: str
    type_: str            # as written, e.g. "TemperatureSensor", "Sensor *", "float"
    line_number: int
    is_static: bool = False

@dataclass
class MethodDecl:
    name: str             # short name, e.g. "process", "~Sensor"
    sig_key: str          # qualified, e.g. "Sensor::process()" (same normalisation as FunctionDef.sig_key)
    line_number: int
    is_virtual: bool      # 'virtual' keyword OR override/final specifier
    is_pure: bool         # declared "= 0"
    is_override: bool
    is_final: bool
    has_body: bool        # defined inline inside the class body

@dataclass
class ClassDef:
    name: str             # qualified with enclosing namespaces/classes, e.g. "hw::Sensor"
    file_path: str
    line_number: int
    end_line_number: int
    bases: List[str]      # base names as written minus access/virtual keywords, e.g. ["Sensor"]
    methods: List[MethodDecl] = field(default_factory=list)
    fields: List[FieldDef] = field(default_factory=list)
    is_struct: bool = False
```

`FunctionDef` new fields: `class_name: Optional[str] = None`, `end_line_number: int = 0`,
`body_start_line: int = 0` (line of the body's `{`), `is_override: bool = False`,
`preproc_guard: Optional[str] = None` (condition text of the nearest enclosing `#if`,
`#ifdef` or `#elif` — NOT `#ifndef`, so include guards are ignored; e.g. `"0"` for `#if 0`).

`ParseResult` new field: `class_defs: List[ClassDef] = field(default_factory=list)`
(keep `classes: List[str]` populated as today).

### 1.3 Parser changes (`rag_engine/core/parser.py`)

- In-class method definitions are qualified with the enclosing class chain and namespaces:
  the inline `float read() override { ... }` inside `class AltitudeSensor` becomes
  `AltitudeSensor::read`, sig_key `AltitudeSensor::read()`. Destructor inside class
  `Sensor` → `Sensor::~Sensor`. Out-of-line definitions inside a `namespace` block get the
  namespace prefix too (`namespace hw { void Sensor::process() {} }` → `hw::Sensor::process`).
  This removes the collision where two classes' inline `read()` collapsed into one.
- `class_name`: enclosing class for in-class definitions; for out-of-line `X::m` set it to
  the qualifier `X` (Task 2 corrects it to None when `X` is a namespace, not a class).
- `is_virtual` is True for `virtual` keyword OR an `override`/`final` virtual_specifier
  (note: tree-sitter puts `virtual_specifier` inside the `function_declarator`, which is why
  `void process() override {}` is currently reported non-virtual). `is_override` True for
  `override`.
- `end_line_number` = node end line (1-based); `body_start_line` = start line of the
  compound_statement.
- `preproc_guard` as defined above.
- `class_defs`: one `ClassDef` per `class_specifier`/`struct_specifier` with a body. Bases
  from `base_class_clause`. Methods from `field_declaration`s with a function declarator
  (declarations, incl. pure `= 0`) and from inline `function_definition`s. Fields from
  non-function `field_declaration`s (arrays like `Sensor* sensors[2]` → name `sensors`,
  type `Sensor *`).
- `data_reads` / `data_writes` computed from the function body AST (not regex):
  - writes: left operand of `assignment_expression` (incl. compound `+=` etc.) and operand
    of `update_expression` (`++`/`--`). Take the base name: identifier → itself;
    `this->x` → `x`; `obj.f` / `obj->f` → `obj`; `arr[i]` → `arr`; `*p` → `p`.
  - reads: every other identifier use in the body. Compound assignment and `++/--` count as
    read AND write. A call receiver counts as a read (`sensor.read()` reads `sensor`); the
    callee name itself does not (`foo()` does not read `foo`). `this->x` reads `x`.
  - Exclude variables declared inside the body (locals, for-loop vars, catch params).
    Parameters ARE included.
  - Expected for `SensorManager::updateTemperature()`: writes == {"temperature"},
    reads == {"sensor", "temperature", "validator", "display"}.

### 1.4 Cache invalidation (`rag_engine/knowledge_base/cache_manager.py`)

Pickled `ParseResult`s from the old shape would load without the new attributes. Add a
module constant `CACHE_VERSION = 2` that is part of the stored hash (e.g. hash string
`f"v{CACHE_VERSION}:{sha256}"`), so old cache entries miss. Add a test.

### 1.5 Fix baseline failures and latent bugs

Currently failing: `tests/test_dead_code.py::test_orphan_func_detected`,
`tests/test_graph_builder.py::test_find_unreachable_includes_orphan` (both compare bare names
against sig_key node ids — update the tests to the sig_key contract, e.g. `'orphanFunc()'`),
and `tests/test_standards_validator.py::test_high_complexity_flagged` (find the root cause
and fix the code; do not weaken the test).

Latent bugs to fix:
- `StandardsValidator._check_naming` matches the regex against `fn.name`, which is now
  `Class::method` for every method → every method fails PascalCase/camelCase. Match against
  the short name (`fn.name.rsplit('::', 1)[-1]`); skip constructors/destructors/operators.
- Recursion is never detected: `CallGraphBuilder._extract_calls` drops self-calls
  (`sig_key != fn.sig_key`) and the validator checks `fn.name in fn.calls` (bare name vs
  sig_keys). Fix: when a call in the body resolves (after the existing arity/type filtering)
  to the function's own sig_key via a free-function or own-class call, keep it in
  `fn.calls` and as a self-edge; validator checks `fn.sig_key in fn.calls`. Do NOT create
  self-edges from the conservative fan-out paths. Test with
  `int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }`.

### 1.6 Tests

- New `tests/test_class_model.py` against the sensors fixture: function sig_keys include
  `TemperatureSensor::read()`, `PressureSensor::read()`, `AltitudeSensor::read()`,
  `AltitudeSensor::selfTest()`, `Sensor::selfTest()`, `Sensor::~Sensor()`,
  `Sensor::getValue()`; `AltitudeSensor::read()` is_virtual and is_override; ClassDef for
  `TemperatureSensor` has bases `["Sensor"]`; ClassDef `Sensor` has a MethodDecl `read`
  with `is_pure=True`, and fields `value`, `lastValue`, `lastUpdateMs`; ClassDef
  `SensorManager` field `sensor` has type `TemperatureSensor` and field `sensors` exists;
  `disabledFeature()` has `preproc_guard == "0"`; end_line_number > line_number for
  multi-line functions; data reads/writes for `SensorManager::updateTemperature()` exactly as
  in 1.3; `Sensor::process()` writes {"lastValue", "value"}.
- Update existing tests that asserted bare names for in-class methods (e.g. virtual analysis
  tests using `brake`, `start`, `getSpeed`) to the qualified contract (`Vehicle::brake` ...).
- Full fast suite green.

---

## Task 2: Class hierarchy, virtual dispatch in the call graph, R1 Virtual Function Analysis

**Depends on Task 1.**

### 2.1 `rag_engine/core/class_hierarchy.py` (new)

`ClassHierarchy(parse_results: Dict[str, ParseResult])`:
- Merges `ClassDef`s across files by qualified name (header declaration + out-of-line
  definitions in .cpp files).
- Post-pass over all FunctionDefs (idempotent): set `class_name=None` when the qualifier is
  not a known class; set `is_virtual=True` for an out-of-line definition whose MethodDecl
  is virtual, and for any method that overrides a virtual method of an ancestor (same short
  name + same param types) even without the keyword; set `is_override` likewise.
- API: `classes: Dict[str, ClassDef]`, `bases(c) -> List[str]`, `derived(c)` (direct),
  `descendants(c)` (transitive), `ancestors(c)` (transitive, nearest first),
  `find_method(c, short_name, param_types=None) -> Optional[str]` (sig_key on `c` or the
  nearest ancestor declaring it — static lookup), `is_virtual_method(sig_key) -> bool`,
  `overrides_of(sig_key) -> List[str]` (sig_keys in descendants with same short name and
  param types), `root_virtual_methods() -> List[str]` (sig_keys of virtual methods whose
  class introduces them: no ancestor declares the same method), `field_type(c, name)`
  (searches ancestors too; returns the bare class name without `*`/`&`/`const`, or None),
  `inheritance_tree(root) -> str` rendered exactly like:

```text
Sensor
├── AltitudeSensor
├── PressureSensor
└── TemperatureSensor
```
  (children sorted by name, nested levels indented with `│   ` / `    `).

### 2.2 Call graph changes (`rag_engine/core/graph_builder.py`)

- `build()` first creates `self.hierarchy = ClassHierarchy(self._results)`.
- New shared model in `models.py`:
```python
@dataclass
class CallSite:
    caller: str                 # sig_key of the calling function
    target: str                 # sig_key of the statically resolved callee (may be a declaration-only method, e.g. pure virtual "Sensor::read()")
    file_path: str
    line: int                   # absolute 1-based line in file_path, 0 if unknown
    call_type: str              # 'direct' | 'virtual' | 'callback' | 'constructor' | 'unresolved'
    receiver: Optional[str] = None   # e.g. "sensor" for sensor.read(), "s" for s->process()
```
- `self.call_sites: List[CallSite]` — every resolved call, including targets that have no
  graph node (pure virtual declarations). Graph edges carry `call_type` and `lines`
  attributes. Line = `fn.body_start_line` + newline count before the match in the body.
- Receiver type inference additionally uses the enclosing class's fields (incl. inherited)
  via `hierarchy.field_type`, so `sensor.read()` inside SensorManager resolves to
  `TemperatureSensor::read()` and `validator.check(temperature)` to `Validator::check(float)`.
- Implicit this-calls: inside a method of class C, a bare `m(...)` or `this->m(...)`
  resolves via `hierarchy.find_method(C, m)` before falling back to free functions.
- Instantiated classes: `self.instantiated_classes: Set[str]` = classes appearing as local
  non-pointer declarations (`T x;`, `T x(...)`, `T x = ...`), `new T`, or as a non-pointer,
  non-reference field type of any class.
  `# shortcut: field types count as instantiated even if the owner never is; upgrade with whole-program construction analysis.`
- Virtual dispatch: when a call statically resolves to method `T::m` (via receiver static
  type T or this-call) and `T::m` is virtual, add the edge to the static target if it has a
  node (`call_type='direct'`), plus `call_type='virtual'` edges to the most-derived
  implementation of `m` for every instantiated class X in `{T} ∪ descendants(T)`. Overrides
  in classes that are never instantiated get no edge. Record one CallSite with the static
  target (`call_type` 'direct' if T::m non-virtual else 'virtual').
- Conservative fan-out edges (unknown receiver type) get `call_type='unresolved'`; function
  names passed as arguments (existing Pass 6) get `call_type='callback'`; constructor passes
  get `call_type='constructor'`.
- Expected on the fixture: `SensorManager::update()` → `Sensor::process()` (direct) and →
  `TemperatureSensor::process()`, `PressureSensor::process()` (virtual); no edge to any
  `AltitudeSensor::*`. `Sensor::process()` → `TemperatureSensor::read()` and
  `PressureSensor::read()` (virtual), not `AltitudeSensor::read()`. `SensorManager::updateTemperature()`
  → `TemperatureSensor::read()`, `Validator::check(float)`, `Display::update(float)`, and
  NOT `PressureSensor::read()`.

### 2.3 Virtual Function Analysis (`rag_engine/features/virtual_analysis.py`)

`VirtualAnalyzer(base_results, current_results, graph_builder=None, entry_points=None)` —
builds a CallGraphBuilder for `current_results` when none is given; `entry_points` default
`['main']`. Keep the existing base-vs-current §12 change classification, keyed by sig_key.

Add to the result (new dataclasses in this module):
```python
@dataclass
class OverrideInfo:
    function: str               # "TemperatureSensor::process()"
    class_name: str
    file_path: str
    line_number: int
    class_instantiated: bool
    reachable: bool             # reachable from entry points in the call graph
    call_sites: List[CallSite]  # calls whose static target is this override

@dataclass
class VirtualFunctionInfo:
    function: str               # "Sensor::process()"
    class_name: str
    short_name: str
    file_path: str
    line_number: int
    is_pure: bool
    has_definition: bool
    overridden_by: List[OverrideInfo]
    call_sites: List[CallSite]  # calls whose static target is this function
    reachable: bool             # base implementation reachable (False when pure / no definition)
    requirement_trace: Optional[str]
```
`VirtualAnalysisResult` gains `inventory: List[VirtualFunctionInfo]`,
`hierarchy_trees: Dict[str, str]` (root class → tree text, only for roots that have
derived classes), `entry_points: List[str]` — all with defaults.

Expected on the fixture (entry `main`): inventory contains `Sensor::process()` overridden by
TemperatureSensor/PressureSensor (both reachable, instantiated) with a call site from
`SensorManager::update()` at the line of `s->process();`; `Sensor::read()` is_pure,
overridden by Temperature/Pressure/Altitude, the Altitude override `class_instantiated=False`
and `reachable=False`; `Sensor::selfTest()` overridden by `AltitudeSensor::selfTest()`, no
call sites, not reachable; `hierarchy_trees['Sensor']` equals the tree in 2.1.

### 2.4 Report

`rag_engine/templates/virtual_analysis.jinja2`: add a "Virtual Function Inventory" section
per root virtual function in the R1 example shape (declared in file:line, pure/implemented,
overridden by — each with file:line, instantiated yes/no, reachable yes/no — called from
caller + file:line, inheritance tree in a code block), plus a table "Overrides not reachable
from entry points". `ReportGenerator.generate_virtual_analysis` passes the new fields.

### 2.5 Tests

`tests/test_class_hierarchy.py` (hierarchy API + tree text), additions to
`tests/test_graph_builder.py` (dispatch expectations in 2.2, call_sites lines/receivers),
additions to `tests/test_virtual_analysis.py` (inventory expectations in 2.3; rendered
markdown contains "Overridden by" and "TemperatureSensor::process()"). Existing tests green.

---

## Task 3: Dead code analysis overhaul

**Depends on Tasks 1–2.**

### 3.1 Config (`rag_engine/config.py`, `config.yaml`)

- Fix `load_config`: it filters keys with `hasattr(Config, k)`, which is False for
  `default_factory` fields, so `entry_points` and `lru_names` from config.yaml are silently
  ignored today. Use `{f.name for f in dataclasses.fields(Config)}`. Add a test.
- New fields: `entry_point_patterns: List[str]` default
  `['*_IRQHandler', '*_ISR', 'ISR_*', '*Task']` (fnmatch on the short function name);
  `dead_code_dispositions: str = ''` (path to the disposition YAML). Add both to config.yaml.

### 3.2 Dead-branch-aware call extraction (`graph_builder.py`)

Before call extraction, blank (replace with spaces, keep newlines) the bodies of
`if (0) {...}` / `if (false) {...}` / `while (0) {...}`. Only constant-false branches are
blanked here; code after `return` is handled by the fragment analyzer (3.3), not here.
So `legacyCalibrate()` (called only inside `if (0)`) has no incoming edge.

### 3.3 `rag_engine/features/control_flow_analyzer.py` (new)

Tree-sitter based over `ParseResult.raw_source`:
- `find_dead_fragments(result) -> List[DeadFragment]`: statements following an unconditional
  `return`/`throw`/`break`/`continue`/`goto` in the same compound statement (case labels end
  the dead region); `if` with constant-false condition (`0`, `false`) → the consequence block;
  `while (0)` body. Each has containing function sig_key, start/end lines, reason, snippet
  (first line, stripped). Skip functions inside `#if 0`.
- `find_unused_variables(result) -> List[UnusedVariable]`: locals declared in a function body
  and never referenced afterwards (`(void)x;` counts as a use; skip `volatile`, skip
  `[[maybe_unused]]`); parameters never referenced in the body (skip unnamed params and
  functions without a body). Each has function sig_key, variable name, line, kind
  ('local' | 'parameter').

### 3.4 Detector (`rag_engine/features/dead_code_detector.py`)

Signature: `DeadCodeDetector(graph_builder, parse_results, entry_points=None,
entry_point_patterns=None, dispositions_file=None)`.

`DeadCodeItem` (models.py) gains, all defaulted: `finding_id: str = ''`, `end_line: int = 0`,
`symbol_type: str = 'function'` (function | method | fragment | variable | parameter),
`function_name: Optional[str] = None`, `reason: str = ''`,
`confidence: Literal['high','medium','low'] = 'medium'`, `evidence: List[str]`,
`removal_impact: List[str]`, `review_status: str = 'open'`,
`reviewer_justification: Optional[str] = None`. Category literal adds `'unused_variable'`.

`DeadCodeReport` gains (defaulted): `warnings: List[str]`, `entry_points_used: List[str]`,
`counts_by_category: Dict[str, int]`. Keep `dead_count`, `deactivated_count`,
`false_positive_notes`, `structural_coverage_impact`.

Algorithm:
1. Entry points = configured names (bare name, qualified name, or sig_key) + functions whose
   short name matches `entry_point_patterns` + constructors/destructors of instantiated
   classes and their ancestors. Warn for each configured name not found.
2. Reachable = entry points ∪ their descendants (all edge types).
3. Classify every unreachable function (no silent exclusions — everything is reported):
   | condition (first match wins) | category | disposition | confidence | reason gist |
   |---|---|---|---|---|
   | `preproc_guard` is `0`/`false` | deactivated_code | Justify as Deactivated | high | excluded from build by `#if 0` |
   | other `preproc_guard` | deactivated_code | Justify as Deactivated | medium | only built when `<guard>` |
   | test/mock/stub name pattern | deactivated_code | Justify as Deactivated | low | test-support code |
   | virtual, class never instantiated | unused_export | Investigate | medium | override in class never instantiated |
   | virtual, otherwise | unused_export | Investigate | medium | virtual method never called or dispatched |
   | callback-like name (existing `_CALLBACK_PAT`) | unused_export | Investigate | low | name suggests callback/ISR; add to entry_points if invoked externally |
   | otherwise (method or free function) | dead_code | Remove | high | no call path from entry points `[...]` |
4. Fragments → `dead_fragment` / `Fix Fragment` / high. Unused locals → `unused_variable` /
   `Remove` / medium. Unused parameters → `unused_variable` / `Investigate` / low.
5. Evidence per function finding: "callers: none" or "only called by unreachable: X, Y";
   edge types seen. Removal impact: callees that would lose their only caller; members/
   globals written (`data_writes`); requirement trace (`do178c_requirement_trace`) → "update
   trace matrix"; declaration in a header → "remove declaration in <header>".
6. Warnings: configured entry point not found; graph has 0 edges; > 50 % of functions
   unreachable ("check entry_points configuration").
7. Stable IDs: `DC-<KIND>-<8 upper hex>` where KIND ∈ FUNC (free fn), METH (method),
   FRAG, VAR, CFG (deactivated by preprocessor), hash = sha1 of
   `f"{Path(file).name}|{symbol}|{KIND}|{snippet_or_var}"` (no line numbers, so IDs survive
   line shifts).
8. Dispositions YAML (`findings: {<id>: {status, justification, reviewer}}`): apply status +
   justification; an entry without justification is ignored with a warning; IDs not matching
   any finding produce a "stale disposition" warning.
9. `structural_coverage_impact` = `f"{n} findings; ~{lines} uncoverable source lines"` using
   line spans of function and fragment findings.

Expected on the sensors fixture with `entry_points=['main']` and default patterns:
- dead_code (Remove, high): `orphanHelper()`, `legacyCalibrate()`,
  `SensorManager::unusedDiagnostic()`, `Display::clear(int)`, `Sensor::getValue()`.
- unused_export (Investigate): `AltitudeSensor::read()`, `AltitudeSensor::selfTest()`
  (class never instantiated), `Sensor::selfTest()` (never called).
- deactivated_code: `disabledFeature()` (high, `#if 0`).
- dead_fragment: in `computeChecksum(int)` (`x = 0;`) and in `main()` (the `if (0)` block).
- unused_variable: `unused` in `orphanHelper()` (local), `code` in
  `SensorManager::unusedDiagnostic()` (local), `mode` in `Display::clear(int)` (parameter, low).
- NOT reported as function findings: `main()`, `TIMER0_IRQHandler()` (pattern entry),
  `onTimerTick()`, `registerCallback(Callback)`, `SensorManager::init()/update()/updateTemperature()`,
  all `TemperatureSensor::*`, `PressureSensor::*`, `Validator::check(float)`,
  `Display::update(float)`, `Sensor::process()`, `Sensor::~Sensor()`, `computeChecksum(int)`.
- Warning when entry point `doesNotExist` is configured; no false "not found" warning for main.

### 3.5 Report outputs

`dead_code_report.jinja2`: sections — analysis scope & entry points, warnings, summary by
category and confidence, findings table `| ID | File | Function/Symbol | Lines | Type |
Reason | Confidence | Status |`, then per-finding details (evidence, removal impact).
`ReportGenerator.generate_dead_code` additionally writes `<output_dir>/dead_code_findings.json`
(items as dicts + warnings + entry points). Checklist: `DC-03` "No open high-confidence dead
code findings" (PASS when none open with confidence high and category dead_code).

### 3.6 Tests

`tests/test_dead_code.py` rewritten/extended against the sensors fixture with all
expectations in 3.4 (one test per bullet group), plus: ID stability (same IDs after
inserting a blank line at the top of a copied fixture file in tmp_path), dispositions file
applied / missing-justification warning / stale-ID warning, JSON file written, config fix
test, `control_flow_analyzer` unit tests (code after `break` in a loop, `case` label ends
dead region, `(void)x` counts as use). Keep the old base_build tests working (sig_key names).

---

## Task 4: R2 Function Effect & Coupling Analysis

**Depends on Tasks 1–2.**

`rag_engine/features/function_effect.py` (new):

```python
@dataclass
class VariableAccess:
    name: str
    kind: Literal['member', 'global', 'parameter', 'unknown']
    type_: str = ''
    owner: Optional[str] = None      # owning class for members

@dataclass
class EffectCall:
    callee: str                      # sig_key
    callee_class: Optional[str]
    receiver: Optional[str]
    call_type: str
    line: int

@dataclass
class CoupledComponent:
    name: str                        # class name, or file stem for free functions
    coupling: Literal['control', 'data']
    reason: str                      # e.g. "calls TemperatureSensor::read() via member 'sensor'"

@dataclass
class FunctionEffect:
    function: str
    class_name: Optional[str]
    file_path: str
    start_line: int
    end_line: int
    return_type: str
    parameters: List[Parameter]
    description: str
    reads: List[VariableAccess]
    writes: List[VariableAccess]
    calls: List[EffectCall]
    callers: List[str]
    transitive_callers: List[str]
    coupled_components: List[CoupledComponent]
    data_coupled_functions: List[str]
    affected_components: List[str]
```

`FunctionEffectAnalyzer(parse_results, graph_builder)`; `analyze(method: str) ->
List[FunctionEffect]` — accepts a short name (`updateTemperature`), qualified name
(`SensorManager::updateTemperature`) or sig_key; returns every match (overloads, same short
name in several classes); empty list when nothing matches. `format_effect(effect) -> str`
returns the R2 text block (Method / Reads / Writes / Calls / Called by / Coupled components /
Affected if changed).

Rules:
- Variable kinds: member if the name is a field of the function's class or an ancestor
  (`hierarchy`), parameter if a parameter name, global if a file-scope variable declared in
  any parsed file (tree-sitter `declaration` at translation-unit or namespace level, incl.
  `static`/`const`), else unknown. Use `fn.data_reads` / `fn.data_writes` from Task 1.
- Calls from `graph_builder.call_sites` where `caller == fn.sig_key` (sorted by line).
- Callers: direct predecessors in the graph; transitive callers: `nx.ancestors`.
- Coupled components (control): the class of every callee (and that class's ancestors, reason
  "base class of X"); data: classes that read or write the same member/global variables.
  No numeric score anywhere.
- Data-coupled functions: functions in the same class (members) or any file (globals) that
  read a variable this function writes, or write a variable it reads.
- Affected components: sorted unique class names (file stem for free functions) of
  transitive callers + data-coupled functions + the function's own class.
- Description: first line of the docstring (comment markers stripped) when present; else a
  generated sentence, e.g. "Writes temperature; reads sensor, validator, display; calls
  TemperatureSensor::read(), Validator::check(float), Display::update(float)."

Expected on the fixture for `analyze('updateTemperature')` → one effect for
`SensorManager::updateTemperature()`: writes `temperature` (member, type float); reads
include `sensor`, `validator`, `display`, `temperature` (members); calls (in line order)
`TemperatureSensor::read()` via `sensor`, `Validator::check(float)` via `validator`,
`Display::update(float)` via `display`; callers `['main()']`; coupled components include
`TemperatureSensor`, `Sensor` (base class), `Validator`, `Display`; affected components
include `SensorManager` and `main`. `analyze('process')` returns the three `process`
methods (`Sensor::process()`, `TemperatureSensor::process()`, `PressureSensor::process()`). `analyze('nope')` → `[]`.

Report: `rag_engine/templates/function_effect.jinja2` + `ReportGenerator.generate_function_effect(effects, format)`
writing `function_effect_<short name>.md|docx`.

Tests: `tests/test_function_effect.py` with the expectations above, a global-variable case
(`onTimerTick()` writes global `cycleCount`; `main()` reads it → data-coupled), and the
rendered text containing "Coupled components".

---

## Task 5: R3 Sensor Data Analysis (templates, static checks, sample checker)

**Depends on Tasks 1–2 (and uses Task 4's global-variable helper if convenient).**

### 5.1 Template (`rag_engine/features/sensor_template.py`)

```python
@dataclass
class SensorTemplate:
    name: str
    type: str = ''
    data_type: str = 'float'
    unit: str = ''
    class_name: str = ''                  # C++ class to match; '' means use name
    parameter_group: str = ''
    io_group: str = ''
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
    sampling_interval_ms: Optional[float] = None
    stale_threshold_ms: Optional[float] = None
    stale_detect_frozen: bool = True
    debounce_enabled: bool = True
    debounce_threshold: Optional[float] = None
    debounce_count: int = 3
    corrupt_range_check: bool = True
```
`load_sensor_templates(path) -> List[SensorTemplate]`: YAML with optional `defaults:` (deep
merged into each sensor), and either `sensors:` (list) or a single `sensor:` mapping (the PDF
format). Nested keys as in the PDF: `valid_range: {min, max}`, `stale: {threshold_ms,
detect_frozen}`, `debounce: {enabled, threshold, count}`, `corrupt: {range_check}`.
Validation errors raise `ValueError` naming the sensor and field: missing name, min >= max,
negative thresholds, count < 1.

Files: `sensor_templates.yaml` (repo root, documented example: TemperatureSensor,
PressureSensor, AltitudeSensor, SpeedSensor) and `tests/fixtures/sensor_templates.yaml`:
```yaml
defaults:
  data_type: float
  debounce: {enabled: true, count: 3}
  corrupt: {range_check: true}
sensors:
  - name: TemperatureSensor
    type: temperature
    unit: Celsius
    parameter_group: Temperature Parameter
    io_group: Temperature I/O Group
    valid_range: {min: -54, max: 54}
    sampling_interval_ms: 100
    stale: {threshold_ms: 500}
    debounce: {threshold: 2.0}
  - name: PressureSensor
    type: pressure
    unit: Pascal
    parameter_group: Pressure Parameter
    io_group: Pressure I/O Group
    valid_range: {min: 0, max: 110000}
    sampling_interval_ms: 50
    stale: {threshold_ms: 200}
    debounce: {threshold: 500.0}
  - name: AltitudeSensor
    type: altitude
    unit: Feet
    valid_range: {min: -1000, max: 60000}
    stale: {threshold_ms: 1000}
    debounce: {threshold: 50.0}
  - name: SpeedSensor
    type: speed
    unit: Knots
    valid_range: {min: 0, max: 600}
    stale: {threshold_ms: 300}
    debounce: {threshold: 5.0}
```
Config: `sensor_templates_file: str = ''` in `Config` and config.yaml.

### 5.2 Static analysis (`rag_engine/features/sensor_analysis.py`)

`SensorAnalyzer(parse_results, graph_builder, templates).analyze() -> SensorAnalysisResult`
with per-template `SensorFinding`:
- `template`, `class_name` (matched class or None → `status='not_found'`), `file_path`,
  `line_number`, `instantiated` (in `graph_builder.instantiated_classes`), `hierarchy`
  (list: parameter_group → io_group → class → "data: <fields>", empty groups skipped),
  `data_members` (fields incl. inherited, e.g. value, lastValue, lastUpdateMs).
- Data flow (all lists of `"<sig_key> (<file>:<line>)"`):
  `sources` = the class's `read()` implementation(s) (methods named read/get*/sample/acquire);
  `processors` = callers of the sources (call graph predecessors, incl. via dispatch);
  `validators` = callees of processors or methods of the class whose short name matches
  `(?i)check|valid|range|stale|debounce|filter|limit`;
  `consumers` = other callees of processors that are not sources/validators and not methods
  of any class sharing the sensor's root base class (so `PressureSensor::read()` reached from
  `Sensor::process()` is not a consumer of TemperatureSensor data).
- Checks: `Dict[str, CheckResult]` keyed `'corrupt'`, `'stale'`, `'debounce'`:
```python
@dataclass
class CheckResult:
    status: Literal['implemented', 'mismatch', 'missing', 'not_required']
    expected: str                 # e.g. "range [-54, 54]", "threshold 500 ms"
    found: List[float]
    evidence: List[str]           # "<file>:<line> <sig_key>: <comparison text>"
```
  Scope = methods of the class (incl. inherited definitions) + processors + validators.
  Collect relational comparisons (`<`, `<=`, `>`, `>=`) from the tree-sitter AST where one
  operand is a numeric literal or a named constant resolved from file-scope
  `const <type> NAME = <literal>;` / `#define NAME <literal>` in any parsed file (strip
  `f/F/u/U/l/L` suffixes; a leading unary minus is part of the literal, so `-54.0f` → -54.0
  and `v > -100.0f` yields -100.0). Classify each comparison (first match wins):
  stale if the expression contains a time-like identifier
  (`(?i)(ms\b|_ms\b|ms_|time|stamp|elapsed|timeout|stale|tick|\bage\b|\bnow)`, matched on
  identifiers — `lastUpdateMs`, `nowMs`, `TEMP_STALE_MS` qualify);
  debounce if it contains `abs(`/`fabs(` of a difference or an identifier matching
  `(?i)debounce|prev|last|delta|diff|stable`;
  otherwise range.
  Status: corrupt → implemented if both valid_min and valid_max are among the range
  constants, mismatch if range comparisons exist but not both match, missing otherwise
  (not_required when `corrupt_range_check` false). stale → implemented if
  stale_threshold_ms is among the stale constants, mismatch if stale comparisons exist with
  other values, missing otherwise. debounce → same with debounce_threshold
  (not_required when `debounce_enabled` false). Float equality with tolerance
  `1e-6 * max(1, |expected|)`.
  `# shortcut: identifier-pattern classification, upgrade to data-flow tracking of the sensor value.`
- `unmapped_sensor_classes`: classes deriving (transitively) from a matched sensor's base
  class, or named `*Sensor`, that no template maps to.

Expected on the fixture with `tests/fixtures/sensor_templates.yaml`:
- TemperatureSensor: corrupt implemented (found includes -54, 54), stale implemented (500),
  debounce implemented (2.0); sources include `TemperatureSensor::read()`; processors include
  `TemperatureSensor::process()` and `SensorManager::updateTemperature()`; validators include
  `TemperatureSensor::inRange(float)`, `TemperatureSensor::isStale(unsigned long)`,
  `TemperatureSensor::debounce(float)`, `Validator::check(float)`; consumers include
  `Display::update(float)`; hierarchy starts `["Temperature Parameter", "Temperature I/O Group", "TemperatureSensor", ...]`.
- PressureSensor: corrupt mismatch (found includes 0 and 120000, expected [0, 110000]),
  stale missing, debounce missing.
- AltitudeSensor: class found, `instantiated=False`, all three checks missing.
- SpeedSensor: `status='not_found'`.

### 5.3 Sample-data checker (`rag_engine/features/sensor_data_checker.py`)

`check_samples(template, samples: List[Tuple[float, float]]) -> List[SampleResult]` where
`SampleResult(timestamp_ms, value, accepted_value, flags: List[str])`, flags ⊆
{'corrupt', 'stale', 'debounce'}. `load_samples_csv(path) -> Dict[str, List[Tuple[float, float]]]`
for CSV with header `sensor,timestamp_ms,value`. `summarize(results) -> Dict[str, int]`
(counts per flag + total).

Algorithm per sample, in timestamp order (state: prev_t, last_good, last_change_t,
accepted, run):
1. `gap_stale = prev_t is not None and t - prev_t > stale_threshold_ms`; `prev_t = t`.
2. Corrupt if value is NaN/inf, or `corrupt_range_check` and outside [valid_min, valid_max]:
   flags `['corrupt']` (+ `'stale'` if gap_stale), accepted unchanged, state otherwise
   unchanged; next sample.
3. If `last_good is None or abs(v - last_good) > 1e-9`: `last_change_t = t`. `last_good = v`.
4. `frozen_stale = stale_detect_frozen and t - last_change_t > stale_threshold_ms`.
5. Debounce (when enabled and threshold set): if `accepted is None` → `accepted = v`; elif
   `abs(v - accepted) > debounce_threshold` → `run += 1`; if `run >= debounce_count` →
   `accepted = v, run = 0` else flag `'debounce'`; else → `run = 0, accepted = v`.
   When debounce disabled: `accepted = v`.
6. Flag `'stale'` if gap_stale or frozen_stale. Stale checks skipped when
   `stale_threshold_ms` is None.

Exact test vector (TemperatureSensor template: range [-54, 54], stale 500 ms, frozen on,
debounce threshold 2.0 count 3):

| t | value | accepted | flags |
|---|---|---|---|
| 0 | 25.0 | 25.0 | [] |
| 100 | 25.2 | 25.2 | [] |
| 200 | 80.0 | 25.2 | ['corrupt'] |
| 300 | 25.2 | 25.2 | [] |
| 400 | 25.2 | 25.2 | [] |
| 500 | 25.2 | 25.2 | [] |
| 600 | 25.2 | 25.2 | [] |
| 700 | 25.2 | 25.2 | ['stale'] |
| 800 | 25.2 | 25.2 | ['stale'] |
| 900 | 30.0 | 25.2 | ['debounce'] |
| 1000 | 30.1 | 25.2 | ['debounce'] |
| 1100 | 30.0 | 30.0 | [] |
| 1800 | 30.0 | 30.0 | ['stale'] |

Ship this vector as `tests/fixtures/sensor_samples.csv` (sensor column `TemperatureSensor`)
plus two PressureSensor rows (`0,101325`, `100,150000` → second is corrupt).

### 5.4 Report

`rag_engine/templates/sensor_analysis.jinja2` + `ReportGenerator.generate_sensor_analysis(result, format)`:
per sensor — hierarchy chain, data members, data-flow lists, check table
(`| Check | Status | Expected | Found | Evidence |`), unmapped sensor classes.

### 5.5 Tests

`tests/test_sensor_template.py` (loader: list form, single `sensor:` form, defaults merge,
each validation error), `tests/test_sensor_analysis.py` (expectations in 5.2),
`tests/test_sensor_data_checker.py` (the exact vector, CSV loader, summary, NaN corrupt,
debounce disabled, stale disabled when threshold None).

---

## Task 6: Pipeline integration, CLI commands, reports, checklist, R4 Q&A

**Depends on Tasks 1–5.**

- `rag_engine/pipeline.py` (new): move `AnalysisResults` here from `tui/app.py` (re-export it
  from `tui/app.py` for compatibility) and add fields `sensors`, `graph`, `parse_results`,
  `report_paths`. `run_pipeline(cfg, codebase, base_build=None, lru_docs=None,
  output_format='markdown', progress=None, use_semantic_search=True, write_reports=True)
  -> AnalysisResults`. `progress(step: str, pct: int)` is optional. Steps: parse → call graph →
  virtual → dead code → standards → coupling → sensors (only when
  `cfg.sensor_templates_file`) → optional LLM narratives (existing logic, plus a sensors
  narrative) → reports + checklist when `write_reports`. If SemanticSearch construction
  fails (model unavailable offline) or `use_semantic_search=False`, continue with
  `search=None` and log a warning.
- `CouplingAnalyzer` accepts `search=None`: then LRU-related functions are found by keyword
  overlap between the LRU document text and function names/docstrings.
- `main.py`: `analyze` uses `run_pipeline` (keep options; add `--sensor-templates`); new
  commands: `virtuals CODEBASE` (prints inventory + trees), `effect CODEBASE METHOD`
  (prints `format_effect` for each match, writes the report), `sensors CODEBASE --templates`
  (prints per-sensor check table), `sensor-check SAMPLES_CSV --templates` (prints per-sample
  flags and summary), `ask CODEBASE QUESTION` (offline answer; LLM explanation when
  `ollama_enabled` and reachable). Commands that only need the code model run with
  `use_semantic_search=False, write_reports=False` except where stated.
- `ReportGenerator.generate_all` gains optional `sensors` (writes the sensor report when given).
  `ChecklistFiller.fill_sqa_checklist` gains optional `sensors`: `SEN-01` every template mapped
  to a code class, `SEN-02` no `missing`/`mismatch` checks (FAIL lists them), `VA-03` every
  override reachable or justified (OPEN lists unreachable overrides).
- `rag_engine/llm/offline_qa.py` (new): move `_offline_answer` from `tui/app.py` as
  `offline_answer(question, results)` and add handlers checked FIRST:
  - "virtual functions implemented/overridden by <Class>" → methods of that class that are
    virtual, each with the base function it overrides;
  - "affected if <method> changes" / "effect of <method>" / "coupling of <method>" →
    `format_effect` of `FunctionEffectAnalyzer(results.parse_results, results.graph)`;
  - "checks applied to <Sensor>" / "validation ... <Sensor>" → the sensor's check table and
    data flow;
  - "inheritance/hierarchy of <Class>" → tree text.
  Unknown class/method → a clear "not found" message listing close names.
  LLM path (`ChatScreen._ask` and `ask` command): the offline answer is passed to
  `RAGNarrativeGenerator.answer_question` as the structured-facts context; without an LLM the
  offline answer is returned as is.
- Tests: `tests/test_pipeline.py` (run_pipeline on the sensors fixture with
  `use_semantic_search=False`, writes reports to tmp output dir incl. `dead_code_findings.json`
  and sensor report), `tests/test_offline_qa.py` (the three R4 questions from the spec give
  answers containing `TemperatureSensor::process()`, `Validator`/`Display`, and the three
  check statuses respectively), `tests/test_cli.py` using `typer.testing.CliRunner` for
  `virtuals`, `effect`, `sensors`, `sensor-check`, `ask`.

---

## Task 7: TUI for interactive testing

**Depends on Task 6.**

`rag_engine/tui/app.py`:
- `AnalysisScreen` uses `run_pipeline` with a progress callback (remove the duplicated
  orchestration); progress rows: Parsing, Virtual, Dead code, Standards, Coupling, Sensors.
- `ConfigScreen`: add a "Sensor templates file" input (optional) passed through.
- `ResultsScreen` tabs (keep existing ones):
  - "Virtual Functions": a `Tree` — root classes → virtual functions → overrides with
    reachable ✓/✗ and instantiated flag; call sites as leaves; plus the existing changes table.
  - "Dead Code": table columns ID, File, Symbol, Lines, Type, Confidence, Reason; a Static
    listing warnings and entry points.
  - "Function Effect": `Input` (method name) + `Button` "Analyze" → `Static` with
    `format_effect` output for every match ("not found" message otherwise).
  - "Sensors": DataTable sensor × (corrupt, stale, debounce, instantiated) statuses; a
    "Sample tester" area: `Select` of template names, `Input` accepting either a CSV path
    or inline samples `t:v, t:v, ...`, `Button` "Run checks" → DataTable
    (t, value, accepted, flags) + summary line. Disabled with a hint when no templates loaded.
- `ChatScreen` uses `offline_answer` / LLM path from Task 6; suggestion text lists the three
  R4 questions.
- Tests: `tests/test_tui.py` using `App.run_test()` pilot inside `asyncio.run(...)` (no
  pytest-asyncio dependency): build `AnalysisResults` via `run_pipeline(..., use_semantic_search=False,
  write_reports=False)` on the fixture with the fixture templates; push `ResultsScreen`;
  assert the Virtual tree contains `Sensor::process()`; type `updateTemperature` in the
  effect input, press Analyze, assert output contains `Validator`; enter inline samples
  `0:25.0, 100:80.0` for TemperatureSensor, run, assert a row flagged `corrupt`.

---

## Task 8: Documentation

**Depends on Tasks 1–7.** Mechanical edits only.

- `README.md` and `USAGE.md`: new features (R1–R4), new CLI commands with examples using
  `tests/fixtures/cpp/sensors` and `tests/fixtures/sensor_templates.yaml`, sensor template
  format (copy of the fixture template), new config keys (`entry_point_patterns`,
  `dead_code_dispositions`, `sensor_templates_file`), dead-code finding fields and the
  dispositions file format, TUI tabs.
- `CLAUDE.md`: add the new modules to the Project Structure tree and one line each under
  Feature Implementations.
- Mention open questions for the mentor (PDF section 14) in `USAGE.md` under "Assumptions":
  stale/debounce algorithm and thresholds are configurable defaults pending Honeywell
  confirmation; no numeric coupling score; coverage not yet integrated.
