# DO-178C RAG Engine — Usage Guide

Complete reference for running all four analysis objectives, the TUI, the CLI, Docker, and the interactive chat.

---

## Table of Contents

1. [Install](#1-install)
2. [Quick Start](#2-quick-start)
3. [Four Analysis Objectives](#3-four-analysis-objectives)
   - [Objective 1 — Virtual Function Change Analysis](#objective-1--virtual-function-change-analysis-do-178c-12)
   - [Objective 2 — Coupling Analysis](#objective-2--coupling-analysis-do-178c-631b)
   - [Objective 3 — Dead Code Detection](#objective-3--dead-code-detection-do-178c-6422)
   - [Objective 4 — Standards Validation](#objective-4--standards-validation-do-178c-51--misra-c)
   - [R1 — Virtual Function Analysis](#r1--virtual-function-analysis)
   - [R2 — Function Effect & Coupling](#r2--function-effect--coupling)
   - [R3 — Sensor Data Analysis](#r3--sensor-data-analysis)
   - [R4 — Questions and Answers](#r4--questions-and-answers)
4. [TUI Interface](#4-tui-interface)
5. [CLI Reference](#5-cli-reference)
6. [Chat Interface](#6-chat-interface)
7. [Docker — Laptop Mode](#7-docker--laptop-mode)
8. [Docker — Production Mode](#8-docker--production-mode)
9. [Output Files](#9-output-files)
10. [Configuration](#10-configuration)
11. [Assumptions and Open Questions](#11-assumptions-and-open-questions)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Install

```bash
# Python 3.10+ required
pip install -r requirements.txt
```

```bash
# Install project package (needed for imports)
pip install -e .
```

```bash
# Verify everything works
python main.py validate-config
```

**Windows only** — set these before any command if you see TensorFlow import errors:
```powershell
$env:USE_TF = "0"
$env:USE_TORCH = "1"
```

---

## 2. Quick Start

```bash
# Run all four analyses on a C++ directory, get reports in output/
python main.py analyze /path/to/your/cpp/code
```

```bash
# Or launch the interactive TUI (press c on the results screen to open chat)
python tui.py
```

---

## 3. Four Analysis Objectives

### Objective 1 — Virtual Function Change Analysis (DO-178C §12)

**What it does:**
Compares two builds of your C++ codebase, identifies every virtual function that was added, removed, or modified, and classifies each change as **Category 1** or **Category 2** per DO-178C §12.

| Category | Meaning | Reverification Required |
|----------|---------|------------------------|
| Category 1 | Comment/formatting change only | Regression tests only |
| Category 2 | Signature, implementation, or existence change | Full reverification |

**How to run (CLI):**
```bash
# With base build comparison (recommended)
python main.py analyze /path/to/current_build \
  --base-build /path/to/base_build
```

```bash
# Without base build — compares codebase to itself (shows all virtuals, no changes)
python main.py analyze /path/to/cpp_code
```

**What to provide:**
- `current_build/` — the new version of the code (what you're certifying)
- `base_build/` — the previously certified version

**Output:** `output/virtual_analysis.docx` (or `.md`)

Fields in the report:
- Function name, file, line
- Change type: `added` / `removed` / `modified` / `unchanged`
- DO-178C category (Cat1 / Cat2)
- Reverification scope (which test suites need re-running)

---

### Objective 2 — Coupling Analysis (DO-178C §6.3.1.b)

**What it does:**
Maps all control coupling (call dependencies) and data coupling (shared global variables) between your code and each LRU (Line Replaceable Unit) in the avionics system. Assigns a risk level (low / medium / high) per LRU. An LRU is analysed only when you give the engine a specification document for it.

**How to run (CLI):**
```bash
# Without LRU documents: no LRU is analysed, so the per-LRU section of the report is empty
python main.py analyze /path/to/cpp_code
```

```bash
# With LRU specification documents (needed for per-LRU coupling)
python main.py analyze /path/to/cpp_code \
  --lru-docs /path/to/lru_specifications/
```

**What to provide:**
- `lru_docs/` — folder containing LRU PDF or DOCX specification documents
  - Naming: `ADS.pdf`, `FCS.docx`, `FMS.pdf`, etc. (file name = LRU name)
  - If omitted, no LRU is analysed. The report has no per-LRU entries

**Output:** `output/coupling_analysis.docx`

Fields in the report:
- LRU name and risk level: HIGH if either list has more than 5 items, MEDIUM if either has more than 2, otherwise LOW. This is a count rule, not a score.
- Control coupling list (functions that call LRU-related code)
- Data coupling list (`function -> global` pairs for shared globals the matched functions read or write)

No numeric coupling score is produced. The matched code comes from semantic search when it is available, and from keyword matching otherwise (`--no-semantic-search`).

**`lru_names` in `config.yaml`:** the engine currently does not read this key. LRUs come only from the files in `--lru-docs`, so listing a name there does not add an LRU to the analysis. The key is kept for a later release.

---

### Objective 3 — Dead Code Detection (DO-178C §6.4.2.2)

**What it does:**
Finds code with no call path from the entry points, and classifies each finding with a stable ID, a confidence, evidence and removal impact. Call-graph reachability covers functions and methods; dead fragments and unused variables come from a per-file control-flow pass.

| Category | Disposition | Meaning |
|----------|------------|---------|
| `dead_code` | Remove | No call path from any entry point |
| `deactivated_code` | Justify as Deactivated | Excluded from the build (`#if 0`, or only built under an `#if`/`#ifdef` guard), only reachable from such guarded code, or test-support code |
| `unused_export` | Investigate | Referenced only by name outside the call graph (function pointer, table, macro), a virtual that is never dispatched or sits in a class that is never instantiated, an `extern "C"` function, an operator, or a callback-like name |
| `dead_fragment` | Fix Fragment | Statements after an unconditional `return`, `throw`, `break`, `continue` or `goto`, or the body of `if (0)` |
| `unused_variable` | Remove (local) / Investigate (parameter) | A local variable never used after its declaration, or a parameter never used in the body |

**Confidence** is `high`, `medium` or `low`: how strongly the evidence supports the verdict. Name and reference heuristics give `low`; a plain function with no call path and no other evidence gives `high`.

**How to run (CLI):**
```bash
python main.py analyze /path/to/cpp_code
```

**Configuring entry points** (critical for accuracy):
```yaml
# config.yaml
entry_points:
  - main          # standard C++ entry
  - initSystem    # your system init function
  - handleISR     # interrupt service routines
  - onBoot        # RTOS task entry points
```

If entry points are wrong, all code will appear dead. Set them to your actual program entry points.

Interrupt handlers and RTOS tasks can also be matched by name with `entry_point_patterns` (fnmatch on the short function name; the defaults are shown):

```yaml
entry_point_patterns:
  - '*_IRQHandler'
  - '*_ISR'
  - 'ISR_*'
  - '*Task'
```

Constructors and destructors of instantiated classes are implicit entry points, and so are the functions called from file-scope initialisers (`static Foo g = make();`), which run before `main`. The report lists the entry points it used.

The report warns with `entry_point_pattern '<pattern>' matches nothing` when one of your `entry_point_patterns` matches no function (the built-in defaults are not checked), and with `entry_point_patterns match N of M functions — check the configuration` when the patterns together match more than half of all functions (then almost nothing can be dead).

Calls are resolved by the receiver's type, and global-scope calls (`::f()`) too. A chained call such as `Mgr::instance().compute()` has no plain receiver, so `compute` becomes a conservative `unresolved` edge to every function of that name, as does any call whose receiver type is unknown. A possible caller is never missed this way. These calls are listed under `unresolved_calls` in the JSON.

**Output:** `output/dead_code_report.docx` (`.md` with `--output-format markdown`) and `output/dead_code_findings.json` (the same findings, machine-readable).

The report has warnings, summary tables by category and confidence, a findings table (ID, file, symbol, lines, category, disposition, reason, confidence, status) and a detail block per finding:
- **ID**: `DC-<kind>-<8 hex>`. The kind is `FUNC` (free function), `METH` (method), `CFG` (excluded by the build configuration), `FRAG` (dead fragment) or `VAR` (variable or parameter). The 8 hex digits are the first 8 digits of a SHA-1 of the file name, symbol, kind and detail. The symbol is the function's signature key (for a fragment or variable, its enclosing function). The detail is the fragment's text or the variable's name. Line numbers are not hashed, so an ID survives edits above it. Renaming the function, renaming the file or changing a fragment's text gives a new ID. Moving a file to another directory keeps it, because only the file name is hashed.
- **File, lines, function/symbol, category, disposition**
- **Confidence** and **Reason**, for example `no call path from entry points [main()]`
- **Coverage impact**: an estimate of source lines and decision points, not measured coverage (see Assumptions)
- **Evidence**, for example `callers: none`, `only called by unreachable: ...` or `follows unconditional return at line 30`
- **Removal impact**, for example `would lose its only caller`, `writes members/globals: ...`, `remove declaration in sensor.h`, or `none identified`
- **Review status** and justification, from the dispositions file below (`open` until reviewed)

Fields in `dead_code_findings.json` per item: `finding_id`, `name`, `file_path`, `line_number`, `end_line`, `category`, `do178c_disposition`, `confidence`, `symbol_type`, `function_name`, `reason`, `coverage_impact`, `evidence`, `removal_impact`, `review_status`, `reviewer_justification`, `reviewer`. The file also has `warnings`, `entry_points` and `unresolved_calls`.

**Recording reviews (dispositions file).** A reviewer records a decision per finding in a YAML file, and `dead_code_dispositions` in the config points at it (path relative to the config file):

```yaml
findings:
  DC-CFG-6F03AB16:
    status: deactivated
    justification: "disabledFeature() is excluded by #if 0 in the released build; kept for the feature-B variant."
    reviewer: J. Doe
```

- Each entry needs `status` and `justification`. An entry without them is ignored, with a warning.
- An ID that matches no finding gives a `stale disposition` warning. The ID changes when the function or file is renamed, so update the file.
- `status` must be one of `open`, `confirmed_dead`, `false_positive` or `deactivated`; other values are ignored with a warning. Any value other than `open`, with a justification, marks the finding as reviewed. The report shows the status and justification. The SQA checklist accepts an unreachable virtual override when its finding is reviewed (VA-03), and DC-03 counts only open high-confidence dead-code findings.
- A disposition records the review. It does not change the category or the disposition.

---

### Objective 4 — Standards Validation (DO-178C §5.1 + MISRA C++)

**What it does:**
Checks all C++ source files against the DO-178C Software Code Standard and MISRA C++:2008 rules. Produces a compliance score and a list of violations with severity and correction guidance.

**Severity levels:**

| Severity | Examples | Action |
|----------|---------|--------|
| CRITICAL | `goto`, `new`/`delete`/`malloc` post-init | Must fix before certification |
| MAJOR | Recursion, exceptions, unbounded loops | Must fix or justify deviation |
| MEDIUM | Cyclomatic complexity > 10 | Fix or annotate |
| MINOR | Missing docstrings, naming violations, nesting > 5 | Best effort |

**How to run (CLI):**
```bash
python main.py analyze /path/to/cpp_code

# Override DAL level (affects which rules are enforced)
# Edit config.yaml: dal_level: A  (strictest)
```

**Configuring thresholds:**
```yaml
# config.yaml
cyclomatic_complexity_max: 10   # Flag functions above this
function_length_max: 50         # Flag functions longer than this (lines)
nesting_depth_max: 5            # Flag deeper nesting
param_count_max: 7              # Flag functions with more parameters
```

**Output:** `output/standards_report.docx`

Exit code behavior:
- Exit 0 — analysis complete, no CRITICAL violations
- Exit 1 — CRITICAL violations found (blocks CI pipeline)

---

### R1 — Virtual Function Analysis

Lists every virtual function with its declaration, its overrides, whether each override is reachable from the entry points, its call sites, and the class hierarchies. It needs one build. `analyze --base-build` still produces the Category 1 / Category 2 change report.

```bash
python main.py virtuals tests/fixtures/cpp/sensors
```

Excerpt (real output on the fixture; `Sensor::process()` and `Sensor::read()` from the full list, which also contains `Sensor::~Sensor()` and `Sensor::selfTest()`):

```text
Virtual Function: Sensor::process()
  Declared in:   tests\fixtures\cpp\sensors\sensor.h:9 (implemented)
  Overridden by: PressureSensor::process(), TemperatureSensor::process()
  Inheritance:
    Sensor
    ├── AltitudeSensor
    ├── PressureSensor
    └── TemperatureSensor
  Usage:         Called from SensorManager::update() (tests\fixtures\cpp\sensors\sensor_manager.cpp:11)

Virtual Function: Sensor::read()
  Declared in:   tests\fixtures\cpp\sensors\sensor.h:10 (pure virtual)
  Overridden by: AltitudeSensor::read() [not reachable], PressureSensor::read(), TemperatureSensor::read()
  Inheritance:   ...   (same tree as above; omitted here)
  Usage:         Called from Sensor::process() (tests\fixtures\cpp\sensors\sensor.cpp:5)
...
Class hierarchies:
Sensor
├── AltitudeSensor
├── PressureSensor
└── TemperatureSensor
```

`[not reachable]` marks an override that no entry point reaches. `no call sites found` means nothing calls the function. The TUI shows the same data as a tree (see section 4).

### R2 — Function Effect & Coupling

Takes one method name (`updateTemperature`, or the qualified `SensorManager::updateTemperature`) and reports what it does, what it reads and writes, the functions it calls, its callers, the components it is coupled to (control or data), and what is affected if it changes. There is no numeric coupling score.

```bash
python main.py effect tests/fixtures/cpp/sensors updateTemperature
```

Excerpt (real output):

```text
Method: SensorManager::updateTemperature()
  Location: tests\fixtures\cpp\sensors\sensor_manager.cpp:15-19
  Returns: void    Parameters: none
  Does: Writes temperature; reads sensor, validator, display; calls TemperatureSensor::read(), Validator::check(float), Display::update(float).
...
Coupled components:
  - TemperatureSensor (control): calls TemperatureSensor::read() via member 'sensor'
  - Validator (control): calls Validator::check(float) via member 'validator'
  - Display (control): calls Display::update(float) via member 'display'
  - Sensor (control): base class of TemperatureSensor
Data-coupled functions: none
Affected if changed: SensorManager, main

Report: output\function_effect_updateTemperature.md
```

The report goes to `output/function_effect_<method>.md`. `--output-format docx` writes DOCX instead. An unknown method prints a not-found message with close names and exits with status 1.

### R3 — Sensor Data Analysis

Sensors are described in a YAML template, not in code. One engine checks every sensor against its template, so adding a sensor means adding a template entry. The hierarchy is Parameter Group → I/O Group → Sensor → Sensor Data.

The three checks:
- **Corrupt**: a value outside `valid_range` (`min`, `max`). The value is not accepted.
- **Stale**: the gap between two consecutive samples (corrupt ones included) is longer than `stale.threshold_ms`. With `stale.detect_frozen` (default true), the check also fires when the raw non-corrupt value has not changed for longer than the threshold.
- **Debounce**: a sample that differs from the currently accepted value by more than `debounce.threshold` is held back. The new value is accepted once `debounce.count` consecutive non-corrupt samples have each differed from the accepted value by more than the threshold. The samples need not agree with each other. A sample within the threshold of the accepted value resets the count. Small fluctuations are accepted immediately. In the `sensor-check` run below, the values 30 and 30.1 (t=900 and t=1000) are held back, and 30 at t=1100 is the third and is accepted.

**Template format.** This is the fixture template (`tests/fixtures/sensor_templates.yaml`). Copy it and adapt it, or point `sensor_templates_file` at your own file.

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

| Key | Meaning |
|-----|---------|
| `name` | Template name. Also the C++ class to match, unless `class_name` is set |
| `class_name` | C++ class to match (optional) |
| `type`, `data_type`, `unit` | Descriptive metadata |
| `parameter_group`, `io_group` | The two upper levels of the hierarchy |
| `valid_range` `{min, max}` | Corrupt check bounds; `min` must be less than `max` |
| `sampling_interval_ms` | Expected update period. Recorded for reference; the checks do not use it |
| `stale` `{threshold_ms, detect_frozen}` | Stale check (see above) |
| `debounce` `{enabled, threshold, count}` | Debounce check (see above); `count` must be at least 1 |
| `corrupt` `{range_check}` | Turns the corrupt range check on or off |
| `defaults:` | Deep-merged into every sensor; a sensor's own keys win |
| `sensor:` | A single sensor as a mapping, instead of a `sensors:` list |

Unknown keys are rejected, because a typo would silently leave a check unconfigured.

**Static checks (`sensors`).** For each template, the engine looks for the three checks in the code of the matching class and compares them with the template. Each check gets a status:
- `implemented`: the code has the check and its values match the template.
- `mismatch`: the code has the check, but its range or threshold differs. The expected and found values are shown.
- `missing`: no such check was found in the class.

```bash
python main.py sensors tests/fixtures/cpp/sensors --templates tests/fixtures/sensor_templates.yaml
```

Excerpt (real output):

```text
Sensor             Class              Corrupt      Stale        Debounce
TemperatureSensor  TemperatureSensor  implemented  implemented  implemented
PressureSensor     PressureSensor     mismatch     missing      missing
AltitudeSensor     AltitudeSensor     missing      missing      missing
SpeedSensor        (not found)        -            -            -
  PressureSensor.corrupt: mismatch (expected range [0, 110000]; found 0, 120000)
  PressureSensor.stale: missing (expected threshold 200 ms)
  PressureSensor.debounce: missing (expected threshold 500)
  AltitudeSensor.corrupt: missing (expected range [-1000, 60000])
  AltitudeSensor.stale: missing (expected threshold 1000 ms)
  AltitudeSensor.debounce: missing (expected threshold 50)
```

`SpeedSensor` has no class in the fixture, so it shows `(not found)`. Sensor classes that match no template are listed on a separate line.

**Sample checks (`sensor-check`).** Replays recorded samples through the template's rules. The CSV needs the header `sensor,timestamp_ms,value`, and the sensor names must match template names. Each sample shows the value the checks accept (`accepted`) and its flags. The CSV must be UTF-8 (a BOM, as Excel writes it, is accepted); another encoding gives an error that names the file.

```text
sensor,timestamp_ms,value
TemperatureSensor,0,25.0
TemperatureSensor,100,25.2
TemperatureSensor,200,80.0
...
```

```bash
python main.py sensor-check tests/fixtures/sensor_samples.csv --templates tests/fixtures/sensor_templates.yaml
```

Excerpt (real output, `TemperatureSensor` complete; `PressureSensor` shortened):

```text
TemperatureSensor
  t=0 ms  value=25  accepted=25  ok
  t=100 ms  value=25.2  accepted=25.2  ok
  t=200 ms  value=80  accepted=25.2  corrupt
  t=300 ms  value=25.2  accepted=25.2  ok
  t=400 ms  value=25.2  accepted=25.2  ok
  t=500 ms  value=25.2  accepted=25.2  ok
  t=600 ms  value=25.2  accepted=25.2  ok
  t=700 ms  value=25.2  accepted=25.2  stale
  t=800 ms  value=25.2  accepted=25.2  stale
  t=900 ms  value=30  accepted=25.2  debounce
  t=1000 ms  value=30.1  accepted=25.2  debounce
  t=1100 ms  value=30  accepted=30  ok
  t=1800 ms  value=30  accepted=30  stale
  Summary: 13 samples, corrupt 1, stale 3, debounce 2

PressureSensor
  ...
  t=100 ms  value=150000  accepted=101325  corrupt
  Summary: 2 samples, corrupt 1, stale 0, debounce 0
```

The TUI Sensors tab has a sample tester that runs the same rules on a CSV path or on inline samples such as `0:25.0, 100:80.0`.

When `sensor_templates_file` is set, `analyze` also writes `sensor_analysis` (DOCX or Markdown) with the same statuses and SQA checklist rows SEN-01 (every template maps to a class) and SEN-02 (no check missing or mismatched).

### R4 — Questions and Answers

`ask` answers a question from the static-analysis results. The answer is built from the facts first. A local LLM (Ollama) only explains those facts, and only when `ollama_enabled` is true and Ollama is reachable. Otherwise the facts are printed as they are.

Question kinds it understands:
- virtual functions implemented or overridden by a class
- what a method affects, and its coupling (the same output as `effect`)
- the checks and validation applied to a sensor
- the inheritance hierarchy of a class
- summary questions about standards, virtual changes, dead code, coupling and sensor check status

A name that does not match anything gets a not-found message with close names, so a typo such as `TemperatureSensr` is reported as a typo, not answered with a summary.

```bash
python main.py ask tests/fixtures/cpp/sensors "What are the virtual functions implemented by TemperatureSensor?"
```

Excerpt (real output):

```text
Virtual functions implemented by TemperatureSensor:
  TemperatureSensor::process() — overrides Sensor::process() (reachable: yes)
  TemperatureSensor::read() — overrides Sensor::read() (reachable: yes)
```

```bash
python main.py ask tests/fixtures/cpp/sensors "What validation checks are applied to TemperatureSensor?" --templates tests/fixtures/sensor_templates.yaml
```

Excerpt (real output):

```text
Sensor TemperatureSensor (class TemperatureSensor, tests\fixtures\cpp\sensors\temperature_sensor.h:10)
Hierarchy: Temperature Parameter -> Temperature I/O Group -> TemperatureSensor -> data: value, lastValue, lastUpdateMs
Checks applied:
  corrupt: implemented (expected range [-54, 54]; found -100, -54, 54, 100)
    tests\fixtures\cpp\sensors\temperature_sensor.cpp:13 TemperatureSensor::inRange(float): v >= TEMP_MIN_C
    tests\fixtures\cpp\sensors\temperature_sensor.cpp:13 TemperatureSensor::inRange(float): v <= TEMP_MAX_C
    tests\fixtures\cpp\sensors\validator.cpp:4 Validator::check(float): v > -100.0f
    tests\fixtures\cpp\sensors\validator.cpp:4 Validator::check(float): v < 100.0f
  stale: implemented (expected threshold 500 ms; found 500)
    tests\fixtures\cpp\sensors\temperature_sensor.cpp:9 TemperatureSensor::isStale(unsigned long): (nowMs - lastUpdateMs) > TEMP_STALE_MS
  debounce: implemented (expected threshold 2; found 2)
    tests\fixtures\cpp\sensors\temperature_sensor.cpp:17 TemperatureSensor::debounce(float): std::fabs(v - lastValue) < TEMP_DEBOUNCE_C
Sources: TemperatureSensor::read() (tests\fixtures\cpp\sensors\temperature_sensor.cpp:4)
Processors: SensorManager::updateTemperature() (tests\fixtures\cpp\sensors\sensor_manager.cpp:15), TemperatureSensor::process() (tests\fixtures\cpp\sensors\temperature_sensor.cpp:23)
Validators: Validator::check(float) (tests\fixtures\cpp\sensors\validator.cpp:3), TemperatureSensor::debounce(float) (tests\fixtures\cpp\sensors\temperature_sensor.cpp:16), TemperatureSensor::inRange(float) (tests\fixtures\cpp\sensors\temperature_sensor.cpp:12), TemperatureSensor::isStale(unsigned long) (tests\fixtures\cpp\sensors\temperature_sensor.cpp:8)
Consumers: Display::update(float) (tests\fixtures\cpp\sensors\display.cpp:3)
```

The question "What components are affected if updateTemperature() changes?" answers with the same block that `effect` prints.

---

## 4. TUI Interface

The TUI (Terminal User Interface) provides interactive access to all four objectives with real-time progress and a results browser.

```bash
python tui.py
```

**Screen 1 — Config**
```
┌─────────────────────────────────────────────┐
│  DO-178C RAG Engine                         │
│                                             │
│  Codebase directory (required)              │
│  [/path/to/cpp/code________________]        │
│                                             │
│  Base build directory (optional)            │
│  [/path/to/base_build______________]        │
│                                             │
│  LRU documents directory (optional)         │
│  [/path/to/lru_docs________________]        │
│                                             │
│  Standards profile file (optional)          │
│  [/path/to/standards.yaml__________]        │
│                                             │
│  Sensor templates file (optional)           │
│  [/path/to/sensor_templates.yaml___]        │
│                                             │
│  [x] Use semantic search                    │
│      (needs sentence-transformers)          │
│                                             │
│  Output format:  [Markdown ▼]               │
│  DAL level:      [DAL B ▼]                  │
│                                             │
│  [Preview Standards] [Run Analysis] [Quit]  │
└─────────────────────────────────────────────┘
```

Fill in the paths, choose the output format and DAL level, then press **Run Analysis**.

- **Codebase directory** is required. **Run Analysis** is blocked with a message when it is empty or does not exist.
- **Standards profile file** and **Sensor templates file** are optional. An empty field uses the value from `config.yaml` (`standards_file`, `sensor_templates_file`). A path that does not exist blocks **Run Analysis** with a message.
- **Use semantic search** is on by default and needs sentence-transformers. When it is off, coupling uses keyword matching, and the progress log says so.
- **Output format** defaults to Markdown. DOCX is the other choice.
- **Preview Standards** uses only the file typed in the **Standards profile file** field. It does not fall back to `standards_file` in `config.yaml`, and it asks for a file when the field is empty. It shows the structured standards profile extracted from that file, and **Back** returns to the Config screen.

**Screen 2 — Analysis Progress**

Progress rows for Parsing, Virtual, Dead code, Standards, Coupling and Sensors, plus a live log showing what the engine is doing. The Sensors row reads "skipped" when no sensor templates are configured. Automatically advances to Results when complete. A failed run prints the error in red in the log; **Esc** returns to the Config screen once the run has finished.

While the TUI runs, log messages are written to `tui.log` in the configured `cache_dir` instead of the terminal, so they do not draw over the screen. If that file cannot be created the TUI says so and discards the log output. The terminal log returns when the TUI exits.

**Screen 3 — Results Browser**

Tabs:

| Tab | Shows |
|-----|-------|
| **Standards** | Code standard violations and the compliance score |
| **Virtual Functions** | A tree: each root class, an `Inheritance` node with its derived classes (shown only when there are any), and the class's virtual functions. A function shows `pure virtual` or a reachable ✓ / ✗ mark; each override shows reachable and instantiated marks. Call sites are leaves. The base-build change table is below the tree |
| **Dead Code** | Table of ID, file, symbol, lines, type, confidence and reason; the warnings and entry points are listed above it |
| **Function Effect** | A method name input and an **Analyze** button. Shows the `effect` output for every match, or a not-found message |
| **Sensors** | Sensor × check status (corrupt, stale, debounce, instantiated). Below it, a **Sample tester**: pick a template, enter a CSV path or inline samples (`t:v, t:v, ...`), then press **Run checks** (or Enter in the input). The summary line covers all samples. The result table shows the first 200 samples plus every flagged one, with a line `showing N of M samples (all flagged rows included)`. A CSV path uses only the selected sensor's rows and reports an error if that sensor is not in the file. Disabled with a hint when no templates are loaded |
| **Coupling** | Per-LRU coupling table |
| **Checklist** | SQA checklist rows |

The tabs hold tables, a tree and text panes. Navigate with the keys below.

Keyboard shortcuts on the results screen:

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Move focus between the tab bar, the tables or tree, and the inputs |
| `←` / `→` (tab bar focused) | Switch result tabs |
| `↑` / `↓` | Move within a table or the virtual-function tree |
| `c` | Open the chat (Ask AI) |
| `r` | Re-run analysis (back to Config screen) |
| `q` | Quit |
| `Esc` | Back to previous screen |
| `Ctrl+C` | Quit (any screen) |

**Screen 4 — Chat** (answers from the analysis facts; the model explains them only when `ollama_enabled` is true and Ollama answers)

Ask questions about your analysis results in natural language. In the chat, **Enter** (or **Send**) asks the question, **Esc** returns to the results, and **Ctrl+L** clears the chat log on screen. See [Section 6](#6-chat-interface).

---

## 5. CLI Reference

```bash
python main.py analyze CODEBASE [OPTIONS]
```

| Argument / Option | Required | Default | Description |
|-------------------|----------|---------|-------------|
| `CODEBASE` | Yes | — | Path to C++ source directory |
| `--base-build PATH` | No | same as CODEBASE | Base build for virtual function diff |
| `--lru-docs PATH` | No | — | Directory with LRU PDF/DOCX specs |
| `--standards-file PATH` | No | — | Custom DAL standards file (YAML, JSON, PDF or DOCX) |
| `--config PATH` | No | `config.yaml` | Config file path |
| `--output-format` | No | `docx` | `docx` or `markdown` |
| `--sensor-templates PATH` | No | `sensor_templates_file` | Sensor template YAML; enables the sensor report |
| `--no-semantic-search` | No | off | Skip the embedding model (coupling uses keyword matching) |

Query commands run the analyses without the embedding model and write no reports, except `effect`, which writes its report. Each one has an example and sample output in its section:

| Command | Purpose | Section |
|---------|---------|---------|
| `virtuals CODEBASE` | Virtual functions, overrides, reachability, call sites, class trees | R1 |
| `effect CODEBASE METHOD` | What a method reads, writes and calls; its coupled components and what is affected if it changes | R2 |
| `sensors CODEBASE --templates FILE` | Static corrupt / stale / debounce check per templated sensor | R3 |
| `sensor-check SAMPLES_CSV --templates FILE` | Replay recorded samples through the rules | R3 |
| `ask CODEBASE QUESTION` | Answer from the static analysis; the local LLM (if enabled) only explains the facts | R4 |
| `validate-config [--config PATH]` | Check that the config file loads | — |

`--templates` is an alias of `--sensor-templates`. These errors exit with status 1, and the messages differ:

- `Error: ...`: an invalid sensor template file, for example `Error: Sensor templates <path>: Sensor 'X': stale.threshold_ms must not be negative, got -5`; a codebase with no C++ files, `Error: No C++ files found in <dir>`; a CSV without the `sensor,timestamp_ms,value` header, `Error: <path>: CSV header must contain sensor,timestamp_ms,value`.
- `Sensor templates file not found: <path>`: the `--templates` path does not exist.
- `No sensor templates: pass --templates or set sensor_templates_file in the config`: `sensors` or `sensor-check` run without any template.
- `No template for sensor(s) in the CSV: <names>`: `sensor-check` found a sensor name with no template. It prints the other sensors' results first.
- `Function '<name>' not found. Close names: ...` (`effect`), for example `Function 'updateTemprature' not found. Close names: updateTemperature, disabledFeature, update`.

Progress and log lines go to stderr, so `2>/dev/null` leaves only the results on stdout.

Example on the sensors fixture (real output from a run with `output_dir: .superpowers/out` and `ollama_enabled: false` in a scratch config; with the default `config.yaml` the reports go to `output/`):

```bash
python main.py analyze tests/fixtures/cpp/sensors --sensor-templates tests/fixtures/sensor_templates.yaml --no-semantic-search --output-format markdown
```

```text
Reports in .superpowers/out/
  .superpowers\out\virtual_analysis.md
  .superpowers\out\coupling_analysis.md
  .superpowers\out\dead_code_report.md
  .superpowers\out\standards_report.md
  .superpowers\out\sensor_analysis.md
  .superpowers\out\sqa_checklist.md
```

With `ollama_enabled: true` (the shipped `config.yaml`), the reports also get an LLM narrative section when Ollama is reachable. The findings are the same.

Rows from `dead_code_report.md` (its findings table; four of fourteen rows):

```text
| ID | File | Function/Symbol | Lines | Category | Disposition | Reason | Confidence | Status |
|----|------|-----------------|-------|----------|-------------|--------|------------|--------|
| DC-METH-CE163B0F | altitude_sensor.h | AltitudeSensor::read() | 8 | unused_export | Investigate | override in class never instantiated | medium | open |
| DC-METH-80BD632A | display.cpp | Display::clear(int) | 7-9 | dead_code | Remove | no call path from entry points [TIMER0_IRQHandler(), main()] | high | open |
| DC-CFG-6F03AB16 | main.cpp | disabledFeature() | 49-51 | deactivated_code | Justify as Deactivated | excluded from the build by `#if 0` | high | open |
| DC-FRAG-67F264C3 | main.cpp | computeChecksum(int) / x = 0; | 30 | dead_fragment | Fix Fragment | statements after unconditional `return` can never execute | high | open |
```

One item from `dead_code_findings.json`:

```json
{
  "name": "Display::clear(int)",
  "file_path": "tests\\fixtures\\cpp\\sensors\\display.cpp",
  "line_number": 7,
  "category": "dead_code",
  "do178c_disposition": "Remove",
  "finding_id": "DC-METH-80BD632A",
  "end_line": 9,
  "symbol_type": "method",
  "reason": "no call path from entry points [TIMER0_IRQHandler(), main()]",
  "confidence": "high",
  "review_status": "open"
}
```

The JSON item is abridged; the file also has `coverage_impact`, `function_name`, `evidence`, `removal_impact`, `reviewer_justification` and `reviewer` for each finding.

```bash
python main.py validate-config
```

```text
Config OK — DAL B, output: output
```

```bash
python main.py validate-config --config /path/to/custom.yaml
```

**Examples:**

```bash
# Minimal — just parse and analyze one directory
python main.py analyze src/
```

```bash
# Full production run
python main.py analyze src/current/ \
  --base-build src/baseline/ \
  --lru-docs docs/lru_specs/ \
  --output-format docx \
  --config production.yaml
```

```bash
# CI pipeline usage (exits 1 on CRITICAL violations)
python main.py analyze src/ && echo "PASS" || echo "CRITICAL VIOLATIONS FOUND"
```

---

## 6. Chat Interface

The chat answers questions about the analysis results. Every answer starts from the static-analysis facts, so the chat works without a model. A local LLM (Ollama) is optional, and it only explains those facts.

**How to open the chat (TUI):**
1. Run the analysis in the TUI (`python tui.py`).
2. On the Results screen, press **`c`**.
3. Type your question and press Enter.

The chat status line shows one of these states:

- `Ollama disabled in config: using structured offline answers`: `ollama_enabled` is false. Ollama is not contacted.
- `Checking Ollama…`: `ollama_enabled` is true and the check when the chat opens has not finished.
- `Ollama ready (<model>): facts from static analysis, explained by the LLM`: the server answers.
- `Ollama unavailable: using structured offline answers`: `ollama_enabled` is true, but the server does not answer.
- `Thinking…`: a question is being answered. It stays until the last queued answer is shown, then the state above returns.

Questions asked while an answer is pending are queued and answered in the order they were asked. Each answer appears when it is ready.

**Optional: explanations from Ollama**

```bash
ollama serve
```

```bash
ollama pull llama3.1:8b   # or your preferred model
```

Set `ollama_url` and `ollama_model` in `config.yaml` to match your server and model, and set `ollama_enabled: true`. The chat uses the model only when `ollama_enabled` is true and the server answers. The chat checks the server once, when it opens, and keeps that result until the chat is closed: start Ollama before opening the chat, or close and reopen the chat after starting it. `ask` follows the same rule and checks the server each time it runs.

**How an answer is built:**
- **Facts** (structural questions): the virtual functions a class implements or overrides, what a method reads, writes and calls and what is affected if it changes, the checks applied to a sensor, and a class's inheritance. These are the same facts that `ask` and `effect` print.
- **Summaries** (counts and lists): standards, virtual changes, dead code and coupling. A dead-code question lists the functions to remove.
- **Not found**: a name that is not in the analysis gets a not-found message with close names. A misspelled name is not answered with a summary, for example `Sensor 'TemperatureSensr' not found. Close names: TemperatureSensor, PressureSensor, SpeedSensor, AltitudeSensor`.
- **Explanation**: with a model, the facts are printed first, then `Explanation (local LLM):` and the model's text. The model only explains the facts. If no specific fact matches, the model is given a summary of all results, and its answer is labelled `Explanation (local LLM — no specific static-analysis facts matched; summary facts used)`. If the model fails, the facts are shown alone.

**Example questions** (checked on `tests/fixtures/cpp/sensors`; there is no base build, so the reverification count is 0):

```text
What are the virtual functions implemented by TemperatureSensor?
What components are affected if updateTemperature() changes?
What validation checks are applied to TemperatureSensor?
How many critical violations are there?
How much dead code was found?
Which functions are deactivated?
Which functions need reverification?
What is the compliance score?
```

---

## 7. Docker — Laptop Mode

Laptop mode uses `llama3.1:8b` (CPU only, ~8 GB RAM). Good for development and testing.

```bash
# 1. Start the stack
docker compose -f docker-compose.yml -f docker-compose.laptop.yml up -d
```

```bash
# 2. Pull the model (first time only, ~5 GB download)
docker compose -f docker-compose.yml -f docker-compose.laptop.yml \
  exec ollama ollama pull llama3.1:8b
```

```bash
# 3. Prepare your data directories
mkdir -p input output lru_docs
```

```bash
# Copy your C++ source files
cp -r /path/to/cpp/source/* input/
```

```bash
# Copy LRU spec documents (optional)
cp /path/to/lru/specs/* lru_docs/
```

```bash
# 4. Run all four analyses
docker compose -f docker-compose.yml -f docker-compose.laptop.yml \
  run rag-engine analyze /data/input \
    --lru-docs /data/lru_docs \
    --output-format markdown
```

```bash
# 5. View results
ls output/
```

```bash
cat output/standards_report.md
```

**Laptop mode with base build comparison:**
```bash
# Mount base build as a second volume
docker compose -f docker-compose.yml -f docker-compose.laptop.yml \
  run -v /path/to/base_build:/data/base \
  rag-engine analyze /data/input \
    --base-build /data/base \
    --output-format markdown
```

---

## 8. Docker — Production Mode

Production mode uses `qwen2.5-coder:72b` on NVIDIA GPU (~32 GB VRAM). For final certification runs.

**Prerequisites:**
- NVIDIA GPU with 32+ GB VRAM (A100, H100, or 2× A6000)
- `nvidia-container-toolkit` installed on the host
- Docker with GPU support enabled

```bash
# 1. Verify GPU is accessible
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

```bash
# 2. Start the production stack
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

```bash
# 3. Pull the production model (first time only, ~45 GB)
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  exec ollama ollama pull qwen2.5-coder:72b
```

```bash
# 4. Run full production analysis
mkdir -p input output lru_docs
```

```bash
cp -r /path/to/certified/source/* input/
```

```bash
cp /path/to/lru/specs/* lru_docs/
```

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  run rag-engine analyze /data/input \
    --base-build /data/base \
    --lru-docs /data/lru_docs \
    --output-format docx
```

```bash
# 5. Retrieve DOCX reports
ls output/
# virtual_analysis.docx
# coupling_analysis.docx
# dead_code_report.docx
# standards_report.docx
# sqa_checklist.md
```

**Stop the stack:**
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml down
```

---

## 9. Output Files

All outputs written to `output/` (configurable via `config.yaml`).

| File | DO-178C Ref | Contents |
|------|-------------|---------|
| `virtual_analysis.docx` | §12 | Virtual function change classification table, Cat1/Cat2, reverification scope per function |
| `coupling_analysis.docx` | §6.3.1.b | Per-LRU control and data coupling items, and a risk level (LOW / MEDIUM / HIGH) from the item counts |
| `dead_code_report.docx` | §6.4.2.2 | Dead/deactivated code list, dispositions, structural coverage impact |
| `standards_report.docx` | §5.1 | All violations by severity, compliance score, correction guidance, MISRA cross-reference |
| `sqa_checklist.md` | §11 | SQA checklist auto-filled with PASS/FAIL/OPEN status and evidence |
| `dead_code_findings.json` | §6.4.2.2 | Every dead-code finding with its ID, confidence, evidence, removal impact and review status; warnings |
| `sensor_analysis.docx` | R3 | Sensor × check status with expected and found values; written when a sensor templates file is set |
| `function_effect_<method>.md` | R2 | Written by the `effect` command for each query |

The checklist also has VA-03 (every virtual override is reachable or reviewed), SEN-01 and SEN-02 when sensor templates are set.

Reports are **DO-178C §11 life cycle data artifacts** — they can be submitted directly to a DER (Designated Engineering Representative) as part of a certification package.

---

## 10. Configuration

`config.yaml` (create in project root or pass via `--config`):

```yaml
# Certification level: A (strictest, MC/DC) → D (least strict)
dal_level: B

# Output directory for reports
output_dir: output

# Parallel analysis workers (reduce if low RAM)
max_workers: 4

# Dead code analysis: where does execution start?
entry_points:
  - main
  - initSystem       # add your actual entry points
  - handleBootEvent
entry_point_patterns:          # interrupt handlers and RTOS tasks, matched by name
  - '*_IRQHandler'
  - '*_ISR'
  - 'ISR_*'
  - '*Task'

# LRU names (currently unused: per-LRU coupling comes from the documents given with --lru-docs)
lru_names:
  - ADS
  - FCS
  - FMS
  - FADEC

# Complexity thresholds (violations flagged when exceeded)
cyclomatic_complexity_max: 10
function_length_max: 50
nesting_depth_max: 5
param_count_max: 7

# Dead-code review decisions (YAML file; see "Recording reviews" in Objective 3).
# Example value: the shipped config.yaml leaves it empty (no dispositions applied).
dead_code_dispositions: reviews/dispositions.yaml

# Sensor templates for the R3 checks (optional; --templates overrides it).
# Example value: the shipped config.yaml leaves it empty (sensor checks disabled).
sensor_templates_file: tests/fixtures/sensor_templates.yaml

# Ollama LLM for narrative enrichment (optional)
ollama_enabled: false           # set true when Ollama is running
ollama_model: llama3.1:8b       # laptop model
# ollama_model: qwen2.5-coder:72b  # production model
ollama_url: http://localhost:11434
```

Relative paths in `standards_file`, `sensor_templates_file` and `dead_code_dispositions` are resolved against the directory of the config file. `entry_point_patterns` and the sensor template fields are described in Objective 3 and R3.

Config values are type-checked. A list field (`entry_points`, `entry_point_patterns`, `lru_names`) may be written as a single string; it is used as a one-item list and a warning is logged. A value of the wrong type stops the run with an error that names the field. A `config.yaml` that is not valid YAML stops it with an error that names the file and line. Neither prints a traceback.

---

## 11. Assumptions and Open Questions

These are assumptions the current implementation makes, and questions for the mentor. They are not yet confirmed by Honeywell.

**Assumptions**
- **Stale and debounce algorithms and thresholds** are configurable defaults pending Honeywell confirmation: `stale` (`threshold_ms`, `detect_frozen`), `debounce` (`threshold`, `count`) and `valid_range` in the sensor template. The algorithms are the ones described in section R3.
- **No numeric coupling score.** Coupling output (R2 and the coupling report) identifies the dependencies (control and data) and the components involved. It does not weight or sum them. The coupling report keeps its existing per-LRU risk level, which is a heuristic.
- **Structural coverage is not integrated.** Dead-code "coverage impact" values are estimates of source lines and decision points from the parse. gcov or llvm-cov results are not read.
- **Dead-code verdicts are heuristics for review.** `unused_export` findings are "Investigate", not removal candidates. Reviews are recorded in the dispositions file; they do not change the verdict.

**Open questions for the mentor** (from the requirements PDF, section 14):
1. Which report fields are needed for virtual functions?
2. What coverage is expected (and which criteria)?
3. Which coupling metric is wanted, if any?
4. What is the sensor hierarchy (Parameter Group → I/O Group → Sensor → Sensor Data)?
5. What counts as "stale"?
6. What are the corrupt ranges for each sensor?
7. Which debounce algorithm should be used?
8. Is there an existing Honeywell sensor template to follow?
9. Which DO-178 checklist should the SQA report follow?

---

## 12. Troubleshooting

### Semantic search unavailable (keyword matching used)

```
Semantic search unavailable (NameError: name 'nn' is not defined); continuing with keyword matching
```

The embedding model needs sentence-transformers. On some machines it cannot be imported, for example transformers 5.x with torch < 2.4. The analysis then continues with keyword matching and logs this one warning. Fix it with `pip install "transformers<5"`, or by upgrading torch to 2.4 or newer. `--no-semantic-search` skips the embedding model altogether.

---

### Dead-code disposition warnings

- `stale disposition DC-...: no finding has this ID`: the function or file was renamed, so the ID changed. Update the ID in the dispositions file.
- `Disposition DC-... ignored: no justification` (or `no status`): add both keys to the entry.

---

### TensorFlow import error on Windows

```
RuntimeError: Failed to import transformers.modeling_utils
```

Fix:
```powershell
$env:USE_TF = "0"
$env:USE_TORCH = "1"
python main.py analyze ...
```

Or add to your shell profile permanently.

---

### No C++ files found

```
No C++ files found in /path/to/code
```

The parser looks for `*.cpp`, `*.h` and `*.hpp`. Files with other extensions (for example `.cc` or `.cxx`) are not analysed. Rename them or copy them to these extensions.

---

### All code appears dead

Entry points are not set correctly. Edit `config.yaml`:
```yaml
entry_points:
  - your_actual_main_function
  - your_rtos_task_entry
```

---

### Coupling analysis shows no results

LRU names must match the signals in your codebase. Either:
1. Provide LRU spec documents in `--lru-docs` so the engine can extract signal names
2. Or add LRU names to `config.yaml` that appear in your variable/function names

---

### Ollama not available or failing (facts are shown instead)

If Ollama is not reachable, or `ollama_enabled` is false, answers and reports use the static-analysis facts only. A failing Ollama call logs a warning, for example `Ollama generate failed: ... 500 Internal Server Error`, and the facts are still printed. This is not a failure of the analysis. To enable the explanations:
```bash
ollama serve              # in a separate terminal
```

```bash
ollama pull llama3.1:8b   # or your model
```

---

### Docker: GPU not found

```
Error: could not select device driver "nvidia"
```

Install nvidia-container-toolkit:
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-container-toolkit
```

```bash
sudo systemctl restart docker
```

---

### Second run is slow (cache not working)

The incremental cache lives in `.rag_cache/`. If it was deleted or the cache dir changed:
```yaml
# config.yaml
cache_dir: .rag_cache   # must be a writable directory
```

Second and subsequent runs on unchanged files should be ~70% faster.
