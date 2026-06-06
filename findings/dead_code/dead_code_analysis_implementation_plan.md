# Proper Dead Code Analysis - Implementation Plan

## 1. Objective

Improve the current dead-code feature so it can identify the possibility of:

- Unused or unreachable functions
- Unused or unreachable methods
- Unused variables
- Unreachable code fragments inside functions
- Code that is present but excluded from a specific build configuration

The analysis must cross-check whether each item can be invoked directly or
indirectly during runtime. It must also explain the possible impact of removing
the item.

The tool should treat every result as a **finding that requires review**, rather
than automatically claiming that the item is confirmed dead code.

---

## 2. Required Report Output

Every finding must include at least:

| Required field | Description |
|---|---|
| Code file name | File containing the finding |
| Function name | Containing function, method, or affected function |
| Line numbers | Start and end lines of the finding |

The improved report should also include:

| Additional field | Description |
|---|---|
| Finding ID | Unique ID such as `DC-FUNC-001` |
| Symbol name | Fully qualified function, method, or variable name |
| Finding type | Function, method, variable, fragment, or configuration code |
| Detection reason | Why the item appears unused or unreachable |
| Reachability evidence | Entry points and discovered call paths |
| Confidence | High, medium, low, or uncertain |
| Removal impact | Possible callers, dependencies, and behavior affected |
| Review status | Open, confirmed dead, false positive, or deactivated |
| Reviewer justification | Human explanation and supporting evidence |

Example:

```text
Finding ID: DC-FUNC-001
File: src/motor.cpp
Function: Motor::unusedDiagnostic
Lines: 120-138
Type: Unreachable method
Reason: No call path exists from any configured entry point
Confidence: Medium
Removal impact: Verify diagnostic interface and maintenance build before removal
Review status: Open
```

---

## 3. Problems in the Current Implementation

The current implementation provides a useful prototype, but it is not yet a
complete dead-code analysis.

### 3.1 Function identity is not unique

The call graph uses only the short function name.

For example, these methods are currently difficult to distinguish:

```cpp
Motor::start()
Battery::start()
```

Both can become a single graph node named `start`.

### 3.2 Call detection uses a regular expression

The current graph searches function bodies for text that looks like a function
call. This cannot reliably resolve:

- Overloaded functions
- Namespaced functions
- Class methods
- Virtual dispatch
- Function pointers
- Lambdas and callbacks
- Template functions
- Macro-generated calls

### 3.3 Missing entry points cause false findings

The current default entry point is only `main`. Embedded software may also have:

- Startup functions
- RTOS task functions
- Interrupt handlers
- Registered callbacks
- Exported APIs
- Test-harness entry points
- Hardware-triggered functions

If these are missing, reachable functions may be incorrectly reported.

### 3.4 Virtual and callback-like functions are automatically excluded

The current detector excludes virtual functions and functions with callback-like
names. This avoids some false positives, but it can also hide real dead code.

These functions should be reported with lower confidence and review evidence,
not silently removed from the findings.

### 3.5 Only functions are analyzed

The current feature does not identify:

- Unused variables
- Unreachable statements after `return`, `break`, `continue`, or `throw`
- Constant-condition branches
- Unused methods and classes
- Build-configuration-specific code

### 3.6 Removal impact is too approximate

The current coverage impact is calculated from function size and complexity.
It does not explain dependencies or the behavior that could be affected by
removal.

---

## 4. Target Analysis Design

Use several evidence sources and combine their results:

```text
C++ source + build configuration
              |
              v
      Compiler-aware parsing
              |
              v
   Symbol table and call graph
              |
      +-------+--------+
      |                |
      v                v
Reachability      Local data-flow and
analysis          control-flow analysis
      |                |
      +-------+--------+
              |
              v
    Candidate dead-code findings
              |
              v
 Linker/build/coverage cross-checks
              |
              v
 Human review and final disposition
```

No single source should independently confirm dead code. The final report must
show which checks support or contradict each finding.

---

## 5. Implementation Phases

## Phase 1 - Make Function Reachability Reliable

### 5.1 Use unique symbol identities

Update `rag_engine/models.py`.

Add these fields to `FunctionDef`:

```python
qualified_name: str       # Example: Motor::start
signature_id: str         # Example: Motor::start(int)
end_line_number: int
namespace: str | None
class_name: str | None
linkage: str              # internal, external, unknown
```

Use `signature_id` as the call-graph node ID instead of the short function name.

### 5.2 Improve parser output

Update `rag_engine/core/parser.py` to extract:

- Fully qualified names
- Class and namespace ownership
- Start and end line numbers
- Overload signatures
- Calls represented by AST call-expression nodes
- Function-address references such as `&handleEvent`
- Callback registrations where statically visible

Do not use body-text regular expressions as the primary call detector.

### 5.3 Rebuild the call graph using resolved symbols

Update `rag_engine/core/graph_builder.py`.

The graph should:

- Use unique function signature IDs
- Store direct-call edges
- Store unresolved-call edges separately
- Mark virtual, indirect, callback, and external-call uncertainty
- Preserve evidence for why each edge exists
- Detect recursion without removing valid graph edges

Suggested edge evidence:

```python
@dataclass
class CallEdge:
    caller_id: str
    callee_id: str | None
    file_path: str
    line_number: int
    call_type: Literal[
        "direct", "virtual", "function_pointer", "callback_registration",
        "external", "unresolved"
    ]
    confidence: Literal["high", "medium", "low"]
```

### 5.4 Support complete entry-point configuration

Update `rag_engine/config.py` and `config.yaml`.

Suggested configuration:

```yaml
dead_code:
  entry_points:
    - main
    - initSystem
  entry_point_patterns:
    - "*Task"
    - "*ISR"
  exported_api_files: []
  callback_registration_functions: []
  analyze_headers: true
  report_virtual_candidates: true
```

Entry points must be validated. The report should warn when:

- No configured entry point exists in the analyzed build
- An entry-point pattern matches nothing
- The call graph has no edges
- A large percentage of functions appears unreachable

### 5.5 Change classification behavior

Update `rag_engine/features/dead_code_detector.py`.

Do not automatically exclude virtual functions or callback-like functions.
Report them as candidates with lower confidence.

Suggested classifications:

| Classification | Meaning |
|---|---|
| `unreachable_function` | No known path from an entry point |
| `unused_method` | Method has no known caller or dispatch evidence |
| `possible_external_entry` | May be called outside the analyzed build |
| `possible_callback` | May be called indirectly |
| `deactivated_code` | Intentionally inactive for a documented configuration |
| `confirmed_dead` | Reviewed and confirmed to have no operational purpose |
| `uncertain` | Evidence is insufficient |

The automated detector should initially use `Open` review status. Only a human
review or an approved workflow should mark a finding as `confirmed_dead`.

---

## Phase 2 - Detect Unreachable Code Fragments

Create:

```text
rag_engine/features/control_flow_analyzer.py
```

Build a control-flow graph for each function and detect:

- Statements after unconditional `return`
- Statements after unconditional `throw`
- Statements after unconditional `break` or `continue` within the same block
- Branches with compile-time constant false conditions
- Unreachable `switch` cases where safely provable
- Empty or impossible paths

Each fragment finding must contain:

- Containing function
- Start and end lines
- Reason the fragment is unreachable
- Source snippet or statement type
- Confidence and limitations

Avoid claiming a branch is dead when macro values or build configuration are
unknown.

---

## Phase 3 - Detect Unused Variables

Create:

```text
rag_engine/features/data_flow_analyzer.py
```

Track declarations, reads, and writes for:

- Local variables
- Parameters
- Class members where practical
- File-level and global variables

Report separate cases:

| Finding | Meaning |
|---|---|
| `never_read` | Variable is assigned but its value is never used |
| `never_written` | Variable is declared but never assigned |
| `unused_parameter` | Parameter is never used |
| `dead_store` | A value is overwritten before it is read |
| `write_only_global` | Global is written but never read in the analyzed build |

Consider volatile variables, hardware registers, assembly interaction, and
externally visible symbols before reporting them with high confidence.

---

## Phase 4 - Add Build and Configuration Awareness

Dead code is meaningful only for a specific build.

### 5.6 Read the real build configuration

Prefer reading `compile_commands.json` generated by CMake or another build
system. Capture:

- Compiler flags
- Include directories
- Preprocessor definitions
- Source files included in the build
- Build variant or target name

Create:

```text
rag_engine/core/build_context.py
```

Suggested model:

```python
@dataclass
class BuildContext:
    target_name: str
    source_files: list[str]
    include_paths: list[str]
    defines: dict[str, str]
    compiler_flags: list[str]
```

Analyze each required build variant separately. Code can be unreachable in one
variant and intentionally active in another.

### 5.7 Cross-check final binary and linker information

Where available, read:

- Linker map files
- Symbol tables
- Object-file references
- Removed-section reports from linker garbage collection

Create:

```text
rag_engine/core/linker_evidence.py
```

Linker evidence can strengthen a finding, but absence from one binary does not
prove the source is dead in every build variant.

---

## Phase 5 - Cross-Check Runtime Coverage

Add optional import of structural-coverage data, such as:

- `gcov` output
- LLVM coverage reports
- Existing project-specific coverage exports

Create:

```text
rag_engine/features/coverage_cross_check.py
```

Coverage data should add evidence such as:

```text
Not executed by the supplied test suite
```

It must not automatically mean:

```text
Confirmed dead code
```

Uncovered code may represent missing tests rather than dead code.

---

## Phase 6 - Model Findings and Removal Impact

Replace or extend `DeadCodeItem` in `rag_engine/models.py`.

Suggested model:

```python
@dataclass
class DeadCodeFinding:
    finding_id: str
    finding_type: str
    symbol_name: str
    qualified_name: str | None
    file_path: str
    function_name: str | None
    start_line: int
    end_line: int
    reason: str
    confidence: Literal["high", "medium", "low", "uncertain"]
    evidence: list[str]
    contradictory_evidence: list[str]
    possible_removal_impact: list[str]
    review_status: Literal[
        "open", "confirmed_dead", "false_positive", "deactivated", "needs_review"
    ]
    reviewer_justification: str | None
```

### Removal-impact checks

For every finding, inspect:

- Known callers and callees
- Data read and written by the item
- Referenced global variables
- Interfaces or exported symbols
- Requirement trace references
- Build variants containing the item
- Tests and documentation referencing the item

The impact statement must clearly say when evidence is incomplete.

---

## Phase 7 - Improve the Report

Update:

```text
rag_engine/templates/dead_code_report.jinja2
rag_engine/reporting/report_generator.py
```

The report should contain:

1. Build and analysis scope
2. Configured and discovered entry points
3. Analysis limitations and warnings
4. Summary by finding type and confidence
5. Detailed findings
6. Unresolved indirect or external calls
7. Human review status
8. Removal-impact assessment

Suggested finding table:

| ID | File | Function/Symbol | Lines | Type | Reason | Confidence | Status |
|---|---|---|---|---|---|---|---|

The detailed section should include evidence and possible removal impact.

Generate both machine-readable and human-readable outputs:

```text
output/dead_code_report.md
output/dead_code_report.docx
output/dead_code_findings.json
```

The JSON output allows findings to be reviewed, filtered, and compared between
builds.

---

## Phase 8 - Add Review and Suppression Support

Create a reviewed-disposition file:

```text
dead_code_dispositions.yaml
```

Example:

```yaml
findings:
  DC-FUNC-001:
    status: false_positive
    justification: Called by the bootloader through exported symbol MotorInit.
    evidence:
      - bootloader_interface.md
    reviewer: reviewer-name
```

Rules:

- Suppressions require a justification.
- Suppressions must use stable finding IDs.
- Stale suppressions must be reported.
- Changed source lines must reopen affected findings for review.

---

## 6. Testing Plan

Expand `tests/test_dead_code.py` and add focused fixtures.

### Required test cases

- Directly reachable function
- Indirectly reachable function
- Truly orphaned function
- Two classes with methods having the same short name
- Overloaded functions
- Namespaced functions
- Virtual method with and without a known implementation
- Registered callback
- Function-pointer call
- External/exported API
- RTOS task and ISR entry points
- Recursive function
- Code after `return`
- Constant false branch
- Unused local variable
- Unused parameter
- Dead store
- Volatile/hardware variable that must not be reported as confirmed dead
- Build-variant-specific code
- Missing-entry-point warning
- Empty-call-graph warning

### Acceptance expectations

- Every finding contains file, function/symbol, and start/end lines.
- Direct and indirect calls are traced correctly for supported call types.
- Unresolved calls are visible in the report.
- Virtual functions and callbacks are not silently ignored.
- Findings contain a reason, confidence, evidence, and removal-impact note.
- No automated finding is marked `confirmed_dead` without review evidence.

---

## 7. Recommended Delivery Order

Implement the work in this order:

1. Add unique function identities and end-line information.
2. Replace regex-based direct-call detection with AST call expressions.
3. Add entry-point validation and explicit uncertainty reporting.
4. Redesign finding models and report fields.
5. Add control-flow analysis for unreachable fragments.
6. Add local-variable data-flow analysis.
7. Add build-context and build-variant support.
8. Add linker and runtime-coverage cross-checks.
9. Add disposition/suppression workflow.
10. Validate against representative production C++ code.

Phases 1 through 4 provide the minimum useful foundation. Later phases increase
confidence and reduce false findings.

---

## 8. Definition of Done

The dead-code feature is ready for practical use when:

- It analyzes a defined build target and configuration.
- It identifies functions, methods, variables, and unreachable fragments.
- It traces supported direct and indirect runtime call paths.
- It clearly reports unresolved calls and analysis limitations.
- It does not silently exclude virtual functions or callbacks.
- Every finding includes file name, function/symbol name, and line range.
- Every finding explains why it was reported.
- Every finding includes confidence, evidence, and possible removal impact.
- Human reviewers can confirm, reject, or justify every finding.
- Automated tests cover the required call and dead-code scenarios.
- Results are produced in Markdown, DOCX, and JSON formats.

The final system should answer:

```text
Why does this item appear unreachable?
What evidence supports or contradicts the finding?
What could be affected if it is removed?
What does a reviewer need to verify before making a decision?
```
