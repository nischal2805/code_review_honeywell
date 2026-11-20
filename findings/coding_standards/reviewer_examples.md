# Coding Standards Evaluation Examples

## Test setup

Run the sample corpus with the uploaded PDF:

```powershell
python main.py validate-config --coding-standards-pdf samples/coding_standards_strict.pdf
python main.py analyze sample_codebase --coding-standards-pdf samples/coding_standards_strict.pdf --output-format markdown
```

The validator extracts rule rows from the PDF, resolves the limits for the selected DAL, parses every C++ function, and emits one `Violation` for each detected rule breach. The report records the PDF path under `Coding Standards Source`.

Each report row has both `Function start line` and `Violation line`. For example, `DO178-GOTO` in `violations.cpp` reports function start line `3` and violation line `12`.

## Examples to show a reviewer

| Finding | Source example | Evaluation | Expected flag |
|---|---|---|---|
| Naming | `void bad_name(...)` | Function name is checked against the configured PascalCase regex. | `DO178-NAMING`, MINOR |
| Parameters | `bad_name` has 8 parameters; strict DAL B limit is 6. | Parsed parameter count is compared with `SCS-M-04` for DAL B. | `DO178-PARAM-COUNT`, MINOR |
| Complexity | The chained `if` statements in `bad_name`. | Tree-sitter calculates cyclomatic complexity; it is compared with `SCS-M-01`. | `DO178-CC`, MEDIUM |
| Dynamic memory | `new int(a)` and `delete value`. | The function body is scanned for allocation/deallocation constructs and the PDF policy is applied. | `DO178-DYNAMIC-MEM`, CRITICAL for FORBIDDEN |
| Goto | `goto done;`. | The function body is scanned for a goto statement and matched to `SCS-L-02`. | `DO178-GOTO`, CRITICAL |
| Unbounded loop | `while (true)`. | The source file is scanned for unconditional loops without a proof-bearing condition. | `DO178-UNBOUNDED-LOOP`, MAJOR |
| Exceptions | `try`, `throw`, and `catch` in `ThrowError`. | Exception tokens are checked against `SCS-L-03`. | `DO178-EXCEPTION`, MAJOR or CRITICAL by policy |
| Recursion | `CountDown` calls itself. | The call graph records the self-call and the DAL policy from `SCS-L-04` determines severity. | `DO178-RECURSION` when the call graph resolves the self-call |
| RTTI | `dynamic_cast` in `UseRtti`. | RTTI tokens are checked against `SCS-L-05`. | `DO178-RTTI` |
| Documentation | Functions in `violations.cpp` have no preceding documentation comments. | Missing documentation is compared with the document's documentation policy. | `DO178-NO-DOCSTRING` |

## How to explain a report row

Use the row's rule ID, file, line, element, and message together. For example:

> `DO178-DYNAMIC-MEM` in `violations.cpp`, element `bad_name`: the parser found `new`/`delete` in the function body. The uploaded PDF marks dynamic memory as `FORBIDDEN` for DAL B, so the validator emitted a CRITICAL violation and recommends static allocation or an approved memory pool.

The compliant functions in `sample_codebase/src/compliant.cpp` include documentation comments, use PascalCase names, stay within the default/relaxed limits, and use neither prohibited constructs nor unbounded loops. They provide a useful control when demonstrating that a flag is rule-driven rather than file-wide.

The current sample run with the strict PDF and DAL B produced 12 findings and 8 CRITICAL findings. The CLI therefore exits with status 1 after writing the reports; that status is the intentional certification gate for critical violations, not a PDF ingestion failure. The isolated standards smoke check does not build the call graph, so it does not emit the recursion row; use the full `analyze` command when reviewing call-graph-dependent rules.

## Important interpretation note

A flag is an automated screening result, not a final certification disposition. Review the source line, the extracted PDF rule, and the project deviation process before accepting or waiving it.
