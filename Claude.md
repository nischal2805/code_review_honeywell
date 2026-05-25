# CLAUDE.md — On-Device RAG Engine (DO-178C Codebase Analyzer)

**Stack:** Python 3.10+, tree-sitter, sentence-transformers (all-MiniLM-L6-v2), FAISS, networkx, jinja2, python-docx  
**Standard:** RTCA DO-178C / DO-178B  
**Mode:** Fully offline — no API calls, no network dependency

---

## Project Structure

```
rag_engine/
├── core/
│   ├── parser.py           # C++ → AST via tree-sitter
│   ├── ast_builder.py      # AST utilities
│   ├── graph_builder.py    # Call graph + data flow graph (networkx DiGraph)
│   └── embeddings.py       # Local FAISS index + sentence-transformer encoding
├── features/
│   ├── virtual_analysis.py     # Feature 1: Virtual function change detection
│   ├── coupling_analysis.py    # Feature 2: Control/data coupling per LRU
│   ├── dead_code_detector.py   # Feature 3: Dead/deactivated code classification
│   └── standards_validator.py  # Feature 4: DO-178C §5.1 + MISRA C++ rule checking
├── document_processor/
│   ├── code_reader.py      # Parse C++ files
│   ├── doc_reader.py       # Ingest PDF/DOCX LRU specs
│   └── template_engine.py  # Jinja2 report rendering
├── knowledge_base/
│   ├── index_manager.py    # FAISS index lifecycle
│   ├── cache_manager.py    # Hash-based incremental cache
│   └── standards_db.py     # MISRA C++ rules + project rules
├── reporting/
│   ├── report_generator.py     # DO-178C §11 reports (PDF/DOCX)
│   ├── checklist_filler.py     # SQA checklist auto-fill
│   └── output_formatter.py     # Formatting utilities
└── main.py                 # CLI entry + parallel orchestration
```

---

## Core Data Structures

```python
@dataclass
class FunctionDef:
    name: str
    file_path: str
    line_number: int
    return_type: str
    parameters: List['Parameter']
    is_virtual: bool
    is_inline: bool
    is_static: bool
    body: str
    docstring: Optional[str]
    cyclomatic_complexity: int
    line_count: int
    nesting_depth: int
    called_by: Set[str] = field(default_factory=set)
    calls: Set[str] = field(default_factory=set)
    data_reads: Set[str] = field(default_factory=set)
    data_writes: Set[str] = field(default_factory=set)
    do178c_requirement_trace: Optional[str] = None

@dataclass
class VirtualChange:
    change_type: Literal['added', 'removed', 'modified']
    function: FunctionDef
    base_version: Optional[FunctionDef]
    current_version: Optional[FunctionDef]
    do178c_category: Literal['Category 1', 'Category 2']
    reverification_scope: Optional[str] = None

@dataclass
class Violation:
    rule: str
    misra_ref: Optional[str]
    file: str
    line: int
    element: str
    message: str
    severity: Literal['MINOR', 'MEDIUM', 'MAJOR', 'CRITICAL']
    disposition: Optional[str] = None   # 'Fix Required' | 'Deviation'

@dataclass
class DeadCodeItem:
    name: str
    file_path: str
    line_number: int
    category: Literal['dead_code', 'deactivated_code', 'unused_export', 'dead_fragment']
    do178c_disposition: Literal['Remove', 'Justify as Deactivated', 'Investigate', 'Fix Fragment']
    coverage_impact: str
```

---

## Module Interfaces

### core/parser.py — `CodeParser`
```python
def parse_file(self, file_path: str) -> ParseResult: ...
def parse_directory(self, dir_path: str, pattern: str = "*.cpp") -> Dict[str, ParseResult]: ...
# ParseResult has: tree, file_path, functions: List[FunctionDef], classes, includes, metadata
```

### core/graph_builder.py — `CallGraphBuilder`
```python
def build(self) -> CallGraph: ...
def find_reachable_from(self, function_name: str) -> Set[str]: ...  # DFS via nx.descendants
def find_unreachable(self, entry_points: List[str]) -> Set[str]: ...
```

### core/embeddings.py — `SemanticSearch`
```python
# Model: all-MiniLM-L6-v2 (384-dim, 22M params, offline)
def index_functions(self, functions: List[FunctionDef]): ...  # Embeds name+params+docstring
def find_related(self, query: str, k: int = 10) -> List[FunctionDef]: ...  # FAISS search, threshold 0.6
```

---

## Feature Implementations

### Feature 1 — `VirtualAnalyzer` (virtual_analysis.py)
**Purpose:** Compare base vs current build; classify virtual function changes for DO-178C §12 reverification.

```python
def analyze(self) -> VirtualAnalysisResult:
    base_virtuals = self._extract_virtual_methods(self.base)
    current_virtuals = self._extract_virtual_methods(self.current)
    changes = self._compare_semantically(base_virtuals, current_virtuals)
    # changes keys: 'new'(Cat2), 'removed'(Cat2), 'modified'(Cat1orCat2), 'unchanged'
    impact = self._analyze_impact(changes)
    return VirtualAnalysisResult(changes, impact, do178c_categories, ...)

def _semantic_signature(self, method) -> str:
    # SHA256 of: return_type | name | param_types | virtual/const flags | cyclomatic_complexity | line_count
    # Ignores: comments, whitespace, internal var names
```

**Category rules:** `new` / `removed` → always Category 2. `modified` → Category 1 if only comments/formatting changed, else Category 2.

---

### Feature 2 — `CouplingAnalyzer` (coupling_analysis.py)
**Purpose:** Map control coupling (call dependencies) and data coupling (shared data) per LRU.

```python
def analyze(self) -> CouplingAnalysisResult:
    lru_specs = self._parse_lru_documents()
    lru_impacts = {name: self._analyze_lru_coupling(name, spec) for name, spec in lru_specs.items()}
    return CouplingAnalysisResult(lru_impacts, control_coupling_map, data_coupling_map, ...)

def _analyze_lru_coupling(self, lru_name, lru_spec) -> LRUCoupling:
    # Control coupling: direct_callers, indirect_callers, mode_dependencies
    # Data coupling: shared_globals, parameter_passing, timing_dependencies
    # Uses SemanticSearch to find code related to LRU signals (finds ~15-20% more than text search)

def _find_related_code(self, signal_name: str) -> List[CodeLocation]:
    # Embed signal name → FAISS search → return matches with score > 0.6
```

---

### Feature 3 — `DeadCodeDetector` (dead_code_detector.py)
**Purpose:** Multi-level reachability to classify dead vs deactivated code per DO-178C §6.4.2.2.

```python
def analyze(self) -> DeadCodeReport:
    entry_points = self._identify_entry_points()  # main, exports, virtuals, callbacks, ctors
    # Level 1: DFS reachability from entry_points
    # Level 2: Subtract functions reachable via callbacks/function pointers
    # Level 3: Subtract exported symbols
    # Level 4: Subtract test/doc references → likely deactivated code
    classification = self._classify_dead_code(l1, l2, l3, l4)
    return DeadCodeReport(classification, do178c_dispositions, structural_coverage_impact, ...)

# Classification → disposition mapping:
# dead_code           → 'Remove'
# deactivated_code    → 'Justify as Deactivated'
# unused_export       → 'Investigate'
# dead_fragment       → 'Fix Fragment'
```

---

### Feature 4 — `StandardsValidator` (standards_validator.py)
**Purpose:** Enforce DO-178C §5.1 Software Code Standard + MISRA C++:2008.

```python
def analyze(self) -> ComplianceReport:
    violations = {}
    violations['naming']          = self._validate_naming_conventions()
    violations['complexity']      = self._validate_complexity_metrics()
    violations['do178c_prohibited'] = self._validate_do178c_prohibited_constructs()
    violations['misra_cpp']       = self._validate_misra_cpp_rules()  # clang-tidy integration
    violations['documentation']   = self._validate_documentation()
    return ComplianceReport(violations, compliance_score, corrections, ...)
```

**Prohibited constructs (CRITICAL):** dynamic memory allocation post-init, `goto`  
**Restricted (MAJOR):** recursion without bounded stack, exception handling  
**Complexity thresholds:** cyclomatic < 10, function length < 50 lines, nesting < 5, params < 7

---

## Execution Pipeline (main.py)

```python
# 1. Load config (DAL level, code standard rules, output dir)
# 2. Init knowledge base: load all-MiniLM-L6-v2 offline, build FAISS index, load standards DB
# 3. Parse codebase: CodeParser.parse_directory() → build call graph + data flow graph
# 4. Run features in parallel (ProcessPoolExecutor, max_workers=4):
#    Feature 1: VirtualAnalyzer(base, current).analyze()
#    Feature 2: CouplingAnalyzer(parse_results, lru_docs).analyze()
#    Feature 3: DeadCodeDetector(call_graph, parse_results).analyze()
#    Feature 4: StandardsValidator(rules, parse_results).analyze()
# 5. ReportGenerator: populate DO-178C §11 Jinja2 templates → PDF/DOCX
# 6. ChecklistFiller: auto-fill SQA checklists; write to config-controlled output dir
```

---

## Incremental Cache (knowledge_base/cache_manager.py)

```python
class AnalysisCache:
    def check_cache(self, file_path: str) -> Optional[ParseResult]:
        # Returns None if SHA256 hash changed → triggers re-analysis
    def cache_result(self, file_path: str, result: ParseResult): ...
# ~70-80% speedup on unchanged codebases
# Cache dir: .rag_cache/  (exclude from version control)
```

---

## Risk Scoring

```python
# risk_score = (impact × complexity × likelihood) / max(mitigation, 0.1)  — capped at 1.0
# impact      = (affected_lrus / total_lrus + affected_functions / total_funcs) / 2
# Used to prioritize reverification scope in Feature 1 output
```

---

## Key Dependencies

```
tree-sitter          # AST parsing (primary)
sentence-transformers # Offline embeddings (all-MiniLM-L6-v2)
faiss-cpu            # Vector similarity search
networkx             # Call/dependency graphs
python-docx          # DOCX report output
jinja2               # Template rendering
pypdf                # PDF input (LRU docs)
loguru               # Structured logging
tqdm                 # Progress bars
# External (project-configured): clang-tidy (MISRA C++ checking), gcov/llvm-cov (structural coverage)
```

---

## DO-178C Quick Reference

| Feature | Standard Ref | Key Requirement |
|---------|-------------|-----------------|
| Virtual analysis | §12, §12.1, §12.3 | Classify changes Cat1/Cat2; scope reverification |
| Coupling analysis | §6.3.1.b | Document all control + data coupling per LRU |
| Dead code | §6.4.2.2 | Remove dead code or formally justify as deactivated |
| Code standards | §5.1 | Enforce naming, complexity, prohibited constructs |
| Reports | §11 | All outputs are life cycle data artifacts |
| Tool qualification | DO-330 | Required if tool output used in certification process |

**DAL levels supported:** A (MC/DC), B (Decision), C (Statement)  
**False positive target:** < 2% for dead code detection