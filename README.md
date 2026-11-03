# DO-178C RAG Engine — Codebase Analyzer

Fully offline Python tool that parses C++ codebases and generates DO-178C §11-compliant reports. No internet, no API calls, no cloud.

Covers four analysis domains required for avionics software certification:

| Feature | DO-178C Reference | What it produces |
|---------|-------------------|-----------------|
| Virtual Analysis | §12 change impact | Cat1/Cat2 classification of virtual function changes between builds |
| Coupling Analysis | §6.3.1.b | Control + data coupling map per LRU with risk levels |
| Dead Code Detection | §6.4.2.2 | Dead, deactivated, fragment and unused-variable findings with stable IDs, confidence, evidence, removal impact and a reviewer dispositions file |
| Standards Validation | §5.1 + MISRA C++ | Violation report with compliance score and corrections |
| Virtual Function Analysis (R1) | §12 | Virtual functions, overrides, reachability, call sites and class trees (`virtuals`) |
| Function Effect & Coupling (R2) | §6.3.1.b | What a method reads, writes and calls, and what is affected if it changes (`effect`); no numeric score |
| Sensor Data Analysis (R3) | — | Corrupt, stale and debounce checks per sensor from a YAML template (`sensors`, `sensor-check`) |
| Questions and Answers (R4) | — | Answers from the static-analysis facts; an optional local LLM explains them (`ask`) |

**Full usage guide:** [USAGE.md](USAGE.md) — the four objectives and R1–R4 with examples and sample output, the dead-code review workflow, TUI tabs, CLI, Docker, chat, assumptions and troubleshooting.

---

## Requirements

- Python 3.10+
- 4 GB RAM minimum (8 GB recommended for large codebases)
- Docker + Docker Compose (for containerized use)
- NVIDIA GPU (optional, production Docker only)

---

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

```bash
# Verify install
python main.py validate-config
```

```text
Config OK — DAL B, output: output
```

**Semantic search environment note.** Coupling analysis uses sentence-transformers for semantic search. On machines where it cannot be imported (for example transformers 5.x with torch < 2.4), the engine logs one warning (`Semantic search unavailable ...`) and continues with keyword matching. Fix it with `pip install "transformers<5"`, or by upgrading torch to 2.4 or newer. `--no-semantic-search` skips the embedding model.

---

## Usage

### Basic analysis

```bash
python main.py analyze /path/to/codebase
```

### With base build comparison (virtual analysis diff)

```bash
python main.py analyze /path/to/current_build \
  --base-build /path/to/base_build \
  --lru-docs /path/to/lru_documents \
  --output-format docx
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-build` | same as codebase | Base build directory for virtual function diff |
| `--lru-docs` | none | Directory containing LRU PDF/DOCX specification documents |
| `--config` | `config.yaml` | Path to configuration file |
| `--standards-file` | none | Optional custom standards file (`.yaml`, `.json`, `.pdf`, `.docx`, `.txt`) |
| `--output-format` | `docx` | Output format: `docx` or `markdown` |
| `--sensor-templates PATH` (alias `--templates`) | `sensor_templates_file` | Sensor template YAML; enables the sensor report |
| `--semantic-search` / `--no-semantic-search` | on | Use the embedding model for coupling analysis (keyword matching without it) |

### Output

All reports written to `output/` (configurable in `config.yaml`):

```
output/
├── virtual_analysis.docx       # DO-178C §12 — virtual function change impact
├── coupling_analysis.docx      # DO-178C §6.3.1.b — LRU control/data coupling
├── dead_code_report.docx       # DO-178C §6.4.2.2 — dead/deactivated code
├── dead_code_findings.json     # every dead-code finding: ID, confidence, evidence, review status
├── sensor_analysis.docx        # R3 sensor checks (only when a sensor templates file is set)
├── standards_report.docx       # DO-178C §5.1 — code standard violations
└── sqa_checklist.md            # SQA checklist auto-filled with pass/fail status
```

### Example on the sensors fixture

```bash
python main.py analyze tests/fixtures/cpp/sensors --sensor-templates tests/fixtures/sensor_templates.yaml --no-semantic-search --output-format markdown
```

Output (from a run with `output_dir: .superpowers/out`; the default config writes to `output/`):

```text
Reports in .superpowers/out/
  .superpowers\out\virtual_analysis.md
  .superpowers\out\coupling_analysis.md
  .superpowers\out\dead_code_report.md
  .superpowers\out\standards_report.md
  .superpowers\out\sensor_analysis.md
  .superpowers\out\sqa_checklist.md
```

USAGE.md shows rows of `dead_code_report.md` and an item of `dead_code_findings.json` from the same run.

### Feature commands

Query commands run the analyses without the embedding model and write no reports, except `effect`. Each example uses the test fixtures; USAGE.md has sample output for each.

```bash
python main.py virtuals tests/fixtures/cpp/sensors
```

```bash
python main.py effect tests/fixtures/cpp/sensors updateTemperature
```

```bash
python main.py sensors tests/fixtures/cpp/sensors --templates tests/fixtures/sensor_templates.yaml
```

```bash
python main.py sensor-check tests/fixtures/sensor_samples.csv --templates tests/fixtures/sensor_templates.yaml
```

```bash
python main.py ask tests/fixtures/cpp/sensors "What validation checks are applied to TemperatureSensor?" --templates tests/fixtures/sensor_templates.yaml
```

Dead-code review: run the analysis, then record a decision per finding ID in a YAML file that `dead_code_dispositions` points to (see USAGE.md, Objective 3).

```bash
python main.py analyze tests/fixtures/cpp/sensors --templates tests/fixtures/sensor_templates.yaml --no-semantic-search
```

The interactive TUI has results tabs for virtual functions, dead code, function effect and sensors (with a sample tester). Press `c` on the results screen to open the chat:

```bash
python tui.py
```

---

## Configuration

Edit `config.yaml` to match your project:

```yaml
dal_level: B              # DAL level: A, B, C, or D
output_dir: output        # Where reports are written
max_workers: 4            # Parallel analysis workers

# Complexity thresholds (violations flagged when exceeded)
cyclomatic_complexity_max: 10
function_length_max: 50
nesting_depth_max: 5
param_count_max: 7

# Entry points for dead code reachability analysis
entry_points:
  - main
entry_point_patterns:     # interrupt handlers and RTOS tasks, matched by name
  - '*_IRQHandler'
  - '*_ISR'
  - 'ISR_*'
  - '*Task'

# LRU names for coupling analysis
lru_names:
  - ADS
  - FCS
  - FMS
  - FADEC
  # ... add your LRU names

# Dead-code review decisions (YAML; path relative to this file).
# Example value: the shipped config.yaml leaves it empty (no dispositions applied).
dead_code_dispositions: reviews/dispositions.yaml

# Sensor templates for the R3 checks (optional).
# Example value: the shipped config.yaml leaves it empty (sensor checks disabled).
sensor_templates_file: tests/fixtures/sensor_templates.yaml

# Optional Ollama LLM for richer report narratives
ollama_enabled: false
ollama_model: llama3.1:8b
ollama_url: http://localhost:11434
```

---

## Docker

### Laptop / Dev — llama3.1:8b, CPU only (~8 GB RAM)

```bash
# Start
docker compose -f docker-compose.yml -f docker-compose.laptop.yml up -d
```

```bash
# Pull model (first time only, ~5 GB)
docker compose -f docker-compose.yml -f docker-compose.laptop.yml \
  exec ollama ollama pull llama3.1:8b
```

```bash
# Run analysis
mkdir -p input output lru_docs
```

```bash
cp -r /path/to/cpp/code input/
```

```bash
docker compose -f docker-compose.yml -f docker-compose.laptop.yml \
  run rag-engine analyze /data/input --output-format markdown
```

```bash
# View results
ls output/
```

### Production — deepseek-coder-v2:16b, NVIDIA GPU (~32 GB VRAM)

```bash
# Start (requires nvidia-container-toolkit)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

```bash
# Pull model (first time only, ~9 GB)
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  exec ollama ollama pull deepseek-coder-v2:16b
```

```bash
# Run analysis
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  run rag-engine analyze /data/input \
    --lru-docs /data/lru_docs \
    --output-format docx
```

### Docker volume mounts

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./input/` | `/data/input` | C++ source files to analyze (read-only) |
| `./output/` | `/data/output` | Generated reports |
| `./lru_docs/` | `/data/lru_docs` | LRU PDF/DOCX specification documents |

---

## Architecture

```
C++ source files
       │
       ▼
  CodeParser (tree-sitter)
       │ ParseResult × N files
       ▼
  CallGraphBuilder (networkx DiGraph)
  SemanticSearch   (FAISS + all-MiniLM-L6-v2)
       │
       ├──────────────────────────────────────────┐
       │                                          │
  ┌────▼─────┐  ┌──────────────┐  ┌───────────┐  │  ┌────────────────┐
  │ Virtual  │  │  Coupling    │  │ Dead Code │  │  │   Standards    │
  │ Analyzer │  │  Analyzer    │  │ Detector  │  │  │   Validator    │
  │ DO-178C  │  │  DO-178C     │  │ DO-178C   │  │  │   DO-178C §5.1 │
  │ §12      │  │  §6.3.1.b    │  │ §6.4.2.2  │  │  │   MISRA C++    │
  └────┬─────┘  └──────┬───────┘  └─────┬─────┘  │  └───────┬────────┘
       └───────────────┴────────────────┴─────────┘          │
                                    │                         │
                            ┌───────▼─────────────────────────┘
                            │      ReportGenerator            │
                            │      Jinja2 → DOCX/Markdown     │
                            │      ChecklistFiller → SQA      │
                            └─────────────────────────────────┘
```

**Key design decisions:**

- **Fully offline**: all-MiniLM-L6-v2 (22M params, 384-dim) downloaded once at Docker build time, cached locally. No outbound network calls during analysis.
- **Incremental cache**: SHA-256 hash per source file. Unchanged files skip re-parse (~70% speedup on second run).
- **Semantic coupling detection**: FAISS cosine similarity finds LRU-relevant code that text search misses (~15-20% more coverage).
- **Ollama optional**: LLM enriches report narrative sections. Falls back to template-only output if unavailable — zero hard dependency.
- **One pipeline**: `rag_engine/pipeline.py` (`run_pipeline`) runs every analysis. The CLI commands and the TUI both call it.
- **Facts first**: `ask` and the chat answer from the static-analysis results. The LLM, when enabled, only explains those facts.

---

## DO-178C Quick Reference

| Analysis | Standard | Key output |
|----------|----------|-----------|
| Virtual Analysis | §12, §12.1, §12.3 | Category 1 / Category 2 classification; reverification scope per changed function |
| Coupling Analysis | §6.3.1.b | Control coupling map (call graph), data coupling map (shared globals), risk per LRU |
| Dead Code | §6.4.2.2 | Five categories, each with a disposition: `dead_code` (Remove), `deactivated_code` (Justify as Deactivated), `unused_export` (Investigate), `dead_fragment` (Fix Fragment), `unused_variable` (Remove for locals, Investigate for parameters). Each finding has a stable ID (`DC-…`), a confidence (high / medium / low), evidence and removal impact; structural coverage impact |
| Code Standards | §5.1 | MISRA C++ violations with severity, correction, and MISRA rule cross-reference |
| Reports | §11 | All outputs are DO-178C life cycle data artifacts (§11.9, §11.12, §11.14, §11.16) |

**DAL levels supported:** A (MC/DC), B (Decision), C (Statement)

**False positive target:** < 2% for dead code detection

---

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

```bash
pip install -e .
```

```bash
# Run tests
pytest -v
```

```bash
# Run with coverage
pytest --cov=rag_engine --cov-report=html
```

### Project structure

```
rag_engine/
├── models.py                   Core dataclasses (FunctionDef, DeadCodeItem, Violation, etc.)
├── config.py                   Config dataclass + YAML loader
├── pipeline.py                 run_pipeline(): the one analysis sequence used by the CLI and TUI
├── core/
│   ├── parser.py               C++ → FunctionDef via tree-sitter
│   ├── ast_builder.py          AST traversal helpers, cyclomatic complexity
│   ├── class_hierarchy.py      Class tree and virtual override resolution
│   ├── graph_builder.py        Call graph + reachability (networkx)
│   └── embeddings.py           FAISS index + all-MiniLM-L6-v2
├── features/
│   ├── virtual_analysis.py     R1: virtual function changes and overrides
│   ├── function_effect.py      R2: reads, writes, calls, coupled components of a method
│   ├── coupling_analysis.py    LRU coupling map
│   ├── dead_code_detector.py   Dead code findings, IDs, dispositions
│   ├── control_flow_analyzer.py  Dead fragments and unused variables
│   ├── sensor_template.py      R3: sensor YAML template loader
│   ├── sensor_analysis.py      R3: corrupt / stale / debounce checks in the code
│   ├── sensor_data_checker.py  R3: replay of recorded sensor samples
│   └── standards_validator.py  DO-178C §5.1 + MISRA C++
├── knowledge_base/
│   ├── index_manager.py        FAISS index lifecycle
│   ├── cache_manager.py        SHA-256 incremental parse cache
│   └── standards_db.py         MISRA C++ rules + DO-178C prohibited constructs
├── document_processor/
│   ├── code_reader.py          Cached directory scanner
│   ├── doc_reader.py           LRU PDF/DOCX ingestion
│   └── template_engine.py      Jinja2 environment
├── reporting/
│   ├── report_generator.py     Orchestrates all feature outputs → §11 reports
│   ├── checklist_filler.py     Auto-fills DO-178C SQA checklists
│   └── output_formatter.py     Markdown → DOCX conversion
├── llm/
│   ├── ollama_client.py        Optional Ollama client
│   ├── rag_narrator.py         Optional narrative enrichment of reports
│   └── offline_qa.py           R4: facts-first answers to questions
├── tui/
│   └── app.py                  Textual TUI (config, progress, results, chat)
└── templates/                  Jinja2 report templates (one per analysis)
main.py                         Typer CLI entry point (analyze, virtuals, effect, sensors, sensor-check, ask, validate-config)
```

---

## Limitations and Known Issues

- **Virtual detection**: tree-sitter-cpp represents `virtual` differently across class hierarchies. Detection is reliable for direct class members; deeply nested inheritance may need manual review.
- **Coupling analysis accuracy**: data coupling uses global variable heuristics. Pointer aliasing and struct member access are not tracked.
- **Dead code false positives**: function pointers, tables and macro uses that are not statically analyzable are reported as `unused_export` (Investigate), not as removal candidates. Check each finding's confidence, evidence and reason in the report, and record the review in the dispositions file.
- **Sensor checks**: the static checks find range, stale and debounce code by pattern. `mismatch` and `missing` results need a check against the source. Template thresholds are defaults pending confirmation (see USAGE.md, section 11).
- **MISRA C++ coverage**: the built-in rule set covers the highest-impact rules. Full MISRA C++:2008 compliance requires clang-tidy integration (`clang-tidy` must be on PATH).
- **Python 3.10 on Windows**: set `USE_TF=0 USE_TORCH=1` environment variables if you see TensorFlow import errors (NumPy version conflict with system TF).

---

## Applicable Standards

- RTCA DO-178C — Software Considerations in Airborne Systems and Equipment Certification
- RTCA DO-330 — Software Tool Qualification (applicable if tool output used in certification)
- MISRA C++:2008 — Guidelines for the use of C++ in critical systems
- ARP4761 — Safety Assessment Process Guidelines (for DAL determination)
