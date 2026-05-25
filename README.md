# DO-178C RAG Engine — Codebase Analyzer

Fully offline Python tool that parses C++ codebases and generates DO-178C §11-compliant reports. No internet, no API calls, no cloud.

Covers four analysis domains required for avionics software certification:

| Feature | DO-178C Reference | What it produces |
|---------|-------------------|-----------------|
| Virtual Analysis | §12 change impact | Cat1/Cat2 classification of virtual function changes between builds |
| Coupling Analysis | §6.3.1.b | Control + data coupling map per LRU with risk levels |
| Dead Code Detection | §6.4.2.2 | Dead vs deactivated code with removal/justification disposition |
| Standards Validation | §5.1 + MISRA C++ | Violation report with compliance score and corrections |

**Full usage guide:** [USAGE.md](USAGE.md) — step-by-step for all four objectives, TUI, CLI, Docker, chat interface, and troubleshooting.

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

# Verify install
python main.py validate-config
```

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
| `--output-format` | `docx` | Output format: `docx` or `markdown` |

### Output

All reports written to `output/` (configurable in `config.yaml`):

```
output/
├── virtual_analysis.docx       # DO-178C §12 — virtual function change impact
├── coupling_analysis.docx      # DO-178C §6.3.1.b — LRU control/data coupling
├── dead_code_report.docx       # DO-178C §6.4.2.2 — dead/deactivated code
├── standards_report.docx       # DO-178C §5.1 — code standard violations
└── sqa_checklist.md            # SQA checklist auto-filled with pass/fail status
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

# LRU names for coupling analysis
lru_names:
  - ADS
  - FCS
  - FMS
  - FADEC
  # ... add your LRU names

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

# Pull model (first time only, ~5 GB)
docker compose -f docker-compose.yml -f docker-compose.laptop.yml \
  exec ollama ollama pull llama3.1:8b

# Run analysis
mkdir -p input output lru_docs
cp -r /path/to/cpp/code input/

docker compose -f docker-compose.yml -f docker-compose.laptop.yml \
  run rag-engine analyze /data/input --output-format markdown

# View results
ls output/
```

### Production — deepseek-coder-v2:16b, NVIDIA GPU (~32 GB VRAM)

```bash
# Start (requires nvidia-container-toolkit)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Pull model (first time only, ~9 GB)
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  exec ollama ollama pull deepseek-coder-v2:16b

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

---

## DO-178C Quick Reference

| Analysis | Standard | Key output |
|----------|----------|-----------|
| Virtual Analysis | §12, §12.1, §12.3 | Category 1 / Category 2 classification; reverification scope per changed function |
| Coupling Analysis | §6.3.1.b | Control coupling map (call graph), data coupling map (shared globals), risk per LRU |
| Dead Code | §6.4.2.2 | Dead (Remove) vs Deactivated (Justify) disposition; structural coverage impact |
| Code Standards | §5.1 | MISRA C++ violations with severity, correction, and MISRA rule cross-reference |
| Reports | §11 | All outputs are DO-178C life cycle data artifacts (§11.9, §11.12, §11.14, §11.16) |

**DAL levels supported:** A (MC/DC), B (Decision), C (Statement)

**False positive target:** < 2% for dead code detection

---

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# Run tests
pytest -v

# Run with coverage
pytest --cov=rag_engine --cov-report=html
```

### Project structure

```
rag_engine/
├── models.py                   Core dataclasses (FunctionDef, Violation, etc.)
├── config.py                   Config dataclass + YAML loader
├── core/
│   ├── parser.py               C++ → FunctionDef via tree-sitter
│   ├── ast_builder.py          AST traversal helpers, cyclomatic complexity
│   ├── graph_builder.py        Call graph + reachability (networkx)
│   └── embeddings.py           FAISS index + all-MiniLM-L6-v2
├── features/
│   ├── virtual_analysis.py     Feature 1: virtual change detection
│   ├── coupling_analysis.py    Feature 2: LRU coupling map
│   ├── dead_code_detector.py   Feature 3: reachability-based dead code
│   └── standards_validator.py  Feature 4: DO-178C §5.1 + MISRA C++
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
│   └── ollama_client.py        Optional Ollama narrative enrichment
└── templates/                  Jinja2 report templates (one per analysis)
main.py                         Typer CLI entry point
```

---

## Limitations and Known Issues

- **Virtual detection**: tree-sitter-cpp represents `virtual` differently across class hierarchies. Detection is reliable for direct class members; deeply nested inheritance may need manual review.
- **Coupling analysis accuracy**: data coupling uses global variable heuristics. Pointer aliasing and struct member access are not tracked.
- **Dead code false positives**: function pointers stored in arrays/structs that are not statically analyzable may be flagged as dead. Review `false_positive_notes` in the dead code report.
- **MISRA C++ coverage**: the built-in rule set covers the highest-impact rules. Full MISRA C++:2008 compliance requires clang-tidy integration (`clang-tidy` must be on PATH).
- **Python 3.10 on Windows**: set `USE_TF=0 USE_TORCH=1` environment variables if you see TensorFlow import errors (NumPy version conflict with system TF).

---

## Applicable Standards

- RTCA DO-178C — Software Considerations in Airborne Systems and Equipment Certification
- RTCA DO-330 — Software Tool Qualification (applicable if tool output used in certification)
- MISRA C++:2008 — Guidelines for the use of C++ in critical systems
- ARP4761 — Safety Assessment Process Guidelines (for DAL determination)
