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
4. [TUI Interface](#4-tui-interface)
5. [CLI Reference](#5-cli-reference)
6. [Chat Interface](#6-chat-interface)
7. [Docker — Laptop Mode](#7-docker--laptop-mode)
8. [Docker — Production Mode](#8-docker--production-mode)
9. [Output Files](#9-output-files)
10. [Configuration](#10-configuration)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Install

```bash
# Python 3.10+ required
pip install -r requirements.txt

# Install project package (needed for imports)
pip install -e .

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

# Or launch the interactive TUI
python tui.py

# Or launch with chat interface enabled (requires Ollama)
python tui.py --chat
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
Maps all control coupling (call dependencies) and data coupling (shared global variables) between your code and each LRU (Line Replaceable Unit) in the avionics system. Assigns a risk level (low / medium / high) per LRU.

**How to run (CLI):**
```bash
# Basic — uses LRU names from config.yaml only
python main.py analyze /path/to/cpp_code

# With LRU specification documents (recommended — better coverage)
python main.py analyze /path/to/cpp_code \
  --lru-docs /path/to/lru_specifications/
```

**What to provide:**
- `lru_docs/` — folder containing LRU PDF or DOCX specification documents
  - Naming: `ADS.pdf`, `FCS.docx`, `FMS.pdf`, etc. (file name = LRU name)
  - If omitted, analysis uses LRU names from `config.yaml` only

**Output:** `output/coupling_analysis.docx`

Fields in the report:
- LRU name, risk level
- Control coupling list (functions that call LRU-related code)
- Data coupling list (shared globals read/written near LRU signals)
- Semantic similarity score (how confidently the engine matched code to LRU signals)

**Adding LRU names without documents:**
Edit `config.yaml`:
```yaml
lru_names:
  - ADS
  - FCS
  - FMS
  - FADEC
  - YOUR_LRU_NAME
```

---

### Objective 3 — Dead Code Detection (DO-178C §6.4.2.2)

**What it does:**
Finds all unreachable code via multi-level DFS reachability from entry points, then classifies each item:

| Category | Disposition | Meaning |
|----------|------------|---------|
| `dead_code` | Remove | Unreachable, no justification exists |
| `deactivated_code` | Justify as Deactivated | Unreachable but present in test/doc references |
| `unused_export` | Investigate | Not called internally but exported |
| `dead_fragment` | Fix Fragment | Unreachable branch inside a live function |

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

**Output:** `output/dead_code_report.docx`

Fields in the report:
- Function name, file, line
- Category + disposition
- Structural coverage impact (which coverage criteria are affected)
- False positive notes (function pointer patterns that may have been missed)

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
│  Output format:  [DOCX ▼]                   │
│  DAL level:      [DAL B ▼]                  │
│                                             │
│       [Run Analysis]    [Quit]              │
└─────────────────────────────────────────────┘
```

Fill in paths, select format and DAL level, press **Run Analysis**.

**Screen 2 — Analysis Progress**

Four progress bars (one per objective) + live log showing what the engine is doing. Automatically advances to Results when complete.

**Screen 3 — Results Browser**

Five tabs: **Standards** | **Virtual Changes** | **Dead Code** | **Coupling** | **Checklist**

Each tab is a scrollable table. Navigate with arrow keys.

Keyboard shortcuts:
| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Switch between result tabs |
| `↑` / `↓` | Scroll within table |
| `r` | Re-run analysis (back to Config screen) |
| `q` | Quit |
| `Esc` | Back to previous screen |

**Screen 4 — Chat** (requires Ollama running)

Ask questions about your analysis results in natural language. See [Section 6](#6-chat-interface).

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
| `--config PATH` | No | `config.yaml` | Config file path |
| `--output-format` | No | `docx` | `docx` or `markdown` |

```bash
# Validate config without running analysis
python main.py validate-config
python main.py validate-config --config /path/to/custom.yaml
```

**Examples:**

```bash
# Minimal — just parse and analyze one directory
python main.py analyze src/

# Full production run
python main.py analyze src/current/ \
  --base-build src/baseline/ \
  --lru-docs docs/lru_specs/ \
  --output-format docx \
  --config production.yaml

# CI pipeline usage (exits 1 on CRITICAL violations)
python main.py analyze src/ && echo "PASS" || echo "CRITICAL VIOLATIONS FOUND"
```

---

## 6. Chat Interface

The chat screen lets you ask natural language questions about your analysis results. It works in two modes:

**Mode A — With Ollama (full AI answers)**
Requires Ollama running at `http://localhost:11434` with a model pulled.

```bash
# Start Ollama (separate terminal)
ollama serve
ollama pull llama3.1:8b   # or your preferred model

# Then launch TUI — chat will activate automatically
python tui.py
```

**Mode B — Without Ollama (keyword-based answers)**
Works offline, no model needed. Answers common questions about the analysis results using structured data directly.

**How to access the chat:**
1. Run analysis via TUI
2. On the Results screen, press **`c`** or click **"Ask AI"**
3. Type your question and press Enter

**Example questions you can ask:**

```
# Standards
How many critical violations are there?
Which file has the most violations?
What are the most serious problems?
Show me all goto violations

# Virtual analysis
Which functions need reverification?
What changed between builds?
How many Category 2 changes are there?

# Dead code
What dead code should I remove first?
Which functions are deactivated?
How much dead code was found?

# Coupling
Which LRU has the highest risk?
What data coupling exists in the FCS?
Explain the control coupling map

# General DO-178C
What does Category 2 mean?
What is the compliance score?
Is this codebase ready for certification?
```

**Offline fallback behavior:**
When Ollama is unavailable, the engine searches the analysis results directly and returns structured answers. For example, asking "how many critical violations" returns the count directly from `standards.violations_by_severity['CRITICAL']` without any LLM.

---

## 7. Docker — Laptop Mode

Laptop mode uses `llama3.1:8b` (CPU only, ~8 GB RAM). Good for development and testing.

```bash
# 1. Start the stack
docker compose -f docker-compose.yml -f docker-compose.laptop.yml up -d

# 2. Pull the model (first time only, ~5 GB download)
docker compose -f docker-compose.yml -f docker-compose.laptop.yml \
  exec ollama ollama pull llama3.1:8b

# 3. Prepare your data directories
mkdir -p input output lru_docs

# Copy your C++ source files
cp -r /path/to/cpp/source/* input/

# Copy LRU spec documents (optional)
cp /path/to/lru/specs/* lru_docs/

# 4. Run all four analyses
docker compose -f docker-compose.yml -f docker-compose.laptop.yml \
  run rag-engine analyze /data/input \
    --lru-docs /data/lru_docs \
    --output-format markdown

# 5. View results
ls output/
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

# 2. Start the production stack
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 3. Pull the production model (first time only, ~45 GB)
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  exec ollama ollama pull qwen2.5-coder:72b

# 4. Run full production analysis
mkdir -p input output lru_docs
cp -r /path/to/certified/source/* input/
cp /path/to/lru/specs/* lru_docs/

docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  run rag-engine analyze /data/input \
    --base-build /data/base \
    --lru-docs /data/lru_docs \
    --output-format docx

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
| `coupling_analysis.docx` | §6.3.1.b | Per-LRU control and data coupling maps, risk scores, semantic coverage |
| `dead_code_report.docx` | §6.4.2.2 | Dead/deactivated code list, dispositions, structural coverage impact |
| `standards_report.docx` | §5.1 | All violations by severity, compliance score, correction guidance, MISRA cross-reference |
| `sqa_checklist.md` | §11 | SQA checklist auto-filled with PASS/FAIL/OPEN status and evidence |

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

# LRU names for coupling analysis (match your LRU spec filenames)
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

# Ollama LLM for narrative enrichment (optional)
ollama_enabled: false           # set true when Ollama is running
ollama_model: llama3.1:8b       # laptop model
# ollama_model: qwen2.5-coder:72b  # production model
ollama_url: http://localhost:11434
```

---

## 11. Troubleshooting

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

The parser looks for `*.cpp`, `*.cxx`, `*.cc`, `*.h`, `*.hpp`. Make sure your source files have these extensions.

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

### Ollama not available (chat falls back to offline mode)

```
Ollama not available — using structured answers only
```

This is not an error. Chat still works with structured keyword matching. To enable full AI answers:
```bash
ollama serve              # in a separate terminal
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
