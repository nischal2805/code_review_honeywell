# On-Device RAG Engine for Codebase Analysis and Documentation

## Project Overview

**Project Name:** On-Device RAG (Retrieval-Augmented Generation) Engine for Codebase Analysis and Documentation

**Document Version:** 1.0

**Last Updated:** May 23, 2026

**Project Status:** Specification Phase

**Applicable Standard:** RTCA DO-178C (Software Considerations in Airborne Systems and Equipment Certification) / DO-178B

---

## 1. Executive Summary

This project aims to develop a comprehensive Retrieval-Augmented Generation (RAG) infrastructure that operates entirely in offline mode to analyze C++ codebases and generate accurate, structured technical documentation compliant with DO-178B/C requirements. The system will identify virtual function changes, control flow couplings, data dependencies, dead code, and coding standard violations without requiring internet connectivity or external API dependencies.

All software produced and analyzed shall conform to DO-178C objectives for the applicable Software Level (DAL), including traceability, structural coverage, and coding standards compliance.

### Key Deliverables

- Automated virtual analysis documentation generation (DO-178C Section 11 compliant outputs)
- Control coupling and data coupling analysis reports (per DO-178C Section 6.3.1.b)
- Dead code detection and impact assessment (DO-178C Section 6.4.2.2 — deactivated code handling)
- Coding standard compliance validation (DO-178C Section 5.1 — Software Development Standards)
- Professional documentation in standardized formats (PDF/DOC) traceable to DO-178C objectives

---

## 2. Project Objectives

### Primary Objectives

1. **Virtual Analysis Documentation**
   - Compare base code and current RL builds
   - Identify all virtual function modifications
   - Document newly added and modified functions
   - Generate comprehensive change documentation with DO-178C impact classification

2. **Control and Data Coupling Analysis**
   - Analyze LRU (Line Replaceable Unit) dependencies per DO-178C Section 6.3.1.b
   - Map control coupling and data coupling relationships across modules
   - Document impact analysis for identified modules
   - Track changes in consumption/usage patterns with traceability to software requirements

3. **Dead Code Identification**
   - Perform comprehensive dead code analysis per DO-178C Section 6.4.2.2
   - Distinguish deactivated code (intentional, DO-178C compliant) from dead code (non-compliant)
   - Identify unused functions, methods, and variables
   - Cross-verify code execution paths via structural coverage analysis
   - Quantify impact of code removal on DO-178C structural coverage objectives

4. **Coding Standard Validation**
   - Run automated DO-178C Software Code Standards validation (Section 5.1)
   - Identify and document all violations
   - Provide corrective recommendations aligned with DO-178C objectives
   - Generate compliance reports traceable to the Software Accomplishment Summary (SAS)

---

## 3. Scope Definition

### In Scope

- Analysis of C++ codebases (header and source files) subject to DO-178B/C certification
- Offline processing of code and documentation
- Generation of structured analysis reports meeting DO-178C Software Quality Assurance (SQA) objectives
- Integration with provided checklists and templates (DO-178C Section 11 life cycle data)
- Support for PDF and DOC output formats
- Comprehensive documentation artifacts satisfying DO-178C Section 11 (Software Life Cycle Data)
- Structural coverage analysis support (Statement, Decision, MC/DC per DAL A/B/C)

### Out of Scope

- Real-time code analysis during development
- Cloud-based processing or storage
- Non-C++ language support (initial release)
- Automated code refactoring
- Version control system integration (Phase 2)
- IDE plugin development (Phase 2)
- Hardware/Software Interface (HSI) analysis (separate DO-254 scope)

### Assumptions

- All code files are in C++ (.cpp, .h, .hpp) under DO-178B/C configuration control
- Supporting documents are in PDF or DOC format
- Virtual Analysis Document template will be provided
- Checklists follow predefined DO-178C SQA checklist format
- No external network dependencies required
- Analysis runs on systems with sufficient RAM (minimum 4GB)
- Software Level (DAL) applicable to the project has been determined per ARP4761
- A Software Development Plan (SDP), Software Verification Plan (SVP), and Software Quality Assurance Plan (SQAP) are in place

---

## 4. Technical Architecture

### 4.1 System Components

```
┌─────────────────────────────────────────────────────┐
│         RAG Engine Core Components                  │
│         (DO-178C Compliant Architecture)            │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │ Input Processing Module                     │   │
│  │ - Code file parser (C++)                    │   │
│  │ - Document ingestion (PDF/DOC)              │   │
│  │ - Checklist validator (DO-178C SQA)         │   │
│  └─────────────────────────────────────────────┘   │
│                      │                              │
│  ┌─────────────────────────────────────────────┐   │
│  │ Analysis Engine                             │   │
│  │ - AST (Abstract Syntax Tree) builder        │   │
│  │ - Virtual function analyzer                 │   │
│  │ - Control/data coupling analyzer            │   │
│  │   (DO-178C §6.3.1.b)                       │   │
│  │ - Dead/deactivated code detector            │   │
│  │   (DO-178C §6.4.2.2)                       │   │
│  │ - DO-178C Code Standards validator          │   │
│  │   (DO-178C §5.1)                           │   │
│  └─────────────────────────────────────────────┘   │
│                      │                              │
│  ┌─────────────────────────────────────────────┐   │
│  │ Knowledge Base (Local Embeddings)           │   │
│  │ - Code snippet vectorization                │   │
│  │ - Function signature database               │   │
│  │ - Dependency graph                          │   │
│  │ - DO-178C standards reference               │   │
│  │ - Requirements traceability index           │   │
│  └─────────────────────────────────────────────┘   │
│                      │                              │
│  ┌─────────────────────────────────────────────┐   │
│  │ Report Generation Module                    │   │
│  │ - Template engine (DO-178C §11 formats)     │   │
│  │ - Markdown/HTML generator                   │   │
│  │ - PDF/DOC converter                         │   │
│  │ - DO-178C SQA checklist formatter           │   │
│  └─────────────────────────────────────────────┘   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 4.2 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Language** | Python 3.10+ | Rapid development, excellent parsing libraries |
| **Code Parsing** | Tree-sitter / Clang-libtools | Robust C++ AST generation; supports DO-178C structural coverage analysis |
| **Embeddings** | Sentence-Transformers (offline) | Local vector generation without API calls; no external data dependency |
| **Vector Storage** | FAISS or Chroma | Lightweight, offline-capable similarity search |
| **Document Processing** | python-docx, pypdf | PDF/DOC ingestion and generation |
| **DO-178C Code Analysis** | clang-tidy, PC-lint, LDRA, or custom rules | Automated DO-178C code standard and MISRA C++ rule checking |
| **Coverage Analysis** | gcov / llvm-cov with MC/DC instrumentation | Structural coverage per DO-178C Table A-7 (Statement, Decision, MC/DC) |
| **Report Generation** | Jinja2, Markdown | Template-based documentation compliant with DO-178C §11 life cycle data |

---

## 5. Detailed Feature Specifications

### 5.1 Feature 1: Virtual Analysis Documentation

**Objective:** Compare code builds and document virtual function changes in accordance with DO-178C change impact analysis requirements (Section 12 — Software Life Cycle Data for Modified Software)

**Input Requirements:**
- Base Code Build (reference version under configuration control)
- Current Code Build (new version under configuration control)
- Base Virtual Analysis Document (template)
- DO-178C Validation Checklist

**Processing Steps:**

1. **Code Parsing and Comparison**
   - Parse both code builds into AST structures
   - Extract virtual function declarations
   - Identify function signatures and parameters
   - Build inheritance hierarchies
   - Map each function to its DO-178C requirement trace (if available)

2. **Difference Analysis**
   - Compare virtual function definitions
   - Identify new functions (added in current build)
   - Identify removed functions
   - Identify modified functions (signature/implementation changes)
   - Track overrides and polymorphic relationships
   - Classify changes by DO-178C change impact category (Category 1 / Category 2 per Section 12.1)

3. **Documentation Generation**
   - Populate template with identified changes
   - Document before/after signatures
   - Include rationale for modifications (from comments if available)
   - Generate impact analysis section with DO-178C reverification scope
   - Flag functions requiring re-verification per DO-178C Section 12.3

**Output Requirements:**
- Updated Virtual Analysis Document (PDF/DOC), traceable to DO-178C §11.9 (Problem Reports) or §11.16 (Software Change Notices) as applicable
- Filled DO-178C SQA Validation Checklist
- Change summary report with reverification scope

**Acceptance Criteria:**
- All virtual function changes captured accurately
- Documentation matches provided template format
- Checklist items verified and marked complete per DO-178C SQA objectives
- Output contains no grammatical or formatting errors
- Each change classified as Category 1 or Category 2 per DO-178C Section 12.1

**Unique Approach:**
- Implement semantic comparison rather than text-based diffing
- Use inheritance graph visualization for clarity
- Provide impact radius analysis (what else calls these functions?)
- Tag changes requiring MC/DC re-analysis (DAL A/B)

---

### 5.2 Feature 2: Control and Data Coupling Analysis

**Objective:** Analyze LRU dependencies and document control coupling and data coupling relationships per DO-178C Section 6.3.1.b

**Input Requirements:**
- Base Code Build
- Current Code Build
- All LRU documents (ADS, AGMCAL, AGM, APM, BCU, CLOCK, FADEC, FCS, FECU, FMS, GGF, MWS, TACTICAL)
- VRR Checklist (DO-178C Verification Results Record)

**DO-178C Coupling Definitions:**
- **Control Coupling:** A software component that controls the execution sequence of another component (e.g., via flags, modes, function pointers)
- **Data Coupling:** A software component that shares or exchanges data with another component (e.g., global variables, shared memory, passed parameters)

**Processing Steps:**

1. **Dependency Mapping**
   - Build call graphs for all modules
   - Identify control coupling: branches, flags, function pointer calls, mode-based dispatch
   - Identify data coupling: shared globals, parameter passing, message queues, memory maps
   - Map external library dependencies
   - Create call chain visualization with coupling type annotations

2. **LRU Impact Analysis**
   - For each LRU document, extract:
     - Input signals/data (data coupling inputs)
     - Output signals/data (data coupling outputs)
     - Control flows (control coupling)
     - Timing dependencies
   - Cross-reference with code to identify coupling
   - Map each coupling to a DO-178C-traceable software requirement

3. **Change Impact Evaluation**
   - For each code change identified in Feature 1:
     - Determine affected LRUs
     - Assess control and data coupling impact
     - Evaluate risk level (low/medium/high) per DO-178C §12.3
     - Document reverification evidence required
     - Document mitigation strategies

**Output Requirements:**
- Updated LRU documents with impact analysis (DO-178C §11 compliant)
- VRR checklist (completed) with coupling analysis sign-off
- Coupling relationship diagram (control and data coupling clearly distinguished)
- Risk assessment matrix with DO-178C reverification scope

**Acceptance Criteria:**
- All LRU control and data coupling impacts documented per DO-178C §6.3.1.b
- Code lines captured in documents with file/line traceability
- VRR checklist fully completed and traceable to verification objectives
- Risk levels justified with evidence and linked to DO-178C objectives

**Unique Approach:**
- Implement bidirectional traceability (code ↔ LRU ↔ DO-178C requirement)
- Use color-coded risk visualization distinguishing control vs. data coupling
- Provide automated test case recommendations for re-verification per DO-178C §6.4

---

### 5.3 Feature 3: Dead Code Identification

**Objective:** Identify dead and deactivated code and assess removal impact per DO-178C Section 6.4.2.2

**DO-178C Context:**
DO-178C distinguishes between:
- **Dead Code (non-compliant):** Code that cannot be executed during any operational condition. DO-178C prohibits dead code in certified software. It must be removed or justified.
- **Deactivated Code (potentially compliant):** Code that is not intended to be executed in the current operational configuration but is present by design (e.g., for future use, test modes). Must be formally justified per DO-178C §2.4 and §6.4.2.2.

**Input Requirements:**
- Code Build (executable and source) under configuration control
- Compiler instrumentation data (optional) for structural coverage
- DO-178C Dead Code Analysis template

**Processing Steps:**

1. **Static Analysis**
   - Identify all function definitions
   - Identify all variable declarations
   - Identify all code blocks and fragments
   - Build reachability graph from entry points

2. **Reachability Analysis**
   - Perform depth-first traversal from main() and exported APIs
   - Mark all reachable code
   - Identify unreachable code paths
   - Handle indirect calls through function pointers (control coupling)
   - Flag code not exercised by any structural coverage criterion

3. **Dead Code Classification (DO-178C Aligned)**
   - **Dead Code (DO-178C non-compliant):** Never referenced, no indirect calls, no design justification — must be removed
   - **Deactivated Code (requires DO-178C justification):** Intentionally present but not executed in certification configuration — requires formal rationale per §6.4.2.2
   - **Dead Code Fragments:** Unreachable code within functions (e.g., code after unconditional return)
   - **Unused Variables:** Declared but never read — coding standard violation

4. **Structural Coverage Impact Assessment**
   - For each dead/deactivated code item, assess impact on:
     - Statement Coverage (DAL C and above)
     - Decision Coverage (DAL B and above)
     - MC/DC Coverage (DAL A)
   - Evaluate removal safety (no side effects?)
   - Estimate code size reduction
   - Check for test dependencies
   - Verify no documentation references

**Output Requirements:**
- Dead code report (DO-178C §11 compliant) with:
  - File names
  - Function/variable names
  - Line numbers
  - DO-178C classification (Dead / Deactivated / Fragment / Unused Variable)
  - Impact on structural coverage objectives
  - Recommended disposition (Remove / Justify as Deactivated)

**Acceptance Criteria:**
- Report contains all required fields
- All dead code elements listed with file/line traceability
- Each item classified per DO-178C §6.4.2.2
- Impact on structural coverage documented for each item
- False positives minimized through cross-verification (target: <2% false positive rate)

**Unique Approach:**
- Implement context-aware detection (callbacks, indirect calls, deactivated code patterns)
- Provide removal priority scoring (safety + DO-178C compliance impact)
- Generate before/after structural coverage metrics
- Flag items requiring Problem Report (PR) per DO-178C §11.9

---

### 5.4 Feature 4: Coding Standard Validation

**Objective:** Validate codebase against DO-178C Software Code Standards (Section 5.1) and document violations

**DO-178C Context:**
DO-178C Section 5.1 requires a **Software Code Standard** to be defined, covering at minimum:
- Programming language restrictions and allowable constructs
- Complexity metrics (cyclomatic complexity, nesting depth)
- Naming and documentation conventions
- Use of dynamic memory, recursion, and other restricted constructs
- MISRA C++ compliance (commonly applied subset for avionics C++ code)
- Prohibited language features (e.g., unbounded loops, implicit type conversions)

**Input Requirements:**
- Code Build
- DO-178C Software Code Standard document (project-specific)
- Violation analysis template (DO-178C §11 SQA record)

**Processing Steps:**

1. **Standard Rule Loading**
   - Parse provided DO-178C Software Code Standard document
   - Extract individual rules (including MISRA C++ rules if applicable)
   - Categorize by type (naming, structure, complexity, prohibited constructs, etc.)
   - Assign severity levels (critical/major/minor) aligned with DO-178C DAL

2. **Rule Application**
   - Run automated validation tools (clang-tidy, PC-lint, LDRA, or equivalent)
   - Apply custom rules through pattern matching
   - Check naming conventions (files, functions, variables) per code standard
   - Validate structure requirements
   - Assess code complexity metrics
   - Flag prohibited constructs (dynamic memory allocation, recursion, unbounded loops, etc.)

3. **DO-178C Prohibited Construct Checks**
   - **Dynamic memory allocation:** Flagged as prohibited in many DO-178C projects post-initialization
   - **Recursion:** Flagged; stack depth must be bounded and verified
   - **Unbounded loops:** `while(true)` without verified exit — flagged
   - **Implicit type conversions:** Flagged per MISRA C++ Rule 5-0-x
   - **Use of `goto`:** Flagged per MISRA C++ Rule 6-6-1
   - **Exception handling:** Restricted; `try/catch` flagged unless explicitly allowed by code standard
   - **Global variables (uncontrolled):** Flagged as data coupling risk

4. **Violation Detection**
   - Identify all rule violations
   - Map violations to specific code locations (file, line, column)
   - Classify by severity and DO-178C objective impact
   - Collect context (surrounding code)

5. **Correction Proposal**
   - For each violation, suggest correction
   - Provide before/after examples
   - Estimate refactoring effort
   - Flag high-risk corrections requiring re-verification

**Output Requirements:**
- DO-178C SQA Violations Report (§11 life cycle data) containing:
  - Rule name and DO-178C / MISRA C++ reference
  - Severity level
  - Violation count
  - File locations and line numbers
  - Suggested corrections
  - Priority ranking
  - Disposition (Fix Required / Deviation Justified)

**Acceptance Criteria:**
- All violations identified and documented per DO-178C §5.1 code standard objectives
- Corrective recommendations provided for each violation
- Report includes findings, corrections, and DO-178C objective traceability
- Output is actionable, clear, and suitable for DO-178C SQA sign-off
- Deviations from code standard formally justified with rationale

**Unique Approach:**
- Implement customizable DO-178C-aligned rule engine
- Provide automated fix suggestions with MISRA C++ rule cross-references
- Create trend analysis (violations over time across builds)
- Generate complexity heat maps with DO-178C MC/DC impact overlay

---

## 6. Data Flow and Integration

### 6.1 Input Processing Flow

```
Code Files (C++)          →  Parser/Tokenizer  →  AST Generation
                                    ↓
                     DO-178C Semantic Analysis
                     (Requirements Traceability)
                                    ↓
              ┌─────────────────────┴──────────────────────┐
              ↓                                             ↓
        Function Database                          Variable Database
        Signature Storage                          Type Information
        Call Graph (Control Coupling)              Data Flow Graph (Data Coupling)
        DO-178C Requirement Links                  Structural Coverage Map
```

### 6.2 Analysis Integration Flow

```
Feature 1: Virtual Analysis    Feature 2: Coupling Analysis
  (DO-178C §12 change impact)    (DO-178C §6.3.1.b)
         ↓                              ↓
    [Change Detection]          [Control/Data Dependency Mapping]
         ↓                              ↓
    ┌────────────────────────────────────────┐
    │    Knowledge Base Integration          │
    │  (Shared AST, Call Graphs,            │
    │   DO-178C Requirements Trace)         │
    └────────────────────────────────────────┘
         ↓                              ↓
    Feature 3: Dead/Deactivated   Feature 4: DO-178C Code Standards
    Code (DO-178C §6.4.2.2)        Validation (DO-178C §5.1)
    [Reachability + Coverage]      [Rule Application + MISRA C++]
         ↓                              ↓
    ┌────────────────────────────────────────┐
    │    Report Generation                   │
    │    (DO-178C §11 Life Cycle Data)       │
    └────────────────────────────────────────┘
```

---

## 7. Development Phases

### Phase 1: Core Infrastructure (Weeks 1–3)
- Set up project structure with DO-178C configuration management hooks
- Implement code parser and AST builder
- Create local embedding system
- Develop report template engine (DO-178C §11 compliant formats)
- Build test framework (DO-178C verification objectives)

### Phase 2: Feature Implementation (Weeks 4–8)
- Implement Virtual Analysis feature (DO-178C §12 change impact)
- Implement Control/Data Coupling Analysis feature (DO-178C §6.3.1.b)
- Implement Dead/Deactivated Code Detection (DO-178C §6.4.2.2)
- Implement DO-178C Code Standards Validation (DO-178C §5.1)
- Integrate all features

### Phase 3: Testing and Refinement (Weeks 9–11)
- Unit and integration testing per DO-178C verification objectives
- Performance optimization
- Documentation completeness verification (DO-178C §11 life cycle data)
- User acceptance testing with DO-178C SQA sign-off

### Phase 4: Deployment and Support (Week 12)
- Final documentation (DO-178C Software Accomplishment Summary inputs)
- User training materials
- Deployment package creation
- Support handoff

---

## 8. Quality Assurance

### Testing Strategy

| Test Type | Coverage | Tool/Method |
|-----------|----------|------------|
| **Unit Tests** | All modules | pytest |
| **Integration Tests** | Feature workflows | Custom test harness |
| **Regression Tests** | Previous versions | Automated suite |
| **Performance Tests** | Large codebases | Benchmark datasets |
| **DO-178C Structural Coverage Tests** | Statement / Decision / MC/DC (per DAL) | gcov / llvm-cov with MC/DC instrumentation |
| **Documentation Tests** | Output accuracy vs. DO-178C §11 | Manual verification + SQA checklist |

### Quality Metrics

- Code coverage: ≥85% (unit tests); structural coverage per applicable DAL (Statement/Decision/MC/DC)
- Analysis accuracy: ≥95%
- Documentation completeness: 100% (traceable to DO-178C §11 life cycle data)
- Performance: <5 minutes per 1M LOC
- False positive rate: <2%
- DO-178C code standard compliance: 0 unresolved critical violations at release

---

## 9. Deliverables Checklist

### Code and Documentation
- [ ] Complete Python source code (commented and documented per DO-178C §5.1 code standard)
- [ ] claude.md implementation guide
- [ ] API documentation (if applicable)
- [ ] Configuration file templates
- [ ] Requirements Traceability Matrix (RTM) — code to DO-178C objectives

### User Documentation
- [ ] User Guide (step-by-step)
- [ ] Administrator Guide (setup and configuration)
- [ ] Troubleshooting Guide
- [ ] Glossary of terms (including DO-178C terms)

### DO-178C Life Cycle Data (Section 11)
- [ ] Software Development Plan (SDP) — reference/alignment
- [ ] Software Verification Plan (SVP) — reference/alignment
- [ ] Software Code Standard compliance evidence
- [ ] Verification Results (VR) for each analysis feature
- [ ] Software Quality Assurance Records (SQAR)
- [ ] Problem Reports (PR) for identified violations

### Project Artifacts
- [ ] Requirements traceability matrix (RTM)
- [ ] Architecture diagrams
- [ ] Data flow diagrams (control and data coupling)
- [ ] Test case documentation (DO-178C verification objectives)
- [ ] Structural coverage report (per DAL)
- [ ] Release notes

### Sample Outputs
- [ ] Example Virtual Analysis Document (DO-178C §12 change impact)
- [ ] Example Control/Data Coupling Analysis Report (DO-178C §6.3.1.b)
- [ ] Example Dead/Deactivated Code Report (DO-178C §6.4.2.2)
- [ ] Example DO-178C Code Standards Violation Report (DO-178C §5.1)

---

## 10. Success Criteria

### Functional Success
- System analyzes C++ codebases without internet connectivity
- All four features produce outputs meeting DO-178C acceptance criteria
- Documentation is accurate, complete, and professional
- DO-178C SQA checklists are properly filled with verification evidence

### Performance Success
- Analysis completes within acceptable timeframes
- Resource usage is optimized
- Scalable to large codebases (>10M LOC)

### DO-178C Compliance Success
- Zero unresolved dead code items at release (all items either removed or formally justified as deactivated code per §6.4.2.2)
- All code standard violations resolved or formally deviated with rationale
- Structural coverage objectives met per applicable DAL
- All outputs traceable to DO-178C §11 life cycle data requirements

### Quality Success
- Zero critical defects in production
- ≥95% accuracy in analysis results
- 100% documentation coverage
- User satisfaction rating ≥4.5/5

### Business Success
- Delivered on schedule and budget
- Meets all stakeholder requirements
- Professional presentation suitable for DER (Designated Engineering Representative) review
- Positioned for future expansion

---

## 11. Risk Management

### Identified Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| C++ parsing complexity | Medium | High | Use proven libraries (Tree-sitter/Clang) |
| Large codebase performance | Medium | High | Implement incremental analysis, caching |
| Virtual function aliasing edge cases | Medium | Medium | Comprehensive test suite with real examples |
| False positives in dead code detection | High | Medium | Multi-level verification per DO-178C §6.4.2.2; user override with justification |
| Coupling analysis accuracy | Medium | High | Extensive validation with domain experts; DO-178C §6.3.1.b cross-check |
| DO-178C structural coverage gaps | Medium | High | MC/DC instrumentation; coverage gap analysis prior to release |
| MISRA C++ rule deviation without justification | Low | High | Automated flagging; deviation log with DER approval workflow |

---

## 12. Appendices

### A. Reference Standards and Rules

- **RTCA DO-178C** — Software Considerations in Airborne Systems and Equipment Certification (primary standard)
- **RTCA DO-178B** — Previous revision (applicable for legacy projects)
- **RTCA DO-330** — Software Tool Qualification Considerations (applicable to this RAG tool if used in the certification process)
- **RTCA DO-331** — Model-Based Development and Verification Supplement to DO-178C
- **MISRA C++:2008** — Guidelines for the use of the C++ language in critical systems (commonly applied under DO-178C projects)
- **ISO/IEC 14882:2020** — C++ Language Standard
- **ARP4761** — Guidelines and Methods for Conducting the Safety Assessment Process (for DAL determination)
- **ARP4754A** — Guidelines for Development of Civil Aircraft and Systems (system-level context)
- Company-specific DO-178C Software Code Standard

### B. Sample LRU List
ADS, AGMCAL, AGM, APM, BCU, CLOCK, FADEC, FCS, FECU, FMS, GGF, MWS, TACTICAL

### C. Template and Checklist References (DO-178C §11 Life Cycle Data)
- Base Virtual Analysis Document (§11.16 Software Change Notice)
- DO-178C Virtual Analysis SQA Checklist (§11.14 Software Quality Assurance Records)
- LRU Documents for each LRU (Coupling Analysis input)
- VRR Checklist (§11.12 Verification Results)
- Dead/Deactivated Code Analysis Template (§11.14 SQA Records)
- DO-178C Code Standards Compliance Template (§11.14 SQA Records)

### D. Glossary

| Term | Definition |
|------|-----------|
| **RAG** | Retrieval-Augmented Generation |
| **AST** | Abstract Syntax Tree |
| **LRU** | Line Replaceable Unit |
| **VRR** | Verification Results Record |
| **DO-178C** | RTCA DO-178C — Software Considerations in Airborne Systems and Equipment Certification |
| **DO-178B** | Previous revision of DO-178C |
| **DAL** | Design Assurance Level (A through E); determines rigor of DO-178C objectives |
| **MC/DC** | Modified Condition/Decision Coverage — structural coverage required for DAL A |
| **DER** | Designated Engineering Representative — FAA-appointed representative for certification |
| **SDP** | Software Development Plan (DO-178C §11.1) |
| **SVP** | Software Verification Plan (DO-178C §11.3) |
| **SQAP** | Software Quality Assurance Plan (DO-178C §11.5) |
| **SAS** | Software Accomplishment Summary (DO-178C §11.20) |
| **Control Coupling** | A software component that controls execution sequence of another (DO-178C §6.3.1.b) |
| **Data Coupling** | A software component that shares or exchanges data with another (DO-178C §6.3.1.b) |
| **Dead Code** | Code that cannot be executed in any operational condition; prohibited by DO-178C |
| **Deactivated Code** | Code intentionally not executed in current configuration; requires formal justification per DO-178C §6.4.2.2 |
| **MISRA C++** | Motor Industry Software Reliability Association C++ guidelines; widely applied under DO-178C |
| **RTM** | Requirements Traceability Matrix |
| **PR** | Problem Report (DO-178C §11.9) |

---

**Document Approval:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Project Manager | _________________ | ______ | _____________ |
| Technical Lead | _________________ | ______ | _____________ |
| QA Lead (DO-178C SQA) | _________________ | ______ | _____________ |
| DER / Client Representative | _________________ | ______ | _____________ |

---

*End of Project Specification Document*
