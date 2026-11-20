# Uploadable Coding Standards PDFs

Upload either PDF directly through the TUI, or pass it to the CLI:

```powershell
python main.py validate-config --coding-standards-pdf samples/coding_standards_strict.pdf
python main.py analyze sample_codebase --coding-standards-pdf samples/coding_standards_strict.pdf --output-format markdown
```

- `coding_standards_strict.pdf` uses tight limits and forbids all prohibited constructs.
- `coding_standards_relaxed.pdf` uses different DAL limits and allows recursion at DAL C.

The PDFs are generated from `tools/create_sample_standards_pdfs.py`, so they can be regenerated without editing binary files manually.
