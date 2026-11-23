# Fix SciPy `_minpack` DLL Application Control Error

## Problem

The project fails when running:

```powershell
python main.py validate-config
```

with the following error:

```text
ImportError: DLL load failed while importing _minpack:
An Application Control policy has blocked this file.
```

The import chain is:

```text
main.py
→ rag_engine.core.embeddings.SemanticSearch
→ sentence_transformers
→ sklearn
→ scipy
→ scipy.optimize
→ _minpack
→ Windows Application Control blocks the native SciPy binary
```

This is currently believed to be an environment/dependency issue rather than an error in `main.py` or the `validate-config` implementation.

---

## Objective

Diagnose and fix the SciPy import failure **without unnecessarily changing the project's application code or downgrading unrelated packages**.

Work inside the existing virtual environment:

```text
C:\Users\Admin\Desktop\clean\code_review_honeywell\venv
```

The project root is:

```text
C:\Users\Admin\Desktop\clean\code_review_honeywell
```

---

# Step 1 — Confirm the SciPy failure

Make sure the virtual environment is activated:

```powershell
.\venv\Scripts\Activate.ps1
```

Run:

```powershell
python -c "import scipy; print(scipy.__version__)"
```

### Expected possibilities

If this produces the same:

```text
ImportError: DLL load failed while importing _minpack:
An Application Control policy has blocked this file.
```

then the problem is confirmed to be SciPy/native DLL loading.

If SciPy imports successfully, continue by testing:

```powershell
python -c "import sklearn; print(sklearn.__version__)"
```

and:

```powershell
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"
```

Do not modify application code unless these dependency checks indicate a different problem.

---

# Step 2 — Record the installed package versions

Before changing anything, collect:

```powershell
python --version
python -m pip --version
python -m pip show scipy
python -m pip show scikit-learn
python -m pip show sentence-transformers
```

Also run:

```powershell
python -m pip check
```

Record the output so that the original environment can be reconstructed if necessary.

---

# Step 3 — Locate the SciPy installation

Run:

```powershell
python -c "import pathlib, site; print(site.getsitepackages())"
```

Then inspect the SciPy optimize directory:

```powershell
Get-ChildItem .\venv\Lib\site-packages\scipy\optimize\*minpack*
```

The directory should contain a native extension similar to:

```text
_minpack*.pyd
```

The exact filename may vary by SciPy/Python version.

Do NOT manually delete the `.pyd` file.

---

# Step 4 — Try a clean SciPy reinstall

If Step 1 confirms that SciPy itself cannot be imported, reinstall SciPy inside the active virtual environment:

```powershell
python -m pip uninstall scipy -y
```

Then clear pip's cache:

```powershell
python -m pip cache purge
```

Reinstall:

```powershell
python -m pip install scipy
```

Do not install SciPy globally.

Always use:

```powershell
python -m pip
```

so that pip operates on the currently active virtual environment.

---

# Step 5 — Test SciPy immediately

Run:

```powershell
python -c "import scipy; print('SciPy OK:', scipy.__version__)"
```

If successful, test the next dependency:

```powershell
python -c "import sklearn; print('scikit-learn OK:', sklearn.__version__)"
```

Then:

```powershell
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"
```

Only if all three succeed should the project itself be tested.

---

# Step 6 — Run the original command

Run:

```powershell
python main.py validate-config
```

If it works, stop. Do not make additional dependency changes.

---

# Step 7 — If the same Application Control error remains

If SciPy was freshly reinstalled and this still occurs:

```text
Application Control policy has blocked this file
```

then treat this as a **Windows application-control/security-policy issue**, not simply a corrupt Python package.

Do NOT repeatedly reinstall packages.

First determine the exact native binary being blocked.

Run:

```powershell
Get-ChildItem .\venv\Lib\site-packages\scipy\optimize\*minpack*
```

Also run:

```powershell
python -c "import pathlib; print(pathlib.Path('venv/Lib/site-packages/scipy/optimize').resolve())"
```

The relevant file will normally be a `.pyd` native extension associated with `_minpack`.

---

# Step 8 — Check whether the issue is specific to the project

Create a minimal SciPy import test:

```powershell
python -c "from scipy.optimize import leastsq; print('scipy.optimize OK')"
```

If this produces the same Application Control error, the project code is not responsible.

Also test:

```powershell
python -c "import scipy.stats; print('scipy.stats OK')"
```

If both fail with the same Application Control message, the issue is definitely occurring while Windows loads SciPy's native components.

---

# Step 9 — Do NOT make these changes without evidence

Do NOT:

* Rewrite `main.py`
* Remove `sentence-transformers`
* Replace SciPy with another library
* Randomly downgrade `scikit-learn`
* Randomly downgrade `sentence-transformers`
* Delete `.pyd` files manually
* Disable Windows security features
* Disable application-control policies
* Modify organization security settings
* Install packages globally to bypass the virtual environment

The error specifically mentions an Application Control policy, so bypassing or disabling security controls is not an appropriate troubleshooting step.

---

# Step 10 — If Windows is blocking the freshly installed SciPy binary

If all of the following are true:

1. The virtual environment is active.
2. SciPy was freshly installed.
3. `scipy` still fails to import.
4. The error explicitly says:

```text
An Application Control policy has blocked this file.
```

5. The blocked file is a SciPy native `.pyd`/DLL.

Then report the problem as an environment/security-policy restriction.

The useful information to provide to the system administrator/IT team is:

```text
Python version:
Virtual environment:
SciPy version:
Blocked binary:
Full error:
Project:
```

The key issue is that Windows application control is preventing the Python process from loading SciPy's native extension.

Do not attempt to circumvent the policy.

---

# Step 11 — Final validation

Once SciPy can be imported successfully, run these in order:

```powershell
python -c "import scipy; print('SciPy OK:', scipy.__version__)"
```

```powershell
python -c "import sklearn; print('scikit-learn OK:', sklearn.__version__)"
```

```powershell
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"
```

Then:

```powershell
python main.py validate-config
```

Finally:

```powershell
python -m pip check
```

All dependency checks should complete without errors.

---

## Important Diagnostic Conclusion

The original traceback does **not** indicate that `validate-config` itself is broken.

The failure occurs during module import:

```text
sentence_transformers
→ sklearn
→ scipy
→ scipy.optimize
→ _minpack
```

The critical error is:

```text
An Application Control policy has blocked this file.
```

Therefore, first establish whether reinstalling SciPy resolves the native extension loading problem. If a clean SciPy installation is still blocked with the same policy message, stop modifying Python dependencies and treat it as a Windows application-control restriction.
