# Implementation Plan: Integrating the New Software Code Standard

This document outlines the necessary code changes to implement the rules defined in `coding_standards_to_use.md`.

## Step 1: Update `config.yaml` with the New Rules

The first step is to translate the rules from your Markdown standard into the `config.yaml` file. This makes the rules machine-readable for your Python script.

**Action:** Replace the content of your `config.yaml` with the following:

```yaml
# config.yaml
# This configuration is based on the rules in PHX-SCS-001.

# Universal rules that apply to all DALs
universal_standards:
  severities:
    prohibited_constructs:
      goto: FORBIDDEN # Rule SCS-L-02
    documentation:
      missing_docstring: MAJOR # Based on Rule SCS-D-01

# DAL-specific standards, matching the tables in the standard
dal_specific_standards:
  # DAL C is the baseline
  C:
    complexity:
      cyclomatic_complexity_max: 20 # SCS-M-01
      function_length_max: 100      # SCS-M-02
      nesting_depth_max: 5          # SCS-M-03
      parameter_count_max: 6        # SCS-M-04
    severities:
      prohibited_constructs:
        dynamic_memory: Restricted  # SCS-L-01
        exceptions: Restricted      # SCS-L-03
        recursion: Allowed          # SCS-L-04
        rtti: Restricted            # SCS-L-05

  # DAL B is stricter
  B:
    complexity:
      cyclomatic_complexity_max: 15 # SCS-M-01
      function_length_max: 75       # SCS-M-02
      nesting_depth_max: 4          # SCS-M-03
      parameter_count_max: 5        # SCS-M-04
    severities:
      prohibited_constructs:
        dynamic_memory: FORBIDDEN   # SCS-L-01
        exceptions: FORBIDDEN       # SCS-L-03
        recursion: Restricted       # SCS-L-04
        rtti: FORBIDDEN             # SCS-L-05

  # DAL A is the strictest
  A:
    complexity:
      cyclomatic_complexity_max: 10 # SCS-M-01
      function_length_max: 50       # SCS-M-02
      nesting_depth_max: 3          # SCS-M-03
      parameter_count_max: 4        # SCS-M-04
    severities:
      prohibited_constructs:
        dynamic_memory: FORBIDDEN   # SCS-L-01
        exceptions: FORBIDDEN       # SCS-L-03
        recursion: FORBIDDEN        # SCS-L-04
        rtti: FORBIDDEN             # SCS-L-05

# Naming convention rules from SCS 4.1
naming_conventions:
  function:
    regex: '^[A-Z][a-zA-Z0-9]*$'
    convention: 'PascalCase'
  variable:
    regex: '^[a-z][a-zA-Z0-9]*$'
    convention: 'camelCase'
  class:
    regex: '^[A-Z][a-zA-Z0-9]*$'
    convention: 'PascalCase'
  constant:
    regex: '^[A-Z][A-Z0-9_]*$'
    convention: 'SCREAMING_SNAKE_CASE'
```

---

## Step 2: Modify `rag_engine/features/standards_validator.py`

Now, update the `StandardsValidator` class to read from the new `config.yaml` structure and use the DAL-specific rules.

### 2.1. Update the `__init__` Method and Add a Rule Loader

Modify the class to accept the `dal` and load the correct rules by merging the universal and DAL-specific configurations.

**Action:** Update the `StandardsValidator` class as follows.

```python
# In rag_engine/features/standards_validator.py
import copy # Add this import at the top of the file

class StandardsValidator:
    def __init__(self, parse_results: dict, config: dict, dal: str = 'C'):
        self.parse_results = parse_results
        self.dal = dal.upper()
        self.rules = self._load_rules_for_dal(config)
        self.violations = []

    def _load_rules_for_dal(self, config: dict) -> dict:
        """Loads rules by merging universal and DAL-specific configs."""
        # Start with universal rules
        rules = copy.deepcopy(config.get('universal_standards', {}))
        
        # Get the rules for the specific DAL
        dal_rules = config.get('dal_specific_standards', {}).get(self.dal, {})
        
        # Merge DAL-specific rules into the base rules
        for key, value in dal_rules.items():
            if key in rules and isinstance(rules[key], dict):
                rules[key].update(value)
            else:
                rules[key] = value
        
        # Add naming conventions to the ruleset
        rules['naming_conventions'] = config.get('naming_conventions', {})
        
        return rules

    def analyze(self) -> list:
        # (This method should call your individual validation methods)
        self._validate_complexity_metrics()
        self._validate_prohibited_constructs()
        self._validate_naming_conventions()
        self._validate_documentation()
        return self.violations
```

### 2.2. Update the Validation Methods to Use the Loaded Rules

Now, go through each validation method and replace the hardcoded values with the values from `self.rules`.

**Action:** Modify your validation methods to use the loaded rules.

**Example for Complexity Rules:**

```python
# In rag_engine/features/standards_validator.py

def _validate_complexity_metrics(self):
    complexity_rules = self.rules.get('complexity', {})
    max_complexity = complexity_rules.get('cyclomatic_complexity_max', 999)
    max_length = complexity_rules.get('function_length_max', 999)
    max_nesting = complexity_rules.get('nesting_depth_max', 999)
    max_params = complexity_rules.get('parameter_count_max', 999)

    for fn in self.parse_results.get('functions', []):
        if fn.cyclomatic_complexity > max_complexity:
            self.violations.append(Violation(
                rule='SCS-M-01',
                file=fn.file_path,
                line=fn.line_number,
                element=fn.name,
                message=f'Cyclomatic complexity {fn.cyclomatic_complexity} exceeds DAL {self.dal} limit of {max_complexity}',
                severity='MEDIUM'
            ))
        # Add similar checks for max_length, max_nesting, and max_params
```

**Example for Prohibited Constructs:**

```python
# In rag_engine/features/standards_validator.py

def _validate_prohibited_constructs(self):
    """
    Validates code against prohibited constructs based on DAL-specific rules.
    """
    severities = self.rules.get('severities', {}).get('prohibited_constructs', {})

    for fn in self.parse_results.get('functions', []):
        # Rule SCS-L-01: Check for dynamic memory
        if 'new' in fn.body or 'malloc' in fn.body:
            severity = severities.get('dynamic_memory', 'CRITICAL')
            if severity != 'Allowed':
                self.violations.append(Violation(
                    rule='SCS-L-01',
                    file=fn.file_path, line=fn.line_number, element=fn.name,
                    message=f'Dynamic memory allocation (`new`/`malloc`) is {severity} for DAL {self.dal}.',
                    severity=severity
                ))

        # Rule SCS-L-02: Check for goto
        if 'goto' in fn.body:
            severity = severities.get('goto', 'CRITICAL')
            if severity != 'Allowed':
                self.violations.append(Violation(
                    rule='SCS-L-02',
                    file=fn.file_path, line=fn.line_number, element=fn.name,
                    message='Use of `goto` is prohibited.',
                    severity=severity
                ))

        # Rule SCS-L-03: Check for exceptions
        if 'try' in fn.body or 'catch' in fn.body or 'throw' in fn.body:
            severity = severities.get('exceptions', 'CRITICAL')
            if severity != 'Allowed':
                self.violations.append(Violation(
                    rule='SCS-L-03',
                    file=fn.file_path, line=fn.line_number, element=fn.name,
                    message=f'Use of exceptions (`try`/`catch`/`throw`) is {severity} for DAL {self.dal}.',
                    severity=severity
                ))

        # Rule SCS-L-04: Check for recursion
        if fn.name in fn.calls: # A function is recursive if it calls itself
            severity = severities.get('recursion', 'CRITICAL')
            if severity != 'Allowed':
                self.violations.append(Violation(
                    rule='SCS-L-04',
                    file=fn.file_path, line=fn.line_number, element=fn.name,
                    message=f'Recursion is {severity} for DAL {self.dal}.',
                    severity=severity
                ))

        # Rule SCS-L-05: Check for RTTI
        if 'dynamic_cast' in fn.body or 'typeid' in fn.body:
            severity = severities.get('rtti', 'CRITICAL')
            if severity != 'Allowed':
                self.violations.append(Violation(
                    rule='SCS-L-05',
                    file=fn.file_path, line=fn.line_number, element=fn.name,
                    message=f'Use of RTTI (`dynamic_cast`/`typeid`) is {severity} for DAL {self.dal}.',
                    severity=severity
                ))
```
