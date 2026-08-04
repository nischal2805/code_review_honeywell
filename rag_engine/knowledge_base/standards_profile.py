from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def _read_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == '.pdf':
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return '\n'.join(page.extract_text() or '' for page in reader.pages)
    if suffix in {'.docx', '.doc'}:
        from docx import Document

        doc = Document(str(path))
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    if suffix in {'.txt', '.md', '.yaml', '.yml', '.json'}:
        return path.read_text(encoding='utf-8', errors='replace')
    raise ValueError(f'Unsupported standards document format: {path.suffix}')


def _compact_text(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', text.lower())


def _extract_triplet(text: str, pattern: str) -> tuple[str, str, str] | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).upper(), match.group(2).upper(), match.group(3).upper()


def _extract_severity_triplet(text: str, rule_id: str, construct_key: str) -> tuple[str, str, str] | None:
    pattern = (
        rf'{rule_id}.*?{construct_key}.*?'
        rf'(forbidden|restricted|allowed)\D+'
        rf'(forbidden|restricted|allowed)\D+'
        rf'(forbidden|restricted|allowed)'
    )
    return _extract_triplet(text, pattern)


def _extract_int_triplet(text: str, pattern: str) -> tuple[int, int, int] | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _extract_limits(text: str, row_id: str, metric_key: str) -> tuple[int, int, int] | None:
    pattern = (
        rf'{row_id}.*?{metric_key}.*?'
        rf'(?:≤|=)\s*(\d+)\D+'
        rf'(?:≤|=)\s*(\d+)\D+'
        rf'(?:≤|=)\s*(\d+)'
    )
    return _extract_int_triplet(text, pattern)


def _extract_naming_conventions(text: str) -> Dict[str, Dict[str, str]]:
    return {
        'function': {'regex': r'^[A-Z][a-zA-Z0-9]*$', 'convention': 'PascalCase'},
        'variable': {'regex': r'^[a-z][a-zA-Z0-9]*$', 'convention': 'camelCase'},
        'class': {'regex': r'^[A-Z][a-zA-Z0-9]*$', 'convention': 'PascalCase'},
        'constant': {'regex': r'^[A-Z][A-Z0-9_]*$', 'convention': 'SCREAMING_SNAKE_CASE'},
    }


def _infer_policy(value: str | None, fallback: str) -> str:
    return (value or fallback).upper()


def parse_standards_document(text: str) -> Dict[str, Any]:
    compact = _compact_text(text)
    normalized = re.sub(r'\s+', ' ', text)
    normalized_lower = normalized.lower()

    universal = {
        'prohibited_constructs': {
            'goto': 'FORBIDDEN',
        },
        'documentation': {
            'missing_docstring': 'FORBIDDEN',
        },
    }

    dal_specific: Dict[str, Any] = {}

    rule_01 = _extract_severity_triplet(normalized_lower, r'scs-l-01', r'dynamic memory')
    rule_02 = _extract_severity_triplet(normalized_lower, r'scs-l-02', r'goto statement')
    rule_03 = _extract_severity_triplet(normalized_lower, r'scs-l-03', r'exceptions')
    rule_04 = _extract_severity_triplet(normalized_lower, r'scs-l-04', r'recursion')
    rule_05 = _extract_severity_triplet(normalized_lower, r'scs-l-05', r'rtti')

    complexity_a = {
        'cyclomatic_complexity_max': 10,
        'function_length_max': 50,
        'nesting_depth_max': 5,
        'parameter_count_max': 7,
    }
    complexity_b = {
        'cyclomatic_complexity_max': 15,
        'function_length_max': 75,
        'nesting_depth_max': 4,
        'parameter_count_max': 7,
    }
    complexity_c = {
        'cyclomatic_complexity_max': 20,
        'function_length_max': 100,
        'nesting_depth_max': 5,
        'parameter_count_max': 7,
    }

    cc_limits = _extract_limits(normalized_lower, r'scs-m-01', r'cyclomatic complexity')
    length_limits = _extract_limits(normalized_lower, r'scs-m-02', r'function length')
    nesting_limits = _extract_limits(normalized_lower, r'scs-m-03', r'control flow nesting depth')
    param_limits = _extract_limits(normalized_lower, r'scs-m-04', r'number of function parameters')
    if cc_limits:
        complexity_a['cyclomatic_complexity_max'], complexity_b['cyclomatic_complexity_max'], complexity_c['cyclomatic_complexity_max'] = cc_limits
    if length_limits:
        complexity_a['function_length_max'], complexity_b['function_length_max'], complexity_c['function_length_max'] = length_limits
    if nesting_limits:
        complexity_a['nesting_depth_max'], complexity_b['nesting_depth_max'], complexity_c['nesting_depth_max'] = nesting_limits
    if param_limits:
        complexity_a['parameter_count_max'], complexity_b['parameter_count_max'], complexity_c['parameter_count_max'] = param_limits

    dal_specific['A'] = {
        'complexity': copy.deepcopy(complexity_a),
        'severities': {
            'prohibited_constructs': {
                'dynamic_memory': _infer_policy(rule_01[0] if rule_01 else None, 'FORBIDDEN'),
                'exceptions': _infer_policy(rule_03[0] if rule_03 else None, 'FORBIDDEN'),
                'recursion': _infer_policy(rule_04[0] if rule_04 else None, 'FORBIDDEN'),
                'rtti': _infer_policy(rule_05[0] if rule_05 else None, 'FORBIDDEN'),
            },
        },
    }
    dal_specific['B'] = {
        'complexity': copy.deepcopy(complexity_b),
        'severities': {
            'prohibited_constructs': {
                'dynamic_memory': _infer_policy(rule_01[1] if rule_01 else None, 'FORBIDDEN'),
                'exceptions': _infer_policy(rule_03[1] if rule_03 else None, 'FORBIDDEN'),
                'recursion': _infer_policy(rule_04[1] if rule_04 else None, 'RESTRICTED'),
                'rtti': _infer_policy(rule_05[1] if rule_05 else None, 'FORBIDDEN'),
            },
        },
    }
    dal_specific['C'] = {
        'complexity': copy.deepcopy(complexity_c),
        'severities': {
            'prohibited_constructs': {
                'dynamic_memory': _infer_policy(rule_01[2] if rule_01 else None, 'RESTRICTED'),
                'exceptions': _infer_policy(rule_03[2] if rule_03 else None, 'RESTRICTED'),
                'recursion': _infer_policy(rule_04[2] if rule_04 else None, 'ALLOWED'),
                'rtti': _infer_policy(rule_05[2] if rule_05 else None, 'RESTRICTED'),
            },
        },
    }
    dal_specific['D'] = copy.deepcopy(dal_specific['C'])

    return {
        'universal_standards': universal,
        'naming_conventions': _extract_naming_conventions(text),
        'dal_specific_standards': dal_specific,
    }


def build_default_standards_profile(
    cyclomatic_complexity_max: int,
    function_length_max: int,
    nesting_depth_max: int,
    param_count_max: int,
) -> Dict[str, Any]:
    baseline_dal = {
        'complexity': {
            'cyclomatic_complexity_max': cyclomatic_complexity_max,
            'function_length_max': function_length_max,
            'nesting_depth_max': nesting_depth_max,
            'parameter_count_max': param_count_max,
        },
        'severities': {
            'prohibited_constructs': {
                'dynamic_memory': 'FORBIDDEN',
                'exceptions': 'FORBIDDEN',
                'recursion': 'FORBIDDEN',
                'rtti': 'FORBIDDEN',
            },
        },
    }

    return {
        'universal_standards': {
            'prohibited_constructs': {
                'goto': 'FORBIDDEN',
            },
            'documentation': {
                'missing_docstring': 'MINOR',
            },
        },
        'naming_conventions': {
            'function': {'regex': r'^[A-Z][a-zA-Z0-9]*$', 'convention': 'PascalCase'},
            'variable': {'regex': r'^[a-z][a-zA-Z0-9]*$', 'convention': 'camelCase'},
            'class': {'regex': r'^[A-Z][a-zA-Z0-9]*$', 'convention': 'PascalCase'},
            'constant': {'regex': r'^[A-Z][A-Z0-9_]*$', 'convention': 'SCREAMING_SNAKE_CASE'},
        },
        'dal_specific_standards': {
            'A': copy.deepcopy(baseline_dal),
            'B': {
                'complexity': {
                    'cyclomatic_complexity_max': max(15, cyclomatic_complexity_max),
                    'function_length_max': max(75, function_length_max),
                    'nesting_depth_max': max(4, nesting_depth_max),
                    'parameter_count_max': max(5, param_count_max),
                },
                'severities': {
                    'prohibited_constructs': {
                        'dynamic_memory': 'FORBIDDEN',
                        'exceptions': 'FORBIDDEN',
                        'recursion': 'RESTRICTED',
                        'rtti': 'FORBIDDEN',
                    },
                },
            },
            'C': {
                'complexity': {
                    'cyclomatic_complexity_max': max(20, cyclomatic_complexity_max),
                    'function_length_max': max(100, function_length_max),
                    'nesting_depth_max': max(5, nesting_depth_max),
                    'parameter_count_max': max(6, param_count_max),
                },
                'severities': {
                    'prohibited_constructs': {
                        'dynamic_memory': 'RESTRICTED',
                        'exceptions': 'RESTRICTED',
                        'recursion': 'ALLOWED',
                        'rtti': 'RESTRICTED',
                    },
                },
            },
            'D': copy.deepcopy(baseline_dal),
        },
    }


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged: Dict[str, Any] = copy.deepcopy(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def load_raw_standards_profile(path: str | Path) -> Dict[str, Any]:
    standards_path = Path(path).expanduser()
    if not standards_path.exists():
        raise FileNotFoundError(str(standards_path))
    suffix = standards_path.suffix.lower()
    if suffix == '.json':
        return json.loads(standards_path.read_text(encoding='utf-8'))
    if suffix in {'.yaml', '.yml'}:
        loaded = yaml.safe_load(standards_path.read_text(encoding='utf-8')) or {}
        if not isinstance(loaded, dict):
            raise ValueError('Standards profile must be a mapping')
        return loaded
    if suffix in {'.pdf', '.docx', '.doc', '.txt', '.md'}:
        return parse_standards_document(_read_document_text(standards_path))
    raise ValueError(f'Unsupported standards profile format: {standards_path.suffix}')


def extract_standards_profile_sections(data: Mapping[str, Any]) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    for key in ('universal_standards', 'dal_specific_standards', 'naming_conventions'):
        value = data.get(key)
        if isinstance(value, dict):
            profile[key] = copy.deepcopy(value)
    nested = data.get('standards_profile')
    if isinstance(nested, dict):
        profile = deep_merge(profile, nested)
    return profile


def resolve_standards_profile(
    profile: Dict[str, Any],
    cyclomatic_complexity_max: int,
    function_length_max: int,
    nesting_depth_max: int,
    param_count_max: int,
) -> Dict[str, Any]:
    base = build_default_standards_profile(
        cyclomatic_complexity_max=cyclomatic_complexity_max,
        function_length_max=function_length_max,
        nesting_depth_max=nesting_depth_max,
        param_count_max=param_count_max,
    )
    return deep_merge(base, profile or {})