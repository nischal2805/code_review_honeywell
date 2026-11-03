from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from rag_engine.knowledge_base.standards_profile import deep_merge


@dataclass
class SensorTemplate:
    name: str
    type: str = ''
    data_type: str = 'float'
    unit: str = ''
    class_name: str = ''                  # C++ class to match; '' means use name
    parameter_group: str = ''
    io_group: str = ''
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
    sampling_interval_ms: Optional[float] = None
    stale_threshold_ms: Optional[float] = None
    stale_detect_frozen: bool = True
    debounce_enabled: bool = True
    debounce_threshold: Optional[float] = None
    debounce_count: int = 3
    corrupt_range_check: bool = True


def _num(sensor: str, label: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Sensor '{sensor}': {label} must be a number, got {value!r}")
    return float(value)


def _int(sensor: str, label: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Sensor '{sensor}': {label} must be an integer, got {value!r}")
    return value


def _flag(sensor: str, label: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Sensor '{sensor}': {label} must be true or false, got {value!r}")
    return value


# field -> (nested section, key in section, converter, human label)
_NESTED = {
    'valid_min': ('valid_range', 'min', _num),
    'valid_max': ('valid_range', 'max', _num),
    'stale_threshold_ms': ('stale', 'threshold_ms', _num),
    'stale_detect_frozen': ('stale', 'detect_frozen', _flag),
    'debounce_enabled': ('debounce', 'enabled', _flag),
    'debounce_threshold': ('debounce', 'threshold', _num),
    'debounce_count': ('debounce', 'count', _int),
    'corrupt_range_check': ('corrupt', 'range_check', _flag),
}
_PLAIN_STR = ('type', 'data_type', 'unit', 'class_name', 'parameter_group', 'io_group')
_SECTION_KEYS: Dict[str, set] = {}
for _section, _key, _ in _NESTED.values():
    _SECTION_KEYS.setdefault(_section, set()).add(_key)
_SENSOR_KEYS = {'name', 'sampling_interval_ms', *_PLAIN_STR, *_SECTION_KEYS}
_FILE_KEYS = {'defaults', 'sensors', 'sensor'}


def _check_keys(label: str, mapping: Dict[str, Any], allowed: set, where: str = '') -> None:
    # A typo'd key would silently leave a safety check unconfigured, so reject it.
    unknown = sorted(str(k) for k in mapping if k not in allowed)
    if unknown:
        raise ValueError(f"{label}: unknown key {unknown[0]!r}{where} (allowed: {', '.join(sorted(allowed))})")


def _check_sensor_keys(label: str, raw: Dict[str, Any]) -> None:
    _check_keys(label, raw, _SENSOR_KEYS)
    for section, keys in _SECTION_KEYS.items():
        if isinstance(raw.get(section), dict):
            _check_keys(label, raw[section], keys, f' in {section}')


def _build(raw: Any, defaults: Dict[str, Any]) -> SensorTemplate:
    if not isinstance(raw, dict):
        raise ValueError(f'Sensor entry must be a mapping, got {raw!r}')
    name = raw.get('name')
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Sensor template is missing required field 'name': {raw!r}")
    _check_sensor_keys(f"Sensor '{name}'", raw)
    raw = deep_merge(defaults, raw)

    kwargs: Dict[str, Any] = {'name': name}
    for key in _PLAIN_STR:
        if raw.get(key) is not None:
            kwargs[key] = str(raw[key])
    if raw.get('sampling_interval_ms') is not None:
        kwargs['sampling_interval_ms'] = _num(name, 'sampling_interval_ms', raw['sampling_interval_ms'])
    for field_name, (section, key, convert) in _NESTED.items():
        body = raw.get(section)
        if body is not None and not isinstance(body, dict):
            raise ValueError(f"Sensor '{name}': {section} must be a mapping, got {body!r}")
        if body and body.get(key) is not None:
            kwargs[field_name] = convert(name, f'{section}.{key}', body[key])

    tpl = SensorTemplate(**kwargs)
    if tpl.valid_min is not None and tpl.valid_max is not None and tpl.valid_min >= tpl.valid_max:
        raise ValueError(
            f"Sensor '{name}': valid_range min ({tpl.valid_min:g}) must be less than max ({tpl.valid_max:g})")
    for label, value in (('sampling_interval_ms', tpl.sampling_interval_ms),
                         ('stale.threshold_ms', tpl.stale_threshold_ms),
                         ('debounce.threshold', tpl.debounce_threshold)):
        if value is not None and value < 0:
            raise ValueError(f"Sensor '{name}': {label} must not be negative, got {value:g}")
    if tpl.debounce_count < 1:
        raise ValueError(f"Sensor '{name}': debounce.count must be >= 1, got {tpl.debounce_count}")
    return tpl


def load_sensor_templates(path: str | Path) -> List[SensorTemplate]:
    """Load sensor templates from YAML: optional `defaults:` plus `sensors:` (list) or `sensor:` (single)."""
    data = yaml.safe_load(Path(path).expanduser().read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'{path}: top level must be a mapping')
    _check_keys(str(path), data, _FILE_KEYS)
    defaults = data.get('defaults') or {}
    if not isinstance(defaults, dict):
        raise ValueError(f'{path}: defaults must be a mapping, got {defaults!r}')
    _check_sensor_keys('defaults', defaults)
    raws = list(data.get('sensors') or [])
    if data.get('sensor'):
        raws.append(data['sensor'])
    if not raws:
        raise ValueError(f"{path}: no sensors defined (expected 'sensors:' list or 'sensor:' mapping)")
    return [_build(raw, defaults) for raw in raws]
