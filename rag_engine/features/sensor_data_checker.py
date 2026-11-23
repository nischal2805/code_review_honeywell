from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from rag_engine.features.sensor_template import SensorTemplate
from rag_engine.features.sensor_validation import FLAGS, SampleResult, SensorValidator   # SampleResult re-exported

Sample = Tuple[float, float]  # (timestamp_ms, value)


def check_samples(template: SensorTemplate, samples: List[Sample]) -> List[SampleResult]:
    """Replay samples (sorted by time) through the generic validation engine with the template's rules."""
    validator = SensorValidator(template)
    return [validator.validate(t, v) for t, v in sorted(samples, key=lambda s: s[0])]


@dataclass
class SensorReplay:
    sensor: str
    results: List[SampleResult]
    counts: Dict[str, int]


@dataclass
class ReplayResult:
    """CSV replay of every sensor: shows how the configured rules treat the supplied samples. It does not
    show that the C++ runtime behaves the same way."""
    source: str
    sensors: Dict[str, SensorReplay] = field(default_factory=dict)
    unknown_sensors: List[str] = field(default_factory=list)      # in the CSV, no YAML configuration
    without_samples: List[str] = field(default_factory=list)      # configured, no samples in the CSV


def replay_samples(templates: List[SensorTemplate], samples: Dict[str, List[Sample]], source: str = '') -> ReplayResult:
    """Associate each CSV sensor with its configuration (by name, else by class_name) and replay it."""
    by_name = {t.name: t for t in templates}
    by_class = {t.class_name: t for t in templates if t.class_name}
    result = ReplayResult(source)
    for sensor, rows in samples.items():
        tpl = by_name.get(sensor) or by_class.get(sensor)
        if tpl is None:
            result.unknown_sensors.append(sensor)
            continue
        replayed = check_samples(tpl, rows)
        result.sensors[tpl.name] = SensorReplay(tpl.name, replayed, summarize(replayed))
    result.without_samples = [t.name for t in templates if t.name not in result.sensors]
    return result


def load_samples_csv(path: str | Path) -> Dict[str, List[Sample]]:
    """Read a `sensor,timestamp_ms,value` CSV into per-sensor sample lists (file order).

    Raises ValueError, naming the line, for a missing column, a short or blank row, a non-numeric value or
    a field the csv module refuses (larger than its 128 KB field limit, NUL bytes) or text that is not UTF-8
    (a leading BOM, as Excel writes, is accepted)."""
    samples: Dict[str, List[Sample]] = {}
    with Path(path).expanduser().open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        try:
            if not {'sensor', 'timestamp_ms', 'value'} <= set(reader.fieldnames or []):
                raise ValueError(f'{path}: CSV header must contain sensor,timestamp_ms,value')
            for row in reader:
                try:
                    sensor = row['sensor'].strip()   # None (AttributeError) when the row is shorter than the header
                    if not sensor:
                        raise ValueError('empty sensor name')
                    samples.setdefault(sensor, []).append((float(row['timestamp_ms']), float(row['value'])))
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError(f'{path}:{reader.line_num}: bad sample row {row!r}') from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f'{path}: not UTF-8 text ({exc.reason} at byte {exc.start})') from exc
        except csv.Error as exc:
            raise ValueError(f'{path}:{reader.line_num + 1}: {exc}') from exc   # line_num lags: the bad record is the next
    return samples


def parse_inline_samples(text: str) -> List[Sample]:
    """'t:v, t:v, ...' (t in ms) -> [(t, v)]. Raises ValueError on a missing colon, a non-number or no samples."""
    samples: List[Sample] = []
    for item in text.split(','):
        if not item.strip():
            continue
        t, sep, v = item.partition(':')
        if not sep:
            raise ValueError(f"bad sample {item.strip()!r}: expected 't:v'")
        samples.append((float(t), float(v)))
    if not samples:
        raise ValueError('no samples')
    return samples


def summarize(results: List[SampleResult]) -> Dict[str, int]:
    counts = {flag: 0 for flag in FLAGS}
    for r in results:
        for flag in r.flags:
            counts[flag] += 1
    return {**counts, 'total': len(results)}
