from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rag_engine.features.sensor_template import SensorTemplate

Sample = Tuple[float, float]  # (timestamp_ms, value)


@dataclass
class SampleResult:
    timestamp_ms: float
    value: float
    accepted_value: Optional[float]   # None until the first non-corrupt sample
    flags: List[str] = field(default_factory=list)   # subset of corrupt / stale / debounce


def check_samples(template: SensorTemplate, samples: List[Sample]) -> List[SampleResult]:
    """Replay samples through the template's corrupt / stale / debounce rules."""
    stale_ms = template.stale_threshold_ms
    use_debounce = template.debounce_enabled and template.debounce_threshold is not None
    prev_t = last_good = last_change_t = accepted = None
    run = 0
    results: List[SampleResult] = []

    for t, v in sorted(samples, key=lambda s: s[0]):
        gap_stale = stale_ms is not None and prev_t is not None and t - prev_t > stale_ms
        prev_t = t

        corrupt = not math.isfinite(v) or (template.corrupt_range_check and (
            (template.valid_min is not None and v < template.valid_min)
            or (template.valid_max is not None and v > template.valid_max)))
        if corrupt:
            results.append(SampleResult(t, v, accepted, ['corrupt'] + (['stale'] if gap_stale else [])))
            continue

        if last_good is None or abs(v - last_good) > 1e-9:
            last_change_t = t
        last_good = v
        frozen_stale = (stale_ms is not None and template.stale_detect_frozen
                        and t - last_change_t > stale_ms)

        debounced = False
        if not use_debounce or accepted is None:
            accepted = v
        elif abs(v - accepted) > template.debounce_threshold:
            run += 1
            if run >= template.debounce_count:
                accepted, run = v, 0
            else:
                debounced = True
        else:
            accepted, run = v, 0

        flags = (['stale'] if gap_stale or frozen_stale else []) + (['debounce'] if debounced else [])
        results.append(SampleResult(t, v, accepted, flags))
    return results


def load_samples_csv(path: str | Path) -> Dict[str, List[Sample]]:
    """Read a `sensor,timestamp_ms,value` CSV into per-sensor sample lists (file order).

    Raises ValueError, naming the line, for a missing column, a short or blank row, a non-numeric value or
    a field the csv module refuses (larger than its 128 KB field limit, NUL bytes)."""
    samples: Dict[str, List[Sample]] = {}
    with Path(path).expanduser().open(newline='', encoding='utf-8') as f:
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
    counts = {flag: 0 for flag in ('corrupt', 'stale', 'debounce')}
    for r in results:
        for flag in r.flags:
            counts[flag] += 1
    return {**counts, 'total': len(results)}
