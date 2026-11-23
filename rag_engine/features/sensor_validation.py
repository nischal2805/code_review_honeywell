"""Generic sensor validation engine: the corrupt/range, stale, frozen and debounce rules, written once and
parameterised per sensor by its YAML template.

Two consumers use it:
* the static C++ verification (``sensor_analysis``) reads ``check_requirements`` to know which checks a
  sensor's configuration enables and which limits the code must use;
* CSV replay (``sensor_data_checker``) runs ``SensorValidator`` over recorded samples.

Adding a sensor therefore needs only a YAML entry: nothing here is specific to one sensor or category.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from rag_engine.features.sensor_template import SensorTemplate

CHECKS = ('corrupt', 'stale', 'debounce')     # verified in the C++ code; 'frozen' is a replay-only rule
FLAGS = ('corrupt', 'stale', 'frozen', 'debounce')


def fmt(x: float) -> str:
    return f'{x:.10g}'


@dataclass(frozen=True)
class CheckRequirement:
    """What a sensor's configuration asks of one check."""
    check: str
    state: Literal['enabled', 'disabled', 'not_configured']
    expected: str                      # e.g. "range [-54, 54]", "threshold 500 ms", "debounce disabled"
    values: Tuple[float, ...] = ()     # the limits an implementation must use


def check_requirements(tpl: SensorTemplate) -> Dict[str, CheckRequirement]:
    """The single place where a template's YAML switches checks on or off and fixes their limits."""
    bounds = tuple(b for b in (tpl.valid_min, tpl.valid_max) if b is not None)
    low = fmt(tpl.valid_min) if tpl.valid_min is not None else '-inf'
    high = fmt(tpl.valid_max) if tpl.valid_max is not None else 'inf'
    if not tpl.corrupt_range_check:
        corrupt = CheckRequirement('corrupt', 'disabled', 'range check disabled')
    elif not bounds:
        corrupt = CheckRequirement('corrupt', 'not_configured', 'no valid_range configured')
    else:
        corrupt = CheckRequirement('corrupt', 'enabled', f'range [{low}, {high}]', bounds)
    if tpl.stale_threshold_ms is None:
        stale = CheckRequirement('stale', 'not_configured', 'no stale threshold configured')
    else:
        stale = CheckRequirement('stale', 'enabled', f'threshold {fmt(tpl.stale_threshold_ms)} ms',
                                 (tpl.stale_threshold_ms,))
    if not tpl.debounce_enabled:
        debounce = CheckRequirement('debounce', 'disabled', 'debounce disabled')
    elif tpl.debounce_threshold is None:
        debounce = CheckRequirement('debounce', 'not_configured', 'no debounce threshold configured')
    else:
        debounce = CheckRequirement('debounce', 'enabled', f'threshold {fmt(tpl.debounce_threshold)}',
                                    (tpl.debounce_threshold,))
    return {'corrupt': corrupt, 'stale': stale, 'debounce': debounce}


# ── The generic rules ────────────────────────────────────────────────────────

def is_corrupt(tpl: SensorTemplate, value: float) -> bool:
    """Not a finite number, or (range check on) outside the sensor's valid_range."""
    if not math.isfinite(value):
        return True
    return tpl.corrupt_range_check and (
        (tpl.valid_min is not None and value < tpl.valid_min)
        or (tpl.valid_max is not None and value > tpl.valid_max))


def is_stale(tpl: SensorTemplate, timestamp_ms: float, last_update_ms: Optional[float]) -> bool:
    """No update for longer than the stale threshold. An unchanged value is *not* stale (see is_frozen)."""
    return (tpl.stale_threshold_ms is not None and last_update_ms is not None
            and timestamp_ms - last_update_ms > tpl.stale_threshold_ms)


def is_frozen(tpl: SensorTemplate, timestamp_ms: float, last_change_ms: Optional[float]) -> bool:
    """Updates keep arriving but the value has not changed for longer than the stale threshold
    (``stale.detect_frozen``, a separate check from staleness)."""
    return (tpl.stale_detect_frozen and tpl.stale_threshold_ms is not None and last_change_ms is not None
            and timestamp_ms - last_change_ms > tpl.stale_threshold_ms)


class Debouncer:
    """A jump larger than the threshold is held back until ``count`` consecutive samples confirm it."""

    def __init__(self, tpl: SensorTemplate) -> None:
        self.enabled = tpl.debounce_enabled and tpl.debounce_threshold is not None
        self.threshold = tpl.debounce_threshold or 0.0
        self.count = tpl.debounce_count
        self.accepted: Optional[float] = None
        self._run = 0

    def update(self, value: float) -> bool:
        """Feed one valid sample; True when it is held back (a debounce finding)."""
        if not self.enabled or self.accepted is None:
            self.accepted = value
            return False
        if abs(value - self.accepted) > self.threshold:
            self._run += 1
            if self._run >= self.count:
                self.accepted, self._run = value, 0
                return False
            return True
        self.accepted, self._run = value, 0
        return False


@dataclass
class SampleResult:
    timestamp_ms: float
    value: float
    accepted_value: Optional[float]   # None until the first non-corrupt sample
    flags: List[str] = field(default_factory=list)   # subset of FLAGS


class SensorValidator:
    """Stateful generic validator for one sensor; feed samples in time order."""

    def __init__(self, tpl: SensorTemplate) -> None:
        self.template = tpl
        self._debounce = Debouncer(tpl)
        self._prev_t: Optional[float] = None
        self._last_good: Optional[float] = None
        self._last_change_t: Optional[float] = None

    def validate(self, timestamp_ms: float, value: float) -> SampleResult:
        tpl = self.template
        stale = is_stale(tpl, timestamp_ms, self._prev_t)
        self._prev_t = timestamp_ms
        if is_corrupt(tpl, value):
            return SampleResult(timestamp_ms, value, self._debounce.accepted, ['corrupt'] + (['stale'] if stale else []))
        if self._last_good is None or abs(value - self._last_good) > 1e-9:
            self._last_change_t = timestamp_ms
        self._last_good = value
        frozen = is_frozen(tpl, timestamp_ms, self._last_change_t)
        held = self._debounce.update(value)
        flags = (['stale'] if stale else []) + (['frozen'] if frozen else []) + (['debounce'] if held else [])
        return SampleResult(timestamp_ms, value, self._debounce.accepted, flags)
