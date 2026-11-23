"""Okapi BM25 over a small domain-aware tokenizer (camelCase/snake_case split, light stemming)."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

_WORD = re.compile(r'[A-Za-z]+|\d+')
_CAMEL = re.compile(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+')
STOPWORDS = frozenset('a an and are as at be been by can for from has have in into is it its may of on or '
                      'shall should must that the their there these this to was were which will with not no '
                      'all any each every than then also only such what does do how why where when who'.split())


def _stem(t: str) -> str:
    for suffix in ('ations', 'ation', 'ings', 'ing', 'ies', 'ied', 'es', 'ed', 's'):
        if len(t) > len(suffix) + 3 and t.endswith(suffix):
            return t[:-len(suffix)] + ('y' if suffix in ('ies', 'ied') else '')
    return t


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    out = []
    for word in _WORD.findall(text):
        parts = [p.lower() for p in _CAMEL.findall(word)] or [word.lower()]
        if len(parts) > 1:
            parts.append(word.lower())          # also keep the whole identifier
        for p in parts:
            if keep_stopwords or p not in STOPWORDS:
                out.append(_stem(p))
    return out


class BM25:
    def __init__(self, docs: Sequence[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df: Dict[str, int] = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: List[str]) -> List[float]:
        out = []
        for tf, length in zip(self.tf, self.len):
            s = 0.0
            for t in query:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
            out.append(s)
        return out
