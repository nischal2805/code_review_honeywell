"""Format-aware ingestion and structure-aware chunking.

Documents become ``Block``s (a line or paragraph with its page and heading flag), then sentences carrying
their page and section, then overlapping ``Chunk``s that never cross a section boundary. Chunk ids are
content hashes, so an unchanged chunk keeps its id (and cached embedding) when the document changes."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import yaml

SUPPORTED_SUFFIXES = ('.pdf', '.docx', '.txt', '.md', '.markdown', '.rst', '.yaml', '.yml', '.json')

# "4.2 Function length", "# Naming", "SECTION 3: MEMORY", "Rule 7 - Recursion"
_HEADING = re.compile(r'^\s*(?:#{1,6}\s+\S.*|(?:\d+(?:\.\d+)*\.?|[A-Z]\.\d+(?:\.\d+)*)\s+[A-Z][^.!?]{0,90}'
                      r'|(?:section|chapter|part|rule|appendix)\s+[\w.]+\s*[:\-–]?\s*[^.!?]{0,90})\s*$', re.IGNORECASE)
_CAPS_HEADING = re.compile(r'^[A-Z][A-Z0-9 ,/&()\-]{3,80}$')       # "MEMORY MANAGEMENT" (case-sensitive)
_SENTENCE_END = re.compile(r'(?<=[.!?;])\s+(?=[\"\'(\[]?[A-Z0-9•\-])')
_BULLET = re.compile(r'^\s*(?:[-*•▪◦]|\(?[a-z0-9]{1,3}[.)])\s+')


@dataclass(frozen=True)
class Block:
    text: str
    page: Optional[int] = None
    heading: bool = False


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: Optional[int] = None
    section: str = ''
    index: int = 0

    def cite(self) -> str:
        parts = [Path(self.source).name or self.source]
        if self.page is not None:
            parts.append(f'p.{self.page}')
        if self.section:
            parts.append(f'§ {self.section}')
        return ', '.join(parts)


def _hash(*parts: str) -> str:
    return hashlib.sha1('\x1f'.join(parts).encode('utf-8')).hexdigest()[:16]


def read_blocks(path: str | Path) -> List[Block]:
    """Lines/paragraphs of a document with page numbers (PDF) and heading flags (DOCX styles, Markdown)."""
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if suffix == '.pdf':
        from pypdf import PdfReader
        out: List[Block] = []
        for number, page in enumerate(PdfReader(str(p)).pages, start=1):
            out += [Block(line, number) for line in (page.extract_text() or '').splitlines()]
            out.append(Block('', number))
        return out
    if suffix == '.docx':
        from docx import Document
        out = []
        for para in Document(str(p)).paragraphs:
            style = (para.style.name if para.style is not None else '') or ''
            out.append(Block(para.text, None, style.lower().startswith(('heading', 'title'))))
            out.append(Block(''))
        for table in Document(str(p)).tables:
            for row in table.rows:
                out.append(Block(' | '.join(cell.text.strip() for cell in row.cells)))
            out.append(Block(''))
        return out
    if suffix in ('.yaml', '.yml', '.json'):
        text = p.read_text(encoding='utf-8', errors='replace')
        data = json.loads(text) if suffix == '.json' else yaml.safe_load(text)
        return [Block(line) for line in _flatten(data)]
    if suffix in SUPPORTED_SUFFIXES:
        return [Block(line, None, line.lstrip().startswith('#')) for line in
                p.read_text(encoding='utf-8', errors='replace').splitlines()]
    raise ValueError(f'Unsupported document format {p.suffix!r} (supported: {", ".join(SUPPORTED_SUFFIXES)})')


def _flatten(data, prefix: str = '') -> Iterator[str]:
    """Structured standards (YAML/JSON) as 'key path: value' sentences so they can be searched too."""
    if isinstance(data, dict):
        for k, v in data.items():
            yield from _flatten(v, f'{prefix}{k} ' if prefix else f'{k} ')
    elif isinstance(data, list):
        for v in data:
            yield from _flatten(v, prefix)
    else:
        yield f'{prefix.strip().replace("_", " ")}: {data}.'


def _is_heading(block: Block) -> bool:
    text = block.text.strip()
    if block.heading:
        return bool(text)
    if not text or len(text) > 100 or text.endswith(('.', ',', ';', ':')) and not text.startswith('#'):
        return False
    if '|' in text or re.search(r'\b(?:shall|must|should)\b', text, re.IGNORECASE):
        return False
    return bool(_HEADING.match(text) or _CAPS_HEADING.match(text))


def _clean_heading(text: str) -> str:
    return re.sub(r'^#+\s*', '', text.strip())[:100]


def paragraphs(blocks: Iterable[Block]) -> Iterator[Tuple[str, Optional[int], str]]:
    """(paragraph text, page of its first line, current section). Wrapped lines are joined; blank lines,
    headings, bullets and table rows end a paragraph."""
    section, buf, page = '', [], None

    def flush():
        nonlocal buf, page
        text = ' '.join(buf).strip()
        buf, start = [], page
        page = None
        return (text, start, section) if text else None

    for b in blocks:
        text = b.text.strip()
        if not text:
            item = flush()
            if item:
                yield item
            continue
        if _is_heading(b):
            item = flush()
            if item:
                yield item
            section = _clean_heading(text)
            continue
        if _BULLET.match(text) or '|' in text:
            item = flush()
            if item:
                yield item
        if page is None:
            page = b.page
        buf.append(text)
        if '|' in text:          # a table row is a paragraph of its own
            item = flush()
            if item:
                yield item
    item = flush()
    if item:
        yield item


def sentences(blocks: Iterable[Block]) -> Iterator[Tuple[str, Optional[int], str]]:
    """(sentence, page, section) for every sentence of the document, in order."""
    for para, page, section in paragraphs(blocks):
        for s in _SENTENCE_END.split(para):
            s = _BULLET.sub('', s).strip()
            if len(s) > 2:
                yield s, page, section


def chunk_blocks(blocks: Iterable[Block], source: str, max_words: int = 160, overlap_words: int = 40) -> List[Chunk]:
    """Sentence-packed chunks of at most ~max_words, overlapping by ~overlap_words, within one section."""
    chunks: List[Chunk] = []
    current: List[Tuple[str, Optional[int]]] = []
    section = None

    def emit():
        if not current:
            return
        text = ' '.join(s for s, _ in current)
        page = next((p for _, p in current if p is not None), None)
        heading = section or ''
        body = f'{heading}: {text}' if heading else text
        chunks.append(Chunk(_hash(source, body), body, source, page, heading, len(chunks)))

    for sent, page, sec in sentences(blocks):
        if sec != section:
            emit()
            current, section = [], sec
        words = len(sent.split())
        if current and sum(len(s.split()) for s, _ in current) + words > max_words:
            emit()
            keep: List[Tuple[str, Optional[int]]] = []
            for s, p in reversed(current):          # overlap: the tail sentences of the previous chunk
                if sum(len(x.split()) for x, _ in keep) + len(s.split()) > overlap_words:
                    break
                keep.insert(0, (s, p))
            current = keep
        current.append((sent, page))
    emit()
    return chunks


def chunk_functions(functions, max_body_words: int = 120) -> List[Chunk]:
    """One chunk per function: signature, documentation and the start of the body."""
    out = []
    for fn in functions:
        params = ', '.join(f'{p.type_} {p.name}'.strip() for p in fn.parameters)
        body = ' '.join((fn.body or '').split()[:max_body_words])
        text = f'{fn.return_type} {fn.name}({params}) {fn.docstring or ""} {body}'.strip()
        out.append(Chunk(_hash(fn.file_path, fn.name, text), text, fn.file_path, None, fn.name, len(out)))
    return out
