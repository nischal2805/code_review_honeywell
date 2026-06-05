from __future__ import annotations
from pathlib import Path
from typing import Dict
from loguru import logger


class DocReader:
    def read_lru_documents(self, doc_dir: str) -> Dict[str, str]:
        results: Dict[str, str] = {}
        base = Path(doc_dir)
        if not base.exists():
            logger.warning(f"LRU document directory not found: {doc_dir}")
            return results
        for path in base.iterdir():
            suffix = path.suffix.lower()
            try:
                if suffix == '.pdf':
                    results[path.stem.upper()] = self._read_pdf(path)
                elif suffix in ('.docx', '.doc'):
                    results[path.stem.upper()] = self._read_docx(path)
                elif suffix == '.txt':
                    results[path.stem.upper()] = path.read_text(encoding='utf-8', errors='replace')
            except Exception as exc:
                logger.warning(f"Read failed {path}: {exc}")
        logger.info(f"Loaded {len(results)} LRU documents from {doc_dir}")
        return results

    @staticmethod
    def _read_pdf(path: Path) -> str:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return '\n'.join(page.extract_text() or '' for page in reader.pages)

    @staticmethod
    def _read_docx(path: Path) -> str:
        from docx import Document
        doc = Document(str(path))
        return '\n'.join(p.text for p in doc.paragraphs)
