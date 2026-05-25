from __future__ import annotations
from typing import Optional
import httpx
from loguru import logger


class OllamaClient:
    def __init__(self, base_url: str = 'http://localhost:11434', model: str = 'llama3.1:8b') -> None:
        self._url = base_url.rstrip('/')
        self._model = model
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                r = httpx.get(f"{self._url}/api/tags", timeout=3)
                self._available = r.status_code == 200
            except Exception:
                self._available = False
        return self._available

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.is_available():
            return ''
        try:
            r = httpx.post(f"{self._url}/api/generate",
                           json={'model': self._model, 'prompt': prompt, 'stream': False,
                                 'options': {'num_predict': max_tokens, 'temperature': 0.1}},
                           timeout=120)
            r.raise_for_status()
            return r.json().get('response', '').strip()
        except Exception as exc:
            logger.warning(f"Ollama generate failed: {exc}")
            return ''
