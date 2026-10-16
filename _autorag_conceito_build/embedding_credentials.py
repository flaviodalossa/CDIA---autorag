"""Preencha diretamente aqui ou use variáveis de ambiente / .env.local.

Precedência: resolvedor configurável, ambiente, arquivo local, chave direta.
Ollama e Sentence Transformers não usam essas chaves.
"""
from pathlib import Path
import os

OPENAI_API_KEY_DIRETA = ""
GEMINI_API_KEY_DIRETA = ""
RESOLVEDOR_SEGURO = None  # callable(provedor) -> str | None


def obter_chave(provedor: str) -> str:
    if provedor not in {"openai", "gemini"}:
        return ""
    if RESOLVEDOR_SEGURO:
        value = RESOLVEDOR_SEGURO(provedor)
        if value:
            return str(value).strip()
    name = "OPENAI_API_KEY" if provedor == "openai" else "GEMINI_API_KEY"
    value = os.getenv(name) or (os.getenv("GOOGLE_API_KEY") if provedor == "gemini" else None)
    if not value:
        path = Path(__file__).resolve().parent / ".env.local"
        if path.exists():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                key, separator, content = line.partition("=")
                if separator and key.strip() == name:
                    value = content.strip().strip('"').strip("'")
                    break
    return (value or (OPENAI_API_KEY_DIRETA if provedor == "openai" else GEMINI_API_KEY_DIRETA)).strip()
