"""Verificação real dos perfis Ollama, sem modificar índices ou dados de estudo."""
from pathlib import Path
import numpy as np
from autorag.core import Project, normalize
from autorag.providers import get_provider

def main():
    project = Project(Path(__file__).resolve().parent)
    profiles = [p for p in project.config()["perfis"].values() if p["provedor"]=="ollama"]
    if not profiles:
        raise ValueError("Cadastre pelo menos um modelo Ollama.")
    for profile in profiles:
        backend = get_provider(profile)
        documents = ["O consentimento livre e esclarecido protege os participantes da pesquisa.", "A previsão do tempo indica chuva amanhã."]
        matrix = normalize(backend.embed(documents), profile["dimensoes"])
        query = normalize(backend.embed(["Como proteger os participantes de uma pesquisa?"], "consulta"), profile["dimensoes"])[0]
        print(profile["modelo"], profile["dimensoes"], "dimensões; escores:", (matrix@query).round(6).tolist())
        assert matrix.shape==(2,profile["dimensoes"]) and np.all(np.isfinite(matrix))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
