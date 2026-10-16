"""Operações analíticas persistentes, sem misturar espaços de embeddings."""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from .core import file_hash, normalize, now, uid, write_json, write_npz
from .experiments import load_experiment


def load_vectors(project, references, original=False):
    """Cada referência inclui perfil_id, arquivo_npz, linha, embedding_id.

    Referências de resultados herdam perfil_id da execução. Para originais
    legados use original_disponivel=False; eles nunca são reconstruídos.
    """
    if not references:
        raise ValueError("Selecione ao menos um vetor.")
    profiles = {ref.get("perfil_id") for ref in references}
    if len(profiles) != 1 or None in profiles:
        raise ValueError("A operação exige referências de um único perfil de modelo.")
    profile_id = next(iter(profiles))
    profile = project.config()["perfis"].get(profile_id)
    if not profile:
        raise ValueError("Perfil não cadastrado.")
    vectors, cache = [], {}
    for ref in references:
        path = project.path(ref["arquivo_npz"])
        # A própria árvore do perfil limita a leitura a seu espaço vetorial.
        if f"/{profile_id}/" not in "/" + project.relative(path):
            raise ValueError("Arquivo não pertence ao perfil indicado.")
        if ref.get("checksum") and file_hash(path) != ref["checksum"]:
            raise ValueError("Checksum divergente.")
        key = str(path)
        if key not in cache:
            with np.load(path, allow_pickle=False) as package:
                cache[key] = {name: package[name].copy() for name in package.files}
        package = cache[key]
        row = ref["linha"]
        if not isinstance(row, int) or row < 0 or row >= len(package["vetores"]):
            raise ValueError("Linha inválida na referência.")
        if str(package["embedding_ids"][row]) != ref["embedding_id"]:
            raise ValueError("ID não corresponde à linha do NPZ.")
        if original and (ref.get("original_disponivel") is not True or "vetores_originais" not in package):
            raise ValueError("Coordenadas originais não disponíveis para esta unidade.")
        vectors.append(package["vetores_originais" if original else "vetores"][row])
    matrix = np.stack(vectors)
    normalize(matrix, profile["dimensoes"])
    return profile, matrix


def experiment_references(project, execution_id, categories=None, minimum=None, maximum=None):
    experiment = load_experiment(project, execution_id, verify=True)
    refs = []
    for row in experiment["universo"]:
        if categories and row["categoria"] not in categories:
            continue
        score = row["similaridade_cosseno"]
        if minimum is not None and score < minimum or maximum is not None and score > maximum:
            continue
        refs.append({**row["vetor_recuperado"], "perfil_id": experiment["perfil_id"],
                     "categoria": row["categoria"], "topico_id": row["topico_id"],
                     "escore_origem": score,
                     "original_disponivel": row.get("original_disponivel", False),
                     "checksum": experiment["indices_utilizados"][row["categoria"]]["checksum"]})
    return refs


def compare(project, left, right):
    _, matrix = load_vectors(project, [left, right])
    normalized = normalize(matrix)
    score = float(np.clip(normalized[0] @ normalized[1], -1, 1))
    return {"similaridade_cosseno": score, "distancia_cosseno": 1-score,
            "distancia_euclidiana": float(np.linalg.norm(matrix[0]-matrix[1])),
            "distancia_euclidiana_normalizada": math.sqrt(max(0, 2-2*score))}


def derive(project, references, operation="centroide", weights=None):
    profile, matrix = load_vectors(project, references)
    if operation == "centroide":
        result = matrix.mean(axis=0, keepdims=True)
    elif operation in {"soma", "diferenca", "combinacao"}:
        if operation == "soma":
            weights = np.ones(len(matrix))
        elif operation == "diferenca":
            if len(matrix) != 2:
                raise ValueError("Diferença exige dois vetores.")
            weights = [1, -1]
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(matrix),) or not np.all(np.isfinite(weights)):
            raise ValueError("Informe um peso finito para cada vetor.")
        result = (weights @ matrix)[None, :]
    else:
        raise ValueError("Operação não disponível.")
    normalized = normalize(result)  # operação nula é rejeitada, nunca normalizada em silêncio.
    derived_id = uid("derivado")
    folder = project.root/"estudos"/"derivados"/profile["perfil_id"]/derived_id
    with project.lock():
        path = folder/"vetores.npz"
        write_npz(path, vetores=normalized, vetores_originais=result,
                  embedding_ids=np.asarray([derived_id], dtype=np.str_))
        metadata = {"derivado_id": derived_id, "perfil_id": profile["perfil_id"], "data": now(),
                    "operacao": operation, "pesos": np.asarray(weights).tolist() if weights is not None else None,
                    "origens": deepcopy(references), "normalizacao": "l2",
                    "vetor": {"arquivo_npz": project.relative(path), "linha": 0,
                              "embedding_id": derived_id, "perfil_id": profile["perfil_id"],
                              "original_disponivel": True}, "npz_sha256": file_hash(path)}
        write_json(folder/"metadados.json", metadata)
    return metadata


def pairwise(project, references, maximum_vectors=5000):
    profile, matrix = load_vectors(project, references)
    if len(matrix) > maximum_vectors:
        raise ValueError(f"Matriz quadrática limitada a {maximum_vectors} vetores; reduza o conjunto.")
    matrix = normalize(matrix)
    scores = np.clip(matrix @ matrix.T, -1, 1)
    study_id = uid("matriz")
    folder = project.root/"estudos"/"analises"/study_id
    with project.lock():
        path = folder/"matrizes.npz"
        write_npz(path, similaridades=scores, distancias_cosseno=1-scores,
                  distancias_euclidianas=np.sqrt(np.maximum(0, 2-2*scores)))
        metadata = {"analise_id": study_id, "perfil_id": profile["perfil_id"], "data": now(),
                    "operacao": "matrizes_pares", "origens": references,
                    "arquivo_npz": project.relative(path), "npz_sha256": file_hash(path)}
        write_json(folder/"metadados.json", metadata)
    return metadata


def cluster(project, references, method="kmeans", parameters=None):
    """Opcionais científicos instalados pelo script 07. Parâmetros explícitos."""
    try:
        from sklearn import cluster as algorithms
    except ImportError as exc:
        raise RuntimeError("Instale as dependências de análise com 07_instalar_analise.bat.") from exc
    profile, matrix = load_vectors(project, references)
    methods = {"kmeans": algorithms.KMeans, "minibatch": algorithms.MiniBatchKMeans,
               "aglomerativo": algorithms.AgglomerativeClustering, "dbscan": algorithms.DBSCAN,
               "hdbscan": algorithms.HDBSCAN, "espectral": algorithms.SpectralClustering}
    if method not in methods:
        raise ValueError("Método de clusterização desconhecido.")
    params = dict(parameters or {})
    if method in {"kmeans", "minibatch", "espectral"}:
        params.setdefault("random_state", 42)
        params.setdefault("n_clusters", min(8, len(matrix)))
    if method in {"kmeans", "minibatch"}:
        params.setdefault("n_init", 10)
    labels = methods[method](**params).fit_predict(normalize(matrix))
    study_id = uid("cluster")
    folder = project.root/"estudos"/"analises"/study_id
    with project.lock():
        path = folder/"rotulos.npz"
        write_npz(path, rotulos=labels, embedding_ids=np.asarray([r["embedding_id"] for r in references], dtype=np.str_))
        metadata = {"analise_id": study_id, "perfil_id": profile["perfil_id"], "data": now(),
                    "operacao": "clusterizacao", "metodo": method, "parametros": params,
                    "origens": references, "arquivo_npz": project.relative(path), "npz_sha256": file_hash(path),
                    "grupos": [{"rotulo": int(label), "quantidade": int(np.sum(labels==label))}
                               for label in sorted(set(labels.tolist()))]}
        write_json(folder/"metadados.json", metadata)
    return metadata
