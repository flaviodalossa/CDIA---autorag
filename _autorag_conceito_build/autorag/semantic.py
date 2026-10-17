"""Independent UMAP reduction, HDBSCAN clustering and original-space baseline.

UMAP alone ranks in its unnormalized projection. Clustering ranks in the
original L2-normalized embedding space, with or without UMAP preprocessing.
The input query is fitted together with the chosen document vector categories.
"""
from __future__ import annotations

from collections.abc import Mapping
import html
import json
import math

import numpy as np


DEFAULT_CATEGORIES = ("perguntas_sugeridas", "resposta_sintese", "resumo", "conteudo_fonte")
METRICS = ("cosine", "euclidean", "manhattan")
_OPTION_KEYS = {
    "metric", "umap_metric", "cluster_metric", "n_neighbors", "n_components",
    "min_dist", "min_cluster_size", "min_samples", "random_state",
    "cluster_selection_method", "allow_single_cluster", "dendrogram",
    "use_umap", "use_hdbscan",
}


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} deve ser um inteiro.")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} fora do intervalo permitido.")
    return int(value)


def _parameters(options):
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("Opções da busca devem ser um objeto.")
    opts = dict(options or {})
    if any(not isinstance(key, str) for key in opts):
        raise ValueError("Os nomes das opções devem ser texto.")
    unknown = set(opts) - _OPTION_KEYS
    if unknown:
        raise ValueError("Opções desconhecidas: " + ", ".join(sorted(unknown)))
    params = {
        "metric": "cosine", "umap_metric": "cosine", "cluster_metric": "euclidean",
        "n_neighbors": 15, "n_components": 5, "min_dist": 0.0,
        "min_cluster_size": 5, "min_samples": 2, "random_state": 42,
        "cluster_selection_method": "eom", "allow_single_cluster": False,
        "dendrogram": False,
        "use_umap": True, "use_hdbscan": True,
    }
    params.update(opts)
    for key in ("metric", "umap_metric", "cluster_metric"):
        if params[key] not in METRICS:
            raise ValueError(f"{key}: escolha cosine, euclidean ou manhattan.")
    for name, minimum, maximum in (
        ("n_neighbors", 2, 1000), ("n_components", 2, 50),
        ("min_cluster_size", 2, None), ("min_samples", 1, None),
        ("random_state", 0, 2**32 - 1),
    ):
        params[name] = _integer(params[name], name, minimum, maximum)
    value = params["min_dist"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("min_dist deve ser finito entre 0 e 1.")
    params["min_dist"] = float(value)
    if params["cluster_selection_method"] not in ("eom", "leaf"):
        raise ValueError("cluster_selection_method deve ser eom ou leaf.")
    for key in ("allow_single_cluster", "dendrogram", "use_umap", "use_hdbscan"):
        if not isinstance(params[key], bool):
            raise ValueError(f"{key} deve ser booleano.")
    if params["dendrogram"] and not params["use_hdbscan"]:
        raise ValueError("O dendrograma exige agrupamento por HDBSCAN.")
    return params


def _embedding(value, name, dimensions=None):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu" or raw.ndim != 1 or not raw.size:
            raise ValueError()
        row = raw.astype(np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Embedding inválido: {name}.") from exc
    if dimensions is not None and len(row) != dimensions:
        raise ValueError(f"Dimensão divergente no embedding: {name}.")
    if not np.all(np.isfinite(row)):
        raise ValueError(f"Embedding contém valor não finito: {name}.")
    # Scaling before the norm avoids overflow for large finite coordinates.
    scale = np.max(np.abs(row))
    if scale == 0:
        raise ValueError(f"Embedding nulo: {name}.")
    row = row / scale
    return row / np.linalg.norm(row)


def _records(vectors, dimensions):
    if not isinstance(vectors, (list, tuple)):
        raise ValueError("Vetores devem ser uma lista de registros.")
    records, matrix, embedding_ids = [], [], set()
    for index, vector in enumerate(vectors):
        if not isinstance(vector, Mapping):
            raise ValueError(f"Registro de vetor inválido na linha {index}.")
        topic_id = vector.get("topic_id", vector.get("topico_id"))
        category = vector.get("category", vector.get("categoria"))
        if not isinstance(topic_id, str) or not topic_id.strip() or not isinstance(category, str) or not category.strip():
            raise ValueError(f"Vetor sem tópico ou categoria na linha {index}.")
        embedding_id = vector.get("embedding_id", f"{topic_id}::{category}::{index}")
        if not isinstance(embedding_id, str) or not embedding_id or embedding_id in embedding_ids:
            raise ValueError("embedding_id ausente, inválido ou duplicado.")
        embedding_ids.add(embedding_id)
        source = vector.get("source", vector.get("fonte", {})) or {}
        try:
            if not isinstance(source, Mapping):
                raise ValueError()
            source = json.loads(json.dumps(dict(source), allow_nan=False))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Metadados da fonte devem ser um objeto JSON finito.") from exc
        document_id = vector.get("document_id", vector.get("documento_id"))
        if document_id is not None and not isinstance(document_id, str):
            raise ValueError("document_id deve ser texto ou nulo.")
        matrix.append(_embedding(vector.get("embedding", vector.get("vetor")), embedding_id, dimensions))
        records.append({
            "topic_id": topic_id, "category": category, "embedding_id": embedding_id,
            "document_id": document_id, "source": source,
            "title": str(vector.get("title", vector.get("titulo", ""))),
            "text": str(vector.get("text", vector.get("texto_segmento", ""))),
        })
    return records, np.stack(matrix) if matrix else np.empty((0, dimensions), dtype=np.float64)


def _distances(matrix, query, metric):
    if metric == "cosine":
        return 1.0 - np.clip(matrix @ query, -1.0, 1.0)
    if metric == "euclidean":
        return np.linalg.norm(matrix - query, axis=1)
    return np.abs(matrix - query).sum(axis=1)


def _rank(records, matrix, query, metric, indices, top_k, by_coverage=False,
          minimum_similarity=None, maximum_distance=None, ranking_distances=None,
          projected_distances=None, distance_space="original_l2_normalized"):
    if not indices:
        return [], 0
    original_distances = _distances(matrix, query, metric)
    distances = original_distances if ranking_distances is None else ranking_distances
    if minimum_similarity is not None:
        indices = [i for i in indices if float(np.clip(matrix[i] @ query, -1, 1)) >= minimum_similarity]
    if maximum_distance is not None:
        indices = [i for i in indices if float(distances[i]) <= maximum_distance]
    groups = {}
    for index in indices:
        record = records[index]
        groups.setdefault(record["topic_id"], []).append(index)
    results = []
    for topic_id, rows in groups.items():
        rows.sort(key=lambda index: (float(distances[index]), records[index]["embedding_id"]))
        best = rows[0]
        first = records[best]
        categories = sorted({records[index]["category"] for index in rows})
        matches = [{**records[index], "distance": float(distances[index]),
                    "distance_space": distance_space,
                    "original_distance": float(original_distances[index]),
                    "original_cosine_similarity": float(np.clip(matrix[index] @ query, -1, 1)),
                    "cosine_similarity_space": "original_l2_normalized",
                    **({"projected_distance": float(projected_distances[index])} if projected_distances is not None else {}),
                    "cosine_similarity": float(np.clip(matrix[index] @ query, -1, 1))}
                   for index in rows]
        results.append({
            "topic_id": topic_id, "document_id": first["document_id"],
            "title": first["title"], "text": first["text"], "source": first["source"],
            "distance": float(distances[best]),
            "distance_space": distance_space,
            "original_distance": float(original_distances[best]),
            "original_cosine_similarity": matches[0]["original_cosine_similarity"],
            "cosine_similarity_space": "original_l2_normalized",
            **({"projected_distance": float(projected_distances[best])} if projected_distances is not None else {}),
            "cosine_similarity": matches[0]["cosine_similarity"],
            "category_count": len(categories), "categories": categories,
            "vector_count": len(rows), "matched_vectors": matches,
        })
    eligible = len(results)
    # Coverage chooses which topics enter the result set; distance orders that set.
    results.sort(key=lambda row: ((-row["category_count"] if by_coverage else 0), row["distance"], row["topic_id"]))
    results = results[:top_k]
    results.sort(key=lambda row: (row["distance"], row["topic_id"]))
    for position, result in enumerate(results, 1):
        result["position"] = position
    return results, eligible


def _fit_umap(matrix, params):
    try:
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError("Instale umap-learn para a redução UMAP.") from exc
    reducer = UMAP(
        n_neighbors=params["n_neighbors"], n_components=params["n_components"],
        metric=params["umap_metric"], min_dist=params["min_dist"],
        random_state=params["random_state"], transform_seed=params["random_state"],
        n_jobs=1, init="random", low_memory=True,
    )
    projected = np.asarray(reducer.fit_transform(matrix), dtype=np.float64)
    if projected.shape != (len(matrix), params["n_components"]) or not np.all(np.isfinite(projected)):
        raise RuntimeError("UMAP produziu uma projeção inválida.")
    return projected


def _fit_hdbscan(matrix, params):
    try:
        from hdbscan import HDBSCAN
    except ImportError as exc:
        raise RuntimeError("Instale hdbscan para o agrupamento HDBSCAN.") from exc
    # Cosine requires the generic distance backend; Euclidean/Manhattan can
    # use efficient trees. An exact MST and one worker retain repeatability.
    model = HDBSCAN(
        min_cluster_size=params["min_cluster_size"], min_samples=params["min_samples"],
        metric=params["cluster_metric"],
        algorithm="generic" if params["cluster_metric"] == "cosine" else "best", core_dist_n_jobs=1,
        approx_min_span_tree=False, cluster_selection_method=params["cluster_selection_method"],
        allow_single_cluster=params["allow_single_cluster"],
    ).fit(matrix)
    return model


def _fit_clusters(matrix, params):
    """Legacy combined fit, retained for old experiments and integrations."""
    projected = _fit_umap(matrix, params)
    return projected, _fit_hdbscan(projected, params)


def _projection_distances(matrix, query, metric):
    """Distances in raw UMAP coordinates; cosine alone divides by the norms.

    A zero coordinate has cosine similarity zero, including zero vs zero.
    UMAP coordinates are never recentered or normalized for L1/L2 distances.
    """
    if metric != "cosine":
        distances = _distances(matrix, query, metric)
    else:
        def unit(row):
            scale = np.max(np.abs(row))
            return row if scale == 0 else (row / scale) / np.linalg.norm(row / scale)
        rows = np.asarray([unit(row) for row in matrix])
        distances = 1 - np.clip(rows @ unit(query), -1, 1)
    if not np.all(np.isfinite(distances)):
        raise RuntimeError("A distância no espaço UMAP não é finita.")
    return distances


def _dendrogram(model, point_labels):
    linkage = np.asarray(model.single_linkage_tree_.to_numpy(), dtype=float)
    condensed = model.condensed_tree_.to_numpy()
    edges = [{"parent": int(row["parent"]), "child": int(row["child"]),
              "lambda": float(row["lambda_val"]) if math.isfinite(row["lambda_val"]) else None,
              "lambda_infinite": bool(math.isinf(row["lambda_val"])),
              "child_size": int(row["child_size"])} for row in condensed]
    # The SVG is drawn directly from the fitted HDBSCAN single linkage tree,
    # with no second clustering or invented hierarchy.
    svg = _linkage_svg(linkage, point_labels)
    serial_linkage = [[int(row[0]), int(row[1]), float(row[2]) if math.isfinite(row[2]) else None, int(row[3])]
                      for row in linkage]
    return {"method": "hdbscan_single_linkage", "point_labels": point_labels,
            "single_linkage": serial_linkage, "condensed_tree": edges, "svg": svg}


def _linkage_svg(linkage, labels):
    count = len(labels)
    width, height, padding = max(720, min(6000, count * 38)), 420, 45
    finite_heights = [row[2] for row in linkage if math.isfinite(row[2])]
    max_height = max(finite_heights, default=1.0) or 1.0
    children = {count + offset: (int(row[0]), int(row[1])) for offset, row in enumerate(linkage)}
    leaf_order, pending = [], [count + len(linkage) - 1] if len(linkage) else list(range(count))
    while pending:
        node = pending.pop()
        if node < count:
            leaf_order.append(node)
        else:
            left, right = children[node]
            pending.extend((right, left))
    positions = {index: (padding + order * (width - 2 * padding) / max(1, count - 1), height - padding)
                 for order, index in enumerate(leaf_order)}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Dendrograma HDBSCAN">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g fill="none" stroke="#64748b" stroke-width="1.5">']
    for offset, row in enumerate(linkage):
        left, right = positions[int(row[0])], positions[int(row[1])]
        distance = row[2] if math.isfinite(row[2]) else max_height
        y = height - padding - (height - 2 * padding) * min(1.0, distance / max_height)
        parts.append(f'<path d="M {left[0]:.2f} {left[1]:.2f} V {y:.2f} H {right[0]:.2f} V {right[1]:.2f}"/>')
        positions[count + offset] = ((left[0] + right[0]) / 2, y)
    parts.append('</g><g fill="#334155" font-family="sans-serif" font-size="9">')
    for index, label in enumerate(labels):
        x, y = positions[index]
        safe_label = html.escape(str(label), quote=True)
        color = '#e11d48' if index == count - 1 else '#334155'
        parts.append(f'<text x="{x:.2f}" y="{y+14:.2f}" text-anchor="middle" fill="{color}"><title>{safe_label}</title>{index}</text>')
    parts.append('</g><text x="12" y="20" fill="#334155" font-family="sans-serif" font-size="12">HDBSCAN — distância de alcançabilidade mútua; último índice = consulta</text></svg>')
    return "".join(parts)


def search_semantic(vectors, query_embedding, categories=None, top_k=6, options=None,
                    minimum_similarity=None, maximum_distance=None):
    """Fit independent reduction/clustering stages including the query.

    Only clustering uses category coverage in the query cluster. UMAP alone
    ranks by minimum projected distance. Similarity limits always refer to the
    original cosine; maximum distance refers to the ranking space. The source
    baseline has its own original-space distances and thresholds throughout.
    """
    top_k = _integer(top_k, "top_k", 1, 100)
    params = _parameters(options)
    _thresholds(minimum_similarity, maximum_distance)
    query = _embedding(query_embedding, "consulta")
    records, matrix = _records(vectors, len(query))
    if categories is not None and not isinstance(categories, (list, tuple)):
        raise ValueError("Categorias devem ser uma lista.")
    categories = list(DEFAULT_CATEGORIES if categories is None else categories)
    if not categories or any(not isinstance(category, str) or not category.strip() for category in categories):
        raise ValueError("Selecione pelo menos uma categoria de vetor.")
    categories = list(dict.fromkeys(categories))
    selected = [index for index, row in enumerate(records) if row["category"] in categories]
    source_rows = [index for index, row in enumerate(records) if row["category"] == "conteudo_fonte"]
    use_umap, use_hdbscan = params["use_umap"], params["use_hdbscan"]
    distance_space = "umap_projection" if use_umap and not use_hdbscan else "original_l2_normalized"
    cluster_space = ("umap_projection" if use_umap else "original_l2_normalized") if use_hdbscan else None
    baseline, baseline_eligible = _rank(records, matrix, query, params["metric"], source_rows, top_k,
                                        minimum_similarity=minimum_similarity, maximum_distance=maximum_distance)
    result = {
        "status": "empty" if not selected else "insufficient_data",
        "semantic_results": [], "baseline_results": baseline,
        "parameters": {**params, "categories": categories, "top_k": top_k,
                       "minimum_similarity": minimum_similarity, "maximum_distance": maximum_distance,
                       "distance_space": distance_space, "cluster_space": cluster_space,
                       "cosine_similarity_space": "original_l2_normalized",
                       "minimum_similarity_space": "original_l2_normalized",
                       "maximum_distance_space": distance_space,
                       "baseline_distance_space": "original_l2_normalized",
                       "topic_distance": "minimum_member_distance",
                       "selection": "distinct_categories_then_distance" if use_hdbscan else "minimum_selected_vector_distance",
                       "baseline_category": "conteudo_fonte", "query_in_fit": use_umap or use_hdbscan},
        "cluster": {"query_label": None, "query_probability": None, "counts": [], "points": [], "space": cluster_space},
        "projection": None,
        "coverage": {"semantic_eligible_topics": 0, "baseline_eligible_topics": baseline_eligible,
                     "selected_vectors": len(selected), "baseline_vectors": len(source_rows)},
        "warnings": [], "dendrogram": None,
    }
    if not source_rows:
        result["warnings"].append("Baseline indisponível: gere vetores de conteudo_fonte.")
    if not selected:
        result["warnings"].append("Não há vetores das categorias selecionadas.")
        return result
    sample_count = len(selected) + 1
    if (use_umap and sample_count < 4) or (use_hdbscan and
            (sample_count < params["min_cluster_size"] or sample_count <= params["min_samples"])):
        stages = "UMAP/HDBSCAN" if use_umap and use_hdbscan else "UMAP" if use_umap else "HDBSCAN"
        result["warnings"].append(f"Base insuficiente para {stages} com os parâmetros escolhidos; consulte o baseline ou ajuste os parâmetros.")
        return result
    effective = dict(params)
    if use_umap:
        effective["n_neighbors"] = min(params["n_neighbors"], sample_count - 1)
        effective["n_components"] = min(params["n_components"], sample_count - 2)
        result["parameters"].update({"effective_n_neighbors": effective["n_neighbors"],
                                     "effective_n_components": effective["n_components"]})
    fit_matrix = np.vstack([matrix[selected], query[None, :]])
    projected, model = None, None
    if use_umap and use_hdbscan:
        projected, model = _fit_clusters(fit_matrix, effective)
    elif use_umap:
        projected = _fit_umap(fit_matrix, effective)
    elif use_hdbscan:
        model = _fit_hdbscan(fit_matrix, effective)
    projected_distances = None
    if projected is not None:
        projected = np.asarray(projected, dtype=np.float64)
        if projected.shape != (sample_count, effective["n_components"]) or not np.all(np.isfinite(projected)):
            raise RuntimeError("UMAP produziu uma projeção inválida.")
        projected_distances = np.zeros(len(records), dtype=np.float64)
        projected_distances[selected] = _projection_distances(projected[:-1], projected[-1], params["metric"])
        projection_points = [{**records[index], "coordinates": projected[local].tolist(), "is_query": False}
                             for local, index in enumerate(selected)]
        projection_points.append({"embedding_id": "__query__", "topic_id": None, "category": "consulta",
                                  "coordinates": projected[-1].tolist(), "is_query": True})
        result["projection"] = {"space": "umap_projection", "dimensions": effective["n_components"],
                                "query_embedding_id": "__query__", "points": projection_points,
                                "normalization": "none", "cosine_zero_policy": "zero_similarity"}
    members = selected
    if model is not None:
        labels = np.asarray(model.labels_, dtype=int)
        probabilities = np.asarray(model.probabilities_, dtype=float)
        if labels.shape != (sample_count,) or probabilities.shape != (sample_count,) or not np.all(np.isfinite(probabilities)):
            raise RuntimeError("HDBSCAN produziu rótulos ou probabilidades inválidos.")
        query_label = int(labels[-1])
        coordinates = projected if projected is not None else fit_matrix
        points = [{**records[index], "cluster": int(labels[local]),
                   "probability": float(probabilities[local]),
                   **({"coordinates": coordinates[local].tolist()} if projected is not None else {}),
                   "is_query": False} for local, index in enumerate(selected)]
        points.append({"embedding_id": "__query__", "topic_id": None, "category": "consulta",
                       "cluster": query_label, "probability": float(probabilities[-1]),
                       **({"coordinates": coordinates[-1].tolist()} if projected is not None else {}), "is_query": True})
        result["cluster"] = {"space": cluster_space, "dimensions": coordinates.shape[1],
            "coordinates_included": projected is not None,
            "query_label": query_label, "query_probability": float(probabilities[-1]), "points": points,
            "counts": [{"label": int(label), "count": int(np.sum(labels == label)), "is_noise": bool(label == -1)}
                       for label in sorted(set(labels.tolist()))]}
        if result["projection"]:
            result["projection"]["points"] = points
        if params["dendrogram"]:
            result["dendrogram"] = _dendrogram(model, [row["embedding_id"] for row in points])
        if query_label == -1:
            result["status"] = "query_noise"
            result["warnings"].append("HDBSCAN classificou a consulta como ruído; nenhum agrupamento semântico foi atribuído.")
            return result
        members = [index for local, index in enumerate(selected) if labels[local] == query_label]
    ranked, eligible = _rank(records, matrix, query, params["metric"], members, top_k, by_coverage=use_hdbscan,
                             minimum_similarity=minimum_similarity, maximum_distance=maximum_distance,
                             ranking_distances=projected_distances if distance_space == "umap_projection" else None,
                             projected_distances=projected_distances, distance_space=distance_space)
    result["semantic_results"] = ranked
    result["coverage"]["semantic_eligible_topics"] = eligible
    result["status"] = "ok"
    return result


def vector_baseline(vectors, query_embedding, top_k=6, metric="cosine"):
    """Pure source-text retrieval, without importing or running UMAP/HDBSCAN."""
    if top_k is not None:
        top_k = _integer(top_k, "top_k", 1, 100)
    if metric not in METRICS:
        raise ValueError("Métrica inválida: escolha cosine, euclidean ou manhattan.")
    query = _embedding(query_embedding, "consulta")
    records, matrix = _records(vectors, len(query))
    indices = [i for i, row in enumerate(records) if row["category"] == "conteudo_fonte"]
    results, _ = _rank(records, matrix, query, metric, indices, top_k)
    return results


def _thresholds(minimum_similarity=None, maximum_distance=None):
    for value, name in ((minimum_similarity, "minimum_similarity"), (maximum_distance, "maximum_distance")):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"{name} deve ser finito.")
    if minimum_similarity is not None and not -1 <= minimum_similarity <= 1:
        raise ValueError("minimum_similarity deve estar entre -1 e 1.")
    if maximum_distance is not None and maximum_distance < 0:
        raise ValueError("maximum_distance deve ser não negativo.")


def vector_search(vectors, query_embedding, categories=None, top_k=6, metric="cosine",
                  minimum_similarity=None, maximum_distance=None):
    """Rank topics by their nearest selected vector, filtering before top-k."""
    if top_k is not None:
        top_k = _integer(top_k, "top_k", 1, 100)
    if metric not in METRICS:
        raise ValueError("Métrica inválida.")
    _thresholds(minimum_similarity, maximum_distance)
    query = _embedding(query_embedding, "consulta")
    records, matrix = _records(vectors, len(query))
    chosen = set(categories or ["conteudo_fonte"])
    indices = [i for i, row in enumerate(records) if row["category"] in chosen]
    result, _ = _rank(records, matrix, query, metric, indices, top_k,
                      minimum_similarity=minimum_similarity, maximum_distance=maximum_distance)
    return result
