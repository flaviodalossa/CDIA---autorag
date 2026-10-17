"""Application services shared by the HTTP API and offline integration tests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .core import (CATEGORIES, LABELS, category_text, digest, documents, now,
                   read_json, topics, uid, write_json)
from .experiments import save_input, search as laboratory_search
from .service import inventory, require_profile, sync_base

DEFAULT_CATEGORIES = ("conteudo_fonte",)
SEMANTIC_CATEGORIES = ("perguntas_sugeridas", "resposta_sintese", "resumo", "conteudo_fonte")


def status(project):
    base = project.base() if project.base_path.exists() else {"documentos": []}
    return {"name": "AutoRAG", "version": "0.2.0", "documents": len(documents(base)),
            "topics": sum(1 for _ in topics(base)),
            "profiles": list(project.config()["perfis"].values()),
            "active_profile": project.config().get("perfil_ativo_id"),
            "categories": [{"id": c, "label": LABELS[c]} for c in CATEGORIES],
            "default_categories": list(DEFAULT_CATEGORIES), "default_top_k": 6,
            "inventory": inventory(project) if project.base_path.exists() else []}


def collect_vectors(project, profile_id, categories):
    """Only active topics and entirely current category segments are eligible."""
    mapping = {t["topico_id"]: (t, d) for t, d in topics(project.base())}
    vectors, references, missing, excluded = [], {}, [], 0
    for category in dict.fromkeys([*categories, "conteudo_fonte"]):
        stored = project.load_index(profile_id, category)
        if stored is None:
            missing.append(category)
            continue
        manifest, matrix, _ = stored
        references[category] = {"manifesto": project.current(profile_id, category)["manifesto"],
                                "arquivo_npz": manifest["arquivo_npz"],
                                "checksum": manifest["npz_sha256"],
                                "revisao": manifest["revisao_indice"]}
        valid = {}
        for record in manifest["itens"]:
            topic_id = record["topico_id"]
            pair = mapping.get(topic_id)
            current = bool(pair) and digest(category_text(*pair, category)) == record["hash_texto_fonte"]
            valid[topic_id] = valid.get(topic_id, True) and current
        for record, vector in zip(manifest["itens"], matrix):
            if not valid[record["topico_id"]]:
                excluded += 1
                continue
            topic, document = mapping[record["topico_id"]]
            vectors.append({"topic_id": topic["topico_id"], "category": category,
                            "embedding": vector.tolist(), "embedding_id": record["embedding_id"],
                            "document_id": document["documento_id"],
                            "title": topic.get("titulo", ""), "text": record["texto_segmento"],
                            "source": {"name": document.get("nome_documento", ""),
                                       "url": document.get("metadados", {}).get("url") or (document.get("fonte", {}).get("referencia") if str(document.get("fonte", {}).get("referencia", "")).startswith("https://") else None),
                                       **topic.get("fonte", {})}})
    return vectors, references, missing, excluded


def present_results(rows, mapping):
    result = []
    for row in rows:
        topic_id = row.get("topic_id", row.get("topico_id"))
        topic, document = mapping.get(topic_id, ({}, {}))
        categories = row.get("categories", row.get("categorias", []))
        if not categories and row.get("categoria"):
            categories = [row["categoria"]]
        comparisons = row.get("comparacoes", [])
        if not categories and comparisons:
            categories = sorted({r["categoria"] for r in comparisons})
        distance = row.get("distance", row.get("distancia", row.get("distancia_cosseno_media", row.get("distancia_cosseno"))))
        result.append({**row, "topic_id": topic_id, "title": topic.get("titulo", row.get("title", row.get("titulo", ""))),
                       "text": topic.get("conteudo_fonte", row.get("text", "")),
                       "summary": topic.get("resumo", ""), "distance": distance,
                       "categories": categories,
                       "category_count": row.get("category_count", len(categories)),
                       "source": {"name": document.get("nome_documento", ""),
                                  "url": document.get("metadados", {}).get("url") or (document.get("fonte", {}).get("referencia") if str(document.get("fonte", {}).get("referencia", "")).startswith("https://") else None), **topic.get("fonte", {})}})
    return result


def run_search(project, request, provider=None):
    """Snapshot a single profile. Never mix coordinate spaces from distinct models."""
    from .semantic import search_semantic
    profile_id, query = request["profile_id"], request["query"]
    mode = request.get("mode", "vector")
    categories = request.get("categories") or list(SEMANTIC_CATEGORIES if mode == "semantic" else DEFAULT_CATEGORIES)
    if any(c not in CATEGORIES for c in categories):
        raise ValueError("Categoria desconhecida.")
    if mode not in {"vector", "semantic", "laboratory"}:
        raise ValueError("Modo de recuperação desconhecido.")
    profile = require_profile(project, profile_id)
    top_k = request.get("top_k", 6)
    with project.lock():
        vectors, references, missing, excluded = collect_vectors(project, profile_id, categories)
        mapping = {t["topico_id"]: (t, d) for t, d in topics(project.base())}
        revision = project.base().get("controle_embeddings", {}).get("revisao_base")
        if not vectors:
            raise ValueError("Gere os embeddings das categorias escolhidas antes de consultar.")
        entry, generation, query_vector = save_input(project, query, profile, provider)
    options = {k: request[k] for k in ("metric", "n_neighbors", "n_components", "min_dist", "min_cluster_size",
               "min_samples", "random_state", "dendrogram") if k in request}
    if mode == "laboratory":
        # Preserve the existing filters, ranking, saved scores and index provenance.
        experiment = laboratory_search(project, profile_id, categories, query,
            view=request.get("view", "topicos"), top_k=top_k, minimum=request.get("minimum"),
            maximum=request.get("maximum"), partial=request.get("partial", False),
            individual_limits=request.get("individual_limits"),
            individual_rule=request.get("individual_rule", "todos"), provider=provider)
        metric = request.get("metric", "cosine")
        distance_key = "distancia_euclidiana_normalizada" if metric == "euclidean" else "distancia_cosseno"
        # Both metrics produce the same ordering for normalized individual vectors;
        # topic means are recalculated explicitly for Euclidean laboratory ranking.
        if metric == "euclidean" and request.get("view", "topicos") == "topicos":
            from .experiments import rank_results
            all_rows, _ = rank_results(experiment["universo"], categories, "topicos", None,
                request.get("minimum"), request.get("maximum"), request.get("partial", False),
                request.get("individual_limits"), request.get("individual_rule", "todos"))
            all_rows.sort(key=lambda r: (r["distancia_euclidiana_media"], r["topico_id"]))
            selected = all_rows[:top_k] if top_k is not None else all_rows
            for number, row in enumerate(selected, 1):
                row.update(posicao=number, distance=row["distancia_euclidiana_media"])
        else:
            selected = experiment["resultados"]
            for row in selected:
                row["distance"] = row.get(distance_key, row.get("distancia_cosseno_media"))
        # Baseline never uses the composite legacy topic category.
        from .semantic import vector_baseline
        baseline = vector_baseline(vectors, query_vector, top_k=top_k, metric=metric)
        output = {"semantic_results": present_results(selected, mapping),
                  "baseline_results": present_results(baseline, mapping),
                  "status": "ok", "cluster": {}, "projection": [], "warnings": [],
                  "laboratory_execution_id": experiment["execucao_id"], "parameters": request}
    elif mode == "vector":
        from .semantic import vector_search
        metric = request.get("metric", "cosine")
        limits = {"minimum_similarity": request.get("minimum_similarity", request.get("minimum")),
                  "maximum_distance": request.get("maximum_distance")}
        selected = vector_search(vectors, query_vector, categories, top_k, metric, **limits)
        baseline = vector_search(vectors, query_vector, ["conteudo_fonte"], top_k, metric, **limits)
        output = {"semantic_results": present_results(selected, mapping),
                  "baseline_results": present_results(baseline, mapping),
                  "status": "ok", "cluster": {}, "projection": [], "warnings": [],
                  "parameters": {**request, "distance_space": "original_l2_normalized",
                                 "selection": "minimum_selected_vector_distance"}}
    else:
        output = search_semantic(vectors, query_vector, categories=categories, top_k=top_k, options=options)
        output["semantic_results"] = present_results(output["semantic_results"], mapping)
        output["baseline_results"] = present_results(output["baseline_results"], mapping)
        output["projection"] = output.get("cluster", {}).get("points", [])
    if missing:
        output.setdefault("warnings", []).append("Categorias ainda não geradas: " + ", ".join(LABELS[c] for c in missing))
    if excluded:
        output.setdefault("warnings", []).append(f"{excluded} vetores inativos ou desatualizados foram excluídos.")
    execution_id = uid("web")
    output.update({"execution_id": execution_id, "query": query, "profile": deepcopy(profile),
                   "mode": mode, "date": now(), "base_revision": revision,
                   "input_reference": generation["vetor_entrada"], "input_checksum": generation["npz_sha256"],
                   "indices": references})
    with project.lock():
        write_json(project.root/"estudos"/"web"/execution_id/"resultado.json", output)
    return output


def merge_documents(project, incoming):
    """Upsert by source identity; retain earlier base snapshots and immutable indices."""
    with project.lock():
        base = project.base() if project.base_path.exists() else {"schema_version": "3.0", "documentos": [], "eixos": []}
        if project.base_path.exists():
            write_json(project.root/"historico"/"bases"/(uid("base")+".json"), base)
        current = {d["documento_id"]: d for d in documents(base)}
        for document in incoming:
            current[document["documento_id"]] = document
        base["documentos"] = list(current.values())
        base.setdefault("controle_embeddings", {})["revisao_base"] = uid("base")
        base["data_atualizacao_base"] = now()
        # Eixos are reconstructed as references, never duplicated content.
        axes = {}
        for document in base["documentos"]:
            names = document.get("metadados", {}).get("eixos") or document.get("metadados", {}).get("eixo") or ["Geral"]
            if isinstance(names, str):
                names = [n.strip() for n in names.split(";") if n.strip()]
            for name in names:
                axes.setdefault(name, []).append(document["documento_id"])
        base["eixos"] = [{"eixo_id": "eixo_"+digest(name)[:12], "nome": name, "documentos_ids": ids} for name, ids in axes.items()]
        write_json(project.base_path, base)
        sync_base(project)
    return {"documents": len(incoming), "topics": sum(len(d.get("estrutura", [])) for d in incoming), "committed": True}


def history(project):
    from .experiments import list_experiments
    rows = [{"execution_id": e["execucao_id"], "query": e["texto_original_entrada"],
             "date": e["data"], "mode": "laboratory", "status": e["status"]} for e in list_experiments(project)]
    for path in (project.root/"estudos"/"web").glob("*/resultado.json"):
        e = read_json(path)
        rows.append({k: e.get(k) for k in ("execution_id", "query", "date", "mode", "status")})
    return sorted(rows, key=lambda r: r.get("date", ""), reverse=True)
