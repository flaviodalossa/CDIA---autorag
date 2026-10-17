"""Named, reproducible comparisons over immutable per-model vector snapshots.

Distances are evaluated only within one embedding space. Multiple models are
combined using reciprocal rank fusion over topic IDs, never coordinates/scores.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import io
import hashlib
import json
import math
import re
import platform
import zipfile
from importlib.metadata import version, PackageNotFoundError

import numpy as np

from .core import (CATEGORIES, Project, category_text, digest, file_hash, normalize, now, read_json,
                   topics, uid, write_json, write_npz)
from .experiments import save_input
from .service import require_profile
from .semantic import (_parameters, _thresholds, search_semantic, vector_search,
                       _distances, _embedding, _records, _projection_distances)

SCHEMA_VERSION = "2.0"
RRF_K = 60
CONFIG_KEYS = {
    "id", "name", "mode", "profile_ids", "categories", "model_categories", "metric", "top_k",
    "minimum_similarity", "maximum_distance", "fusion_depth", "n_neighbors",
    "n_components", "min_dist", "min_cluster_size", "min_samples", "random_state",
    "umap_metric", "cluster_metric", "cluster_selection_method", "allow_single_cluster",
    "dendrogram", "partial", "individual_limits", "individual_rule",
}
SEMANTIC_KEYS = {"metric", "n_neighbors", "n_components", "min_dist", "min_cluster_size",
                 "min_samples", "random_state", "umap_metric", "cluster_metric",
                 "cluster_selection_method", "allow_single_cluster", "dendrogram"}
PROFILE_KEYS = {
    "perfil_id", "provedor", "modelo", "dimensoes", "dimensoes_nativas", "limite_tokens",
    "revisao_modelo", "normalizacao", "versao_preparacao", "segmentacao", "reducao",
}


def _identifier(value, label="ID"):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", value):
        raise ValueError(f"{label} inválido.")
    return value


def _integer(value, label, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} deve ser inteiro entre {minimum} e {maximum}.")
    return value


def _analysis(value=None):
    from .evaluation import validate_analysis_config
    return validate_analysis_config(value, resolve_credentials=False)


def validate_config(project, config):
    if not isinstance(config, dict):
        raise ValueError("A configuração do experimento deve ser um objeto.")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError("Parâmetros de experimento desconhecidos: " + ", ".join(sorted(unknown)))
    result = deepcopy(config)
    name = result.get("name", "Experimento").strip() if isinstance(result.get("name", "Experimento"), str) else ""
    if not name or len(name) > 160:
        raise ValueError("O nome do experimento deve conter de 1 a 160 caracteres.")
    result["name"] = name
    mode = result.setdefault("mode", "vector")
    if not isinstance(mode, str) or mode not in {"vector", "umap", "hdbscan", "semantic", "laboratory"}:
        raise ValueError("Modo de recuperação desconhecido.")
    profiles = result.get("profile_ids")
    if not isinstance(profiles, list) or not 1 <= len(profiles) <= 12 or any(not isinstance(p, str) for p in profiles):
        raise ValueError("Selecione de 1 a 12 perfis de embeddings.")
    if len(set(profiles)) != len(profiles):
        raise ValueError("Não repita perfis no mesmo experimento.")
    for profile_id in profiles:
        _identifier(profile_id, "Perfil")
        require_profile(project, profile_id)
    result["profile_ids"] = list(profiles)
    categories = result.get("categories", ["conteudo_fonte"])
    if not isinstance(categories, list) or not categories or any(not isinstance(c, str) or c not in CATEGORIES for c in categories):
        raise ValueError("Selecione categorias de vetores conhecidas.")
    result["categories"] = list(dict.fromkeys(categories))
    if "model_categories" in result:
        choices = result["model_categories"]
        if not isinstance(choices, dict) or set(choices) != set(profiles):
            raise ValueError("As categorias por modelo devem informar exatamente os perfis selecionados.")
        selected = {}
        for profile_id in profiles:
            group = choices[profile_id]
            if not isinstance(group, list) or not group or any(not isinstance(c, str) or c not in CATEGORIES for c in group):
                raise ValueError("Cada modelo deve selecionar uma lista não vazia de categorias conhecidas.")
            selected[profile_id] = list(dict.fromkeys(group))
        result["model_categories"] = selected
        result["categories"] = list(dict.fromkeys(category for profile_id in profiles for category in selected[profile_id]))
    categories = result["categories"]
    result["top_k"] = _integer(result.get("top_k", 6), "Top-k", 1, 100)
    result["fusion_depth"] = _integer(result.get("fusion_depth", max(50, result["top_k"])), "Profundidade de fusão", result["top_k"], 100)
    semantic = _parameters({key: result[key] for key in SEMANTIC_KEYS if key in result})
    semantic.pop("use_umap", None)
    semantic.pop("use_hdbscan", None)
    if mode == "umap" and semantic["dendrogram"]:
        raise ValueError("O dendrograma exige HDBSCAN; selecione HDBSCAN ou UMAP + HDBSCAN.")
    result.update(semantic)
    _thresholds(result.get("minimum_similarity"), result.get("maximum_distance"))
    for key in ("minimum_similarity", "maximum_distance"):
        result[key] = float(result[key]) if result.get(key) is not None else None
    partial = result.setdefault("partial", False)
    if not isinstance(partial, bool):
        raise ValueError("partial deve ser booleano.")
    rule = result.setdefault("individual_rule", "todos")
    if not isinstance(rule, str) or rule not in {"todos", "qualquer"}:
        raise ValueError("Regra de limites individuais inválida.")
    limits = result.setdefault("individual_limits", {})
    if not isinstance(limits, dict) or any(c not in categories for c in limits):
        raise ValueError("Limites individuais devem referir-se às categorias escolhidas.")
    for value in limits.values():
        _thresholds(value)
    if limits and mode != "laboratory":
        raise ValueError("Limites individuais de categoria são usados no modo laboratório.")
    result["id"] = _identifier(result.get("id") or "experiment_" + digest(result)[:20], "Experimento")
    return result


def _model_categories(config, profile_id):
    """The union is metadata only; each model uses its own selected categories."""
    return config.get("model_categories", {}).get(profile_id, config["categories"])


def _configs(project, values, *, empty=False):
    if not isinstance(values, list) or len(values) > 3 or (not values and not empty):
        raise ValueError("Compare de 1 a 3 experimentos.")
    configs = [validate_config(project, config) for config in values]
    if len({config["id"] for config in configs}) != len(configs):
        raise ValueError("Os experimentos da comparação devem ter IDs diferentes.")
    return configs


def _preset_path(project, preset_id):
    return project.root/"estudos"/"presets"/(_identifier(preset_id, "Configuração") + ".json")


def consultation_config(project):
    path = project.root/"estudos"/"consultation-config.json"
    return read_json(path) if path.exists() else {"experiments": [], "analysis": _analysis(), "date": None}


def active_experiments(project):
    return consultation_config(project).get("experiments", [])


def save_consultation_config(project, data):
    if not isinstance(data, dict) or set(data) - {"experiments", "analysis"}:
        raise ValueError("Configuração da consulta inválida.")
    output = {"schema_version": SCHEMA_VERSION, "date": now(),
              "experiments": _configs(project, data.get("experiments", []), empty=True),
              "analysis": _analysis(data.get("analysis"))}
    with project.lock():
        write_json(project.root/"estudos"/"consultation-config.json", output)
    return output


def save_preset(project, data):
    if not isinstance(data, dict) or set(data) - {"id", "name", "experiments", "analysis", "active"}:
        raise ValueError("Configuração salva inválida.")
    name = data.get("name", "").strip() if isinstance(data.get("name"), str) else ""
    if not name or len(name) > 160:
        raise ValueError("Informe um nome de até 160 caracteres para a configuração.")
    if not isinstance(data.get("active", False), bool):
        raise ValueError("active deve ser booleano.")
    record = {"schema_version": SCHEMA_VERSION, "id": _identifier(data.get("id") or uid("preset")),
              "name": name, "date": now(), "experiments": _configs(project, data.get("experiments")),
              "analysis": _analysis(data.get("analysis"))}
    with project.lock():
        write_json(_preset_path(project, record["id"]), record)
        if data.get("active"):
            write_json(project.root/"estudos"/"consultation-config.json",
                       {"schema_version": SCHEMA_VERSION, "date": now(), "preset_id": record["id"],
                        "experiments": record["experiments"], "analysis": record["analysis"]})
    return record


def load_preset(project, preset_id):
    path = _preset_path(project, preset_id)
    if not path.exists():
        raise ValueError("Configuração salva não encontrada.")
    return read_json(path)


def list_presets(project):
    return sorted((read_json(path) for path in (project.root/"estudos"/"presets").glob("*.json")),
                  key=lambda item: item.get("date", ""), reverse=True)


def _public_profile(profile):
    return {key: deepcopy(value) for key, value in profile.items() if key in PROFILE_KEYS}


def _capture(project, profile, categories, mapping):
    vectors, references, missing, excluded = [], {}, [], 0
    # Source vectors are captured for the independent baseline, even when not selected.
    for category in dict.fromkeys([*categories, "conteudo_fonte"]):
        stored = project.load_index(profile["perfil_id"], category)
        if stored is None:
            missing.append(category)
            continue
        manifest, matrix, _ = stored
        if manifest.get("perfil_id") != profile["perfil_id"] or manifest.get("perfil") != profile:
            raise ValueError("O perfil atual diverge do perfil gravado no índice. Gere outro índice para este perfil.")
        references[category] = {"manifesto": project.current(profile["perfil_id"], category)["manifesto"],
                                "arquivo_npz": manifest["arquivo_npz"], "checksum": manifest["npz_sha256"],
                                "manifest_checksum": file_hash(project.path(project.current(profile["perfil_id"], category)["manifesto"])),
                                "revisao": manifest["revisao_indice"]}
        validity = {}
        for record in manifest["itens"]:
            pair = mapping.get(record["topico_id"])
            current = bool(pair) and digest(category_text(*pair, category)) == record["hash_texto_fonte"]
            validity[record["topico_id"]] = validity.get(record["topico_id"], True) and current
        for record, vector in zip(manifest["itens"], matrix):
            if not validity[record["topico_id"]]:
                excluded += 1
                continue
            topic, document = mapping[record["topico_id"]]
            vectors.append({"topic_id": topic["topico_id"], "category": category,
                            "embedding": vector.tolist(), "embedding_id": record["embedding_id"],
                            "document_id": document["documento_id"], "title": topic.get("titulo", ""),
                            "text": record["texto_segmento"],
                            "source": {"name": document.get("nome_documento", ""),
                                       "url": document.get("metadados", {}).get("url"), **topic.get("fonte", {})},
                            "vector_reference": record.get("referencia"),
                            "source_hash": record["hash_texto_fonte"]})
    return {"vectors": vectors, "indices": references, "missing": missing, "excluded": excluded}


def _laboratory(vectors, query_embedding, config, depth):
    query = _embedding(query_embedding, "consulta")
    records, matrix = _records(vectors, len(query))
    distances = _distances(matrix, query, config["metric"])
    groups = {}
    for index, record in enumerate(records):
        if record["category"] in config["categories"]:
            groups.setdefault(record["topic_id"], []).append(index)
    results = []
    for topic_id, indices in groups.items():
        present = {records[i]["category"] for i in indices}
        if not config["partial"] and not set(config["categories"]).issubset(present):
            continue
        similarities = {i: float(np.clip(matrix[i] @ query, -1, 1)) for i in indices}
        tests = [similarities[i] >= config["individual_limits"][records[i]["category"]]
                 for i in indices if records[i]["category"] in config["individual_limits"]]
        if tests and not (all(tests) if config["individual_rule"] == "todos" else any(tests)):
            continue
        similarity = math.fsum(similarities.values())/len(indices)
        distance = math.fsum(float(distances[i]) for i in indices)/len(indices)
        if config["minimum_similarity"] is not None and similarity < config["minimum_similarity"]:
            continue
        if config["maximum_distance"] is not None and distance > config["maximum_distance"]:
            continue
        first = records[indices[0]]
        results.append({"topic_id": topic_id, "document_id": first["document_id"],
                        "title": first["title"], "text": first["text"], "source": first["source"],
                        "distance": distance, "cosine_similarity": similarity,
                        "categories": sorted(present), "category_count": len(present),
                        "vector_count": len(indices), "matched_vectors": [
                            {**records[i], "distance": float(distances[i]), "cosine_similarity": similarities[i]}
                            for i in indices]})
    results.sort(key=lambda row: (row["distance"], row["topic_id"]))
    for position, row in enumerate(results[:depth], 1):
        row["position"] = position
    return results[:depth]


def _enrich(rows, mapping, vectors, profile_id):
    by_id = {row["embedding_id"]: row for row in vectors}
    output = []
    for row in rows:
        topic, document = mapping[row["topic_id"]]
        row = deepcopy(row)
        row.update(text=topic.get("conteudo_fonte", ""), summary=topic.get("resumo", ""),
                   title=topic.get("titulo", ""), profile_id=profile_id, analysis=None)
        for match in row.get("matched_vectors", []):
            original = by_id[match["embedding_id"]]
            match.update(profile_id=profile_id, vector_reference=original.get("vector_reference"),
                         source_hash=original.get("source_hash"))
        output.append(row)
    return output


def _space(snapshot, config, mapping):
    profile_id = snapshot["profile"]["perfil_id"]
    config = {**config, "categories": list(_model_categories(config, profile_id))}
    vectors, query_vector = snapshot["vectors"], snapshot["query_vector"]
    depth = config["fusion_depth"] if len(config["profile_ids"]) > 1 else config["top_k"]
    mode, metric = config["mode"], config["metric"]
    args = {"top_k": depth, "minimum_similarity": config["minimum_similarity"],
            "maximum_distance": config["maximum_distance"]}
    baseline = vector_search(vectors, query_vector, ["conteudo_fonte"], metric=metric, **args)
    output = {"profile_id": profile_id, "categories": config["categories"], "profile": _public_profile(snapshot["profile"]),
              "profile_fingerprint": digest(snapshot["profile"]), "indices": snapshot["indices"],
              "input_reference": snapshot["generation"]["vetor_entrada"],
              "input_checksum": snapshot["generation"]["npz_sha256"],
              "input_source_reference": snapshot["generation"].get("source_input_reference"),
              "input_source_checksum": snapshot["generation"].get("source_input_checksum"),
              "input_segments": deepcopy(snapshot["generation"].get("segmentos", [])),
              "input_aggregation": snapshot["generation"].get("agregacao"),
              "input_storage": "comparison_query_exact_float_bytes",
              "distance_space": "original_l2_normalized", "metric": metric,
              "warnings": [], "status": "ok", "cluster": {}, "projection": [], "dendrogram": None}
    if mode in {"umap", "hdbscan", "semantic"}:
        try:
            semantic = search_semantic(vectors, query_vector, categories=config["categories"],
                                       options={**{key: config[key] for key in SEMANTIC_KEYS},
                                                "use_umap": mode in {"umap", "semantic"},
                                                "use_hdbscan": mode in {"hdbscan", "semantic"}}, **args)
            output.update({key: semantic[key] for key in ("status", "cluster", "warnings", "dendrogram", "parameters", "coverage")})
            output["projection"] = semantic.get("projection")
            output["distance_space"] = semantic["parameters"]["distance_space"]
            output["selection"] = semantic["parameters"]["selection"]
            rows = semantic["semantic_results"]
        except Exception as exc:
            output.update(status="error", error="Não foi possível concluir as etapas de recuperação selecionadas; a comparação vetorial do conteúdo original foi preservada.",
                          error_type=type(exc).__name__)
            output["warnings"].append(output["error"])
            rows = []
    elif mode == "laboratory":
        rows = _laboratory(vectors, query_vector, config, depth)
        output["selection"] = "mean_distance_over_selected_vectors"
    else:
        rows = vector_search(vectors, query_vector, config["categories"], metric=metric, **args)
        output["selection"] = "minimum_selected_vector_distance"
    if snapshot["excluded"]:
        output["warnings"].append(f'{snapshot["excluded"]} vetores inativos ou desatualizados excluídos.')
    if "conteudo_fonte" in snapshot["missing"]:
        output["warnings"].append("Baseline indisponível: gere vetores de conteudo_fonte.")
    output["results"] = _enrich(rows, mapping, vectors, profile_id)
    output["baseline_results"] = _enrich(baseline, mapping, vectors, profile_id)
    return output


def _fuse(spaces, config, key="results"):
    if len(spaces) == 1:
        return deepcopy(spaces[0][key][:config["top_k"]])
    grouped = {}
    for space in spaces:
        for rank, row in enumerate(space[key], 1):
            group = grouped.setdefault(row["topic_id"], {"row": row, "score": 0.0, "evidence": []})
            contribution = 1.0/(RRF_K+rank)
            group["score"] += contribution
            group["evidence"].append({"profile_id": space["profile_id"], "rank": rank,
                                      "distance": row["distance"], "cosine_similarity": row["cosine_similarity"],
                                      "distance_space": space["distance_space"] if key == "results" else "original_l2_normalized",
                                      "original_distance": row.get("original_distance"),
                                      "original_cosine_similarity": row.get("original_cosine_similarity"),
                                      "projected_distance": row.get("projected_distance"),
                                      "rrf_contribution": contribution, "metric": space["metric"],
                                      "categories": row["categories"], "matched_vectors": row["matched_vectors"]})
    ordered = sorted(grouped.values(), key=lambda group: (-group["score"], group["row"]["topic_id"]))
    output = []
    for position, group in enumerate(ordered[:config["top_k"]], 1):
        row = deepcopy(group["row"])
        row.update(position=position, distance=None, cosine_similarity=None, profile_id=None,
                   distance_space="per_model", original_distance=None, original_cosine_similarity=None, projected_distance=None,
                   fusion_score=group["score"], ranking="rrf", model_evidence=group["evidence"],
                   matched_vectors=[match for proof in group["evidence"] for match in proof["matched_vectors"]])
        row["categories"] = sorted({category for proof in group["evidence"] for category in proof["categories"]})
        row["category_count"] = len(row["categories"])
        row["vector_count"] = len(row["matched_vectors"])
        output.append(row)
    return output


def _software():
    packages = {}
    for package in ("numpy", "scipy", "umap-learn", "hdbscan", "sentence-transformers", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"autorag": "0.2.0", "retrieval_pipeline": "independent-stages-1",
            "python": platform.python_version(), "packages": packages}


def run_comparison(project, request, providers=None):
    if not isinstance(request, dict):
        raise ValueError("A comparação deve ser um objeto.")
    query = request.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 200000:
        raise ValueError("Informe um texto de consulta de até 200000 caracteres.")
    preset = load_preset(project, request["preset_id"]) if request.get("preset_id") else None
    values = request.get("experiments") if "experiments" in request else (preset["experiments"] if preset else [])
    if values == [] or values is None:
        profile_id = request.get("profile_id") or project.config().get("perfil_ativo_id")
        values = [{"name": "Conteúdo original — similaridade vetorial", "mode": "vector",
                   "profile_ids": [profile_id], "categories": ["conteudo_fonte"], "metric": "cosine", "top_k": 6}]
    configs = _configs(project, values)
    analysis_config = _analysis(request.get("analysis", preset.get("analysis") if preset else None))
    comparison_id = uid("comparison")
    folder = project.root/"estudos"/"comparacoes"/comparison_id
    snapshots, capture_errors, query_generations = {}, {}, {}
    with project.lock():
        base = project.base()
        mapping = {topic["topico_id"]: (topic, document) for topic, document in topics(base)}
        for config in configs:
            for profile_id in config["profile_ids"]:
                categories = _model_categories(config, profile_id)
                key = (profile_id, tuple(categories))
                if key in snapshots or key in capture_errors:
                    continue
                try:
                    profile = deepcopy(require_profile(project, profile_id))
                    snapshot = _capture(project, profile, categories, mapping)
                    missing = [category for category in categories if category in snapshot["missing"]]
                    if not snapshot["vectors"]:
                        raise ValueError("Categorias sem índice atual neste perfil: " + ", ".join(categories))
                    if profile_id not in query_generations:
                        _, generation, query_vector = save_input(project, query, profile, (providers or {}).get(profile_id))
                        if generation.get("perfil") != profile:
                            raise ValueError("O perfil do embedding de consulta salvo diverge do perfil atual.")
                        # Store the exact vector actually used, including any cache normalization.
                        # Sharing it across categories avoids a second normalization/inference per model.
                        generation = deepcopy(generation)
                        generation["source_input_reference"] = deepcopy(generation["vetor_entrada"])
                        generation["source_input_checksum"] = generation["npz_sha256"]
                        input_path = folder/"consultas"/profile_id/"vetor.npz"
                        write_npz(input_path, vetores=np.asarray(query_vector)[None, :],
                                  embedding_ids=np.asarray([generation["vetor_entrada"]["embedding_id"]], dtype=np.str_))
                        generation["vetor_entrada"]["arquivo_npz"] = project.relative(input_path)
                        generation["npz_sha256"] = file_hash(input_path)
                        query_generations[profile_id] = generation, query_vector.copy()
                    generation, query_vector = query_generations[profile_id]
                    snapshot.update(profile=profile, generation=deepcopy(generation), query_vector=query_vector.copy())
                    snapshots[key] = snapshot
                    if missing:
                        raise ValueError("Categorias sem índice neste perfil: " + ", ".join(missing))
                    if not any(vector["category"] in categories for vector in snapshot["vectors"]):
                        raise ValueError("Nenhum vetor atual das categorias selecionadas neste perfil.")
                except Exception as exc:
                    capture_errors[key] = {"profile_id": profile_id, "error": str(exc), "error_type": type(exc).__name__}
        # This exact source snapshot survives later edits and contains no provider connection configuration.
        source_snapshot = deepcopy(base)
        source_snapshot.pop("controle_embeddings", None)
        for topic, _ in topics(source_snapshot, active=False):
            topic.pop("embeddings", None)
            topic.pop("embedding", None)
        write_json(folder/"base_snapshot.json", source_snapshot)
    output = {"schema_version": SCHEMA_VERSION, "comparison_id": comparison_id, "query": query,
              "date": now(), "base_revision": base.get("controle_embeddings", {}).get("revisao_base"),
              "base_checksum": digest(source_snapshot), "base_snapshot": project.relative(folder/"base_snapshot.json"),
              "base_snapshot_checksum": file_hash(folder/"base_snapshot.json"), "experiments": [],
              "configuration": {"experiments": configs,
                                "analysis": analysis_config}, "analysis_config": analysis_config,
              "software": _software(),
              "analysis": {"status": "pending" if analysis_config["enabled"] else "disabled"}}
    _execute(output, configs, snapshots, capture_errors, mapping)
    save_comparison(project, output)
    return output


def _execute(output, configs, snapshots, capture_errors, mapping):
    for config in configs:
        experiment = {"experiment_id": config["id"], "config": config, "name": config["name"],
                      "status": "error", "results": [], "baseline_results": [], "spaces": [], "warnings": [],
                      "fusion": {"method": "rrf" if len(config["profile_ids"]) > 1 else "single_space",
                                 "k": RRF_K if len(config["profile_ids"]) > 1 else None,
                                 "candidate_depth_per_model": config["fusion_depth"],
                                 "threshold_scope": "per_model_before_fusion",
                                 "distance_scope": "within_model_only"}}
        errors = [capture_errors[(profile_id, tuple(_model_categories(config, profile_id)))]
                  for profile_id in config["profile_ids"] if (profile_id, tuple(_model_categories(config, profile_id))) in capture_errors]
        if errors:
            experiment.update(error="Um ou mais perfis do experimento estão indisponíveis.", profile_errors=errors)
            # Keep source-only comparisons available, without presenting a partial model fusion as a successful experiment.
            for profile_id in config["profile_ids"]:
                key = (profile_id, tuple(_model_categories(config, profile_id)))
                if key not in snapshots:
                    experiment["spaces"].append({"profile_id": profile_id, "categories": list(_model_categories(config, profile_id)), "status": "error",
                                                  "error": capture_errors.get(key, {}).get("error"),
                                                  "results": [], "baseline_results": []})
                    continue
                baseline_config = {**config, "mode": "vector", "categories": ["conteudo_fonte"]}
                baseline_config.pop("model_categories", None)
                diagnostic = _space(snapshots[key], baseline_config, mapping)
                diagnostic["categories"] = list(_model_categories(config, profile_id))
                diagnostic.update(status="error" if key in capture_errors else "not_combined", results=[])
                diagnostic["error"] = capture_errors.get(key, {}).get("error")
                experiment["spaces"].append(diagnostic)
            if all("profile_fingerprint" in space for space in experiment["spaces"]):
                experiment["baseline_results"] = _fuse(experiment["spaces"], config, "baseline_results")
        else:
            try:
                for profile_id in config["profile_ids"]:
                    experiment["spaces"].append(_space(snapshots[(profile_id, tuple(_model_categories(config, profile_id)))], config, mapping))
                experiment["baseline_results"] = _fuse(experiment["spaces"], config, "baseline_results")
                statuses = {space["status"] for space in experiment["spaces"]}
                if "error" in statuses:
                    experiment.update(status="error", error="Um ou mais espaços de recuperação falharam; o baseline foi preservado.",
                                      profile_errors=[{"profile_id": space["profile_id"], "error": space["error"],
                                                       "error_type": space.get("error_type")}
                                                      for space in experiment["spaces"] if space["status"] == "error"])
                else:
                    experiment["results"] = _fuse(experiment["spaces"], config)
                    experiment["status"] = "ok" if experiment["results"] or statuses == {"ok"} else (next(iter(statuses)) if len(statuses) == 1 else "no_semantic_results")
                experiment["warnings"] = [warning for space in experiment["spaces"] for warning in space["warnings"]]
                if len(config["profile_ids"]) > 1:
                    experiment["warnings"].append("Modelos combinados por RRF (k=60); limites de similaridade e distância são aplicados em cada modelo antes da fusão.")
            except Exception as exc:
                experiment.update(status="error", error=str(exc), error_type=type(exc).__name__, results=[], baseline_results=[])
        output["experiments"].append(experiment)
    output["status"] = "error" if all(experiment["status"] == "error" for experiment in output["experiments"]) else "completed"
    return output


def _folder(project, comparison_id):
    return project.root/"estudos"/"comparacoes"/_identifier(comparison_id, "Comparação")


def _lines(comparison):
    rows = []
    for experiment in comparison.get("experiments", []):
        for row in experiment.get("results", []):
            rows.append({"schema_version": SCHEMA_VERSION, "comparison_id": comparison["comparison_id"],
                         "experiment_id": experiment["experiment_id"], "experiment_name": experiment["config"]["name"],
                         "query": comparison["query"], "date": comparison["date"],
                         "base_revision": comparison.get("base_revision"), "base_checksum": comparison.get("base_checksum"),
                         "config": experiment["config"], "result": row, "analysis": row.get("analysis"),
                         "provenance": [{key: space.get(key) for key in (
                             "profile_id", "profile", "profile_fingerprint", "indices", "input_reference", "input_checksum",
                             "input_source_reference", "input_source_checksum", "distance_space", "parameters")}
                             for space in experiment.get("spaces", [])]})
    return rows


def save_comparison(project, comparison):
    folder = _folder(project, comparison["comparison_id"])
    with project.lock():
        write_json(folder/"resultado.json", comparison)
        write_json(folder/"configuracao.json", {"schema_version": SCHEMA_VERSION,
                   "comparison_id": comparison["comparison_id"], "query": comparison["query"],
                   "base_revision": comparison.get("base_revision"), "base_checksum": comparison.get("base_checksum"),
                   **comparison["configuration"]})
        # Atomic JSONL replacement reuses the same write primitive, with one JSON object per line.
        rows = _lines(comparison)
        text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
        temporary = folder/(".linhas_" + uid("tmp"))
        try:
            temporary.write_text(text, encoding="utf-8", newline="\n")
            temporary.replace(folder/"linhas.jsonl")
        finally:
            temporary.unlink(missing_ok=True)
    export_vector_bundle(project, comparison["comparison_id"])
    return comparison


def load_comparison(project, comparison_id, verify=False):
    path = _folder(project, comparison_id)/"resultado.json"
    if not path.exists():
        raise ValueError("Comparação não encontrada.")
    comparison = read_json(path)
    if verify:
        try:
            if file_hash(project.path(comparison["base_snapshot"])) != comparison["base_snapshot_checksum"]:
                raise ValueError("O snapshot da base foi alterado.")
            for experiment in comparison["experiments"]:
                for space in experiment.get("spaces", []):
                    if not space.get("input_reference"):
                        continue
                    if file_hash(project.path(space["input_reference"]["arquivo_npz"])) != space["input_checksum"]:
                        raise ValueError("O embedding da entrada foi alterado.")
                    origin = space.get("input_source_reference")
                    if origin and space.get("input_source_checksum") and file_hash(project.path(origin["arquivo_npz"])) != space["input_source_checksum"]:
                        raise ValueError("O NPZ original dos segmentos da consulta foi alterado.")
                    for index in space["indices"].values():
                        if file_hash(project.path(index["arquivo_npz"])) != index["checksum"]:
                            raise ValueError("Um índice gravado na comparação foi alterado.")
                        if file_hash(project.path(index["manifesto"])) != index["manifest_checksum"]:
                            raise ValueError("Um manifesto gravado na comparação foi alterado.")
        except OSError as exc:
            raise ValueError("Artefato salvo indisponível; a reprodução exige os arquivos originais da base, dos índices e da consulta.") from exc
    return comparison


def list_comparisons(project):
    output = []
    for path in (project.root/"estudos"/"comparacoes").glob("*/resultado.json"):
        comparison = read_json(path)
        output.append({key: comparison.get(key) for key in ("comparison_id", "query", "date", "base_revision", "status", "analysis")})
        output[-1]["experiments"] = [{"experiment_id": item["experiment_id"], "name": item["config"]["name"],
                                      "status": item["status"], "result_count": len(item.get("results", []))}
                                     for item in comparison.get("experiments", [])]
    return sorted(output, key=lambda item: item.get("date", ""), reverse=True)


def export_comparison(project, comparison_id, format="json"):
    comparison = load_comparison(project, comparison_id, verify=True)
    if format == "json":
        return json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if format == "config":
        return json.dumps({"schema_version": SCHEMA_VERSION, "query": comparison["query"],
                           "experiments": [item["config"] for item in comparison["experiments"]],
                           "analysis": comparison["configuration"].get("analysis"),
                           "base_revision": comparison.get("base_revision"), "base_checksum": comparison.get("base_checksum"),
                           "reproduction": "A reprodução salva usa o snapshot da base, os índices e o vetor de consulta originais. Versões de bibliotecas são registradas; novos embeddings também exigem os pesos e a preparação originais."},
                          ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    rows = _lines(comparison)
    if format == "jsonl":
        return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    if format == "csv":
        stream = io.StringIO(newline="")
        fields = ["comparison_id", "experiment_id", "experiment_name", "query", "date", "base_revision",
                  "position", "topic_id", "title", "distance", "distance_space", "original_distance", "projected_distance", "cosine_similarity", "cosine_similarity_space", "fusion_score",
                  "classification", "relevant", "comment", "utility_rank", "analysis", "config", "provenance", "result"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            result, analysis = row["result"], row["analysis"] or {}
            flat = {key: row.get(key) for key in fields if key in row}
            flat.update({key: result.get(key) for key in ("position", "topic_id", "title", "distance", "distance_space", "original_distance", "projected_distance", "cosine_similarity", "cosine_similarity_space", "fusion_score")})
            flat.update({key: analysis.get(key) for key in ("classification", "relevant", "comment", "utility_rank")})
            for key in ("analysis", "config", "provenance", "result"):
                flat[key] = json.dumps(row.get(key), ensure_ascii=False, allow_nan=False)
            # Neutralize spreadsheet formula cells without changing structured JSON exports.
            for key, value in flat.items():
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                    flat[key] = "'" + value
            writer.writerow(flat)
        return stream.getvalue()
    raise ValueError("Formato de exportação desconhecido: escolha json, jsonl, csv ou config.")


class _PinnedIndices:
    """Read immutable pointers from a saved comparison, never from atual.json."""
    def __init__(self, project, indices):
        self.project, self.indices = project, indices

    def current(self, profile_id, category):
        reference = self.indices.get(category)
        return {"manifesto": reference["manifesto"]} if reference else None

    def path(self, value):
        return self.project.path(value)

    def load_index(self, profile_id, category):
        return Project.load_index(self, profile_id, category)


def reproduce_comparison(project, comparison_id):
    """Re-run retrieval from saved source/NPZ snapshots, without any inference.

    The LLM assessment is not repeated automatically. Package versions of the
    original execution remain recorded so changes to UMAP/HDBSCAN are visible.
    """
    original = load_comparison(project, comparison_id, verify=True)
    source = read_json(project.path(original["base_snapshot"]))
    mapping = {topic["topico_id"]: (topic, document) for topic, document in topics(source)}
    configs = deepcopy(original["configuration"]["experiments"])
    snapshots, errors = {}, {}
    new_id = uid("comparison")
    folder = _folder(project, new_id)
    with project.lock():
        for experiment in original["experiments"]:
            config = experiment["config"]
            spaces = {space["profile_id"]: space for space in experiment.get("spaces", [])}
            recorded_errors = {error["profile_id"]: error for error in experiment.get("profile_errors", [])}
            for profile_id in config["profile_ids"]:
                categories = _model_categories(config, profile_id)
                key = (profile_id, tuple(categories))
                if profile_id in recorded_errors:
                    errors[key] = deepcopy(recorded_errors[profile_id])
                if key in snapshots:
                    continue
                space = spaces.get(profile_id, {})
                if not space.get("indices") or not space.get("input_reference"):
                    errors.setdefault(key, {"profile_id": profile_id,
                                            "error": "A execução original não contém um snapshot válido deste perfil.",
                                            "error_type": "UnavailableSnapshot"})
                    continue
                index = next(iter(space["indices"].values()))
                profile = read_json(project.path(index["manifesto"]))["perfil"]
                if digest(profile) != space["profile_fingerprint"]:
                    raise ValueError("O perfil do manifesto diverge da impressão digital registrada.")
                view = _PinnedIndices(project, space["indices"])
                snapshot = _capture(view, profile, categories, mapping)
                reference = space["input_reference"]
                with np.load(project.path(reference["arquivo_npz"]), allow_pickle=False) as package:
                    query = np.asarray(package["vetores"]).copy()
                    normalize(query, profile["dimensoes"])  # Validate without altering saved coordinates.
                if query.shape != (1, profile["dimensoes"]):
                    raise ValueError("Formato inválido no embedding de consulta gravado.")
                snapshot.update(profile=profile, query_vector=query[0], generation={
                    "vetor_entrada": deepcopy(reference), "npz_sha256": space["input_checksum"],
                    "source_input_reference": deepcopy(space.get("input_source_reference")),
                    "source_input_checksum": space.get("input_source_checksum"),
                    "segmentos": deepcopy(space.get("input_segments", [])),
                    "agregacao": space.get("input_aggregation")})
                snapshots[key] = snapshot
        write_json(folder/"base_snapshot.json", source)
    analysis_config = {**original.get("analysis_config", original["configuration"].get("analysis", _analysis())), "enabled": False}
    result = {"schema_version": SCHEMA_VERSION, "comparison_id": new_id,
              "reproduced_from": comparison_id, "query": original["query"], "date": now(),
              "base_revision": original["base_revision"], "base_checksum": original["base_checksum"],
              "base_snapshot": project.relative(folder/"base_snapshot.json"),
              "base_snapshot_checksum": file_hash(folder/"base_snapshot.json"),
              "experiments": [], "configuration": {"experiments": configs, "analysis": analysis_config},
              "analysis_config": analysis_config, "analysis": {"status": "disabled",
                  "reason": "Reprodução da recuperação com vetores gravados; a avaliação LLM original permanece no registro de origem."},
              "software": _software(), "original_software": original.get("software"),
              "previous_analysis_reference": {"comparison_id": comparison_id,
                  "status": original.get("analysis", {}).get("status"), "reexecuted": False}}
    result["reproduction_exact_environment"] = result["software"] == result["original_software"]
    result["warnings"] = [] if result["reproduction_exact_environment"] else [
        "As versões de software diferem da execução original; resultados UMAP/HDBSCAN podem variar."]
    _execute(result, configs, snapshots, errors, mapping)
    save_comparison(project, result)
    return result


def _bundle_csv(rows):
    fields = ["scope", "origin", "comparison_id", "experiment_id", "experiment_name", "profile_id", "model",
              "topic_id", "title", "category", "embedding_id", "matrix_row", "vectors_file", "query_file",
              "metric", "distance_space", "vector_distance", "original_distance", "original_cosine_similarity",
              "projection_file", "projection_row", "projected_distance", "projected_cosine_distance",
              "projected_euclidean_distance", "projected_manhattan_distance",
              "vector_cosine_similarity", "vector_cosine_distance",
              "vector_euclidean_distance", "vector_manhattan_distance", "result_position", "topic_distance",
              "topic_cosine_similarity", "model_rank", "fusion_score", "fusion_contribution", "fusion_rank",
              "classification", "relevant", "comment", "utility_rank", "analysis", "source_reference"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for association in rows:
        row = {key: association.get(key) for key in fields}
        assessment = association.get("analysis") or {}
        row.update({key: assessment.get(key) for key in ("classification", "relevant", "comment", "utility_rank")})
        for key in ("analysis", "source_reference"):
            row[key] = json.dumps(association.get(key), ensure_ascii=False, allow_nan=False)
        for key, value in row.items():
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                row[key] = "'" + value
        writer.writerow(row)
    return stream.getvalue().encode("utf-8-sig")


def _npz_bytes(**arrays):
    stream = io.BytesIO()
    for array in arrays.values():
        if np.asarray(array).dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            raise ValueError("O pacote vetorial aceita somente coordenadas numéricas finitas.")
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def export_vector_bundle(project, comparison_id, *, persist=True):
    """Export exact saved queries and only the matching returned corpus vectors.

    Arrays remain separated by profile. The primary arrays retain original
    numeric precision; normalized effective arrays make distance formulas clear.
    Every archive is tied to the result digest and verified immutable artifacts.
    """
    folder = _folder(project, comparison_id)
    with project.lock(wait_seconds=2.0):
        comparison = load_comparison(project, comparison_id, verify=True)
        result_digest = digest(comparison)
        archive_path, cache_path = folder/"vetores.zip", folder/"vetores_export.json"
        if archive_path.exists() and cache_path.exists():
            cache = read_json(cache_path)
            if cache.get("result_digest") == result_digest and file_hash(archive_path) == cache.get("archive_sha256"):
                return archive_path.read_bytes()
        if not persist:
            raise ValueError("O pacote vetorial não está disponível ou está desatualizado. Execute ou reproduza a comparação para gerá-lo.")
        files, profiles, associations, source_cache, projections = {}, {}, [], {}, {}

        def json_bytes(value):
            return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")

        def ensure_profile(space):
            profile_id = _identifier(space["profile_id"], "Perfil")
            if profile_id in profiles:
                if profiles[profile_id]["fingerprint"] != space["profile_fingerprint"]:
                    raise ValueError("O mesmo perfil possui impressões digitais divergentes no pacote.")
                if profiles[profile_id]["query_checksum"] != space["input_checksum"]:
                    raise ValueError("O mesmo perfil possui consultas efetivas divergentes no pacote.")
                return profiles[profile_id]
            reference = space["input_reference"]
            with np.load(project.path(reference["arquivo_npz"]), allow_pickle=False) as package:
                query = np.asarray(package["vetores"]).copy()
            dimensions = space["profile"]["dimensoes"]
            normalize(query, dimensions)
            if query.shape != (1, dimensions):
                raise ValueError("Formato inválido no vetor de consulta salvo.")
            query_file = f"consultas/{profile_id}/efetivo.npz"
            effective_query = _embedding(query[0], "consulta")
            files[query_file] = _npz_bytes(vetores=query, vetores_efetivos=effective_query[None, :])
            state = {"profile_id": profile_id, "profile": deepcopy(space["profile"]),
                     "fingerprint": space["profile_fingerprint"], "dimensions": dimensions,
                     "query_file": query_file, "vectors_file": f"correspondentes/{profile_id}/vetores.npz",
                     "query_reference": deepcopy(reference), "query_checksum": space["input_checksum"],
                     "query_segments": deepcopy(space.get("input_segments", [])),
                     "query_aggregation": space.get("input_aggregation"),
                     "query_dtype": str(query.dtype), "query_effective": effective_query,
                     "vectors": [], "effective_vectors": [], "original_vectors": [], "vector_keys": {}, "entries": []}
            origin = space.get("input_source_reference")
            if origin and space.get("input_source_checksum"):
                try:
                    if file_hash(project.path(origin["arquivo_npz"])) != space["input_source_checksum"]:
                        raise ValueError("O NPZ original dos segmentos da consulta foi alterado.")
                    with np.load(project.path(origin["arquivo_npz"]), allow_pickle=False) as package:
                        numeric = {key: np.asarray(package[key]).copy() for key in (
                            "vetores", "vetores_originais", "vetores_segmentos", "vetores_segmentos_originais") if key in package}
                except OSError as exc:
                    raise ValueError("O NPZ original dos segmentos da consulta está indisponível.") from exc
                for array in numeric.values():
                    if array.ndim != 2 or array.shape[1] != dimensions:
                        raise ValueError("Dimensões inválidas nos segmentos da consulta original.")
                origin_file = f"consultas/{profile_id}/origem.npz"
                files[origin_file] = _npz_bytes(**numeric)
                state.update(query_source_file=origin_file, query_source_reference=deepcopy(origin),
                             query_source_checksum=space["input_source_checksum"])
            profiles[profile_id] = state
            return state

        def add_vector(space, match):
            state = ensure_profile(space)
            profile_id, category, embedding_id = space["profile_id"], match["category"], match["embedding_id"]
            index = space["indices"].get(category)
            if index is None:
                raise ValueError("O vetor retornado não pertence aos índices gravados no experimento.")
            cache_key = index["manifesto"]
            if cache_key not in source_cache:
                manifest = read_json(project.path(index["manifesto"]))
                with np.load(project.path(index["arquivo_npz"]), allow_pickle=False) as package:
                    matrix = np.asarray(package["vetores"]).copy()
                    ids = package["embedding_ids"].tolist()
                    originals = np.asarray(package["vetores_originais"]).copy() if "vetores_originais" in package else None
                if ids != [item["embedding_id"] for item in manifest["itens"]] or len(matrix) != len(ids):
                    raise ValueError("IDs ou linhas do índice não correspondem ao manifesto salvo.")
                normalize(matrix, state["dimensions"])
                source_cache[cache_key] = manifest, matrix, originals, {value: row for row, value in enumerate(ids)}
            manifest, matrix, originals, positions = source_cache[cache_key]
            source_row = positions.get(embedding_id)
            if source_row is None:
                raise ValueError("Embedding retornado ausente no índice salvo.")
            record = manifest["itens"][source_row]
            if record["topico_id"] != match["topic_id"] or manifest["categoria"] != category:
                raise ValueError("A associação entre tópico, categoria e vetor foi alterada.")
            expected = {"arquivo_npz": index["arquivo_npz"], "revisao_indice": index["revisao"],
                        "linha": source_row, "embedding_id": embedding_id}
            if match.get("vector_reference") != expected:
                raise ValueError("A referência do vetor retornado diverge do índice salvo.")
            key = (embedding_id, index["checksum"], source_row)
            effective = _embedding(normalize(matrix[source_row:source_row+1], state["dimensions"])[0].tolist(), embedding_id)
            if key not in state["vector_keys"]:
                target_row = len(state["vectors"])
                state["vector_keys"][key] = target_row
                state["vectors"].append(matrix[source_row].copy())
                state["effective_vectors"].append(effective)
                state["original_vectors"].append(originals[source_row].copy() if originals is not None else None)
                state["entries"].append({"matrix_row": target_row, "embedding_id": embedding_id,
                    "topic_id": record["topico_id"], "category": category,
                    "source_reference": expected, "source_checksum": index["checksum"],
                    "manifest_reference": index["manifesto"], "manifest_checksum": index["manifest_checksum"],
                    "source_hash": record["hash_texto_fonte"], "source_dtype": str(matrix.dtype),
                    "original_available": originals is not None})
            return state, state["vector_keys"][key], effective, expected

        def ensure_projection(experiment, space):
            projection = space.get("projection")
            if not isinstance(projection, dict) or projection.get("space") != "umap_projection":
                return None
            key = (experiment["experiment_id"], space["profile_id"])
            if key in projections:
                return projections[key]
            points = projection["points"]
            queries = [point for point in points if point.get("is_query")]
            documents = [point for point in points if not point.get("is_query")]
            dimensions = projection["dimensions"]
            if len(queries) != 1 or type(dimensions) is not int or not 2 <= dimensions <= 50:
                raise ValueError("Projeção salva com consulta ou dimensão inválida.")
            query = np.asarray(queries[0]["coordinates"], dtype=np.float64)
            matrix = np.asarray([point["coordinates"] for point in documents], dtype=np.float64)
            if query.shape != (dimensions,) or matrix.shape != (len(documents), dimensions):
                raise ValueError("Coordenadas divergentes na projeção salva.")
            positions = {point["embedding_id"]: index for index, point in enumerate(documents)}
            if len(positions) != len(documents):
                raise ValueError("IDs duplicados na projeção salva.")
            experiment_id = _identifier(experiment["experiment_id"], "Experimento")
            profile_id = _identifier(space["profile_id"], "Perfil")
            path = f"projecoes/{experiment_id}/{profile_id}.npz"
            files[path] = _npz_bytes(consulta=query[None, :], vetores=matrix)
            state = {"experiment_id": experiment_id, "profile_id": profile_id,
                     "space": "umap_projection", "dimensions": dimensions, "normalization": "none",
                     "file": path, "query_array": "consulta", "vectors_array": "vetores",
                     "query_embedding_id": queries[0]["embedding_id"],
                     "parameters": deepcopy(space.get("parameters", {})),
                     "entries": [{"matrix_row": index, "embedding_id": point["embedding_id"],
                                  "topic_id": point["topic_id"], "category": point["category"]}
                                 for index, point in enumerate(documents)],
                     "positions": positions, "query": query, "matrix": matrix}
            projections[key] = state
            return state

        def add_rows(experiment, rows, scope, origin, forced_space=None):
            spaces = {space["profile_id"]: space for space in experiment.get("spaces", []) if space.get("input_reference")}
            for row in rows:
                for match in row.get("matched_vectors", []):
                    profile_id = match.get("profile_id") or row.get("profile_id") or (forced_space or {}).get("profile_id")
                    space = spaces.get(profile_id)
                    if space is None:
                        raise ValueError("O vetor retornado não possui um espaço de modelo salvo.")
                    state, target_row, effective, reference = add_vector(space, match)
                    proof = next((item for item in row.get("model_evidence", []) if item["profile_id"] == profile_id), {})
                    similarity = float(np.clip(effective @ state["query_effective"], -1, 1))
                    delta = effective-state["query_effective"]
                    projection = ensure_projection(experiment, space)
                    projected = {}
                    if projection is not None and match["embedding_id"] in projection["positions"]:
                        projected_row = projection["positions"][match["embedding_id"]]
                        projected_vector = projection["matrix"][projected_row:projected_row+1]
                        projected.update(projection_file=projection["file"], projection_row=projected_row)
                        for projected_metric in ("cosine", "euclidean", "manhattan"):
                            projected["projected_"+projected_metric+"_distance"] = float(_projection_distances(
                                projected_vector, projection["query"], projected_metric)[0])
                        projected["projected_distance"] = projected["projected_"+space["metric"]+"_distance"]
                    associations.append({"scope": scope, "origin": origin, "comparison_id": comparison_id,
                        "experiment_id": experiment["experiment_id"], "experiment_name": experiment["config"]["name"],
                        "profile_id": profile_id, "model": state["profile"]["modelo"],
                        "topic_id": row["topic_id"], "title": row.get("title", ""), "category": match["category"],
                        "embedding_id": match["embedding_id"], "matrix_row": target_row,
                        "query_file": state["query_file"], "vectors_file": state["vectors_file"],
                        "metric": space["metric"], "distance_space": match.get("distance_space", "original_l2_normalized"),
                        "original_distance": {"cosine": 1-similarity, "euclidean": float(np.linalg.norm(delta)),
                                              "manhattan": float(np.sum(np.abs(delta)))}[space["metric"]],
                        "original_cosine_similarity": similarity, **projected,
                        "vector_distance": match["distance"],
                        "vector_cosine_similarity": match["cosine_similarity"],
                        "vector_cosine_distance": 1-similarity, "vector_euclidean_distance": float(np.linalg.norm(delta)),
                        "vector_manhattan_distance": float(np.sum(np.abs(delta))),
                        "result_position": row.get("position"), "topic_distance": proof.get("distance", row.get("distance")),
                        "topic_cosine_similarity": proof.get("cosine_similarity", row.get("cosine_similarity")),
                        "model_rank": proof.get("rank", row.get("position")),
                        "fusion_score": row.get("fusion_score"), "fusion_contribution": proof.get("rrf_contribution"),
                        "fusion_rank": row.get("position") if row.get("ranking") == "rrf" else None,
                        "source_reference": reference, "analysis": deepcopy(row.get("analysis"))})

        for experiment in comparison["experiments"]:
            for space in experiment.get("spaces", []):
                if space.get("input_reference"):
                    ensure_profile(space)
                    ensure_projection(experiment, space)
            add_rows(experiment, experiment.get("results", []), "retrieval", "experiment")
            add_rows(experiment, experiment.get("baseline_results", []), "baseline", "experiment")
            if experiment["status"] == "error":
                for space in experiment.get("spaces", []):
                    add_rows(experiment, space.get("results", []), "retrieval", "model_partial", space)
                    add_rows(experiment, space.get("baseline_results", []), "baseline", "model_partial", space)
        public_profiles = []
        for state in profiles.values():
            dimensions = state["dimensions"]
            vectors = np.stack(state["vectors"]) if state["vectors"] else np.empty((0, dimensions), dtype=np.float32)
            effective = np.stack(state["effective_vectors"]) if state["effective_vectors"] else np.empty((0, dimensions), dtype=np.float64)
            arrays = {"vetores": vectors, "vetores_efetivos": effective}
            if state["original_vectors"] and all(value is not None for value in state["original_vectors"]):
                arrays["vetores_originais"] = np.stack(state["original_vectors"])
            files[state["vectors_file"]] = _npz_bytes(**arrays)
            public_profiles.append({key: deepcopy(value) for key, value in state.items() if key not in {
                "query_effective", "vectors", "effective_vectors", "original_vectors", "vector_keys"}})
        files["resultado.json"] = json_bytes(comparison)
        files["configuracao.json"] = json_bytes({"query": comparison["query"], **comparison["configuration"]})
        files["metricas.csv"] = _bundle_csv(associations)
        manifest = {"schema_version": "1.1", "comparison_id": comparison_id, "result_digest": result_digest,
                    "query": comparison["query"], "base_revision": comparison["base_revision"],
                    "base_checksum": comparison["base_checksum"], "profiles": public_profiles,
                    "associations": associations, "software": comparison.get("software"),
                    "projections": [{key: value for key, value in projection.items() if key not in {"positions", "query", "matrix"}}
                                    for projection in projections.values()],
                    "numeric_policy": "Exact saved coordinates and dtypes; original effective arrays apply L2 normalization. UMAP arrays retain raw, unnormalized fitted coordinates per experiment and model.",
                    "formulas": {"effective": "L2-normalize within each model; query and corpus have the same model dimensions",
                                 "cosine": "1 - clip(dot(query_effective, vector_effective), -1, 1)",
                                 "euclidean": "norm(query_effective - vector_effective, 2)",
                                 "manhattan": "sum(abs(query_effective - vector_effective))",
                                 "projected_cosine": "1 - clip(dot(query / norm(query), vector / norm(vector)), -1, 1); any zero coordinate has zero similarity",
                                 "projected_euclidean": "norm(query_projection - vector_projection, 2); no normalization",
                                 "projected_manhattan": "sum(abs(query_projection - vector_projection)); no normalization",
                                 "metric_scope": "vector_distance uses distance_space; vector_cosine_distance/vector_euclidean_distance/vector_manhattan_distance always use original_l2_normalized",
                                 "rrf": "sum(1 / (60 + model_rank)); distances are never compared across model spaces"},
                    "files": [{"path": name, "sha256": hashlib.sha256(content).hexdigest()} for name, content in files.items()]}
        files["manifesto.json"] = json_bytes(manifest)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(files):
                archive.writestr(name, files[name])
        data = stream.getvalue()
        temporary = folder/(".vetores_"+uid("tmp"))
        try:
            temporary.write_bytes(data)
            temporary.replace(archive_path)
        finally:
            temporary.unlink(missing_ok=True)
        write_json(cache_path, {"schema_version": "1.0", "result_digest": result_digest,
                               "archive_sha256": hashlib.sha256(data).hexdigest(), "filename": "vetores.zip"})
        return data
