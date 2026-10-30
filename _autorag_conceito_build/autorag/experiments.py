from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import zipfile

import numpy as np

from .core import (CATEGORIES, category_text, digest, file_hash, normalize, now,
                   read_json, topics, uid, write_json, write_npz)
from .providers import get_provider
from .service import require_profile


def validate_limits(top_k=5, minimum=None, maximum=None):
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
        raise ValueError("Top-k deve ser um inteiro positivo ou desativado.")
    for value in (minimum, maximum):
        if value is not None and (not math.isfinite(value) or not -1 <= value <= 1):
            raise ValueError("Escores mínimo/máximo devem ser finitos no intervalo [-1, 1].")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("Escore mínimo maior que máximo.")


def save_input(project, value, profile, provider=None, regenerate=False, *, role="consulta"):
    if role not in {"consulta", "documento"}:
        raise ValueError("Papel do embedding de entrada inválido.")
    if not value.strip():
        raise ValueError("Texto da consulta está vazio.")
    entry_id = "ent_" + digest(value)[:24]
    folder = project.root/"estudos"/"entradas"/entry_id
    metadata_path = folder/"entrada.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {
        "schema_version": "1.0", "entrada_id": entry_id, "texto_original": value,
        "hash_texto_original": digest(value), "data": now(), "geracoes": {}, "execucoes": []}
    key = profile["perfil_id"] + "::" + role
    previous = metadata["geracoes"].get(key, [])
    if previous and not regenerate:
        record = previous[-1]
        path = project.path(record["vetor_entrada"]["arquivo_npz"])
        if file_hash(path) != record["npz_sha256"]:
            raise ValueError("Checksum do embedding de entrada diverge.")
        with np.load(path, allow_pickle=False) as package:
            matrix = normalize(package["vetores"], profile["dimensoes"])
        if matrix.shape != (1, profile["dimensoes"]):
            raise ValueError("Formato inválido no embedding de entrada.")
        return metadata, record, matrix[0]
    # Registra o texto antes da inferência, inclusive se o provedor falhar.
    write_json(metadata_path, metadata)
    try:
        backend = provider or get_provider(profile)
        segments = [part for part, _, _ in backend.segments(value, role)]
        originals = np.asarray(backend.embed(segments, role))
        normalized = normalize(originals, profile["dimensoes"])
        if len(originals) != len(segments):
            raise ValueError("Quantidade de vetores de entrada incorreta.")
    except Exception as exc:
        metadata.setdefault("erros_geracao", []).append({"perfil_id": profile["perfil_id"],
                                                       "data": now(), "erro": str(exc)})
        write_json(metadata_path, metadata)
        raise
    # Entrada longa é registrada com todos os segmentos e agregação explícita.
    original_aggregate = originals.mean(axis=0, keepdims=True)
    aggregate = normalize(original_aggregate, profile["dimensoes"])
    generation_id = uid("ger")
    path = folder/profile["perfil_id"]/role/generation_id/"vetores.npz"
    embedding_id = entry_id+"::"+role+"::"+generation_id
    write_npz(path, vetores=aggregate, vetores_originais=original_aggregate,
              vetores_segmentos=normalized, vetores_segmentos_originais=originals,
              embedding_ids=np.asarray([embedding_id], dtype=np.str_))
    record = {"geracao_id": generation_id, "perfil_id": profile["perfil_id"],
              "perfil": deepcopy(profile), "papel": role, "data": now(),
              "hash_texto_preparado": digest([backend.prepare(t, role) for t in segments]),
              "segmentos": segments, "quantidade_segmentos": len(segments),
              "agregacao": "nenhuma" if len(segments) == 1 else "media_segmentos_originais_seguida_l2",
              "vetor_entrada": {"embedding_id": embedding_id, "arquivo_npz": project.relative(path),
                                "linha": 0, "dimensoes": profile["dimensoes"], "normalizacao": "l2"},
              "npz_sha256": file_hash(path)}
    metadata["geracoes"].setdefault(key, []).append(record)
    write_json(metadata_path, metadata)
    return metadata, record, aggregate[0]


def _within(score, minimum, maximum):
    return (minimum is None or score >= minimum) and (maximum is None or score <= maximum)


def rank_results(universe, categories, view="topicos", top_k=5, minimum=None,
                 maximum=None, partial=False, individual_limits=None, individual_rule="todos"):
    validate_limits(top_k, minimum, maximum)
    if view not in {"topicos", "vetores"} or individual_rule not in {"todos", "qualquer"}:
        raise ValueError("Visão ou regra de filtros individuais inválida.")
    individual_limits = individual_limits or {}
    for category, threshold in individual_limits.items():
        if category not in categories:
            raise ValueError("Filtro individual de categoria não selecionada.")
        validate_limits(None, threshold)
    if view == "vetores":
        output = []
        eligible = 0
        for category in categories:
            group = sorted([row for row in universe if row["categoria"] == category],
                           key=lambda row: (-row["similaridade_cosseno"], row["ordem"], row["embedding_id"]))
            selected = [row for row in group if _within(row["similaridade_cosseno"], minimum, maximum)]
            if category in individual_limits:
                selected = [row for row in selected if row["similaridade_cosseno"] >= individual_limits[category]]
            eligible += len(selected)
            for number, row in enumerate(selected[:top_k] if top_k is not None else selected, 1):
                output.append({**row, "posicao": number, "unidade_resultado": "vetor"})
        return output, {"cobertura_incompleta": 0, "elegiveis": eligible}
    groups = {}
    for row in universe:
        groups.setdefault(row["topico_id"], []).append(row)
    output, incomplete = [], 0
    for topic_id, rows in groups.items():
        present = {row["categoria"] for row in rows}
        absent = [category for category in categories if category not in present]
        if absent:
            incomplete += 1
            if not partial:
                continue
        tests = [row["similaridade_cosseno"] >= individual_limits[row["categoria"]]
                 for row in rows if row["categoria"] in individual_limits]
        if tests and not (all(tests) if individual_rule == "todos" else any(tests)):
            continue
        score = math.fsum(row["similaridade_cosseno"] for row in rows)/len(rows)
        if not _within(score, minimum, maximum):
            continue
        first = rows[0]
        output.append({"topico_id": topic_id, "documento_id": first["documento_id"],
                       "titulo": first["titulo"], "fonte": first["fonte"], "ordem": first["ordem"],
                       "similaridade_media": score, "distancia_cosseno_media": 1-score,
                       "distancia_euclidiana_media": math.fsum(r["distancia_euclidiana_normalizada"] for r in rows)/len(rows),
                       "quantidade_vetores": len(rows), "categorias_ausentes": absent,
                       "comparacoes": rows, "unidade_resultado": "topico"})
    output.sort(key=lambda row: (-row["similaridade_media"], row["ordem"], row["topico_id"]))
    eligible = len(output)
    output = output[:top_k] if top_k is not None else output
    for number, row in enumerate(output, 1):
        row["posicao"] = number
    return output, {"cobertura_incompleta": incomplete, "elegiveis": eligible}


def search(project, profile_id, categories, query, view="topicos", top_k=5,
           minimum=None, maximum=None, partial=False, individual_limits=None,
           individual_rule="todos", provider=None, regenerate_input=False, comparison_id=None):
    validate_limits(top_k, minimum, maximum)
    if not categories or any(category not in CATEGORIES for category in categories):
        raise ValueError("Selecione ao menos uma categoria conhecida.")
    categories = list(dict.fromkeys(categories))
    profile = require_profile(project, profile_id)
    # A trava captura base e ponteiros de todas as categorias consistentemente.
    with project.lock():
        base = project.base()
        mapping = {topic["topico_id"]: (topic, document) for topic, document in topics(base)}
        snapshots = {}
        for category in categories:
            stored = project.load_index(profile_id, category)
            if stored is None:
                raise ValueError(f"Categoria não gerada neste modelo: {category}")
            snapshots[category] = stored
        entry, generation, query_vector = save_input(project, query, profile, provider, regenerate_input)
        execution_id = uid("exp")
        folder = project.root/"estudos"/"execucoes"/execution_id
        params = {"categorias": categories, "visao": view, "top_k": top_k, "limiar": minimum,
                  "escore_maximo": maximum, "cobertura_parcial": partial,
                  "limiares_individuais": individual_limits or {}, "regra_limites_individuais": individual_rule,
                  "metrica_ranking": "similaridade_cosseno_individual" if view == "vetores" else "media_similaridade_cosseno",
                  "escopo_limites": "modelo_categoria" if view == "vetores" else "modelo_combinacao"}
        experiment = {"schema_version": "1.0", "execucao_id": execution_id,
                      "comparacao_id": comparison_id, "entrada_id": entry["entrada_id"],
                      "texto_original_entrada": query, "perfil_id": profile_id, "perfil": deepcopy(profile),
                      "data": now(), "status": "calculando", "parametros": params,
                      "revisao_base": base["controle_embeddings"]["revisao_base"],
                      "vetor_entrada": generation["vetor_entrada"], "hash_npz_entrada": generation["npz_sha256"],
                      "geracao_entrada": generation["geracao_id"], "indices_utilizados": {}, "resultados": []}
        write_json(folder/"resultado.json", experiment)
        entry["execucoes"].append(execution_id)
        write_json(project.root/"estudos"/"entradas"/entry["entrada_id"]/"entrada.json", entry)
        try:
            universe, excluded = [], 0
            for category, (manifest, matrix, _) in snapshots.items():
                experiment["indices_utilizados"][category] = {
                    "revisao": manifest["revisao_indice"], "arquivo_npz": manifest["arquivo_npz"],
                    "checksum": manifest["npz_sha256"], "manifesto": project.current(profile_id, category)["manifesto"]}
                scores = np.clip(matrix @ query_vector, -1.0, 1.0)
                # Categorias com qualquer segmento desatualizado são excluídas por tópico.
                validity = {}
                for record in manifest["itens"]:
                    pair = mapping.get(record["topico_id"])
                    valid = bool(pair) and digest(category_text(*pair, category)) == record["hash_texto_fonte"]
                    validity[record["topico_id"]] = validity.get(record["topico_id"], True) and valid
                for row, record in enumerate(manifest["itens"]):
                    if not validity[record["topico_id"]]:
                        excluded += 1
                        continue
                    topic, document = mapping[record["topico_id"]]
                    score = float(scores[row])
                    universe.append({"embedding_id": record["embedding_id"], "categoria": category,
                                     "perfil_id": profile_id, "modelo": profile["modelo"],
                                     "topico_id": record["topico_id"], "documento_id": document["documento_id"],
                                     "titulo": topic.get("titulo", ""), "segmento": record["segmento"],
                                     "ordem": [document.get("ordem_documento", 0), topic.get("ordem_no_documento", 0)],
                                     "texto_fonte_recuperado": record["texto_segmento"],
                                     "hash_texto_fonte": record["hash_texto_fonte"],
                                     "hash_texto_embeddado": record["hash_texto_embeddado"],
                                     "fonte": {"nome_documento": document.get("nome_documento"),
                                               "url": document.get("metadados", {}).get("url"), **topic.get("fonte", {})},
                                     "vetor_recuperado": record["referencia"],
                                     "original_disponivel": record.get("original_disponivel", False),
                                     "vetor_entrada": generation["vetor_entrada"],
                                     "similaridade_cosseno": score, "distancia_cosseno": 1-score,
                                     "distancia_euclidiana_normalizada": float(math.sqrt(max(0.0, 2-2*score)))})
            results, coverage = rank_results(universe, categories, view, top_k, minimum, maximum,
                                             partial, individual_limits, individual_rule)
            score_path = folder/"escores.npz"
            write_npz(score_path, embedding_ids=np.asarray([r["embedding_id"] for r in universe], dtype=np.str_),
                      categorias=np.asarray([r["categoria"] for r in universe], dtype=np.str_),
                      escores=np.asarray([r["similaridade_cosseno"] for r in universe]),
                      distancias_cosseno=np.asarray([r["distancia_cosseno"] for r in universe]),
                      distancias_euclidianas=np.asarray([r["distancia_euclidiana_normalizada"] for r in universe]))
            reason = "sem_resultados" if not results else (
                "top_k" if top_k is not None and coverage["elegiveis"] > len(results)
                else "limiar" if minimum is not None or maximum is not None else "candidatos_esgotados")
            experiment.update({"status": "concluido", "universo": universe, "resultados": results,
                               "universo_escores_npz": project.relative(score_path), "universo_checksum": file_hash(score_path),
                               "avaliados": len(universe), "excluidos_desatualizados_inativos": excluded,
                               "cobertura": coverage, "motivo_termino": reason,
                               "quantidade_resultados": len(results), "fim": now()})
            write_json(folder/"resultado.json", experiment)
        except Exception as exc:
            experiment.update({"status": "erro", "erro": str(exc), "fim": now()})
            write_json(folder/"resultado.json", experiment)
            raise
    return experiment


def load_experiment(project, execution_id, verify=False):
    path = project.path(f"estudos/execucoes/{execution_id}/resultado.json")
    experiment = read_json(path)
    if verify:
        reference = experiment["vetor_entrada"]
        if file_hash(project.path(reference["arquivo_npz"])) != experiment["hash_npz_entrada"]:
            raise ValueError("Embedding da entrada foi alterado.")
        for record in experiment.get("indices_utilizados", {}).values():
            if file_hash(project.path(record["arquivo_npz"])) != record["checksum"]:
                raise ValueError("Um índice usado no experimento foi alterado.")
        if experiment.get("universo_escores_npz") and file_hash(project.path(experiment["universo_escores_npz"])) != experiment["universo_checksum"]:
            raise ValueError("Escores salvos foram alterados.")
    return experiment


def list_experiments(project):
    folder = project.root/"estudos"/"execucoes"
    return [read_json(path) for path in sorted(folder.glob("*/resultado.json"), reverse=True)] if folder.exists() else []


def score_groups(project, execution_id, boundaries, basis="individual"):
    if basis not in {"individual", "media"}:
        raise ValueError("Base de faixa deve ser individual ou media.")
    if len(boundaries) < 2 or any(not math.isfinite(v) or not -1<=v<=1 for v in boundaries):
        raise ValueError("Defina ao menos dois limites finitos em [-1, 1].")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("Os limites devem estar em ordem crescente, sem repetição.")
    original = load_experiment(project, execution_id, verify=True)
    if original["status"] != "concluido":
        raise ValueError("Execução não concluída.")
    rows = original["universo"]
    if basis == "media":
        params = original["parametros"]
        rows, _ = rank_results(rows, params["categorias"], "topicos", None, None, None,
                               params.get("cobertura_parcial", False))
    parts = []
    members = set()
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        last = index == len(boundaries)-2
        selections = {}
        for row in rows:
            score = row["similaridade_cosseno"] if basis == "individual" else row["similaridade_media"]
            if left <= score and (score <= right if last else score < right):
                category = row.get("categoria", "combinacao")
                selections.setdefault(category, []).append(row)
                members.add(row.get("embedding_id", row.get("topico_id")))
        for category, selected in selections.items():
            parts.append({"categoria": category, "minimo_inclusivo": left, "maximo": right,
                          "maximo_inclusivo": last, "quantidade": len(selected), "membros": selected})
    set_id = uid("conjunto")
    result = {"conjunto_id": set_id, "execucao_id": execution_id, "entrada_id": original["entrada_id"],
              "perfil_id": original["perfil_id"], "base_escore": basis, "limites": boundaries,
              "faixas": parts, "fora_das_faixas": len(rows)-len(members), "data": now()}
    with project.lock():
        write_json(project.root/"estudos"/"conjuntos"/(set_id+".json"), result)
    return result


def rerank(project, execution_id, categories=None, view=None, top_k=5, minimum=None, maximum=None, partial=False):
    original = load_experiment(project, execution_id, verify=True)
    categories = categories or original["parametros"]["categorias"]
    if not categories or not set(categories).issubset(original["parametros"]["categorias"]):
        raise ValueError("Categorias novas exigem outra avaliação; escolha categorias já avaliadas.")
    view = view or original["parametros"]["visao"]
    universe = [row for row in original["universo"] if row["categoria"] in categories]
    results, coverage = rank_results(universe, categories, view, top_k, minimum, maximum, partial)
    derived = deepcopy(original)
    derived.update({"execucao_id": uid("exp"), "execucao_origem": execution_id, "data": now(),
                    "resultados": results, "universo": universe, "quantidade_resultados": len(results), "cobertura": coverage})
    derived["parametros"].update({"categorias": categories, "visao": view, "top_k": top_k,
                                  "limiar": minimum, "escore_maximo": maximum, "cobertura_parcial": partial,
                                  "limiares_individuais": {}, "regra_limites_individuais": "todos"})
    derived["parametros"].update({"metrica_ranking": "similaridade_cosseno_individual" if view == "vetores" else "media_similaridade_cosseno",
                                   "escopo_limites": "modelo_categoria" if view == "vetores" else "modelo_combinacao"})
    derived["avaliados"] = len(universe)
    derived["fim"] = now()
    derived["motivo_termino"] = "sem_resultados" if not results else ("top_k" if top_k is not None and coverage["elegiveis"] > len(results) else "limiar" if minimum is not None or maximum is not None else "candidatos_esgotados")
    with project.lock():
        folder = project.root/"estudos"/"execucoes"/derived["execucao_id"]
        write_json(folder/"resultado.json", derived)
        path = project.root/"estudos"/"entradas"/original["entrada_id"]/"entrada.json"
        metadata = read_json(path)
        metadata["execucoes"].append(derived["execucao_id"])
        write_json(path, metadata)
    return derived


def export_experiment(project, execution_id, destination):
    experiment = load_experiment(project, execution_id, verify=True)
    paths = {f"estudos/execucoes/{execution_id}/resultado.json", experiment["vetor_entrada"]["arquivo_npz"]}
    input_path = f"estudos/entradas/{experiment['entrada_id']}/entrada.json"
    entry = read_json(project.path(input_path))
    exported_entry = deepcopy(entry)
    # A exportação não deixa referências a gerações de outros modelos ausentes do ZIP.
    exported_entry["geracoes"] = {key: [record for record in records if record["geracao_id"] == experiment["geracao_entrada"]]
                                   for key, records in entry["geracoes"].items()
                                   if any(record["geracao_id"] == experiment["geracao_entrada"] for record in records)}
    exported_entry["execucoes"] = [execution_id]
    if experiment.get("universo_escores_npz"):
        paths.add(experiment["universo_escores_npz"])
    for reference in experiment["indices_utilizados"].values():
        paths.update([reference["arquivo_npz"], reference["manifesto"]])
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(project.path(path), path)
        import json
        archive.writestr(input_path, json.dumps(exported_entry, ensure_ascii=False, indent=2, allow_nan=False))
        archive.writestr("dados/projeto.json", json.dumps({"schema_version": "1.0", "perfis": {
            experiment["perfil_id"]: experiment["perfil"]}, "perfil_ativo_id": experiment["perfil_id"]}, ensure_ascii=False, indent=2))
    return str(destination)
