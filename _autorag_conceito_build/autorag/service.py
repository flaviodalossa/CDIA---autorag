from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from .core import (CATEGORIES, Project, category_text, digest, documents, full_topic,
                   normalize, now, read_json, topics, uid, write_json)
from .providers import get_provider, profile_spec


def register_profile(project, provider, model, dimensions=None, *, activate=True, **options):
    if not isinstance(activate, bool):
        raise ValueError("activate deve ser booleano.")
    profile = profile_spec(provider, model, dimensions, **options)
    with project.lock():
        config = project.config()
        config["perfis"][profile["perfil_id"]] = profile
        if activate:
            config["perfil_ativo_id"] = profile["perfil_id"]
        project.save_config(config)
        if project.base_path.exists():
            sync_base(project)
    return profile


def require_profile(project, profile_id):
    config = project.config()
    if profile_id not in config["perfis"]:
        raise ValueError(f"Perfil não cadastrado: {profile_id}")
    return config["perfis"][profile_id]


def sync_base(project):
    """Manifestos publicados são a autoridade; JSON é reconstruível após interrupção."""
    base = project.base()
    config = project.config()
    mapping = {topic["topico_id"]: (topic, document) for topic, document in topics(base, active=False)}
    control = base.setdefault("controle_embeddings", {})
    control.update({"schema_version": "2.0", "perfil_ativo_id": config.get("perfil_ativo_id"),
                    "perfis": deepcopy(config["perfis"])})
    for topic, _ in mapping.values():
        topic.pop("embedding", None)
        topic["embeddings"] = {"perfis": {}}
    for profile_id, profile in config["perfis"].items():
        control["perfis"][profile_id]["indices_por_tipo"] = {}
        for category in CATEGORIES:
            stored = project.load_index(profile_id, category)
            if stored is None:
                continue
            manifest = stored[0]
            control["perfis"][profile_id]["indices_por_tipo"][category] = {
                "revisao_indice": manifest["revisao_indice"], "arquivo_npz": manifest["arquivo_npz"],
                "manifesto": project.current(profile_id, category)["manifesto"]}
            for record in manifest["itens"]:
                if record["topico_id"] not in mapping:
                    continue
                topic, document = mapping[record["topico_id"]]
                current = category_text(topic, document, category)
                state = "atualizado" if digest(current) == record["hash_texto_fonte"] else "desatualizado"
                if not topic.get("ativo", True):
                    state = "inativo"
                entry = topic["embeddings"]["perfis"].setdefault(profile_id, {"categorias": {}})
                part = entry["categorias"].setdefault(category, {"unidades": []})
                unit = {key: record.get(key) for key in (
                    "embedding_id", "categoria", "segmento", "inicio", "fim", "hash_texto_fonte",
                    "hash_texto_embeddado", "data_geracao", "revisao_base_geracao", "referencia")}
                unit["status"] = state
                part["unidades"].append(unit)
                part["status"] = state
    base["schema_version"] = "3.0"
    base["data_atualizacao_base"] = now()
    write_json(project.base_path, base)
    return base


def import_base(project, source, legacy_npz=None):
    source = Path(source).expanduser().resolve()
    incoming = read_json(source)
    if not isinstance(incoming, dict):
        raise ValueError("A base deve ser um objeto JSON.")
    incoming["documentos"] = documents(incoming)
    doc_ids = [doc.get("documento_id") for doc in incoming["documentos"]]
    if None in doc_ids or len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Documento sem ID ou documento duplicado.")
    # Um arquivo só de metadados atualiza os documentos sem apagar tópicos.
    metadata_only = not any(doc.get("estrutura") for doc in incoming["documentos"])
    if metadata_only and project.base_path.exists():
        previous = {doc["documento_id"]: doc for doc in documents(project.base())}
        for doc in incoming["documentos"]:
            old = previous.get(doc["documento_id"], {})
            if old.get("estrutura"):
                doc["estrutura"] = deepcopy(old["estrutura"])
    all_topics = list(topics(incoming, active=False))
    old_embeddings = {}
    for topic, _ in all_topics:
        old_embeddings[topic["topico_id"]] = topic.pop("embedding", None)
        topic.pop("embeddings", None)
    incoming["schema_version"] = "3.0"
    incoming["controle_embeddings"] = {"schema_version": "2.0", "revisao_base": uid("base")}
    candidate = Path(legacy_npz).resolve() if legacy_npz else source.with_name("embeddings_etica.npz")
    # Validação antes de substituir a base ativa.
    if candidate.exists() and (legacy_npz or not project.config()["perfis"]):
        with np.load(candidate, allow_pickle=False) as package:
            matrix = np.asarray(package["vetores"])
            ids = package["topico_ids"].tolist()
            dimensions = int(package["dimensoes"].reshape(-1)[0])
            str(package["modelo"].reshape(-1)[0])
            normalize(matrix, dimensions)
            if len(ids) != len(set(ids)) or len(ids) != len(matrix):
                raise ValueError("Índice legado com IDs inválidos.")
            if "hashes_texto" in package and len(package["hashes_texto"]) != len(ids):
                raise ValueError("Quantidade de hashes divergente no índice legado.")
    for old in old_embeddings.values():
        if isinstance(old, dict) and isinstance(old.get("vetor"), list) and old["vetor"]:
            normalize(np.asarray([old["vetor"]]), len(old["vetor"]))
    for profile_id in project.config()["perfis"]:
        for category in CATEGORIES:
            project.load_index(profile_id, category)
    with project.lock():
        if project.base_path.exists():
            backup = project.root/"historico"/"bases"/(uid("base") + ".json")
            write_json(backup, project.base())
        write_json(project.base_path, incoming)
        config = project.config()
        config.setdefault("perfis", {})
        project.save_config(config)
        if candidate.exists() and (legacy_npz or not config["perfis"]):
            _import_legacy_npz(project, incoming, candidate)
        # Excepcionalmente, arquivo antigo contém coordenadas no JSON: preserva no NPZ.
        groups = {}
        for topic, _ in all_topics:
            old = old_embeddings.get(topic["topico_id"])
            if isinstance(old, dict) and isinstance(old.get("vetor"), list) and old["vetor"]:
                key = (old.get("modelo") or "modelo_legado", len(old["vetor"]))
                groups.setdefault(key, []).append((topic, old))
        for (model, dimension), rows in groups.items():
            profile = legacy_profile(model, dimension)
            if project.current(profile["perfil_id"], CATEGORIES[0]):
                continue
            config = project.config()
            config["perfis"][profile["perfil_id"]] = profile
            project.save_config(config)
            records = [legacy_record(topic, full_topic(topic), incoming, old) for topic, old in rows]
            vectors = np.asarray([old["vetor"] for _, old in rows])
            project.publish(profile, CATEGORIES[0], records, vectors, vectors)
        base = sync_base(project)
    return {"documentos": len(doc_ids), "topicos": len(all_topics),
            "topicos_ativos": len(list(topics(base))), "somente_metadados": metadata_only,
            "revisao_base": base["controle_embeddings"]["revisao_base"]}


def legacy_profile(model, dimension):
    profile = {"provedor": "openai", "modelo": model, "dimensoes": dimension,
               "limite_tokens": 8191, "revisao_modelo": model, "endpoint": "",
               "normalizacao": "l2", "versao_preparacao": "1", "segmentacao": "utf8_conservador_v1",
               "legado": True}
    profile["perfil_id"] = "pf_" + digest(profile)[:20]
    return profile


def legacy_record(topic, value, base, old=None, recorded_hash=None):
    old = old or {}
    return {"embedding_id": topic["topico_id"]+"::"+CATEGORIES[0]+"::0000",
            "topico_id": topic["topico_id"], "categoria": CATEGORIES[0], "segmento": 0,
            "hash_texto_fonte": recorded_hash or old.get("hash_texto_embeddado") or digest(value),
            "hash_texto_embeddado": old.get("hash_texto_embeddado") or recorded_hash or digest(value),
            "inicio": 0, "fim": len(value), "texto_segmento": value,
            "data_geracao": old.get("data_geracao") or now(),
            "revisao_base_geracao": base["controle_embeddings"]["revisao_base"], "original_disponivel": False}


def _import_legacy_npz(project, base, path):
    with np.load(path, allow_pickle=False) as package:
        matrix = np.asarray(package["vetores"]).copy()
        ids = [str(value) for value in package["topico_ids"].tolist()]
        model = str(package["modelo"].reshape(-1)[0])
        dimensions = int(package["dimensoes"].reshape(-1)[0])
        hashes = package["hashes_texto"].tolist() if "hashes_texto" in package else None
    if len(ids) != len(set(ids)) or len(ids) != len(matrix):
        raise ValueError("Índice legado com IDs inválidos.")
    normalize(matrix, dimensions)
    profile = legacy_profile(model, dimensions)
    config = project.config()
    config["perfis"][profile["perfil_id"]] = profile
    config["perfil_ativo_id"] = profile["perfil_id"]
    project.save_config(config)
    mapping = {topic["topico_id"]: (topic, document) for topic, document in topics(base, active=False)}
    records, rows = [], []
    for row, topic_id in enumerate(ids):
        if topic_id not in mapping:
            continue
        topic, document = mapping[topic_id]
        record = legacy_record(topic, full_topic(topic), base, recorded_hash=str(hashes[row]) if hashes else None)
        record["documento_id"] = document["documento_id"]
        records.append(record)
        rows.append(row)
    if records:
        project.publish(profile, CATEGORIES[0], records, matrix[rows])


def inventory(project):
    base = project.base()
    active = list(topics(base))
    result = []
    for profile_id, profile in project.config()["perfis"].items():
        for category in CATEGORIES:
            stored = project.load_index(profile_id, category)
            by_topic = {}
            if stored:
                for record in stored[0]["itens"]:
                    by_topic.setdefault(record["topico_id"], []).append(record)
            current_count = outdated = missing = empty = 0
            for topic, document in active:
                value = category_text(topic, document, category)
                if not value:
                    empty += 1
                elif topic["topico_id"] not in by_topic:
                    missing += 1
                elif all(item["hash_texto_fonte"] == digest(value) for item in by_topic[topic["topico_id"]]):
                    current_count += 1
                else:
                    outdated += 1
            result.append({"perfil_id": profile_id, "modelo": profile["modelo"], "provedor": profile["provedor"],
                           "dimensoes": profile["dimensoes"], "categoria": category,
                           "atuais": current_count, "desatualizados": outdated, "faltantes": missing,
                           "sem_texto": empty, "vetores_armazenados": len(stored[1]) if stored else 0,
                           "revisao": stored[0]["revisao_indice"] if stored else None})
    return result


def generate(project, profile_id, categories, operation="atualizar", topic_ids=None,
             document_ids=None, batch_size=16, simulate=False, progress=None, provider=None):
    if not categories or any(category not in CATEGORIES for category in categories):
        raise ValueError("Selecione pelo menos uma categoria conhecida.")
    if operation not in {"faltantes", "atualizar", "regenerar"} or batch_size <= 0:
        raise ValueError("Operação ou tamanho de lote inválido.")
    profile = require_profile(project, profile_id)
    base = project.base()
    active = list(topics(base))
    if topic_ids:
        unknown = set(topic_ids) - {topic["topico_id"] for topic, _ in active}
        if unknown:
            raise ValueError("Tópicos ativos não encontrados: " + ", ".join(sorted(unknown)))
    if document_ids:
        unknown = set(document_ids) - {doc["documento_id"] for doc in documents(base)}
        if unknown:
            raise ValueError("Documentos não encontrados: " + ", ".join(sorted(unknown)))
    scoped = [(topic, doc) for topic, doc in active if
              (not topic_ids or topic["topico_id"] in topic_ids) and
              (not document_ids or doc["documento_id"] in document_ids)]
    plans = []
    for category in dict.fromkeys(categories):
        stored = project.load_index(profile_id, category)
        old = {}
        if stored:
            for row, record in enumerate(stored[0]["itens"]):
                old.setdefault(record["topico_id"], []).append((record, row))
        selected = []
        for topic, doc in scoped:
            previous = old.get(topic["topico_id"], [])
            value = category_text(topic, doc, category)
            changed = not previous or any(item[0]["hash_texto_fonte"] != digest(value) for item in previous)
            if operation == "regenerar" or (operation == "atualizar" and changed) or (operation == "faltantes" and not previous):
                selected.append((topic, doc, value))
        plans.append({"categoria": category, "topicos_selecionados": len(selected),
                      "textos_unicos": len({row[2] for row in selected if row[2]}),
                      "sem_texto": sum(not row[2] for row in selected), "selecionados": selected,
                      "stored": stored})
    summary = [{key: value for key, value in plan.items() if key not in {"selecionados", "stored"}} for plan in plans]
    if simulate:
        return {"simulacao": True, "perfil_id": profile_id, "operacao": operation, "plano": summary}
    if not any(plan["selecionados"] for plan in plans):
        return {"perfil_id": profile_id, "plano": summary, "mensagem": "Nenhuma alteração selecionada."}
    backend = provider or get_provider(profile)
    with project.lock():
        # Evita uma seleção calculada em uma revisão que foi trocada antes da trava.
        if project.base()["controle_embeddings"]["revisao_base"] != base["controle_embeddings"]["revisao_base"]:
            raise ValueError("A base mudou durante a seleção. Repita a operação.")
        for plan in plans:
            category, selected, stored = plan["categoria"], plan["selecionados"], plan["stored"]
            pointer = project.current(profile_id, category)
            previous = stored[0]["revisao_indice"] if stored else None
            current = read_json(project.path(pointer["manifesto"]))["revisao_indice"] if pointer else None
            if previous != current:
                raise ValueError("O índice mudou durante a seleção. Repita a operação.")
            if not selected:
                continue
            selected_ids = {topic["topico_id"] for topic, _, _ in selected}
            records, vectors, originals = [], [], []
            old_original = stored[2] if stored else None
            if stored:
                for row, item in enumerate(stored[0]["itens"]):
                    if item["topico_id"] not in selected_ids:
                        records.append(deepcopy(item))
                        vectors.append(stored[1][row])
                        originals.append(old_original[row] if old_original is not None else stored[1][row])
            pending = []
            for topic, doc, value in selected:
                for number, (segment, start, stop) in enumerate(backend.segments(value)) if value else []:
                    prepared = backend.prepare(segment, "documento")
                    record = {"embedding_id": f"{topic['topico_id']}::{category}::{number:04d}",
                              "topico_id": topic["topico_id"], "documento_id": doc["documento_id"],
                              "categoria": category, "segmento": number, "inicio": start, "fim": stop,
                              "hash_texto_fonte": digest(value), "hash_texto_embeddado": digest(prepared),
                              "texto_segmento": segment, "data_geracao": now(),
                              "revisao_base_geracao": base["controle_embeddings"]["revisao_base"],
                              "original_disponivel": True}
                    pending.append((record, segment))
            checkpoint_path = project.root/"indices"/profile_id/category/"checkpoint.json"
            checkpoint_vectors = checkpoint_path.with_name("checkpoint.npz")
            signature = digest({"base": base["controle_embeddings"]["revisao_base"], "perfil": profile_id,
                                "categoria": category, "operacao": operation,
                                "registros": [(r["embedding_id"], r["hash_texto_embeddado"]) for r, _ in pending]})
            cache = {}
            if checkpoint_path.exists() and checkpoint_vectors.exists():
                checkpoint = read_json(checkpoint_path)
                if checkpoint.get("assinatura") == signature:
                    from .core import file_hash
                    if checkpoint.get("npz_sha256") == file_hash(checkpoint_vectors):
                        with np.load(checkpoint_vectors, allow_pickle=False) as package:
                            cache = {h: package["vetores_originais"][i].copy() for i, h in enumerate(checkpoint["hashes"])}
            unique = list(dict.fromkeys(r["hash_texto_embeddado"] for r, _ in pending))
            values = {r["hash_texto_embeddado"]: segment for r, segment in pending}
            missing = [h for h in unique if h not in cache]
            for start in range(0, len(missing), batch_size):
                batch_hashes = missing[start:start+batch_size]
                try:
                    raw = np.asarray(backend.embed([values[h] for h in batch_hashes], "documento"))
                    normalize(raw, profile["dimensoes"])
                    if len(raw) != len(batch_hashes):
                        raise ValueError("Quantidade de embeddings retornados incorreta.")
                    cache.update({h: raw[i].copy() for i, h in enumerate(batch_hashes)})
                    from .core import file_hash, write_npz
                    hashes = list(cache)
                    write_npz(checkpoint_vectors, vetores_originais=np.stack([cache[h] for h in hashes]))
                    write_json(checkpoint_path, {"assinatura": signature, "hashes": hashes,
                                               "npz_sha256": file_hash(checkpoint_vectors), "data": now()})
                except Exception as exc:
                    write_json(checkpoint_path.with_name("ultimo_erro.json"), {"data": now(), "erro": str(exc), "assinatura": signature})
                    raise
                if progress:
                    progress(f"{category}: {min(start+batch_size,len(missing))}/{len(missing)} textos únicos gerados")
            for record, _ in pending:
                raw = cache[record["hash_texto_embeddado"]]
                records.append(record)
                vectors.append(normalize(raw[None, :], profile["dimensoes"])[0])
                originals.append(raw)
            matrix = np.stack(vectors) if vectors else np.empty((0, profile["dimensoes"]))
            # Unidades legadas sem original mantêm original_disponivel=False;
            # a matriz retém os originais das unidades novas, mesmo num índice misto.
            original = np.stack(originals) if originals else matrix.copy()
            project.publish(profile, category, records, matrix, original)
            checkpoint_path.unlink(missing_ok=True)
            checkpoint_vectors.unlink(missing_ok=True)
            sync_base(project)
            if progress:
                progress(f"{category}: revisão publicada; {len(records)} unidades armazenadas.")
    return {"perfil_id": profile_id, "operacao": operation, "plano": summary}
