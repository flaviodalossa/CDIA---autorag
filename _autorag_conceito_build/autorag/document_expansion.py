"""Expand accepted retrieval seeds to all active topics of their source documents."""
from copy import deepcopy


def full_document_enabled(value=None):
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError("full_document deve ser booleano.")
    return value


def expand_documents(rows, mapping):
    """Keep seed ranking; append unseen topics in document/source order, without inference."""
    seeds = deepcopy(rows)
    grouped, document_order = {}, []
    seen = set()
    result = []
    for row in seeds:
        topic_id = row.get("topic_id", row.get("topico_id"))
        pair = mapping.get(topic_id)
        if pair is None or topic_id in seen:
            continue
        document_id = pair[1]["documento_id"]
        if document_id not in grouped:
            grouped[document_id] = []
            document_order.append(document_id)
        grouped[document_id].append(topic_id)
        seen.add(topic_id)
        row.update(topic_id=topic_id, document_id=document_id, result_origin="retrieval",
                   retrieval_position=row.get("position", row.get("posicao")))
        result.append(row)
    seed_count = len(result)
    for document_id in document_order:
        for topic_id, (topic, document) in mapping.items():
            if document["documento_id"] != document_id or topic_id in seen or not topic.get("ativo", True):
                continue
            seen.add(topic_id)
            result.append({"topic_id": topic_id, "document_id": document_id,
                "title": topic.get("titulo", ""), "text": topic.get("conteudo_fonte", ""),
                "summary": topic.get("resumo", ""), "distance": None, "cosine_similarity": None,
                "original_distance": None, "original_cosine_similarity": None,
                "categories": [], "category_count": 0, "vector_count": 0, "matched_vectors": [],
                "profile_id": None, "analysis": None, "result_origin": "document_expansion",
                "expanded_from_topic_ids": list(grouped[document_id]),
                "retrieval_position": None,
                "source": {"name": document.get("nome_documento", ""),
                           "url": document.get("metadados", {}).get("url") or
                               (document.get("fonte", {}).get("referencia") if str(document.get("fonte", {}).get("referencia", "")).startswith("https://") else None),
                           **topic.get("fonte", {})}})
    for position, row in enumerate(result, 1):
        row["position"] = position
    return result, {"enabled": True, "seed_count": seed_count, "added_count": len(result)-seed_count,
                    "total_topics": len(result), "document_ids": document_order,
                    "scope": "all_active_topics_of_documents_with_accepted_hits",
                    "order": "retrieval_seeds_then_document_source_order"}


def expand_comparison(output, mapping):
    enabled = full_document_enabled(output.get("full_document"))
    output["configuration"]["full_document"] = enabled
    if enabled:
        for experiment in output["experiments"]:
            for key, meta in (("results", "document_expansion"), ("baseline_results", "baseline_document_expansion")):
                experiment[key], experiment[meta] = expand_documents(experiment.get(key, []), mapping)
    return output
