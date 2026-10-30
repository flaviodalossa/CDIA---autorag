from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from autorag.core import (CATEGORIES, Project, category_text, digest, full_topic,
                          normalize, read_json, split_text, write_json, write_npz)
from autorag.providers import Provider, OllamaProvider, GeminiProvider
from autorag.service import generate, import_base, inventory, legacy_profile
from autorag.experiments import (export_experiment, load_experiment, rank_results,
                                 rerank, score_groups, search, validate_limits)
from autorag.analysis import compare, derive, experiment_references, pairwise


class FakeProvider(Provider):
    """Somente teste: não é um modelo oferecido pela aplicação."""
    def __init__(self, profile, fail_call=None):
        super().__init__(profile)
        self.calls = []
        self.fail_call = fail_call

    def embed(self, texts, role="documento"):
        self.calls.append((list(texts), role))
        if len(self.calls) == self.fail_call:
            raise RuntimeError("Falha de inferência simulada")
        if role == "consulta":
            return np.tile([1.0, 0.0, 0.0], (len(texts), 1))
        return np.asarray([[int(digest(t)[i:i+2], 16)+1 for i in (0, 2, 4)] for t in texts], dtype=float)


def fixture():
    return {"schema_version": "2.1", "documentos": [{"documento_id": "doc1", "nome_documento": "Ética em pesquisa",
            "metadados": {"idioma": "português", "eixos": ["ética"]}, "ordem_documento": 1,
            "estrutura": [{"topico_id": f"t{i}", "titulo": f"Título {i}", "ativo": True,
                            "ordem_no_documento": i, "caminho_hierarquico": ["Parte I"],
                            "resumo": f"Resumo {i}", "conteudo_fonte": f"Fonte completa {i}",
                            "perguntas_sugeridas": [f"Pergunta {i}?"], "resposta_sintese": f"Resposta {i}",
                            "palavras_chave": [f"chave {i}"], "base_normativa": ["Norma A"]} for i in range(1, 4)]}]}


class LabTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = Project(self.root/"projeto")
        self.source = self.root/"base.json"
        self.data = fixture()
        write_json(self.source, self.data)
        import_base(self.project, self.source)
        self.profile = {"perfil_id": "pf_teste", "provedor": "teste", "modelo": "teste_offline",
                        "dimensoes": 3, "revisao_modelo": "teste1", "limite_tokens": 512,
                        "normalizacao": "l2", "versao_preparacao": "1"}
        self.project.save_config({"perfis": {"pf_teste": self.profile}, "perfil_ativo_id": "pf_teste"})
        self.provider = FakeProvider(self.profile)

    def tearDown(self):
        self.temporary.cleanup()

    def gen(self, categories, **kwargs):
        return generate(self.project, "pf_teste", categories, provider=self.provider, **kwargs)

    def find(self, categories, **kwargs):
        return search(self.project, "pf_teste", categories, " Consulta com acentuação\n", provider=self.provider, **kwargs)

    def update_source(self):
        write_json(self.source, self.data)
        import_base(self.project, self.source)

    def test_keywords_do_not_change_baseline(self):
        topic, document = self.data["documentos"][0]["estrutura"][0], self.data["documentos"][0]
        before = {cat: digest(category_text(topic, document, cat)) for cat in CATEGORIES}
        topic["palavras_chave"] = ["outra palavra"]
        changed = [cat for cat in CATEGORIES if before[cat] != digest(category_text(topic, document, cat))]
        self.assertEqual(changed, [CATEGORIES[1], "palavras_chave"])

    def test_eight_independent_categories_and_metadata_deduplication(self):
        source_categories = [c for c in CATEGORIES if c != "enriquecimento_recuperacao"]
        for category in source_categories:
            self.gen([category])
            self.assertIsNotNone(self.project.current("pf_teste", category))
        self.assertEqual(len(inventory(self.project)), 9)
        self.assertTrue(all(row["atuais"] == 3 for row in inventory(self.project) if row["categoria"] in source_categories))
        self.assertIsNone(self.project.current("pf_teste", "enriquecimento_recuperacao"))
        self.assertEqual(len(self.provider.calls[-1][0]), 1)  # documento tem metadados comuns
        for path in self.project.root.rglob("*.json"):
            content = read_json(path)
            def verify(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        self.assertNotIn(key, {"vetores", "vetores_originais", "vetor"})
                        verify(value)
                elif isinstance(obj, list):
                    for value in obj:
                        verify(value)
            verify(content)

    def test_selective_regeneration_preserves_other_units_and_category(self):
        self.gen(["resumo", "conteudo_fonte"])
        old, matrix, _ = self.project.load_index("pf_teste", "resumo")
        original_path = self.project.path(old["arquivo_npz"])
        untouched_pointer = self.project.current("pf_teste", "conteudo_fonte")
        self.data["documentos"][0]["estrutura"][0]["resumo"] = "Resumo modificado"
        self.update_source()
        self.assertEqual([r for r in inventory(self.project) if r["categoria"]=="resumo"][0]["desatualizados"], 1)
        self.gen(["resumo"], topic_ids=["t1"])
        new, revised, _ = self.project.load_index("pf_teste", "resumo")
        self.assertNotEqual(new["revisao_indice"], old["revisao_indice"])
        self.assertTrue(original_path.exists())
        old_map = {r["topico_id"]: matrix[i] for i, r in enumerate(old["itens"])}
        new_map = {r["topico_id"]: revised[i] for i, r in enumerate(new["itens"])}
        np.testing.assert_allclose(old_map["t2"], new_map["t2"])
        np.testing.assert_allclose(old_map["t3"], new_map["t3"])
        self.assertEqual(self.project.current("pf_teste", "conteudo_fonte"), untouched_pointer)

    def test_failed_regeneration_preserves_pointer_and_resumes_checkpoint(self):
        self.gen(["resumo"])
        pointer = self.project.current("pf_teste", "resumo")
        self.provider = FakeProvider(self.profile, fail_call=2)
        with self.assertRaises(RuntimeError):
            self.gen(["resumo"], operation="regenerar", batch_size=1)
        self.assertEqual(pointer, self.project.current("pf_teste", "resumo"))
        self.provider = FakeProvider(self.profile)
        self.gen(["resumo"], operation="regenerar", batch_size=1)
        self.assertEqual(sum(len(t) for t, _ in self.provider.calls), 2)
        self.assertNotEqual(pointer, self.project.current("pf_teste", "resumo"))

    def test_update_never_generates_automatically_and_excludes_stale(self):
        self.gen(["resumo"])
        old = self.project.current("pf_teste", "resumo")
        self.data["documentos"][0]["estrutura"][0]["resumo"] = "Modificado"
        self.update_source()
        self.assertEqual(old, self.project.current("pf_teste", "resumo"))
        result = self.find(["resumo"], top_k=None)
        self.assertEqual(result["avaliados"], 2)
        self.assertEqual(result["excluidos_desatualizados_inativos"], 1)
        self.assertNotIn("t1", [r["topico_id"] for r in result["resultados"]])

    def test_queries_and_all_scores_persist_reuse_no_inference_on_rerank(self):
        self.gen(["resumo", "palavras_chave"])
        result = self.find(["resumo", "palavras_chave"], top_k=1)
        self.assertEqual(result["avaliados"], 6)
        self.assertEqual(len(result["resultados"]), 1)
        self.assertEqual(result["texto_original_entrada"], " Consulta com acentuação\n")
        count = len(self.provider.calls)
        other = self.find(["resumo"], top_k=None)
        self.assertEqual(count, len(self.provider.calls))
        self.assertEqual(other["vetor_entrada"], result["vetor_entrada"])
        ranked = rerank(self.project, result["execucao_id"], view="vetores", top_k=None)
        self.assertEqual(len(ranked["resultados"]), 6)
        self.assertEqual(ranked["parametros"]["escopo_limites"], "modelo_categoria")
        self.assertEqual(count, len(self.provider.calls))
        load_experiment(self.project, result["execucao_id"], verify=True)
        with np.load(self.project.path(result["vetor_entrada"]["arquivo_npz"]), allow_pickle=False) as package:
            self.assertEqual(package["vetores"].shape, (1, 3))

    def test_zero_results_still_record_query_and_universe(self):
        self.gen(["resumo"])
        result = self.find(["resumo"], minimum=1, top_k=None)
        self.assertEqual(result["resultados"], [])
        self.assertEqual(len(result["universo"]), 3)
        self.assertTrue(self.project.path(result["vetor_entrada"]["arquivo_npz"]).exists())
        self.assertEqual(result["motivo_termino"], "sem_resultados")

    def test_failed_query_generation_keeps_original_text_and_error(self):
        self.gen(["resumo"])
        self.provider = FakeProvider(self.profile, fail_call=1)
        with self.assertRaises(RuntimeError):
            self.find(["resumo"])
        records = list((self.project.root/"estudos"/"entradas").glob("*/entrada.json"))
        self.assertEqual(len(records), 1)
        entry = read_json(records[0])
        self.assertEqual(entry["texto_original"], " Consulta com acentuação\n")
        self.assertEqual(len(entry["erros_geracao"]), 1)
        self.assertEqual(entry["geracoes"], {})

    def test_score_groups_and_export_include_vectors_without_json_coordinates(self):
        self.gen(["resumo", "palavras_chave"])
        result = self.find(["resumo", "palavras_chave"], top_k=1)
        group = score_groups(self.project, result["execucao_id"], [-1, .5, 1])
        self.assertEqual(sum(g["quantidade"] for g in group["faixas"]), 6)
        self.assertEqual(group["fora_das_faixas"], 0)
        dest = self.root/"estudo.zip"
        export_experiment(self.project, result["execucao_id"], dest)
        with zipfile.ZipFile(dest) as package:
            self.assertEqual(len([n for n in package.namelist() if n.endswith(".npz")]), 4)
            self.assertIn(result["vetor_entrada"]["arquivo_npz"], package.namelist())
            replay = Project(self.root/"exportado")
            package.extractall(replay.root)
        load_experiment(replay, result["execucao_id"], verify=True)
        references = experiment_references(replay, result["execucao_id"])
        center = derive(replay, references)
        self.assertEqual(len(center["origens"]), 6)

    def test_partial_coverage_explicit(self):
        self.gen(["resumo"])
        self.gen(["palavras_chave"], topic_ids=["t1"])
        complete = self.find(["resumo", "palavras_chave"], top_k=None)
        partial = self.find(["resumo", "palavras_chave"], top_k=None, partial=True)
        self.assertEqual(len(complete["resultados"]), 1)
        self.assertEqual(len(partial["resultados"]), 3)
        self.assertEqual(complete["cobertura"]["cobertura_incompleta"], 2)

    def test_segments_preserve_offsets_and_individual_mean(self):
        topic = self.data["documentos"][0]["estrutura"][0]
        topic["resumo"] = "ação ética consentimento "*200
        self.update_source()
        self.gen(["resumo"])
        manifest, _, _ = self.project.load_index("pf_teste", "resumo")
        parts = [r for r in manifest["itens"] if r["topico_id"]=="t1"]
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(r["texto_segmento"] for r in parts), topic["resumo"].strip())
        result = self.find(["resumo"], top_k=None)
        row = next(r for r in result["resultados"] if r["topico_id"]=="t1")
        self.assertAlmostEqual(row["similaridade_media"], np.mean([r["similaridade_cosseno"] for r in row["comparacoes"]]))

    def test_missing_only_does_not_replace_stale(self):
        self.gen(["resumo"])
        self.data["documentos"][0]["estrutura"][0]["resumo"] = "Novo"
        self.update_source()
        pointer = self.project.current("pf_teste", "resumo")
        self.gen(["resumo"], operation="faltantes")
        self.assertEqual(pointer, self.project.current("pf_teste", "resumo"))

    def test_simulation_does_not_call_provider(self):
        self.gen(["resumo"], simulate=True)
        self.assertEqual(self.provider.calls, [])
        self.assertIsNone(self.project.current("pf_teste", "resumo"))

    def test_empty_field_is_not_embedded_and_old_snapshot_retained(self):
        self.gen(["resumo"])
        old = self.project.load_index("pf_teste", "resumo")[0]
        self.data["documentos"][0]["estrutura"][0]["resumo"] = ""
        self.update_source()
        self.gen(["resumo"])
        new = self.project.load_index("pf_teste", "resumo")[0]
        self.assertEqual(len(new["itens"]), 2)
        self.assertTrue(self.project.path(old["arquivo_npz"]).exists())

    def test_derived_vectors_and_pairwise_provenance(self):
        self.gen(["resumo"])
        result = self.find(["resumo"], top_k=None)
        refs = experiment_references(self.project, result["execucao_id"])
        center = derive(self.project, refs)
        self.assertEqual(len(center["origens"]), 3)
        self.assertTrue(self.project.path(center["vetor"]["arquivo_npz"]).exists())
        self.assertAlmostEqual(compare(self.project, refs[0], refs[0])["similaridade_cosseno"], 1)
        pair = pairwise(self.project, refs)
        with np.load(self.project.path(pair["arquivo_npz"]), allow_pickle=False) as package:
            np.testing.assert_allclose(np.diag(package["similaridades"]), 1)
        other = deepcopy(refs[0])
        other["perfil_id"] = "outro"
        with self.assertRaises(ValueError):
            derive(self.project, [refs[0], other])

    def test_corrupt_npz_detected_before_search(self):
        self.gen(["resumo"])
        manifest = self.project.load_index("pf_teste", "resumo")[0]
        self.project.path(manifest["arquivo_npz"]).write_bytes(b"corrompido")
        with self.assertRaises(ValueError):
            self.find(["resumo"])

    def test_duplicate_topics_rejected_before_base_changes(self):
        previous = self.project.base_path.read_bytes()
        self.data["documentos"][0]["estrutura"][1]["topico_id"] = "t1"
        write_json(self.source, self.data)
        with self.assertRaises(ValueError):
            import_base(self.project, self.source)
        self.assertEqual(previous, self.project.base_path.read_bytes())

    def test_metadata_only_import_preserves_topics_and_vectors(self):
        self.gen(["resumo", "metadados"])
        pointer = self.project.current("pf_teste", "resumo")
        self.data["documentos"][0]["estrutura"] = []
        self.data["documentos"][0]["nome_documento"] = "Novo nome"
        self.update_source()
        self.assertEqual(len(self.project.base()["documentos"][0]["estrutura"]), 3)
        self.assertEqual(pointer, self.project.current("pf_teste", "resumo"))
        states = {row["categoria"]: row for row in inventory(self.project)}
        self.assertEqual(states["resumo"]["atuais"], 3)
        self.assertEqual(states["metadados"]["desatualizados"], 3)

    def test_invalid_legacy_rejected_without_replacing_base(self):
        previous = self.project.base_path.read_bytes()
        path = self.root/"legado.npz"
        write_npz(path, vetores=np.zeros((3, 3)), topico_ids=np.asarray(["t1", "t2", "t3"]),
                  modelo=np.asarray(["text-embedding-3-small"]), dimensoes=np.asarray([3]))
        with self.assertRaises(ValueError):
            import_base(self.project, self.source, path)
        self.assertEqual(previous, self.project.base_path.read_bytes())

    def test_legacy_npz_transferred_without_reembedding(self):
        path = self.root/"legado.npz"
        matrix = normalize(np.asarray([[1, 2, 3], [3, 1, 2], [2, 3, 1]], dtype=float))
        hashes = [digest(full_topic(t)) for t in self.data["documentos"][0]["estrutura"]]
        write_npz(path, vetores=matrix, topico_ids=np.asarray(["t1", "t2", "t3"]), hashes_texto=np.asarray(hashes),
                  modelo=np.asarray(["text-embedding-3-small"]), dimensoes=np.asarray([3]))
        import_base(self.project, self.source, path)
        profile = legacy_profile("text-embedding-3-small", 3)
        manifest, stored, original = self.project.load_index(profile["perfil_id"], CATEGORIES[0])
        np.testing.assert_allclose(stored, matrix)
        self.assertIsNone(original)
        self.assertEqual([r["hash_texto_fonte"] for r in manifest["itens"]], hashes)


class RankingTests(unittest.TestCase):
    def rows(self):
        def row(topic, category, score):
            return {"topico_id": topic, "documento_id": "doc", "titulo": topic, "categoria": category,
                    "embedding_id": topic+category, "ordem": [1, 1], "fonte": {},
                    "similaridade_cosseno": score, "distancia_cosseno": 1-score,
                    "distancia_euclidiana_normalizada": float(np.sqrt(2-2*score))}
        return [row("t1", "resumo", .9), row("t1", "conteudo_fonte", .1),
                row("t2", "resumo", .65), row("t2", "conteudo_fonte", .65)]

    def test_topic_mean_used_before_top_k(self):
        rows, _ = rank_results(self.rows(), ["resumo", "conteudo_fonte"], top_k=1)
        self.assertEqual(rows[0]["topico_id"], "t2")
        self.assertAlmostEqual(rows[0]["similaridade_media"], .65)

    def test_vector_top_k_per_category(self):
        rows, _ = rank_results(self.rows(), ["resumo", "conteudo_fonte"], "vetores", 1)
        self.assertEqual([(r["categoria"], r["topico_id"]) for r in rows], [("resumo", "t1"), ("conteudo_fonte", "t2")])

    def test_individual_threshold_and_mean_threshold(self):
        rows, _ = rank_results(self.rows(), ["resumo", "conteudo_fonte"], minimum=.6,
                               individual_limits={"resumo": .8})
        self.assertEqual(rows, [])
        rows, _ = rank_results(self.rows(), ["resumo", "conteudo_fonte"], top_k=None,
                               individual_limits={"resumo": .8, "conteudo_fonte": .6}, individual_rule="qualquer")
        self.assertEqual(len(rows), 2)

    def test_limits_validate_nan_infinite_and_invalid_k(self):
        for kwargs in [{"top_k": 0}, {"top_k": True}, {"minimum": float("nan")},
                       {"maximum": float("inf")}, {"minimum": .8, "maximum": .1}]:
            with self.assertRaises(ValueError):
                validate_limits(**kwargs)

    def test_split_preserves_unicode_exactly(self):
        value = "ética em pesquisa, ação e compreensão. "*20
        parts = list(split_text(value, 90))
        self.assertEqual("".join(t for t, _, _ in parts), value)
        for part, left, right in parts:
            self.assertLessEqual(len(part.encode("utf-8")), 90)
            self.assertEqual(part, value[left:right])


class ProviderProtocolTests(unittest.TestCase):
    def test_gemma_reduction_is_explicit_and_bge_cannot_crop(self):
        profile = {"modelo": "embeddinggemma:300m-qat-q4_0", "endpoint": "http://local",
                   "revisao_modelo": "abc", "dimensoes": 256, "reducao": "matryoshka"}
        with patch("autorag.providers.http_json", return_value={"models": [{"name": profile["modelo"], "digest": "abc"}]}):
            backend = OllamaProvider(profile)
        self.assertEqual(backend._dimension(np.ones((2, 768)), 2).shape, (2, 256))
        backend.profile["reducao"] = "nenhuma"
        with self.assertRaises(ValueError):
            backend._dimension(np.ones((2, 768)), 2)

    def test_gemini_embedding2_never_aggregates_documents_implicitly(self):
        profile = {"modelo": "gemini-embedding-2", "dimensoes": 3}
        with patch("autorag.providers.obter_chave", return_value="chave_de_teste"), patch("autorag.providers.http_json", return_value={"embedding": {"values": [1, 2, 3]}}) as request:
            output = GeminiProvider(profile).embed(["texto1", "texto2"])
        self.assertEqual(output.shape, (2, 3))
        self.assertEqual(request.call_count, 2)
        self.assertEqual(len(request.call_args.args[1]["content"]["parts"]), 1)

    def test_e5_role_prefix_and_gemma_role_prefix(self):
        self.assertEqual(Provider({"modelo": "intfloat/multilingual-e5-small"}).prepare("ação", "consulta"), "query: ação")
        self.assertEqual(Provider({"modelo": "embeddinggemma"}).prepare("ação", "documento"), "title: none | text: ação")


if __name__ == "__main__":
    unittest.main()
