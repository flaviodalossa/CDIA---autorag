"""Native index transfers preserve data and reject untrusted ZIP/NPZ payloads."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

from autorag.core import Project, category_text, digest, write_json
from autorag.index_transfer import (export_index_bundle, import_index_bundle,
                                   validate_index_bundle)
from autorag.providers import Provider
from autorag.service import generate


class OfflineProvider(Provider):
    def embed(self, texts, role="documento"):
        return np.asarray([np.arange(1, self.profile["dimensoes"] + 1, dtype=float)
                           + int(digest(value)[:4], 16) / 65536 for value in texts])


def unpack(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def pack(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return stream.getvalue()


class IndexTransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = Project(self.root / "source")
        self.target = Project(self.root / "target")
        self.profile = {"perfil_id": "pf_native", "provedor": "ollama",
                        "modelo": "embeddinggemma:300m-qat-q4_0", "dimensoes": 128,
                        "dimensoes_nativas": 768, "reducao": "matryoshka",
                        "limite_tokens": 96, "revisao_modelo": "a" * 64,
                        "normalizacao": "l2", "versao_preparacao": "1",
                        "segmentacao": "utf8_conservador_v1",
                        "endpoint": "http://desktop-private.example:11434"}
        self.category = "conteudo_fonte"
        base = {"schema_version": "3.0", "controle_embeddings": {"revisao_base": "base_fixture"},
                "documentos": [{"documento_id": "doc1", "nome_documento": "Documento",
                    "metadados": {"url": "https://example.org/norma", "idioma": "português"},
                    "estrutura": [{"topico_id": "t1", "ativo": True, "titulo": "Árvore",
                                   "conteudo_fonte": "Árvore e informação íntegra. " * 5},
                                  {"topico_id": "t2", "ativo": True, "titulo": "Segundo",
                                   "conteudo_fonte": "Outro artigo preservado."},
                                  {"topico_id": "empty", "ativo": True, "conteudo_fonte": ""},
                                  {"topico_id": "inactive", "ativo": False,
                                   "conteudo_fonte": "Dispositivo revogado."}]}]}
        for project in (self.source, self.target):
            write_json(project.base_path, deepcopy(base))
            profile = deepcopy(self.profile)
            if project is self.target:
                profile["endpoint"] = "http://127.0.0.1:11434"
            other = {**deepcopy(profile), "perfil_id": "pf_other", "modelo": "other"}
            project.save_config({"perfis": {"pf_native": profile, "pf_other": other},
                                 "perfil_ativo_id": "pf_other"})
        generate(self.source, "pf_native", [self.category], provider=OfflineProvider(self.profile))
        self.bundle = export_index_bundle(self.source, "pf_native", self.category)

    def tearDown(self):
        self.temporary.cleanup()

    def snapshot(self):
        return {path.relative_to(self.target.root).as_posix(): path.read_bytes()
                for path in self.target.root.rglob("*") if path.is_file()}

    def reject(self, data):
        before = self.snapshot()
        with self.assertRaises(ValueError) as error:
            import_index_bundle(self.target, data)
        self.assertNotIn("desktop-private", str(error.exception))
        self.assertEqual(self.snapshot(), before)

    def alter_manifest(self, function):
        files = unpack(self.bundle)
        manifest = json.loads(files["manifesto.json"])
        function(manifest)
        files["manifesto.json"] = json.dumps(manifest, ensure_ascii=False).encode()
        return pack(files)

    def alter_npz(self, function):
        files = unpack(self.bundle)
        with np.load(io.BytesIO(files["vetores.npz"]), allow_pickle=False) as package:
            arrays = {name: package[name].copy() for name in package.files}
        function(arrays)
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        files["vetores.npz"] = stream.getvalue()
        manifest = json.loads(files["manifesto.json"])
        manifest["npz_sha256"] = hashlib.sha256(files["vetores.npz"]).hexdigest()
        files["manifesto.json"] = json.dumps(manifest).encode()
        return pack(files)

    def test_roundtrip_revalidates_preserves_content_config_and_active_profile(self):
        before_base = deepcopy(self.target.base())
        before_config = self.target.config_path.read_bytes()
        before_files = self.snapshot()
        with patch("autorag.providers.obter_chave", side_effect=AssertionError("credential lookup")), \
             patch("autorag.providers.get_provider", side_effect=AssertionError("provider discovery")), \
             patch("autorag.providers.http_json", side_effect=AssertionError("network")):
            checked = validate_index_bundle(self.target, self.bundle)
            self.assertEqual(self.snapshot(), before_files)
            result = import_index_bundle(self.target, self.bundle)
        self.assertEqual(result["topic_ids"], ["t1", "t2"])
        self.assertEqual(result["topics"], 2)
        self.assertGreater(result["vectors"], result["topics"])
        self.assertEqual(result["embedding_ids"], checked["embedding_ids"])
        self.assertEqual(self.target.config_path.read_bytes(), before_config)
        after_base = self.target.base()
        for old_doc, new_doc in zip(before_base["documentos"], after_base["documentos"]):
            self.assertEqual(old_doc["metadados"], new_doc["metadados"])
            for old, new in zip(old_doc["estrutura"], new_doc["estrutura"]):
                self.assertEqual(old, {key: value for key, value in new.items() if key != "embeddings"})
        src_manifest, src_matrix, src_originals = self.source.load_index("pf_native", self.category)
        dst_manifest, dst_matrix, dst_originals = self.target.load_index("pf_native", self.category)
        self.assertNotEqual(src_manifest["revisao_indice"], dst_manifest["revisao_indice"])
        np.testing.assert_allclose(src_matrix, dst_matrix)
        np.testing.assert_array_equal(src_originals, dst_originals)
        self.assertEqual(self.target.config()["perfil_ativo_id"], "pf_other")
        for number, record in enumerate(dst_manifest["itens"]):
            self.assertEqual(record["referencia"]["linha"], number)
            self.assertEqual(record["referencia"]["revisao_indice"], result["revision"])

    def test_export_omits_endpoints_paths_and_transfers_only_two_selector_fields(self):
        files = unpack(self.bundle)
        self.assertEqual(set(files), {"transfer.json", "manifesto.json", "vetores.npz"})
        self.assertEqual(json.loads(files["transfer.json"]), {"profile_id": "pf_native", "category": self.category})
        manifest = json.loads(files["manifesto.json"])
        self.assertNotIn("endpoint", manifest["perfil"])
        self.assertNotIn("caminho_local", manifest["perfil"])
        self.assertNotIn(b"desktop-private", files["manifesto.json"])
        self.assertEqual(manifest["perfil"]["revisao_modelo"], "a" * 64)
        self.assertEqual(manifest["perfil"]["dimensoes_nativas"], 768)
        self.assertEqual(manifest["perfil"]["reducao"], "matryoshka")

    def test_profile_identity_cannot_be_created_switched_or_downgraded(self):
        for field, value in (("modelo", "other"), ("provedor", "openai"), ("dimensoes", 256),
                             ("revisao_modelo", "b" * 64), ("normalizacao", "none"),
                             ("versao_preparacao", "2"), ("segmentacao", "other"),
                             ("limite_tokens", 512), ("reducao", "none"), ("legado", True)):
            with self.subTest(field=field):
                self.reject(self.alter_manifest(lambda manifest: manifest["perfil"].__setitem__(field, value)))
        files = unpack(self.bundle)
        files["transfer.json"] = b'{"profile_id":"unregistered","category":"conteudo_fonte"}'
        self.reject(pack(files))
        self.reject(self.alter_manifest(lambda manifest: manifest["perfil"].__setitem__("endpoint", "http://attacker")))

    def test_corruption_ids_dimensions_and_invalid_coordinates_rejected(self):
        files = unpack(self.bundle)
        files["vetores.npz"] += b"corruption"
        self.reject(pack(files))
        mutations = (
            lambda arrays: arrays["embedding_ids"].__setitem__(0, arrays["embedding_ids"][1]),
            lambda arrays: arrays.__setitem__("embedding_ids", arrays["embedding_ids"][::-1]),
            lambda arrays: arrays.__setitem__("vetores", arrays["vetores"][:, :-1]),
            lambda arrays: arrays["vetores"].__setitem__((0, 0), np.nan),
            lambda arrays: arrays["vetores"].__setitem__(0, 0),
            lambda arrays: arrays["vetores_originais"].__setitem__((0, 0), np.inf),
            lambda arrays: arrays.__setitem__("embedding_ids", np.asarray(["unsafe"] * len(arrays["embedding_ids"]), dtype=object)),
        )
        for change in mutations:
            with self.subTest(change=change):
                self.reject(self.alter_npz(change))

    def test_inactive_unknown_topic_wrong_document_and_stale_hash_rejected(self):
        for field, value in (("topico_id", "inactive"), ("topico_id", "missing"),
                             ("documento_id", "wrong"), ("hash_texto_fonte", "0" * 64),
                             ("hash_texto_embeddado", "0" * 64), ("categoria", "resumo")):
            with self.subTest(field=field, value=value):
                self.reject(self.alter_manifest(lambda manifest: manifest["itens"][0].__setitem__(field, value)))
        before = self.target.base()
        before["documentos"][0]["estrutura"][0]["ativo"] = False
        write_json(self.target.base_path, before)
        self.reject(self.bundle)

    def test_segments_offsets_preparation_and_completeness_are_checked(self):
        for field, value in (("inicio", 1), ("fim", 1), ("segmento", True),
                             ("texto_segmento", "Texto trocado")):
            with self.subTest(field=field):
                self.reject(self.alter_manifest(lambda manifest: manifest["itens"][0].__setitem__(field, value)))
        files = unpack(self.bundle)
        manifest = json.loads(files["manifesto.json"])
        with np.load(io.BytesIO(files["vetores.npz"]), allow_pickle=False) as package:
            arrays = {name: package[name][1:] for name in package.files}
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        files["vetores.npz"] = stream.getvalue()
        manifest["itens"] = manifest["itens"][1:]
        manifest["total_embeddings"] -= 1
        manifest["npz_sha256"] = hashlib.sha256(files["vetores.npz"]).hexdigest()
        files["manifesto.json"] = json.dumps(manifest).encode()
        self.reject(pack(files))

    def test_zip_traversal_extra_duplicate_symlink_encryption_and_limits_rejected(self):
        files = unpack(self.bundle)
        for name in ("../manifesto.json", "/manifesto.json", "folder/manifesto.json", "folder\\manifesto.json"):
            changed = dict(files)
            changed[name] = changed.pop("manifesto.json")
            self.reject(pack(changed))
        self.reject(pack({**files, "extra.txt": b"extra"}))
        stream = io.BytesIO()
        with warnings.catch_warnings(), zipfile.ZipFile(stream, "w") as archive:
            warnings.simplefilter("ignore", UserWarning)
            for name, value in files.items():
                archive.writestr(name, value)
            archive.writestr("manifesto.json", files["manifesto.json"])
        self.reject(stream.getvalue())
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            for name, value in files.items():
                info = zipfile.ZipInfo(name)
                if name == "manifesto.json":
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, value)
        self.reject(stream.getvalue())
        encrypted = bytearray(self.bundle)
        central = encrypted.index(b"PK\x01\x02")
        encrypted[central + 8:central + 10] = struct.pack("<H", 1)
        self.reject(bytes(encrypted))
        with patch("autorag.index_transfer.MAX_ZIP_BYTES", len(self.bundle) - 1):
            self.reject(self.bundle)
        with patch("autorag.index_transfer.MAX_UNCOMPRESSED_BYTES", 100):
            self.reject(self.bundle)

    def test_malicious_nested_npy_shape_cannot_allocate_and_json_duplicates_rejected(self):
        files = unpack(self.bundle)
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False,
                                                    "shape": (2**40, 128)})
        nested = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(files["vetores.npz"])) as old, zipfile.ZipFile(nested, "w") as archive:
            for name in old.namelist():
                archive.writestr(name, header.getvalue() if name == "vetores.npy" else old.read(name))
        files["vetores.npz"] = nested.getvalue()
        manifest = json.loads(files["manifesto.json"])
        manifest["npz_sha256"] = hashlib.sha256(files["vetores.npz"]).hexdigest()
        files["manifesto.json"] = json.dumps(manifest).encode()
        with patch("autorag.index_transfer.np.load") as loader:
            self.reject(pack(files))
            loader.assert_not_called()
        files = unpack(self.bundle)
        files["transfer.json"] = b'{"profile_id":"pf_native","profile_id":"pf_native","category":"conteudo_fonte"}'
        self.reject(pack(files))

    def test_partial_topic_coverage_cannot_replace_a_complete_category(self):
        import_index_bundle(self.target, self.bundle)
        files = unpack(self.bundle)
        manifest = json.loads(files["manifesto.json"])
        keep = [index for index, row in enumerate(manifest["itens"]) if row["topico_id"] == "t1"]
        manifest["itens"] = [manifest["itens"][index] for index in keep]
        manifest["total_embeddings"] = len(keep)
        with np.load(io.BytesIO(files["vetores.npz"]), allow_pickle=False) as package:
            arrays = {name: package[name][keep] for name in package.files}
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        files["vetores.npz"] = stream.getvalue()
        manifest["npz_sha256"] = hashlib.sha256(files["vetores.npz"]).hexdigest()
        files["manifesto.json"] = json.dumps(manifest).encode()
        self.reject(pack(files))

    def test_empty_category_is_valid_only_when_no_active_topic_has_text(self):
        generate(self.source, "pf_native", ["resumo"], provider=OfflineProvider(self.profile))
        data = export_index_bundle(self.source, "pf_native", "resumo")
        result = import_index_bundle(self.target, data)
        self.assertEqual(result["topics"], 0)
        self.assertEqual(result["vectors"], 0)
        self.assertEqual(result["embedding_ids"], [])
        base = self.target.base()
        base["documentos"][0]["estrutura"][0]["resumo"] = "Agora há texto elegível."
        write_json(self.target.base_path, base)
        self.reject(data)

    def test_publish_failure_keeps_old_pointer_and_commit_revalidates(self):
        import_index_bundle(self.target, self.bundle)
        pointer = self.target.current("pf_native", self.category)
        before_base = self.target.base_path.read_bytes()
        with patch("autorag.index_transfer.sync_base", side_effect=OSError("private failure")):
            with self.assertRaises(ValueError):
                import_index_bundle(self.target, self.bundle)
        self.assertEqual(self.target.current("pf_native", self.category), pointer)
        self.assertEqual(self.target.base_path.read_bytes(), before_base)
        validate_index_bundle(self.target, self.bundle)
        base = self.target.base()
        base["documentos"][0]["estrutura"][0]["conteudo_fonte"] += " atualizado"
        write_json(self.target.base_path, base)
        self.reject(self.bundle)


if __name__ == "__main__":
    unittest.main()
