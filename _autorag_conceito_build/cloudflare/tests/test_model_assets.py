import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from cloudflare.download_models import blob_path, download_blob, fetch_manifest, verify_installation
from cloudflare.install_ollama import cpu_member, extract_cpu


class Response(io.BytesIO):
    pass


def fixture():
    contents = {"config": b'{"model_format":"gguf"}', "model": b'GGUF test weights', "license": b'MIT retained license'}
    def record(name):
        data = contents[name]
        media = "application/vnd.ollama.image." + name
        return {"mediaType": media, "digest": "sha256:" + hashlib.sha256(data).hexdigest(), "size": len(data)}
    manifest = {"schemaVersion": 2, "config": record("config"), "layers": [record("model"), record("license")]}
    encoded = json.dumps(manifest).encode()
    model = {"name": "test", "repository": "library/test", "tag": "latest", "manifest_sha256": hashlib.sha256(encoded).hexdigest(), "model_bytes": len(contents["model"])}
    lock = {"registry": "https://registry.ollama.ai", "models": [model]}
    return contents, manifest, encoded, model, lock


class ModelAssetTests(unittest.TestCase):
    def test_manifest_digest_must_match_audited_local_bytes(self):
        contents, manifest, encoded, model, lock = fixture()
        actual, _ = fetch_manifest(lock, model, lambda url: Response(encoded))
        self.assertEqual(actual, encoded)
        with self.assertRaises(ValueError):
            fetch_manifest(lock, model, lambda url: Response(encoded + b' '))

    def test_corrupt_blob_is_never_published(self):
        contents, manifest, encoded, model, lock = fixture()
        with tempfile.TemporaryDirectory() as directory:
            record = manifest["layers"][0]
            with self.assertRaises(ValueError):
                download_blob(lock, model, record, directory, lambda url: Response(b'wrong weights'))
            self.assertFalse(blob_path(directory, record["digest"]).exists())
            self.assertFalse(list(Path(directory).rglob("*.part")))

    def test_offline_verification_includes_original_license_blob(self):
        contents, manifest, encoded, model, lock = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(lock))
            destination = root / "manifests/registry.ollama.ai/library/test/latest"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(encoded)
            records = [manifest["config"], *manifest["layers"]]
            for name, record in zip(("config", "model", "license"), records):
                path = blob_path(root, record["digest"])
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(contents[name])
            verify_installation(root, lock_path, full=True)
            blob_path(root, manifest["layers"][1]["digest"]).write_bytes(b'altered license')
            with self.assertRaises(ValueError):
                verify_installation(root, lock_path, full=True)

    def test_gpu_runners_and_unsafe_paths_are_excluded(self):
        for name in ("lib/ollama/cuda_v12/libggml-cuda.so", "lib/ollama/libggml-vulkan.so", "lib/ollama/mlx/lib.so", "../bin/ollama", "/bin/ollama", "lib/ollama/../../secret"):
            with self.subTest(name=name):
                self.assertIsNone(cpu_member(name))
        self.assertIsNotNone(cpu_member("bin/ollama"))
        self.assertIsNotNone(cpu_member("lib/ollama/libggml-cpu-haswell.so"))
        self.assertIsNotNone(cpu_member("lib/ollama/libmtmd.so"))
        self.assertIsNotNone(cpu_member("lib/ollama/GO_LICENSE"))

    def test_cpu_extract_retains_licenses_without_gpu_payloads(self):
        package = io.BytesIO()
        names = ["bin/ollama", "lib/ollama/llama-server", "lib/ollama/libggml-base.so", "lib/ollama/libllama.so", "lib/ollama/libggml-cpu-haswell.so", "lib/ollama/GO_LICENSE", "lib/ollama/cuda_v12/large.so"]
        with tarfile.open(fileobj=package, mode="w") as archive:
            for name in names:
                data = b'license' if "LICENSE" in name else b'binary'
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        package.seek(0)
        with tempfile.TemporaryDirectory() as directory, tarfile.open(fileobj=package, mode="r|") as archive:
            installed = extract_cpu(archive, directory)
            self.assertIn("lib/ollama/GO_LICENSE", installed)
            self.assertFalse((Path(directory) / "lib/ollama/cuda_v12").exists())
            if os.name != "nt":
                self.assertEqual((Path(directory) / "bin/ollama").stat().st_mode & 0o111, 0o111)

    @unittest.skipIf(os.name == "nt", "The deployment image uses Linux library symlinks.")
    def test_library_symlink_chains_do_not_depend_on_archive_order(self):
        package = io.BytesIO()
        names = ["bin/ollama", "lib/ollama/llama-server", "lib/ollama/libggml-base.so", "lib/ollama/libllama.so", "lib/ollama/libggml-cpu-haswell.so", "lib/ollama/libgomp.so.1.2"]
        with tarfile.open(fileobj=package, mode="w") as archive:
            for name, target in [("lib/ollama/libgomp.so", "libgomp.so.1"), ("lib/ollama/libgomp.so.1", "libgomp.so.1.2")]:
                member = tarfile.TarInfo(name)
                member.type = tarfile.SYMTYPE
                member.linkname = target
                archive.addfile(member)
            for name in names:
                member = tarfile.TarInfo(name)
                member.size = 3
                archive.addfile(member, io.BytesIO(b'cpu'))
        package.seek(0)
        with tempfile.TemporaryDirectory() as directory, tarfile.open(fileobj=package, mode="r|") as archive:
            extract_cpu(archive, directory)
            self.assertEqual((Path(directory) / "lib/ollama/libgomp.so").read_bytes(), b'cpu')


if __name__ == "__main__":
    unittest.main()
