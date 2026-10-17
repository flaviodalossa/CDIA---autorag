import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
import zipfile

from cloudflare.snapshot import R2State, allowed_member, archive_state, extract_state


class Response:
    def __init__(self, data=b"", headers=None):
        self.data, self.headers = data, headers or {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self, size):
        return self.data[:size]


class SnapshotTests(unittest.TestCase):
    def test_snapshot_round_trip_excludes_secrets_and_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in [("dados/base.json", b'{}'), ("indices/p/v.npz", b'vectors'),
                                  ("dados/.escrita.lock", b'pid'), ("modelos/private", b'model'),
                                  (".env.local", b'secret'), ("jobs/job_a.json", b'{}')]:
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(content)
            content = archive_state(root)
            with zipfile.ZipFile(io.BytesIO(content)) as package:
                self.assertEqual(set(package.namelist()), {"dados/base.json", "indices/p/v.npz"})
            (root / "dados/stale.json").write_text("old")
            extract_state(root, content)
            self.assertFalse((root / "dados/.escrita.lock").exists())
            self.assertFalse((root / "dados/stale.json").exists())
            self.assertEqual((root / "indices/p/v.npz").read_bytes(), b'vectors')
            self.assertEqual((root / ".env.local").read_bytes(), b'secret')

    def test_rejects_unsafe_paths_without_removing_previous_data(self):
        invalid = ["/dados/a", "dados/../a", "dados//a", "dados/./a", "dados/a:b", "dados/a\\b", "jobs/job_x.json", "dados/.secret"]
        for name in invalid:
            with self.subTest(name=name):
                self.assertFalse(allowed_member(name))
                # Windows ZipInfo normalizes its native separator when reading.
                if "\\" in name and os.sep == "\\":
                    continue
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "dados").mkdir()
                    (root / "dados/keep").write_text("existing")
                    package = io.BytesIO()
                    with zipfile.ZipFile(package, "w") as archive:
                        info = zipfile.ZipInfo("safe")
                        info.filename = name
                        archive.writestr(info, "unexpected")
                    with self.assertRaises(ValueError):
                        extract_state(root, package.getvalue())
                    self.assertTrue((root / "dados/keep").exists())

    def test_restore_loads_independent_job_records(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            root = Path(source)
            (root / "dados").mkdir()
            (root / "dados/base.json").write_text("{}")
            data = {"/snapshot": archive_state(root), "/jobs": b'{"keys":["job_test.json"],"cursor":null}',
                    "/jobs/job_test.json": b'{"job_id":"job_test","status":"queued"}'}
            def opener(request, timeout):
                return Response(data[request.full_url.removeprefix("http://storage.autorag")])
            state = R2State(target, opener=opener)
            state.restore()
            self.assertEqual(json.loads((Path(target) / "jobs/job_test.json").read_text())["status"], "queued")
            self.assertTrue((Path(target) / "dados/base.json").exists())

    def test_rejects_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as target:
            state = R2State(target, opener=lambda request, timeout: Response(b"corrupt", {"X-Content-SHA256": "0" * 64}))
            with self.assertRaises(OSError):
                state.request("/snapshot")


if __name__ == "__main__":
    unittest.main()
