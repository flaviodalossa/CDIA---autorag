from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from fastapi import HTTPException
from autorag.core import Project, now, read_json, uid, write_json
from cloudflare.adapter import install


class Store:
    last_checkpoint = None
    def __init__(self):
        self.records = {}
        self.checkpoint_entered = threading.Event()
        self.release_checkpoint = threading.Event()
        self.fail_checkpoint = False
        self.fail_job = False
    def save_job(self, record):
        if self.fail_job:
            raise OSError("R2 offline")
        self.records[record["job_id"]] = deepcopy(record)
    def checkpoint(self, project):
        self.checkpoint_entered.set()
        if self.fail_checkpoint:
            raise OSError("R2 offline")
        if not self.release_checkpoint.wait(10):
            raise TimeoutError("test checkpoint timeout")
        self.last_checkpoint = now()


class DurableJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.api = SimpleNamespace(project=Project(self.temp.name), job_lock=threading.Lock(), jobs={},
            MAX_JOBS=50, uid=uid, now=now, read_json=read_json, write_json=write_json,
            executor=self.executor, app=None)
        self.store = Store()
        self.app = install(self.api, self.store)
    def tearDown(self):
        self.store.release_checkpoint.set()
        self.executor.shutdown(wait=True)
        self.temp.cleanup()
    def wait_for(self, predicate):
        deadline = time.monotonic() + 10
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(predicate())

    def test_success_published_only_after_durable_checkpoint(self):
        response = self.api.submit("search", lambda: {"answer": 42})
        job_id = json.loads(response.body)["job_id"]
        self.assertEqual(response.status_code, 202)
        self.assertIn(job_id, self.store.records)
        self.assertTrue(self.store.checkpoint_entered.wait(10))
        self.assertEqual(self.api.jobs[job_id]["status"], "running")
        self.assertEqual(self.store.records[job_id]["status"], "running")
        self.store.release_checkpoint.set()
        self.wait_for(lambda: job_id not in self.api.jobs)
        durable = self.store.records[job_id]
        self.assertEqual(durable["status"], "succeeded")
        self.assertIn("finished_at", durable)
        self.assertEqual(read_json(self.api.project.root / "jobs" / (job_id + ".json")), durable)

    def test_checkpoint_failure_never_reports_success(self):
        self.store.fail_checkpoint = True
        response = self.api.submit("ingest", lambda: {"documents": 1})
        job_id = json.loads(response.body)["job_id"]
        self.wait_for(lambda: self.api.jobs[job_id]["status"] == "failed")
        self.assertIn("persistence_warning", self.api.jobs[job_id])
        self.assertNotEqual(self.store.records[job_id]["status"], "succeeded")

    def test_queue_requires_r2_acknowledgement(self):
        self.store.fail_job = True
        called = threading.Event()
        with self.assertRaises(HTTPException) as failure:
            self.api.submit("generate", called.set)
        self.assertEqual(failure.exception.status_code, 503)
        self.assertFalse(called.is_set())
        self.assertFalse(self.api.jobs)


if __name__ == "__main__":
    unittest.main()
