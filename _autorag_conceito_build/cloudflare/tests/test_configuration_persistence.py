from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from autorag.core import Project, now, read_json, uid, write_json
from cloudflare.adapter import install


class ConfigurationPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Project(self.temp.name)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.failed = False
        self.saved = []
        routes = FastAPI()
        for method, path in [('POST','/api/profiles'),('POST','/api/experiment-configs'),('PUT','/api/consultation-config')]:
            def save_configuration():
                self.saved.append(True)
                return {'saved': True}
            routes.add_api_route(path, save_configuration, methods=[method])
        def checkpoint(project):
            self.entered.set()
            if self.failed:
                raise OSError('storage unavailable')
            if not self.release.wait(5):
                raise TimeoutError('test checkpoint timeout')
        self.api = SimpleNamespace(app=routes, project=self.project, jobs={}, job_lock=threading.Lock(),
            MAX_JOBS=50, uid=uid, now=now, read_json=read_json, write_json=write_json,executor=self.executor)
        self.store = SimpleNamespace(checkpoint=checkpoint,last_checkpoint=None,save_job=lambda record: None)
        self.client = TestClient(install(self.api,self.store))
        self.env = patch.dict(os.environ,{'AUTORAG_API_TOKEN':'test-configuration-token-32-characters'})
        self.env.start()
        self.headers = {'Authorization':'Bearer test-configuration-token-32-characters'}
    def tearDown(self):
        self.release.set()
        self.executor.shutdown(wait=True)
        self.client.close()
        self.env.stop()
        self.temp.cleanup()

    def test_each_configuration_response_waits_for_durable_checkpoint(self):
        for method,path in [('POST','/api/profiles'),('POST','/api/experiment-configs'),('PUT','/api/consultation-config')]:
            with self.subTest(path=path), ThreadPoolExecutor(max_workers=1) as pool:
                self.entered.clear()
                self.release.clear()
                future = pool.submit(self.client.request,method,path,headers=self.headers,json={})
                self.assertTrue(self.entered.wait(5))
                self.assertFalse(future.done())
                self.release.set()
                self.assertEqual(future.result(timeout=5).status_code,200)

    def test_storage_failure_never_reports_a_saved_configuration(self):
        self.failed = True
        for method,path in [('POST','/api/experiment-configs'),('PUT','/api/consultation-config')]:
            with self.subTest(path=path):
                response = self.client.request(method,path,headers=self.headers,json={})
                self.assertEqual(response.status_code,503)
                self.assertIn('não foi confirmada',response.json()['detail'])


if __name__ == '__main__':
    unittest.main()
