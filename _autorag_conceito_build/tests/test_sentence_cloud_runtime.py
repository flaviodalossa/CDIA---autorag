import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from autorag.providers import SentenceProvider, hf_local_path


class SentenceCloudRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.profile = {'modelo': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                        'dimensoes': 384, 'limite_tokens': 128, 'caminho_local': '/local/model'}
        self.backend = Mock(max_seq_length=128)
        self.backend.tokenizer.encode.side_effect = lambda text, **kwargs: list(range(len(text.split()) + 2))
        self.backend.encode.side_effect = lambda texts, **kwargs: np.ones((len(texts), 384), dtype=np.float32)
        self.constructor = Mock(return_value=self.backend)
        self.fake_modules = {'sentence_transformers': SimpleNamespace(SentenceTransformer=self.constructor),
                             'torch': SimpleNamespace(set_num_threads=Mock())}

    def test_cpu_constraints_preserve_raw_texts_vectors_and_offline_policy(self):
        env = {'AUTORAG_SENTENCE_DEVICE': 'cpu', 'AUTORAG_SENTENCE_NUM_THREADS': '1', 'AUTORAG_SENTENCE_BATCH_SIZE': '1'}
        with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, self.fake_modules):
            provider = SentenceProvider(self.profile)
            values = provider.embed(['Direitos do participante.', 'Participant rights.'], 'consulta')
        self.constructor.assert_called_once_with('/local/model', local_files_only=True, trust_remote_code=False, device='cpu')
        self.fake_modules['torch'].set_num_threads.assert_called_once_with(1)
        self.backend.encode.assert_called_once_with(['Direitos do participante.', 'Participant rights.'],
            normalize_embeddings=False, convert_to_numpy=True, batch_size=1, show_progress_bar=False)
        np.testing.assert_array_equal(values, np.ones((2, 384)))

    def test_desktop_defaults_are_unchanged(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, self.fake_modules):
            SentenceProvider(self.profile).embed(['text'])
        self.constructor.assert_called_once_with('/local/model', local_files_only=True, trust_remote_code=False)
        self.backend.encode.assert_called_once_with(['text'], normalize_embeddings=False, convert_to_numpy=True)
        self.fake_modules['torch'].set_num_threads.assert_not_called()

    def test_over_context_input_fails_before_encode(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, self.fake_modules):
            provider = SentenceProvider(self.profile)
            with self.assertRaisesRegex(ValueError, 'truncamento'):
                provider.embed(['word ' * 129])
        self.backend.encode.assert_not_called()

    def test_invalid_cpu_configuration_fails_before_model_execution(self):
        for name in ('AUTORAG_SENTENCE_NUM_THREADS', 'AUTORAG_SENTENCE_BATCH_SIZE'):
            for value in ('0', '-1', '1.5', '9000'):
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}, clear=True), \
                        patch.dict(sys.modules, self.fake_modules):
                    with self.assertRaises(ValueError):
                        SentenceProvider(self.profile).embed(['text'])
        self.backend.encode.assert_not_called()

    def test_image_model_directory_is_fallback_after_project_directory(self):
        name = self.profile['modelo']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'project'
            image = root / 'image'
            bundled = image / name.replace('/', '--')
            bundled.mkdir(parents=True)
            with patch.dict(os.environ, {'AUTORAG_MODELS_DIR': str(image)}):
                self.assertEqual(hf_local_path(name, project), bundled.resolve())
                local = project / name.replace('/', '--')
                local.mkdir(parents=True)
                self.assertEqual(hf_local_path(name, project), local.resolve())


if __name__ == '__main__':
    unittest.main()
