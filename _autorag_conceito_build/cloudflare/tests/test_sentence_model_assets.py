import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from cloudflare.download_sentence_models import (
    REQUIRED, download_asset, install_models, load_models, model_path,
    verify_installation, verify_settings,
)
from scripts.package_source import source_files


def fixture(root):
    contents = {path: b'original asset' for path in REQUIRED}
    configs = {
        'config.json': {'model_type': 'bert', 'hidden_size': 384},
        'sentence_bert_config.json': {'max_seq_length': 128},
        '1_Pooling/config.json': {'word_embedding_dimension': 384, 'pooling_mode_mean_tokens': True},
        'modules.json': [
            {'path': '', 'type': 'sentence_transformers.models.Transformer'},
            {'path': '1_Pooling', 'type': 'sentence_transformers.models.Pooling'},
        ],
    }
    contents.update({path: json.dumps(value).encode() for path, value in configs.items()})
    model = {
        'name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'repository': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'revision': 'a' * 40, 'license': 'Apache-2.0', 'dimensions': 384, 'max_seq_length': 128,
        'files': [{'path': path, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                  for path, data in sorted(contents.items())],
    }
    lock_path = root / 'lock.json'
    lock_path.write_text(json.dumps({'sentence_transformers': [model]}), encoding='utf-8')
    return model, contents, lock_path


class SentenceModelAssetTests(unittest.TestCase):
    def test_install_is_offline_reusable_and_licenses_are_outside_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, contents, lock_path = fixture(root)
            calls = []
            def opener(url):
                calls.append(url)
                prefix = 'https://huggingface.co/' + model['repository'] + '/resolve/' + model['revision'] + '/'
                self.assertTrue(url.startswith(prefix))
                return io.BytesIO(contents[url.removeprefix(prefix)])
            install_models(root, lock_path, opener)
            self.assertEqual(len(calls), len(REQUIRED))
            self.assertIn(b'END OF TERMS AND CONDITIONS', (root / 'licenses/APACHE-2.0.txt').read_bytes())
            folder = model_path(root, model)
            self.assertTrue((folder / 'README.md').is_file())
            self.assertFalse((folder / 'NOTICE.txt').exists())
            install_models(root, lock_path, lambda url: self.fail('Valid assets must not be downloaded again'))
            verify_installation(root, lock_path, full=True)

    def test_corrupt_or_oversized_download_never_replaces_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, contents, _ = fixture(root)
            record = next(row for row in model['files'] if row['path'] == 'model.safetensors')
            destination = model_path(root, model) / record['path']
            destination.parent.mkdir()
            destination.write_bytes(b'previous file')
            for data in (b'x' * record['size'], b'x' * (record['size'] + 1)):
                with self.subTest(size=len(data)), self.assertRaises(ValueError):
                    download_asset(root, model, record, lambda url: io.BytesIO(data))
                self.assertEqual(destination.read_bytes(), b'previous file')
                self.assertFalse(list(root.rglob('*.part')))

    def test_lock_rejects_unpinned_revision_unsafe_path_or_additional_weights(self):
        for change in ('revision', 'path', 'extra'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, _, lock_path = fixture(root)
                if change == 'revision':
                    model['revision'] = 'main'
                elif change == 'path':
                    model['files'][0]['path'] = '../escape'
                else:
                    model['files'].append({'path': 'pytorch_model.bin', 'size': 1, 'sha256': '0' * 64})
                lock_path.write_text(json.dumps({'sentence_transformers': [model]}))
                with self.assertRaises(ValueError):
                    load_models(lock_path)

    def test_verification_rejects_tampered_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, contents, lock_path = fixture(root)
            for path, data in contents.items():
                destination = model_path(root, model) / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            verify_installation(root, lock_path, full=True)
            (model_path(root, model) / 'model.safetensors').write_bytes(b'x' * len(contents['model.safetensors']))
            with self.assertRaises(ValueError):
                verify_installation(root, lock_path, full=True)

    def test_architecture_enforces_real_sequence_limit_and_mean_pooling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, contents, _ = fixture(root)
            for path, data in contents.items():
                destination = model_path(root, model) / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            folder = model_path(root, model)
            (folder / 'sentence_bert_config.json').write_text('{"max_seq_length":512}')
            with self.assertRaises(ValueError):
                verify_settings(folder, model)
            (folder / 'sentence_bert_config.json').write_bytes(contents['sentence_bert_config.json'])
            (folder / '1_Pooling/config.json').write_text('{"word_embedding_dimension":384,"pooling_mode_mean_tokens":true,"pooling_mode_cls_token":true}')
            with self.assertRaises(ValueError):
                verify_settings(folder, model)

    def test_source_delivery_includes_license_but_no_model_weights(self):
        files = list(source_files())
        self.assertTrue(any(path.as_posix().endswith('cloudflare/licenses/APACHE-2.0.txt') for path in files))
        self.assertFalse(any(path.suffix in {'.safetensors', '.bin', '.gguf'} for path in files))


if __name__ == '__main__':
    unittest.main()
