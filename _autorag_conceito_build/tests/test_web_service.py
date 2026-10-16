"""Integration of existing profiles/indexes with the new query-inclusive retrieval."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from autorag.core import Project, write_json
from autorag.service import generate, import_base
from autorag.web_service import DEFAULT_CATEGORIES, collect_vectors, history, run_search
from test_autorag import FakeProvider, fixture


class WebServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = Project(self.root/'project')
        self.source = self.root/'base.json'
        write_json(self.source, fixture())
        import_base(self.project,self.source)
        self.profile = {'perfil_id':'pf_test','provedor':'teste','modelo':'offline','dimensoes':3,
                        'revisao_modelo':'test1','limite_tokens':512,'normalizacao':'l2','versao_preparacao':'1'}
        self.project.save_config({'perfis':{'pf_test':self.profile},'perfil_ativo_id':'pf_test'})
        self.provider = FakeProvider(self.profile)
        generate(self.project,'pf_test',list(DEFAULT_CATEGORIES),provider=self.provider)
        self.request = {'profile_id':'pf_test','query':' Uma consulta normativa\n','categories':list(DEFAULT_CATEGORIES),'top_k':2}

    def tearDown(self):
        self.temp.cleanup()

    def test_semantic_keeps_sources_and_saves_reproducible_snapshot(self):
        def fitted(matrix,params):
            return np.zeros((len(matrix),params['n_components'])), SimpleNamespace(labels_=np.zeros(len(matrix),dtype=int),probabilities_=np.ones(len(matrix)))
        with patch('autorag.semantic._fit_clusters',side_effect=fitted):
            result = run_search(self.project,self.request,provider=self.provider)
        self.assertEqual(len(result['semantic_results']),2)
        self.assertTrue(all(r['category_count']==4 for r in result['semantic_results']))
        self.assertTrue(all(r['document_id']=='doc1' for r in result['semantic_results']))
        self.assertTrue(all(r['categories']==['conteudo_fonte'] for r in result['baseline_results']))
        self.assertTrue(result['projection'][-1]['is_query'])
        self.assertEqual(set(result['indices']),set(DEFAULT_CATEGORIES))
        self.assertTrue(result['input_checksum'])
        saved = history(self.project)
        self.assertEqual(saved[0]['execution_id'],result['execution_id'])
        self.assertEqual(result['query'],self.request['query'])

    def test_small_base_still_returns_source_only_baseline(self):
        result = run_search(self.project,{**self.request,'min_cluster_size':100},provider=self.provider)
        self.assertEqual(result['status'],'insufficient_data')
        self.assertFalse(result['semantic_results'])
        self.assertEqual(len(result['baseline_results']),2)

    def test_stale_category_excluded_and_laboratory_metric_applied(self):
        base = self.project.base()
        base['documentos'][0]['estrutura'][0]['resumo']='Conteúdo modificado'
        write_json(self.project.base_path,base)
        vectors,refs,missing,excluded=collect_vectors(self.project,'pf_test',list(DEFAULT_CATEGORIES))
        self.assertEqual(excluded,1)
        self.assertFalse(any(v['topic_id']=='t1' and v['category']=='resumo' for v in vectors))
        result=run_search(self.project,{**self.request,'mode':'laboratory','metric':'euclidean','partial':True},provider=self.provider)
        distances=[row['distance'] for row in result['semantic_results']]
        self.assertEqual(distances,sorted(distances))
        self.assertTrue(result['laboratory_execution_id'])
        self.assertTrue(any('desatualizados' in warning for warning in result['warnings']))

if __name__=='__main__':
    unittest.main()
