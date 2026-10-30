"""Full-document expansion: offline retrieval, persistence, API and reproduction."""
from copy import deepcopy
import csv, io, json, tempfile, unittest, zipfile
from pathlib import Path
from unittest.mock import Mock, patch
from autorag.core import Project, topics, write_json
from autorag.document_expansion import expand_documents, full_document_enabled
from autorag.assistance import answer_comparison
from autorag.comparisons import (run_comparison, reproduce_comparison, consultation_config,
    save_consultation_config, save_preset, export_comparison, export_vector_bundle)
from autorag.service import import_base, generate
from autorag.web_service import run_search
from test_autorag import fixture, FakeProvider
import test_comparison_api as http

class DocumentExpansionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.project=Project(Path(self.tmp.name)/'project')
        base=fixture()
        # A second document has an active topic with no embedding and an inactive topic.
        other=deepcopy(base['documentos'][0]);other['documento_id']='doc2';other['nome_documento']='Outro documento'
        other['estrutura']=[{**deepcopy(other['estrutura'][0]),'topico_id':'t4'},
                            {**deepcopy(other['estrutura'][1]),'topico_id':'inactive','ativo':False}]
        base['documentos'].append(other)
        source=Path(self.tmp.name)/'base.json';write_json(source,base);import_base(self.project,source)
        self.profile={'perfil_id':'pf_test','provedor':'teste','modelo':'offline','dimensoes':3,'revisao_modelo':'fixed','limite_tokens':512,'normalizacao':'l2','versao_preparacao':'1'}
        self.project.save_config({'perfis':{'pf_test':self.profile},'perfil_ativo_id':'pf_test'})
        self.provider=FakeProvider(self.profile)
        generate(self.project,'pf_test',['conteudo_fonte'],topic_ids=['t1'],provider=self.provider)
        self.providers={'pf_test':self.provider}
    def tearDown(self):self.tmp.cleanup()
    def run_query(self,**extra):
        return run_comparison(self.project,{'query':'Consulta',**extra},self.providers)
    def test_disabled_keeps_top_k_and_no_new_context(self):
        out=self.run_query();self.assertEqual([r['topic_id'] for r in out['experiments'][0]['results']],['t1'])
        self.assertFalse(out['full_document'])
    def test_expansion_adds_unindexed_topics_once_without_other_documents_or_inactive_topics(self):
        out=self.run_query(full_document=True);e=out['experiments'][0]
        self.assertEqual([r['topic_id'] for r in e['results']],['t1','t2','t3'])
        self.assertEqual(e['document_expansion']['seed_count'],1)
        self.assertEqual(e['document_expansion']['added_count'],2)
        self.assertEqual(e['results'][0]['result_origin'],'retrieval')
        for r in e['results'][1:]:
            self.assertIsNone(r['distance']);self.assertIsNone(r['cosine_similarity']);self.assertEqual(r['matched_vectors'],[])
            self.assertEqual(r['expanded_from_topic_ids'],['t1']);self.assertTrue(r['text'].startswith('Fonte completa'))
        self.assertEqual(len(e['spaces'][0]['results']),1)
        self.assertEqual(len(e['baseline_results']),3)
    def test_multiple_seeds_and_documents_are_deduplicated(self):
        mapping={t['topico_id']:(t,d) for t,d in topics(self.project.base())}
        seeds=[{'topic_id':'t2','position':1,'distance':.2},{'topic_id':'t1','position':2,'distance':.3},{'topic_id':'t4','position':3,'distance':.4}]
        rows,meta=expand_documents(seeds,mapping)
        self.assertEqual([r['topic_id'] for r in rows],['t2','t1','t4','t3'])
        self.assertEqual(rows[-1]['expanded_from_topic_ids'],['t2','t1']);self.assertEqual(meta['document_ids'],['doc1','doc2'])
        self.assertEqual(seeds[0],{'topic_id':'t2','position':1,'distance':.2})
    def test_noise_rejection_cannot_be_bypassed_by_document_expansion(self):
        out=self.run_query(full_document=True,noise_filter={'enabled':True,'margin':2})
        self.assertEqual(out['experiments'][0]['results'],[])
        with patch('autorag.assistance.make_structured_generator',side_effect=AssertionError('No LLM')):
            answer_comparison(out)
        self.assertEqual(out['support']['status'],'noise')
    def test_llm_receives_the_entire_document_and_may_cite_a_topic_without_vector(self):
        out=self.run_query(full_document=True);generate_answer=Mock(return_value={'answer':'Resposta [t3]','topic_ids':['t3']})
        answer_comparison(out,generator=generate_answer)
        self.assertEqual(out['support']['status'],'completed')
        self.assertEqual([r['topic_id'] for r in generate_answer.call_args.args[0]['topics']],['t1','t2','t3'])
    def test_reproduction_uses_original_document_even_after_source_changes(self):
        out=self.run_query(full_document=True)
        b=self.project.base();b['documentos'][0]['estrutura'][1]['conteudo_fonte']='Changed';write_json(self.project.base_path,b)
        with patch('autorag.experiments.get_provider',side_effect=AssertionError('No inference')):
            replay=reproduce_comparison(self.project,out['comparison_id'])
        self.assertTrue(replay['full_document']);self.assertEqual(replay['experiments'],out['experiments'])
        self.assertEqual(replay['experiments'][0]['results'][1]['text'],'Fonte completa 2')
    def test_all_exports_preserve_expansion_and_no_fake_vector(self):
        out=self.run_query(full_document=True);cid=out['comparison_id']
        self.assertTrue(json.loads(export_comparison(self.project,cid,'config'))['full_document'])
        lines=[json.loads(l) for l in export_comparison(self.project,cid,'jsonl').splitlines()]
        self.assertEqual(len(lines),3);self.assertEqual(lines[-1]['result']['result_origin'],'document_expansion')
        rows=list(csv.DictReader(io.StringIO(export_comparison(self.project,cid,'csv'))))
        self.assertEqual(rows[-1]['result_origin'],'document_expansion');self.assertEqual(rows[-1]['distance'],'')
        with zipfile.ZipFile(io.BytesIO(export_vector_bundle(self.project,cid))) as z:
            m=json.loads(z.read('manifesto.json'));self.assertEqual(len(m['context_only_topics']),2)
            self.assertTrue(all(not r['vector_included'] for r in m['context_only_topics']))
    def test_saved_option_preserved_by_older_lab_settings_and_presets(self):
        save_consultation_config(self.project,{'full_document':True})
        save_consultation_config(self.project,{'experiments':[]})
        self.assertTrue(consultation_config(self.project)['full_document'])
        save_preset(self.project,{'name':'Old','active':True,'experiments':[{'name':'x','profile_ids':['pf_test']}]})
        self.assertTrue(consultation_config(self.project)['full_document'])
        save_consultation_config(self.project,{'full_document':False})
        self.assertFalse(consultation_config(self.project)['full_document'])
    def test_explicit_profile_search_expands_before_answer(self):
        out=run_search(self.project,{'profile_id':'pf_test','query':'Consulta','full_document':True},provider=self.provider)
        self.assertEqual([r['topic_id'] for r in out['semantic_results']],['t1','t2','t3'])
    def test_invalid_values_and_empty_expansion(self):
        for v in ['true',1,0,[],{}]:
            with self.assertRaises(ValueError):full_document_enabled(v)
        self.assertEqual(expand_documents([],{} )[0],[])

@unittest.skipUnless(http.WEB_AVAILABLE,'HTTP dependencies optional')
class DocumentExpansionAPITests(unittest.TestCase):
    setUp=http.ComparisonAPITests.setUp
    tearDown=http.ComparisonAPITests.tearDown
    wait_job=http.ComparisonAPITests.wait_job
    def test_query_override_and_saved_defaults(self):
        res=self.client.put('/api/consultation-config',json={'full_document':True,'support':{'enabled':False}})
        self.assertEqual(res.status_code,200,res.text)
        out=self.wait_job(self.client.post('/api/search',json={'query':'Consulta'}))
        self.assertTrue(out['full_document']);self.assertEqual(len(out['experiments'][0]['results']),3)
        off=self.wait_job(self.client.post('/api/search',json={'query':'Consulta','full_document':False}))
        self.assertFalse(off['full_document']);self.assertTrue(self.client.get('/api/consultation-config').json()['full_document'])
    def test_comparison_and_explicit_profile_accept_flag(self):
        for payload in [{'query':'Consulta','full_document':True,'support':{'enabled':False}},
                        {'query':'Consulta','profile_id':'pf_test','full_document':True}]:
            out=self.wait_job(self.client.post('/api/search',json=payload));self.assertTrue(out['full_document'])
        self.assertEqual(self.client.post('/api/search',json={'query':'Consulta','full_document':'yes'}).status_code,422)
    def test_presets_store_flag(self):
        res=self.client.post('/api/experiment-configs',json={'name':'Full','full_document':True,'experiments':[{'name':'x','profile_ids':['pf_test']}]})
        self.assertEqual(res.status_code,200,res.text)
        self.assertTrue(res.json()['full_document'])
        out=self.wait_job(self.client.post('/api/comparisons',json={'query':'Consulta','preset_id':res.json()['id']}))
        self.assertTrue(out['full_document'])
