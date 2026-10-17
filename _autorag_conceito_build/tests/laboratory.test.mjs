import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const context = vm.createContext({});
vm.runInContext(await readFile(new URL('../web/laboratory.js',import.meta.url),'utf8'),context);
const lab = context.AutoRAGLaboratory;
const config = () => ({...lab.defaults(),profile_ids:['pf_gemma'],name:'Perguntas e resumo'});

test('aceita até três experimentos e mantém os espaços separados na configuração', () => {
  const input = {experiments:[config(),{...config(),profile_ids:['pf_gemma','pf_minilm']},config()]};
  const result = lab.normalizeBundle(input);
  assert.equal(result.experiments.length,3);
  assert.equal(result.experiments[1].profile_ids.length,2);
  assert.equal(result.analysis.enabled,false);
  assert.throws(() => lab.normalizeBundle({experiments:[...input.experiments,config()]}),/três/);
});
test('importa o resultado exportado como configuração reutilizável sem copiar conteúdo ou credenciais', () => {
  const result = lab.normalizeBundle({query:'consulta anterior',comparison_id:'comparison_1',experiments:[{config:{...config(),api_key:'never-copy'},results:[{text:'conteúdo'}]}],analysis_config:{enabled:true,provider:'openai',model:'gpt-6-luna'}});
  assert.equal(result.experiments[0].api_key,undefined);
  assert.equal(result.experiments[0].results,undefined);
  assert.equal(result.query,undefined);
  assert.equal(result.analysis.enabled,true);
});
test('rejeita limites inválidos, perfil repetido e configuração sem categoria', () => {
  for (const patch of [{top_k:101},{top_k:1.5},{maximum_distance:-1},{minimum_similarity:2},{minimum_similarity:NaN},{profile_ids:['pf_a','pf_a']},{categories:[]}]) assert.throws(() => lab.normalizeBundle({experiments:[{...config(),...patch}]}));
});
test('não classifica falha de LLM como conteúdo não pertinente', () => {
  assert.equal(lab.analysisLabel({status:'failed',error:'indisponível'}),'Análise não concluída');
  assert.equal(lab.analysisLabel({status:'completed',relevant:false}),'Não pertinente');
  assert.equal(lab.analysisLabel({status:'completed',relevant:true}),'Pertinente');
  assert.equal(lab.analysisLabel(undefined),'');
});
test('o padrão de recuperação é o conteúdo original por cosseno', () => {
  assert.equal(lab.defaults().mode,'vector');
  assert.equal(lab.defaults().metric,'cosine');
  assert.equal(lab.defaults().categories.join(','),'conteudo_fonte');
});

test('mantém categorias específicas de cada modelo e rejeita mapas incompletos', () => {
  const cfg = {...config(),profile_ids:['pf_gemma','pf_minilm'],model_categories:{pf_gemma:['resumo'],pf_minilm:['perguntas','resumo']}};
  const result = lab.normalizeBundle({experiments:[cfg]}).experiments[0];
  assert.equal(result.model_categories.pf_gemma.join(','),'resumo');
  assert.equal(result.model_categories.pf_minilm.join(','),'perguntas,resumo');
  assert.equal(result.categories.join(','),'resumo,perguntas');
  for (const mapping of [{pf_gemma:['resumo']},{pf_gemma:[],pf_minilm:['perguntas']},{pf_gemma:['resumo'],pf_minilm:['perguntas'],pf_extra:['resumo']}]) assert.throws(() => lab.normalizeBundle({experiments:[{...cfg,model_categories:mapping}]}));
});
