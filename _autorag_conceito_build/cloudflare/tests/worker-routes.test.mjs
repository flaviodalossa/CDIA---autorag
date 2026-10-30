import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import ts from 'typescript';
import * as storage from '../src/storage.mjs';
const token='t'.repeat(64);
async function loadWorker(backend) {
 const source=await readFile(new URL('../src/index.ts',import.meta.url),'utf8');
 const compiled=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS}}).outputText;
 const context=vm.createContext({exports:{},Request,Response,URL,Headers,require(name){if(name==='@cloudflare/containers')return {Container:class{},ContainerProxy:class{},getContainer:()=>backend};if(name==='./storage.mjs')return storage;throw new Error(name);}});
 vm.runInContext(compiled,context);return context.exports.default;
}
test('production worker forwards authenticated consultation PUT bodies and rejects unrelated writes',async()=>{
 const calls=[];const worker=await loadWorker({async startAndWaitForPorts(){},async fetch(req){calls.push({method:req.method,url:req.url,body:await req.text(),authorization:req.headers.get('Authorization')});return Response.json({experiments:[]});}});
 const env={AUTORAG_API_TOKEN:token};const headers={Authorization:'Bearer '+token,'Content-Type':'application/json'};const body=JSON.stringify({experiments:[],analysis:{enabled:false}});
 let response=await worker.fetch(new Request('https://autorag.test/api/consultation-config',{method:'PUT',headers,body}),env);
 assert.equal(response.status,200);assert.equal(calls.length,1);assert.equal(calls[0].method,'PUT');assert.equal(calls[0].url,'http://container/api/consultation-config');assert.equal(calls[0].body,body);assert.equal(calls[0].authorization,'Bearer '+token);
 response=await worker.fetch(new Request('https://autorag.test/api/status',{method:'PUT',headers,body:'{}'}),env);assert.equal(response.status,405);
 response=await worker.fetch(new Request('https://autorag.test/api/consultation-config',{method:'PUT',body:'{}'}),env);assert.equal(response.status,401);assert.equal(calls.length,1);
 response=await worker.fetch(new Request('https://autorag.test/api/consultation-config',{method:'PUT',headers,body:'x'.repeat(25*1024*1024+1)}),env);assert.equal(response.status,413);assert.equal(calls.length,1);
});

test('production worker forwards enrichment activation PATCH including its bounded body',async()=>{
 const calls=[];const worker=await loadWorker({async startAndWaitForPorts(){},async fetch(req){calls.push({method:req.method,url:req.url,body:await req.text()});return Response.json({job_id:'job_activation'},{status:202});}});
 const env={AUTORAG_API_TOKEN:token};const headers={Authorization:'Bearer '+token,'Content-Type':'application/json'};const body=JSON.stringify({entry_ids:['enrichment_1'],active:true});
 assert.equal((await worker.fetch(new Request('https://autorag.test/api/enrichments/activation',{method:'PATCH',headers,body}),env)).status,202);
 assert.equal(calls[0].method,'PATCH');assert.equal(calls[0].url,'http://container/api/enrichments/activation');assert.equal(calls[0].body,body);
 assert.equal((await worker.fetch(new Request('https://autorag.test/api/enrichments/activation',{method:'PATCH',body:'{}'}),env)).status,401);
 assert.equal((await worker.fetch(new Request('https://autorag.test/api/status',{method:'PATCH',headers,body:'{}'}),env)).status,405);assert.equal(calls.length,1);
});
