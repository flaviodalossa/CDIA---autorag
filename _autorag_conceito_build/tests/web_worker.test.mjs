import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../web/_worker.js';
const req = (path='/api/status',options={}) => new Request('https://autorag.example'+path,options);
test('serves static assets with browser protections',async()=>{
 const response=await worker.fetch(req('/'),{ASSETS:{fetch:async()=>new Response('<h1>AutoRAG</h1>',{headers:{'Content-Type':'text/html'}})}});
 assert.equal(response.status,200);assert.equal(response.headers.get('X-Frame-Options'),'DENY');assert.match(await response.text(),/AutoRAG/);
});
test('explains missing backend and requires visitor token',async()=>{
 assert.equal((await worker.fetch(req(),{})).status,503);
 assert.equal((await worker.fetch(req(),{BACKEND_URL:'https://api.example'})).status,401);
});
test('rejects insecure origins and embedded credentials',async()=>{
 for(const BACKEND_URL of ['http://api.example','https://user:pass@api.example','https://api.example/path','https://api.example?q=x'])
 assert.equal((await worker.fetch(req(),{BACKEND_URL})).status,503);
});
test('proxies only relevant headers to the fixed backend',async()=>{
 const prior=globalThis.fetch;
 globalThis.fetch=async(target,options)=>{
  assert.equal(target.toString(),'https://api.example/api/search');assert.equal(options.headers.get('Authorization'),'Bearer local-token');
  assert.equal(options.headers.get('Cookie'),null);assert.equal(options.redirect,'manual');
  assert.deepEqual(JSON.parse(new TextDecoder().decode(options.body)),{query:'norma'});
  return new Response(JSON.stringify({job_id:'job_123'}),{status:202,headers:{'Content-Type':'application/json'}});
 };
 try{const response=await worker.fetch(req('/api/search',{method:'POST',headers:{Authorization:'Bearer local-token','Content-Type':'application/json',Cookie:'private=yes'},body:JSON.stringify({query:'norma'})}),{BACKEND_URL:'https://api.example'});
 assert.equal(response.status,202);assert.equal(response.headers.get('Cache-Control'),'no-store');}
 finally{globalThis.fetch=prior;}
});
test('does not follow origin redirects and bounds payloads',async()=>{
 const prior=globalThis.fetch;globalThis.fetch=async()=>new Response(null,{status:302,headers:{Location:'https://evil.example'}});
 try{assert.equal((await worker.fetch(req('/api/status',{headers:{Authorization:'Bearer token'}}),{BACKEND_URL:'https://api.example'})).status,502);}
 finally{globalThis.fetch=prior;}
 const response=await worker.fetch(req('/api/ingest',{method:'POST',headers:{Authorization:'Bearer token','Content-Length':String(26*1024*1024)},body:'x'}),{BACKEND_URL:'https://api.example'});
 assert.equal(response.status,413);
});
