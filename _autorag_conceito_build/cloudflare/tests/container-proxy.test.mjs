import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import vm from 'node:vm';
import ts from 'typescript';
import * as storage from '../src/storage.mjs';

const require = createRequire(import.meta.url);

async function loadWorkerWithInstalledSdk(env) {
  const sdkPath = join(dirname(require.resolve('@cloudflare/containers')), 'lib', 'container.js');
  const source = (await readFile(sdkPath, 'utf8'))
    .replace(/^import .*;\r?\n/gm, '')
    .replace(/^export (?=(?:class|function) )/gm, '');
  class CloudflareBase {
    constructor(ctx, bindings) { this.ctx = ctx; this.env = bindings; }
  }
  const context = vm.createContext({
    DurableObject: CloudflareBase, WorkerEntrypoint: CloudflareBase,
    Response, Request, URL, Headers, setTimeout, clearTimeout,
    fetch() { throw new Error('Unexpected network request: outbound handler was not registered'); },
  });
  vm.runInContext(source + '\n;globalThis.sdk = {Container, ContainerProxy};', context);
  const workerPath = new URL('../src/index.ts', import.meta.url);
  const compiled = ts.transpileModule(await readFile(workerPath, 'utf8'), {
    compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS},
  }).outputText;
  context.exports = {};
  context.require = specifier => {
    if (specifier === '@cloudflare/containers') return context.sdk;
    if (specifier === './storage.mjs') return storage;
    throw new Error('Unexpected dependency: ' + specifier);
  };
  vm.runInContext(compiled, context);
  return new context.sdk.ContainerProxy({props: {
    className: context.exports.AutoRAGContainer.name,
    containerId: 'test-container', enableInternet: true, interceptAll: false,
  }}, env);
}

test('installed ContainerProxy routes bootstrap snapshot and jobs to the private R2 binding', async () => {
  const reads = [];
  const proxy = await loadWorkerWithInstalledSdk({STATE: {
    async get(key) { reads.push(key); return null; },
    async list(options) { assert.equal(options.prefix, 'v1/jobs/'); return {objects:[], truncated:false}; },
  }});
  const snapshot = await proxy.fetch(new Request('http://storage.autorag/snapshot'));
  assert.equal(snapshot.status, 404);
  assert.deepEqual(reads, ['v1/state.zip']);
  const jobs = await proxy.fetch(new Request('http://storage.autorag/jobs'));
  assert.equal(jobs.status, 200);
  assert.deepEqual(await jobs.json(), {keys:[], cursor:null});
});
