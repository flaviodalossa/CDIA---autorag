import test from 'node:test';
import assert from 'node:assert/strict';
import { limitedBody, resolveApiToken, storageRequest, validToken } from '../src/storage.mjs';

const token = 'a'.repeat(64);
test('access token is checked before starting a container', async () => {
  assert.equal(await validToken('Bearer ' + token, token), true);
  assert.equal(await validToken('Bearer ' + 'b'.repeat(64), token), false);
  assert.equal(await validToken(null, token), false);
  assert.equal(await validToken('Bearer short', 'short'), false);
});
test('chunked body limit is enforced without Content-Length', async () => {
  const body = new ReadableStream({start(controller) {controller.enqueue(new Uint8Array(12)); controller.close();}});
  const request = new Request('http://test/api/ingest', {method:'POST', body, duplex:'half'});
  assert.equal(await limitedBody(request, 10), null);
});
test('storage only accepts well formed state and job paths', async () => {
  const bucket = { get() { throw new Error('must not be accessed'); } };
  for (const pathname of ['/secrets', '/jobs/not-a-job.json', '/jobs/job_x.json/more']) {
    assert.equal((await storageRequest(new Request('http://storage.autorag' + pathname), bucket)).status, 404);
  }
});
test('snapshot PUT supplies checksum to atomic R2 write', async () => {
  let stored;
  const checksum = 'a'.repeat(64);
  const bucket = { async put(key, body, options) { stored = {key, options}; } };
  const response = await storageRequest(new Request('http://storage.autorag/snapshot', {
    method:'PUT', body:'zip', headers:{'Content-Length':'3','X-Content-SHA256':checksum},
  }), bucket);
  assert.equal(response.status, 204);
  assert.deepEqual(stored, {key:'v1/state.zip', options:{sha256:checksum, customMetadata:{sha256:checksum}}});
});
test('job listing uses private R2 prefix and opaque cursor', async () => {
  let options;
  const bucket = {async list(value) {options=value; return {objects:[{key:'v1/jobs/job_a.json'}],truncated:true,cursor:'next'};}};
  const response = await storageRequest(new Request('http://storage.autorag/jobs?cursor=previous'),bucket);
  assert.deepEqual(options,{prefix:'v1/jobs/',cursor:'previous'});
  assert.deepEqual(await response.json(),{keys:['job_a.json'],cursor:'next'});
});

test('legacy runtime secret alias uses the same token without changing its value', async () => {
  const env = {api_token_autorag: token};
  assert.equal(resolveApiToken(env), token);
  assert.equal(await validToken('Bearer ' + token, resolveApiToken(env)), true);
  assert.equal(resolveApiToken({}), '');
});
test('canonical runtime secret takes precedence whenever both names are configured', async () => {
  const canonical = 'b'.repeat(64);
  const env = {AUTORAG_API_TOKEN: canonical, api_token_autorag: token};
  assert.equal(resolveApiToken(env), canonical);
  assert.equal(await validToken('Bearer ' + canonical, resolveApiToken(env)), true);
  assert.equal(await validToken('Bearer ' + token, resolveApiToken(env)), false);
  assert.equal(resolveApiToken({AUTORAG_API_TOKEN: '', api_token_autorag: token}), '');
});
