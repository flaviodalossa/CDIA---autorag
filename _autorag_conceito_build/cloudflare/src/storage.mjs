const MAX_SNAPSHOT = 90 * 1024 * 1024;
const MAX_JOB = 8 * 1024 * 1024;

/** Only called by the container outbound proxy, never by a public HTTP route. */
export async function storageRequest(request, bucket) {
  const url = new URL(request.url);
  if (url.pathname === '/jobs' && request.method === 'GET') {
    const result = await bucket.list({ prefix: 'v1/jobs/', cursor: url.searchParams.get('cursor') || undefined });
    return Response.json({ keys: result.objects.map(o => o.key.slice('v1/jobs/'.length)), cursor: result.truncated ? result.cursor : null });
  }
  const job = /^\/jobs\/(job_[A-Za-z0-9_]+\.json)$/.exec(url.pathname);
  const snapshot = url.pathname === '/snapshot';
  if (!snapshot && !job) return new Response('Not found', { status: 404 });
  const key = snapshot ? 'v1/state.zip' : 'v1/jobs/' + job[1];
  if (request.method === 'GET') {
    const object = await bucket.get(key);
    if (!object) return new Response(null, { status: 404 });
    const headers = new Headers({ 'Content-Type': snapshot ? 'application/zip' : 'application/json' });
    if (object.customMetadata?.sha256) headers.set('X-Content-SHA256', object.customMetadata.sha256);
    return new Response(object.body, { headers });
  }
  if (request.method !== 'PUT') return new Response('Method not allowed', { status: 405 });
  const limit = snapshot ? MAX_SNAPSHOT : MAX_JOB;
  const length = Number(request.headers.get('content-length'));
  const checksum = request.headers.get('x-content-sha256');
  if (!Number.isSafeInteger(length) || length < 1 || length > limit || !/^[a-f0-9]{64}$/.test(checksum || '')) {
    return new Response('Invalid length or checksum', { status: 400 });
  }
  // R2 verifies SHA256 before atomically replacing the previous object.
  await bucket.put(key, request.body, { sha256: checksum, customMetadata: { sha256: checksum } });
  return new Response(null, { status: 204 });
}

export async function validToken(supplied, expected) {
  if (!expected || expected.length < 32 || !supplied?.startsWith('Bearer ')) return false;
  const encode = value => new TextEncoder().encode(value);
  const [a, b] = await Promise.all([
    crypto.subtle.digest('SHA-256', encode(supplied.slice(7))),
    crypto.subtle.digest('SHA-256', encode(expected)),
  ]);
  let difference = 0;
  const left = new Uint8Array(a), right = new Uint8Array(b);
  for (let i = 0; i < left.length; i++) difference |= left[i] ^ right[i];
  return difference === 0;
}

export async function limitedBody(request, maximum) {
  const declared = request.headers.get('content-length');
  if (declared && (!/^\d+$/.test(declared) || Number(declared) > maximum)) return null;
  const chunks = []; let size = 0;
  const reader = request.body?.getReader();
  if (reader) {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maximum) { await reader.cancel(); return null; }
      chunks.push(value);
    }
  }
  const result = new Uint8Array(size); let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.byteLength; }
  return result;
}
