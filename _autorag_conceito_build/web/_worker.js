/** Cloudflare Pages advanced-mode proxy; Python remains on the configured origin. */
const MAX_BODY = 25 * 1024 * 1024;
const securityHeaders = {
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'same-origin',
  'X-Frame-Options': 'DENY',
  'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self' https: http://127.0.0.1:* http://localhost:*; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
};
function jsonError(detail, status) {
  return new Response(JSON.stringify({detail}), {status, headers: {'Content-Type':'application/json', 'Cache-Control':'no-store', ...securityHeaders}});
}
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (!url.pathname.startsWith('/api/')) {
      const response = await env.ASSETS.fetch(request);
      const headers = new Headers(response.headers);
      for (const [k,v] of Object.entries(securityHeaders)) headers.set(k,v);
      return new Response(response.body, {status: response.status, headers});
    }
    if (!env.BACKEND_URL) return jsonError('Configure BACKEND_URL no Cloudflare Pages para conectar o serviço Python.', 503);
    let origin;
    try { origin = new URL(env.BACKEND_URL); } catch { return jsonError('BACKEND_URL inválido.', 503); }
    if (origin.protocol !== 'https:' || origin.username || origin.password || origin.search || origin.hash || origin.pathname !== '/') {
      return jsonError('BACKEND_URL deve ser a origem HTTPS do serviço Python, sem caminho ou credenciais.', 503);
    }
    if (!['GET', 'POST'].includes(request.method) && !(request.method === 'PUT' && url.pathname === '/api/consultation-config') && !(request.method === 'PATCH' && url.pathname === '/api/enrichments/activation')) return jsonError('Método não permitido.', 405);
    // Access credentials are supplied by the visitor. No model key is sent to the browser.
    const authorization = request.headers.get('Authorization');
    if (!authorization || !authorization.startsWith('Bearer ')) return jsonError('Informe o token de acesso nas configurações.', 401);
    const contentLength = request.headers.get('Content-Length');
    if (contentLength && (!/^\d+$/.test(contentLength) || Number(contentLength) > MAX_BODY)) return jsonError('Requisição excede 25 MiB.', 413);
    const target = new URL(origin);
    target.pathname = url.pathname;
    target.search = url.search;
    const headers = new Headers({'Authorization':authorization});
    const contentType = request.headers.get('Content-Type');
    if (contentType) headers.set('Content-Type',contentType);
    let body;
    if (request.method === 'POST' || request.method === 'PUT' || request.method === 'PATCH') {
      const chunks = []; let size = 0;
      if (request.body) {
        const reader = request.body.getReader();
        while (true) {
          const {done,value} = await reader.read();
          if (done) break;
          size += value.byteLength;
          if (size > MAX_BODY) { await reader.cancel(); return jsonError('Requisição excede 25 MiB.',413); }
          chunks.push(value);
        }
      }
      body = new Uint8Array(size); let offset = 0;
      for (const chunk of chunks) { body.set(chunk,offset); offset += chunk.byteLength; }
    }
    try {
      const response = await fetch(target, {method:request.method, headers, body, redirect:'manual', signal:AbortSignal.timeout(25000)});
      if (response.status >= 300 && response.status < 400) return jsonError('O serviço Python deve responder sem redirecionamento.',502);
      const resultHeaders = new Headers({'Content-Type':response.headers.get('Content-Type') || 'application/json', 'Cache-Control':'no-store', ...securityHeaders});
      return new Response(response.body, {status:response.status, headers:resultHeaders});
    } catch {
      return jsonError('Não foi possível acessar o serviço Python. Verifique BACKEND_URL e a disponibilidade do backend.',502);
    }
  }
};
