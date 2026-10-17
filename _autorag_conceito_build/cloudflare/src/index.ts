import { Container, ContainerProxy, getContainer } from '@cloudflare/containers';
import { limitedBody, resolveApiToken, storageRequest, validToken } from './storage.mjs';

export { ContainerProxy };

interface Env {
  ASSETS: Fetcher;
  BACKEND: DurableObjectNamespace<AutoRAGContainer>;
  STATE: R2Bucket;
  AUTORAG_API_TOKEN?: string;
  api_token_autorag?: string;
  OPENAI_API_KEY?: string;
  GEMINI_API_KEY?: string;
}

const security = {
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'same-origin',
  'X-Frame-Options': 'DENY',
  'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
};

function jsonError(detail: string, status: number): Response {
  return Response.json({ detail }, { status, headers: { ...security, 'Cache-Control': 'no-store' } });
}

export class AutoRAGContainer extends Container<Env> {
  defaultPort = 8000;
  sleepAfter = '10m';
  enableInternet = true;
  envVars = {
    AUTORAG_API_TOKEN: resolveApiToken(this.env),
    OPENAI_API_KEY: this.env.OPENAI_API_KEY || '',
    GEMINI_API_KEY: this.env.GEMINI_API_KEY || '',
  };

  private activeRequests = 0;
  private idleBarrier: Promise<void> | undefined;

  override async fetch(request: Request): Promise<Response> {
    await this.idleBarrier;
    this.activeRequests += 1;
    try { return await this.containerFetch(request); }
    finally { this.activeRequests -= 1; }
  }

  override async onActivityExpired(): Promise<void> {
    // Background jobs keep running even after their HTTP 202 or a closed browser.
    // Never stop on an ambiguous health/checkpoint response.
    if (this.activeRequests) { this.renewActivityTimeout(); return; }
    let release!: () => void;
    this.idleBarrier = new Promise<void>(resolve => { release = resolve; });
    try {
      const response = await this.containerFetch('http://container/internal/cloud-state', {
        headers: { authorization: 'Bearer ' + resolveApiToken(this.env) },
      });
      if (!response.ok) { this.renewActivityTimeout(); return; }
      const state = await response.json() as { active: boolean };
      if (state.active) { this.renewActivityTimeout(); return; }
      const checkpoint = await this.containerFetch('http://container/internal/checkpoint', {
        method: 'POST', headers: { authorization: 'Bearer ' + resolveApiToken(this.env) },
      });
      if (!checkpoint.ok) { this.renewActivityTimeout(); return; }
      await this.stop();
    } catch { this.renewActivityTimeout(); }
    finally { this.idleBarrier = undefined; release(); }
  }
}

// Use the SDK's inherited setter so ContainerProxy receives this handler.
// A static class field would shadow the setter and leave its registry empty.
AutoRAGContainer.outboundByHost = {
  'storage.autorag': (request: Request, env: Env) => storageRequest(request, env.STATE),
};

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/internal/') || url.pathname.startsWith('/.')) return jsonError('Não encontrado.', 404);
    if (!url.pathname.startsWith('/api/')) {
      const response = await env.ASSETS.fetch(request);
      const headers = new Headers(response.headers);
      for (const [name, value] of Object.entries(security)) headers.set(name, value);
      return new Response(response.body, { status: response.status, headers });
    }
    if (!['GET', 'POST'].includes(request.method)) return jsonError('Método não permitido.', 405);
    const apiToken = resolveApiToken(env);
    if (apiToken.length < 32) return jsonError('Configure o segredo AUTORAG_API_TOKEN no Worker.', 503);
    if (!await validToken(request.headers.get('Authorization'), apiToken)) return jsonError('Informe o token de acesso nas configurações.', 401);
    const headers = new Headers({ Authorization: request.headers.get('Authorization')! });
    if (request.headers.has('Content-Type')) headers.set('Content-Type', request.headers.get('Content-Type')!);
    let body: Uint8Array | undefined;
    if (request.method === 'POST') {
      const content = await limitedBody(request, 25 * 1024 * 1024);
      if (content === null) return jsonError('Requisição excede 25 MiB.', 413);
      body = content;
    }
    const target = new URL(request.url);
    target.protocol = 'http:';
    target.host = 'container';
    try {
      const backend = getContainer(env.BACKEND, 'autorag-project-v1');
      await backend.startAndWaitForPorts({
        ports: 8000,
        cancellationOptions: { portReadyTimeoutMS: 120000, instanceGetTimeoutMS: 30000 },
      });
      const response = await backend.fetch(new Request(target, { method: request.method, headers, body }));
      if (response.status >= 300 && response.status < 400) return jsonError('O backend respondeu com redirecionamento inesperado.', 502);
      return new Response(response.body, { status: response.status, headers: {
        ...security, 'Cache-Control': 'no-store', 'Content-Type': response.headers.get('Content-Type') || 'application/json',
      } });
    } catch {
      return jsonError('O backend está iniciando ou indisponível. Aguarde e tente novamente.', 503);
    }
  },
} satisfies ExportedHandler<Env>;
