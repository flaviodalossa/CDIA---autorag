'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const DEFAULT_CATEGORIES = ['perguntas_sugeridas', 'resposta_sintese', 'resumo', 'conteudo_fonte'];
  const CATEGORY_LABELS = {
    perguntas_sugeridas: 'Perguntas sugeridas', resposta_sintese: 'Resposta-síntese', resumo: 'Resumo', conteudo_fonte: 'Conteúdo-fonte',
    topico_total_sem_palavras_chave: 'Tópico total sem palavras-chave', topico_total_com_palavras_chave: 'Tópico total com palavras-chave', palavras_chave: 'Palavras-chave', metadados: 'Metadados', enriquecimento_recuperacao: 'Enriquecimento de recuperação'
  };
  const DEFAULT = { categories: ['conteudo_fonte'], mode: 'vector', metric: 'cosine', top_k: 6, n_neighbors: 15, min_cluster_size: 5, min_samples: 2, n_components: 5, dendrogram: false };
  const state = { status: null, categories: Object.entries(CATEGORY_LABELS).map(([id, label]) => ({id, label})), config: {...DEFAULT, categories: [...DEFAULT.categories], profile_id: ''}, results: null, resultQuery: '', preview: null, previewForm: null, selectedExperiment: null, selectedFiles: [], searchController: null, activePage: 'consulta', ingestRevision: 0 };
  const session = { get(key) { try { return sessionStorage.getItem(key) || ''; } catch { return ''; } }, set(key, value) { try { sessionStorage.setItem(key, value); } catch {} } };
  let backend = session.get('autorag_backend').replace(/\/+$/, '');
  let token = session.get('autorag_token');
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
  const label = category => state.categories.find(c => c.id === category)?.label || CATEGORY_LABELS[category] || category;
  const number = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value)) ? new Intl.NumberFormat('pt-BR', {maximumFractionDigits: 4, minimumFractionDigits: 4}).format(Number(value)) : '—';
  const list = value => Array.isArray(value) ? value : value && typeof value === 'object' ? Object.values(value) : [];
  const isAbort = error => error?.name === 'AbortError';
  const abortError = () => new DOMException('Espera interrompida.', 'AbortError');
  const delay = (ms, signal) => new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(abortError()); return; }
    const timer = setTimeout(() => { signal?.removeEventListener('abort', onAbort); resolve(); }, ms);
    const onAbort = () => { clearTimeout(timer); reject(abortError()); };
    signal?.addEventListener('abort', onAbort, {once:true});
  });
  function message(text, type = '', target = 'global-message') {
    const element = $(target); element.className = 'message ' + type; element.textContent = text; element.hidden = !text;
  }
  function warnings(items, target) {
    const rows = list(items).filter(Boolean);
    const element = $(target); element.replaceChildren(); element.hidden = !rows.length;
    if (rows.length) { element.className = 'message warning'; rows.forEach(text => { const p = document.createElement('p'); p.textContent = typeof text === 'string' ? text : JSON.stringify(text); element.append(p); }); }
  }
  function apiUrl(path) { return backend + path; }
  async function request(path, {method = 'GET', body, signal, progress} = {}) {
    const controller = new AbortController();
    const onAbort = () => controller.abort();
    if (signal?.aborted) throw abortError();
    signal?.addEventListener('abort', onAbort, {once: true});
    const timeout = setTimeout(() => controller.abort(), 120000);
    let response;
    try {
      const headers = {Accept: 'application/json'};
      if (token) headers.Authorization = 'Bearer ' + token;
      if (body && !(body instanceof FormData)) headers['Content-Type'] = 'application/json';
      response = await fetch(apiUrl(path), {method, headers, body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined, signal: controller.signal, credentials: 'same-origin'});
      const contentType = response.headers.get('content-type') || '';
      if (!contentType.includes('application/json')) {
        if (response.status >= 500) throw new Error('O serviço apresentou uma falha (HTTP ' + response.status + '). Aguarde alguns instantes e tente novamente.');
        throw new Error('A API respondeu em um formato inesperado (HTTP ' + response.status + '). Confira a URL da aplicação nas configurações.');
      }
      const data = await response.json();
      if (!response.ok) {
        const detail = data.detail || data.error || data.message;
        throw new Error(typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : 'A operação não foi concluída (' + response.status + ').');
      }
      if (response.status === 202 && data.job_id) return await pollJob(data.job_id, {signal, progress});
      return data;
    } catch (error) {
      if (isAbort(error) && !signal?.aborted) throw new Error('O serviço demorou para responder. Verifique a conexão e tente novamente.');
      if (error instanceof TypeError) throw new Error('Não foi possível conectar ao serviço. Confira a URL, o acesso e se o backend está em execução.');
      throw error;
    } finally { clearTimeout(timeout); signal?.removeEventListener('abort', onAbort); }
  }
  async function pollJob(id, {signal, progress} = {}) {
    const start = Date.now(); let attempt = 0;
    while (Date.now() - start < 15 * 60 * 1000) {
      await delay(Math.min(2000 + attempt * 250, 5000), signal);
      const job = await request('/api/jobs/' + encodeURIComponent(id), {signal});
      if (job.status === 'succeeded') return job.result;
      if (job.status === 'failed') throw new Error(typeof job.error === 'string' ? job.error : job.error?.message || 'O processamento não foi concluído.');
      if (job.status === 'cancelled') throw new Error('O processamento foi cancelado pelo serviço.');
      if (progress) progress(job.status === 'queued' ? 'Na fila de processamento…' : (job.message || 'Processando no servidor…'));
      attempt += 1;
    }
    throw new Error('O processamento continua no servidor após 15 minutos. Atualize a base e o histórico antes de iniciar uma nova execução.');
  }
  function navigate(page, focus = true) {
    if (!['consulta', 'preparar', 'laboratorio'].includes(page)) page = 'consulta';
    state.activePage = page;
    document.querySelectorAll('.page').forEach(element => element.hidden = element.id !== 'page-' + page);
    document.querySelectorAll('[data-page]').forEach(button => {
      const active = button.dataset.page === page;
      button.classList.toggle('active', active);
      if (active) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
    });
    const names = {consulta: 'Consulta', preparar: 'Preparar base', laboratorio: 'Laboratório'};
    $('breadcrumb').innerHTML = 'Espaço de trabalho <span aria-hidden="true">/</span> <strong>' + names[page] + '</strong>';
    history.replaceState(null, '', '#' + page);
    if (focus) { $('main').focus({preventScroll: true}); window.scrollTo({top: 0, behavior: 'instant'}); }
    if (page === 'laboratorio') { refreshHistory(); if (!state.models) refreshModels(); }
    if (page === 'preparar' && !state.llms) refreshLLMs();
    state.lab?.navigate(page);
    state.assistance?.navigate(page);
  }
  function renderCategories() {
    for (const name of ['search', 'generate']) {
      const selected = name === 'search' ? state.config.categories : DEFAULT_CATEGORIES;
      const container = $(name + '-categories'); container.replaceChildren();
      state.categories.filter(category => name !== 'generate' || category.id !== 'enriquecimento_recuperacao').forEach(category => {
        const wrapper = document.createElement('label'); wrapper.className = 'category-option';
        const input = document.createElement('input'); input.type = 'checkbox'; input.name = name + '-category'; input.value = category.id; input.checked = selected.includes(category.id);
        const text = document.createElement('span'); text.textContent = category.label;
        wrapper.append(input, text); container.append(wrapper);
      });
    }
  }
  function selectedCategories(prefix) { return Array.from(document.querySelectorAll('input[name="' + prefix + '-category"]:checked'), input => input.value); }
  function renderProfiles(profiles) {
    const rows = list(profiles);
    document.querySelectorAll('.profile-select').forEach(select => {
      const old = select.value; select.replaceChildren();
      if (!rows.length) { const option = new Option('Cadastre um perfil no Laboratório', ''); select.append(option); }
      rows.forEach(profile => { const id = profile.perfil_id || profile.id; select.add(new Option((profile.modelo || profile.model || id) + ' · ' + (profile.dimensoes || profile.dimensions || '?') + 'd · ' + (profile.provedor || profile.provider || ''), id)); });
      if (rows.some(p => (p.perfil_id || p.id) === old)) select.value = old;
      else if (rows.some(p => (p.perfil_id || p.id) === state.config.profile_id)) select.value = state.config.profile_id;
    });
    if (!rows.some(p => (p.perfil_id || p.id) === state.config.profile_id)) state.config.profile_id = $('search-profile').value;
    if (state.config.profile_id) $('search-profile').value = state.config.profile_id;
  }
  function renderInventory(rows) {
    rows = list(rows);
    if (!rows.length) { $('inventory').innerHTML = '<p class="empty-inline">Nenhum vetor cadastrado. Incorpore documentos e gere embeddings.</p>'; return; }
    $('inventory').innerHTML = '<table><thead><tr><th>Modelo / dimensão</th><th>Categoria</th><th>Atuais</th><th>Desatualizados</th><th>Faltantes</th><th>Sem texto</th></tr></thead><tbody>' + rows.map(row => '<tr><td>' + esc(row.modelo || row.model || row.perfil_id) + (row.dimensoes ? ' · ' + esc(row.dimensoes) + 'd' : '') + '</td><td>' + esc(label(row.categoria || row.category)) + '</td><td>' + esc(row.atuais ?? row.current ?? 0) + '</td><td>' + esc(row.desatualizados ?? row.stale ?? 0) + '</td><td>' + esc(row.faltantes ?? row.missing ?? 0) + '</td><td>' + esc(row.sem_texto ?? row.empty ?? 0) + '</td></tr>').join('') + '</tbody></table>';
  }
  async function refreshStatus({quiet = false} = {}) {
    $('refresh-status').disabled = true;
    try {
      const data = await request('/api/status'); state.status = data;
      const docCount = Array.isArray(data.documents) ? data.documents.length : Number(data.documents || 0);
      const topicCount = Array.isArray(data.topics) ? data.topics.length : Number(data.topics || 0);
      const categories = list(data.categories).map(item => typeof item === 'string' ? {id: item, label: CATEGORY_LABELS[item] || item} : {id: item.id || item.categoria, label: item.label || item.nome || CATEGORY_LABELS[item.id] || item.id});
      if (categories.length && JSON.stringify(categories) !== JSON.stringify(state.categories)) { state.categories = categories; renderCategories(); }
      $('connection-label').textContent = 'Serviço conectado'; $('connection-dot').className = 'status-dot green';
      $('sidebar-count').textContent = docCount + ' documentos · ' + topicCount + ' tópicos';
      $('onboarding').hidden = docCount > 0;
      $('sources-help').textContent = data.local_paths_enabled === false ? 'Uma URL por linha ou envie arquivos abaixo. Caminhos locais estão desabilitados neste ambiente.' : 'Uma fonte por linha. Caminhos locais devem existir no servidor da aplicação. Para arquivos do seu computador, use o envio abaixo.';
      if (!state.config.profile_id && data.active_profile) state.config.profile_id = data.active_profile;
      renderProfiles(data.profiles || data.perfis); renderInventory(data.inventory); state.lab?.renderProfiles(); state.assistance?.renderProfiles();
      if (state.models) renderModels();
      $('generation-state').textContent = docCount + ' documentos · ' + topicCount + ' tópicos na base atual.';
      if (!quiet) message('');
    } catch (error) {
      $('connection-label').textContent = 'Serviço indisponível'; $('connection-dot').className = 'status-dot red';
      $('sidebar-count').textContent = 'Verifique conexão e acesso';
      if (!quiet) message(error.message, 'error');
    } finally { $('refresh-status').disabled = false; }
  }
  function sourceParts(row) {
    const source = row.source || row.fonte || row.document || row.documento || '';
    const reference = row.url || (typeof source === 'string' ? source : source.url || source.caminho || source.path || '');
    const name = typeof source === 'string' ? source : source.title || source.titulo || source.nome_documento || source.name || reference;
    return {reference: String(reference || ''), name: String(name || 'Fonte não informada')};
  }
  function safeWebUrl(value) { try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) ? url.href : null; } catch { return null; } }
  function sourceMarkup(row) {
    const {reference, name} = sourceParts(row); const href = safeWebUrl(reference);
    return '<div class="source-line"><span aria-hidden="true">▤</span><span>' + (href ? '<a href="' + esc(href) + '" target="_blank" rel="noopener noreferrer">' + esc(name) + ' ↗</a>' : esc(name)) + '</span></div>';
  }
  function resultCard(row, index, semantic) {
    const expanded = row.result_origin === 'document_expansion';
    if (expanded) semantic = false;
    const categories = Array.isArray(row.categories) ? row.categories : Array.isArray(row.categorias) ? row.categorias : [];
    const count = row.category_count ?? row.quantidade_categorias ?? categories.length;
    const text = row.text || row.conteudo_fonte || row.resumo || row.trecho || '';
    const title = row.title || row.titulo || 'Tópico ' + (row.topic_id || row.topico_id || index + 1);
    const space = row.distance_space === 'umap_projection' ? 'Projeção UMAP' : row.distance_space === 'per_model' ? 'Distâncias por modelo' : 'Embeddings originais normalizados';
    const distanceNote = expanded ? '<p class="field-help">Incluído pelo documento completo; não foi um resultado direto da busca e não recebeu escore vetorial.</p>' : '<p class="field-help">' + esc(space) + (row.distance_space === 'umap_projection' ? ' · distância original: ' + number(row.original_distance) + ' · cosseno original: ' + number(row.original_cosine_similarity ?? row.cosine_similarity) : '') + '</p>';
    return '<article class="result-card"><div class="result-topline"><span class="rank">' + String(index + 1).padStart(2, '0') + (expanded ? ' / DOCUMENTO COMPLETO' : ' / TÓPICO') + '</span>' + (expanded ? '<span class="category-chip">Contexto do documento</span>' : '<span class="distance">Distância <b>' + number(row.distance ?? row.distancia) + '</b></span>') + '</div>' + distanceNote + '<h4>' + esc(title) + '</h4><p class="result-text result-snippet">' + esc(text) + '</p>' + (text.length > 250 ? '<details><summary>Ler conteúdo recuperado</summary><p class="result-text">' + esc(text) + '</p></details>' : '') + (semantic || categories.length ? '<div class="category-chips">' + (semantic ? '<span class="category-chip match-count">' + esc(count) + (Number(count) === 1 ? ' categoria no grupo' : ' categorias no grupo') + '</span>' : '') + categories.map(category => '<span class="category-chip">' + esc(label(typeof category === 'string' ? category : category.id || category.categoria)) + '</span>').join('') + '</div>' : '') + sourceMarkup(row) + '</article>';
  }
  function renderResults(data, query) {
    state.assistance?.render(data);
    if (state.lab?.isComparison(data)) { state.lab.render(data,query,'consultation'); return; }
    state.lab?.hideConsultation();
    state.results = data; state.resultQuery = query;
    $('results').hidden = false; $('search-empty').hidden = true; $('result-query').textContent = query;
    const metricName = {cosine:'Cosseno',euclidean:'Euclidiana',manhattan:'Manhattan'}[data.parameters?.metric] || '';
    $('result-context').textContent = [data.profile?.modelo || '',data.profile?.dimensoes ? data.profile.dimensoes + ' dimensões' : '',metricName ? 'Distância: ' + metricName : ''].filter(Boolean).join(' · ');
    const semantic = list(data.semantic_results); const baseline = list(data.baseline_results);
    $('semantic-count').textContent = semantic.length; $('direct-count').textContent = baseline.length;
    const noise = data.status === 'query_noise' || data.cluster?.noise === true || Number(data.cluster?.query_label) === -1;
    const laboratory = (data.mode || data.parameters?.mode) === 'laboratory';
    const vector = (data.mode || data.parameters?.mode) === 'vector';
    const umapOnly = (data.mode || data.parameters?.mode) === 'umap';
    document.querySelector('.result-column.direct').hidden = vector;
    document.querySelector('.result-columns').classList.toggle('single-result',vector);
    $('semantic-title').textContent = vector ? 'Similaridade vetorial' : laboratory ? 'Laboratório de vetores' : umapOnly ? 'Projeção UMAP' : 'Agrupamento semântico';
    $('semantic-subtitle').textContent = vector ? 'Conteúdo original da base' : laboratory ? 'Categorias e métrica selecionadas' : umapOnly ? 'Distância nas coordenadas projetadas, sem agrupamento' : 'Convergência entre categorias de vetores';
    $('semantic-results').innerHTML = semantic.length ? semantic.map((row, index) => resultCard(row, index, !laboratory && !vector && !umapOnly)).join('') : '<div class="empty-result"><strong>' + (noise ? 'A consulta ficou fora dos grupos.' : 'Nenhum tópico neste agrupamento.') + '</strong>' + (noise ? 'O HDBSCAN classificou a consulta como ruído. Consulte a similaridade direta ao lado ou avalie outra combinação no laboratório.' : data.status === 'insufficient_data' ? 'A base tem poucos vetores para esta configuração. Gere embeddings ou ajuste os parâmetros no laboratório.' : 'Verifique o perfil, a disponibilidade das categorias e a documentação da base.') + '</div>';
    $('baseline-results').innerHTML = baseline.length ? baseline.map((row, index) => resultCard(row, index, false)).join('') : '<div class="empty-result"><strong>Nenhum conteúdo disponível.</strong>' + (data.legacy ? 'A recuperação direta não foi registrada nesta execução legada.' : 'Gere vetores da categoria Conteúdo-fonte para habilitar a comparação por distância direta.') + '</div>';
    warnings(data.warnings, 'search-warnings');
    renderProjection(data.projection || (data.cluster?.space !== "original_l2_normalized" ? data.cluster?.points : []), data.cluster?.query_label);
    renderDendrogram(data.dendrogram);
  }
  function renderProjection(points, queryLabel, chart = $('projection-chart'), panel = $('projection-panel')) {
    points = Array.isArray(points) ? points : points?.points || [];
    points = list(points).map(point => ({...point,x:point.x ?? point.coordinates?.[0],y:point.y ?? point.coordinates?.[1]})).filter(point => Number.isFinite(Number(point.x)) && Number.isFinite(Number(point.y)));
    panel.hidden = !points.length; chart.replaceChildren();
    if (!points.length) return;
    const width = 900, height = 390, pad = 35;
    const xs = points.map(point => Number(point.x)), ys = points.map(point => Number(point.y));
    const minX = xs.reduce((a,b) => Math.min(a,b),Infinity), maxX = xs.reduce((a,b) => Math.max(a,b),-Infinity), minY = ys.reduce((a,b) => Math.min(a,b),Infinity), maxY = ys.reduce((a,b) => Math.max(a,b),-Infinity);
    const x = value => pad + (Number(value) - minX) / (maxX - minX || 1) * (width - 2 * pad);
    const y = value => height - pad - (Number(value) - minY) / (maxY - minY || 1) * (height - 2 * pad);
    const ns = 'http://www.w3.org/2000/svg'; const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Projeção semântica dos vetores e da consulta em duas dimensões');
    const rect = document.createElementNS(ns, 'rect'); Object.entries({x:1, y:1, width:898, height:388, rx:8, class:'chart-background'}).forEach(([key, value]) => rect.setAttribute(key,value)); svg.append(rect);
    points.sort((a, b) => Number(Boolean(a.is_query)) - Number(Boolean(b.is_query))).forEach(point => {
      const isQuery = point.is_query || point.category === 'query' || point.category === 'consulta';
      const cluster = point.cluster ?? point.label; const noise = Number(cluster) === -1;
      const inGroup = cluster != null && queryLabel != null && !noise && String(cluster) === String(queryLabel);
      const circle = document.createElementNS(ns, 'circle');
      Object.entries({cx:x(point.x), cy:y(point.y), r:isQuery ? 7 : inGroup ? 4.5 : 3, fill:isQuery ? '#153f4a' : noise ? '#d9dcd5' : inGroup ? '#5a9b76' : '#bacbc7', stroke:isQuery ? '#fff' : 'none', 'stroke-width':2, tabindex:0}).forEach(([key,value]) => circle.setAttribute(key, value));
      const title = document.createElementNS(ns, 'title'); title.textContent = (isQuery ? 'Sua consulta' : (point.title || point.topic_id || 'Vetor') + ' · ' + label(point.category || '')) + (cluster == null ? ' · sem agrupamento' : ' · grupo ' + cluster); circle.append(title); svg.append(circle);
    });
    chart.append(svg);
  }
  function renderDendrogram(data, chart = $('dendrogram-chart'), panel = $('dendrogram-panel')) {
    chart.replaceChildren(); panel.hidden = true;
    if (!data?.svg || typeof data.svg !== 'string') return;
    const parsed = new DOMParser().parseFromString(data.svg, 'image/svg+xml');
    const root = parsed.documentElement;
    if (root.localName !== 'svg' || parsed.querySelector('parsererror')) return;
    const allowed = new Set(['svg','g','path','line','polyline','polygon','rect','circle','ellipse','text','tspan','title','desc','defs','clipPath']);
    const attrs = new Set(['viewBox','width','height','x','y','x1','x2','y1','y2','cx','cy','rx','ry','r','d','points','fill','stroke','stroke-width','stroke-linecap','stroke-linejoin','stroke-dasharray','opacity','fill-opacity','stroke-opacity','transform','text-anchor','font-size','font-family','font-weight','dy','dx','id','clip-path','version','preserveAspectRatio']);
    for (const node of [root, ...root.querySelectorAll('*')]) {
      if (!allowed.has(node.localName)) { node.remove(); continue; }
      for (const attribute of [...node.attributes]) {
        if (!attrs.has(attribute.name) || /(?:javascript:|data:|https?:|url\((?!#[A-Za-z_]))/i.test(attribute.value)) node.removeAttribute(attribute.name);
      }
    }
    root.setAttribute('role','img'); root.setAttribute('aria-label','Dendrograma HDBSCAN desta consulta');
    chart.append(document.importNode(root, true)); panel.hidden = false;
  }
  function updateActiveDescription() {
    if (state.lab) { state.lab.describe(); return; }
    const config = state.config; const semantic = config.mode === 'semantic';
    $('active-method').textContent = semantic ? 'Agrupamento semântico' : 'Laboratório de vetores';
    $('active-selection').textContent = config.categories.map(label).join(' · ');
    $('method-description').textContent = semantic ? 'A consulta participa do agrupamento UMAP + HDBSCAN. Os ' + config.top_k + ' tópicos com mais categorias no mesmo grupo são selecionados e ordenados pela distância vetorial.' : 'A recuperação usa as categorias de vetores e a distância ' + (config.metric === 'cosine' ? 'cosseno' : 'euclidiana') + ' escolhidas no laboratório. O conteúdo-fonte é recuperado em paralelo para comparação.';
  }
  function setBusy(id, busy, progressId) { $(id).disabled = busy; if (progressId) $(progressId).hidden = !busy; }
  async function search(event) {
    event.preventDefault(); const query = $('query').value.trim();
    if (!query) { $('query').focus(); return; }
    if (!state.config.profile_id) { message('Cadastre e selecione um perfil de embeddings no Laboratório para consultar a base.', 'warning'); navigate('laboratorio'); return; }
    if (!state.config.categories.length) { message('Selecione pelo menos uma categoria no Laboratório.', 'warning'); navigate('laboratorio'); return; }
    const controller = new AbortController(); state.searchController = controller;
    setBusy('search-button', true, 'search-progress'); $('search-progress-label').textContent = 'Executando a recuperação configurada…'; message('');
    try {
      await state.assistance?.ready();
      const result = await request('/api/search', {method:'POST', body:{query,...state.assistance?.settings()}, signal:controller.signal, progress: text => $('search-progress-label').textContent = text});
      renderResults(result, query); refreshHistory();
    } catch (error) { message(isAbort(error) ? 'A espera foi interrompida. O processamento já iniciado pode continuar no servidor; consulte o histórico antes de repetir.' : error.message, isAbort(error) ? 'warning' : 'error'); }
    finally { state.searchController = null; setBusy('search-button', false, 'search-progress'); }
  }
  function updateLLMFields() {
    const enabled = $('llm-enabled').checked;
    $('llm-fields').disabled = !enabled;
    $('llm-extractive-note').hidden = enabled;
    const provider = state.llms?.providers?.find(row => row.provider === $('llm-provider').value);
    $('llm-availability').textContent = enabled ? (provider?.detail || 'A disponibilidade do provedor será verificada no servidor.') : 'Modo extrativo selecionado explicitamente, sem uso de LLM.';
    $('llm-custom-choice').hidden = $('llm-model').value !== '__custom__';
  }
  function renderLLMOptions() {
    const provider = $('llm-provider').value;
    const previous = $('llm-model').value;
    const entry = state.llms?.providers?.find(row => row.provider === provider);
    const fallback = provider === 'openai' ? [{id:'gpt-6-luna',label:'GPT-6 Luna'},{id:'gpt-6.1-sol',label:'GPT-6.1 Sol'},{id:'gpt-6-astra',label:'GPT-6 Astra'}] : [];
    const models = list(entry?.models || fallback);
    $('llm-model').replaceChildren();
    models.forEach(model => $('llm-model').add(new Option(model.label || model.id,model.id)));
    $('llm-model').add(new Option('Outro modelo…','__custom__'));
    if (models.some(model => model.id === previous) || previous === '__custom__') $('llm-model').value = previous;
    $('llm-model-choice').hidden = models.length === 0;
    updateLLMFields();
  }
  async function refreshLLMs() {
    if (state.llmsLoading) return;
    state.llmsLoading = true;
    try { state.llms = await request('/api/llms'); renderLLMOptions(); }
    catch(error) { $('llm-availability').textContent = 'Não foi possível verificar o provedor: ' + error.message; }
    finally { state.llmsLoading = false; }
  }
  function readLLM() {
    if (!$('llm-enabled').checked) return {provider:'extractive'};
    const model = ($('llm-model').value === '__custom__' ? $('llm-custom-model').value : $('llm-model').value).trim();
    if (!model) throw new Error('Informe o nome do modelo de linguagem para preparar o conteúdo ou escolha o modo extrativo sem LLM.');
    return {provider:$('llm-provider').value,model};
  }
  function readMetadata() {
    let extra = {}; const raw = $('meta-extra').value.trim();
    if (raw) { try { extra = JSON.parse(raw); } catch { throw new Error('Os metadados adicionais precisam ser um objeto JSON válido.'); } }
    if (!extra || Array.isArray(extra) || typeof extra !== 'object') throw new Error('Informe um objeto JSON nos metadados adicionais.');
    return {...extra, title: $('meta-title').value.trim(), orgao:$('meta-org').value.trim(), eixo:$('meta-axis').value.trim(), idioma:$('meta-lang').value};
  }
  function captureIngestion() {
    const sources = $('sources').value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
    if (!sources.length && !state.selectedFiles.length) throw new Error('Informe ao menos um caminho, URL ou arquivo para pré-processar.');
    return {sources, metadata:readMetadata(), llm:readLLM(), files:[...state.selectedFiles]};
  }
  function ingestBody(snapshot, commit) {
    const form = new FormData(); form.set('sources', JSON.stringify(snapshot.sources)); form.set('metadata', JSON.stringify(snapshot.metadata)); form.set('commit', String(commit));
    form.set('llm',JSON.stringify(snapshot.llm));
    snapshot.files.forEach(file => form.append('files', file, file.name));
    if (commit && state.preview?.preview_id) form.set('preview_id', state.preview.preview_id);
    return form;
  }
  function previewDocuments(data) {
    if (Array.isArray(data.preview)) return data.preview;
    if (data.preview?.documents) return list(data.preview.documents);
    if (Array.isArray(data.documents)) return data.documents;
    return list(data.documentos);
  }
  function previewTopicMarkup(topic) {
    const original = topic.conteudo_fonte || topic.text || '';
    const summary = topic.resumo || topic.summary || '';
    const answer = topic.resposta_sintese || topic.answer || '';
    const rawQuestions = topic.perguntas_sugeridas || topic.suggested_questions || [];
    const questions = Array.isArray(rawQuestions) ? rawQuestions : typeof rawQuestions === 'string' ? rawQuestions.split(/\r?\n/).filter(Boolean) : [];
    const provenance = topic.processamento?.llm || topic.provenance || {};
    const provenanceText = [provenance.provider,provenance.model,provenance.prompt_version ? 'prompt ' + provenance.prompt_version : ''].filter(Boolean).join(' · ');
    return '<section class="preview-topic"><h4>' + esc(topic.titulo || topic.title || topic.topico_id || 'Tópico') + '</h4>' +
      (provenanceText ? '<p class="preview-provenance">Preparação: ' + esc(provenanceText) + '</p>' : '') +
      '<dl><dt>Resumo</dt><dd>' + (summary ? esc(summary) : '<span class="preview-missing">Não fornecido nesta representação.</span>') + '</dd>' +
      '<dt>Resposta-síntese</dt><dd>' + (answer ? esc(answer) : '<span class="preview-missing">Não fornecida nesta representação.</span>') + '</dd></dl>' +
      '<h5>Perguntas sugeridas</h5>' + (questions.length ? '<ul>' + questions.map(question => '<li>' + esc(question) + '</li>').join('') + '</ul>' : '<p class="preview-missing">Nenhuma pergunta fornecida.</p>') +
      (original ? '<details class="preview-original"><summary>Conferir texto original</summary><p>' + esc(original) + '</p></details>' : '') + '</section>';
  }
  function renderPreview(data) {
    const documents = previewDocuments(data);
    $('preview-count').textContent = documents.length + (documents.length === 1 ? ' DOCUMENTO' : ' DOCUMENTOS');
    $('preview-documents').innerHTML = documents.map((doc, index) => {
      const topics = list(doc.topics || doc.topicos); const content = doc.text || doc.conteudo || doc.conteudo_fonte || '';
      return '<article class="preview-document"><h3>' + esc(doc.title || doc.titulo || doc.nome_documento || 'Documento ' + (index + 1)) + '</h3>' + sourceMarkup(doc) + '<p class="preview-text">' + esc(content) + '</p>' + (topics.length ? '<details><summary>' + topics.length + ' tópicos extraídos · revisar amostra de até 8</summary>' + topics.slice(0, 8).map(previewTopicMarkup).join('') + '</details>' : '') + (doc.metadados || doc.metadata ? '<details><summary>Metadados extraídos</summary><pre>' + esc(JSON.stringify(doc.metadata || doc.metadados, null, 2)) + '</pre></details>' : '') + '</article>';
    }).join('') || '<p class="empty-inline">O serviço não retornou documentos para revisão. Consulte os avisos e revise as fontes.</p>';
    $('review-section').hidden = false; $('commit-button').textContent = 'Incorporar à base ✓'; $('review-note').textContent = 'A base será atualizada ao incorporar.'; $('commit-button').disabled = (!documents.length && !Number(data.documents || 0)) || !data.preview_id;
    $('step-review').classList.add('active'); $('step-sources').classList.remove('active');
    warnings(data.warnings, 'ingest-warnings');
  }
  async function preview(event) {
    event.preventDefault(); let snapshot; const revision = state.ingestRevision;
    try { snapshot = captureIngestion(); } catch (error) { message(error.message, 'error'); return; }
    setBusy('preview-button', true, 'ingest-progress'); $('commit-button').disabled = true; message('');
    try {
      const data = await request('/api/ingest', {method:'POST', body:ingestBody(snapshot, false), progress:text => $('ingest-progress-label').textContent = text});
      if (revision !== state.ingestRevision) throw new Error('As fontes ou os metadados foram alterados durante o processamento. Pré-processe novamente para revisar a versão atual.');
      state.preview = data; state.previewForm = snapshot; renderPreview(data);
      $('review-section').scrollIntoView({behavior:'smooth',block:'start'});
    } catch (error) { message(error.message, 'error'); state.preview = null; state.previewForm = null; $('review-section').hidden = true; }
    finally { setBusy('preview-button', false, 'ingest-progress'); }
  }
  async function commit() {
    if (!state.previewForm || !state.preview?.preview_id) { message('Pré-processe novamente para obter uma prévia válida antes de incorporar.', 'warning'); return; }
    setBusy('commit-button', true, 'ingest-progress'); $('ingest-progress-label').textContent = 'Incorporando documentos à base…';
    $('preview-button').disabled = true;
    try {
      const data = await request('/api/ingest', {method:'POST', body:ingestBody(state.previewForm, true), progress:text => $('ingest-progress-label').textContent = text});
      warnings(data.warnings, 'ingest-warnings'); message('Documentos incorporados. Agora gere os embeddings das categorias desejadas.', 'success');
      $('review-note').textContent = 'Importação incorporada à base.'; $('commit-button').textContent = 'Incorporado à base ✓';
      $('step-review').classList.remove('active'); $('step-embeddings').classList.add('active');
      state.previewForm = null; await refreshStatus({quiet:true});
      document.querySelector('.generation-panel').scrollIntoView({behavior:'smooth',block:'start'});
    } catch (error) { message(error.message, 'error'); $('commit-button').disabled = false; }
    finally { $('ingest-progress').hidden = true; $('preview-button').disabled = false; }
  }
  function invalidatePreview() {
    state.ingestRevision += 1;
    if (!state.previewForm) return;
    state.previewForm = null; state.preview = null; $('review-section').hidden = true;
    $('step-review').classList.remove('active'); $('step-sources').classList.add('active');
    $('commit-button').textContent = 'Incorporar à base ✓';
  }
  function renderFiles() {
    $('file-list').replaceChildren();
    state.selectedFiles.forEach((file, index) => {
      const li = document.createElement('li'); const name = document.createElement('span'); name.textContent = file.name + ' · ' + (file.size / 1024).toFixed(0) + ' KB';
      const button = document.createElement('button'); button.type = 'button'; button.textContent = '×'; button.setAttribute('aria-label','Remover ' + file.name); button.addEventListener('click',() => {state.selectedFiles.splice(index,1); renderFiles(); invalidatePreview();});
      li.append(name,button); $('file-list').append(li);
    });
  }
  async function generate(event) {
    event.preventDefault(); const profile = $('generate-profile').value; const categories = selectedCategories('generate');
    if (!profile || !categories.length) { message('Selecione um perfil e pelo menos uma categoria para gerar embeddings.', 'warning'); return; }
    setBusy('generate-button', true, 'generate-progress'); $('generate-result').hidden = true;
    try {
      const data = await request('/api/generate',{method:'POST',body:{profile_id:profile,categories,operation:$('generate-operation').value},progress:text => $('generate-progress-label').textContent = text});
      message('Geração de embeddings concluída. Consulte o inventário atualizado abaixo.', 'success', 'generate-result');
      if (data.warnings?.length) message('Geração concluída com avisos: ' + data.warnings.join(' '), 'warning', 'generate-result');
      await refreshStatus({quiet:true});
    } catch (error) { message(error.message,'error','generate-result'); }
    finally { setBusy('generate-button', false, 'generate-progress'); }
  }
  async function importPreparedBase(event) {
    event.preventDefault();
    const baseFile = $('prepared-base-file').files[0];
    if (!baseFile) { message('Selecione a base preparada em JSON.','warning','base-import-result'); return; }
    const body = new FormData(); body.append('base_file',baseFile);
    const legacyIndex = $('prepared-base-index').files[0];
    if (legacyIndex) body.append('legacy_npz',legacyIndex);
    setBusy('import-base-button',true,'base-import-progress'); $('export-base').disabled = true;
    $('base-import-progress-label').textContent = 'Importando a base preparada…'; message('','','base-import-result');
    try {
      const result = await request('/api/base/import',{method:'POST',body,progress:text => $('base-import-progress-label').textContent = text});
      const count = value => new Intl.NumberFormat('pt-BR').format(Number(value) || 0);
      let summary = 'Importação concluída: ' + count(result.documents) + ' documentos e ' + count(result.topics) + ' tópicos';
      if (Number.isInteger(result.active_topics) && Number.isInteger(result.inactive_topics)) summary += ' (' + count(result.active_topics) + ' ativos e ' + count(result.inactive_topics) + ' inativos)';
      summary += '.';
      if (Number(result.legacy_embeddings) > 0) summary += ' ' + count(result.legacy_embeddings) + ' embeddings legados preservados em perfil próprio.';
      invalidatePreview(); $('base-import-form').reset();
      $('step-sources').classList.remove('active'); $('step-review').classList.remove('active'); $('step-embeddings').classList.add('active');
      message(summary,'success','base-import-result'); await refreshStatus({quiet:true});
    } catch(error) { message(error.message,'error','base-import-result'); }
    finally { setBusy('import-base-button',false,'base-import-progress'); $('export-base').disabled = false; }
  }
  async function exportBase() {
    $('export-base').disabled = true;
    try { const base = await request('/api/base/export'); exportJSON(base,'autorag-base.json'); message('Base atual exportada em JSON.','success','base-import-result'); }
    catch(error) { message(error.message,'error','base-import-result'); }
    finally { $('export-base').disabled = false; }
  }
  function readLab() {
    const config = {
      profile_id:$('search-profile').value, categories:selectedCategories('search'), mode:document.querySelector('input[name=mode]:checked').value,
      metric:$('metric').value, top_k:Number($('top-k').value), n_neighbors:Number($('n-neighbors').value), min_cluster_size:Number($('min-cluster-size').value), min_samples:Number($('min-samples').value), n_components:Number($('n-components').value), dendrogram:$('dendrogram').checked, view:$('lab-view').value,
      minimum:$('score-min').value.trim() ? Number($('score-min').value) : null, maximum:$('score-max').value.trim() ? Number($('score-max').value) : null, partial:$('partial-coverage').checked, individual_rule:$('individual-rule').value
    };
    const rawLimits = $('individual-limits').value.trim();
    if (rawLimits) {
      try { config.individual_limits = JSON.parse(rawLimits); } catch { throw new Error('Informe um objeto JSON válido nos mínimos por categoria.'); }
      if (!config.individual_limits || typeof config.individual_limits !== 'object' || Array.isArray(config.individual_limits) || Object.entries(config.individual_limits).some(([key,value]) => !state.categories.some(category => category.id === key) || typeof value !== 'number' || !Number.isFinite(value) || value < -1 || value > 1)) throw new Error('Os mínimos devem usar IDs de categorias válidas com escores entre -1 e 1.');
    }
    if (config.minimum !== null && config.maximum !== null && config.minimum > config.maximum) throw new Error('O escore mínimo não pode superar o máximo.');
    if (!config.profile_id) throw new Error('Cadastre ou selecione um perfil de embeddings.');
    if (!config.categories.length) throw new Error('Selecione pelo menos uma categoria para a recuperação.');
    return config;
  }
  function applyLab(event) {
    if (state.lab) { state.lab.saveEditor(event); return; }
    event.preventDefault();
    try { state.config = readLab(); updateActiveDescription(); $('lab-state').textContent = 'Configuração aplicada a partir da próxima consulta.'; message('Configuração do laboratório aplicada à próxima consulta.', 'success'); navigate('consulta'); $('query').focus(); }
    catch (error) { message(error.message,'warning'); }
  }
  function resetLab() {
    if (state.lab) { state.lab.resetEditor(); return; }
    state.config = {...DEFAULT,categories:[...DEFAULT_CATEGORIES],profile_id:state.config.profile_id};
    renderCategories();
    Object.entries({'metric':'metric','top_k':'top-k','n_neighbors':'n-neighbors','min_cluster_size':'min-cluster-size','min_samples':'min-samples','n_components':'n-components'}).forEach(([key,id]) => $(id).value = DEFAULT[key]);
    $('lab-view').value = 'topicos'; $('score-min').value = ''; $('score-max').value = ''; $('partial-coverage').checked = false; $('individual-limits').value = ''; $('individual-rule').value = 'todos';
    $('dendrogram').checked = false; document.querySelector('input[name=mode][value=semantic]').checked = true;
    $('lab-state').textContent = 'Configuração padrão restaurada.'; updateActiveDescription();
  }
  async function registerProfile(event) {
    event.preventDefault(); const button = event.submitter; button.disabled = true;
    try {
      const body = {provider:$('provider').value,model:$('model').value.trim(),dimensions:Number($('dimensions').value),activate:false};
      if ($('endpoint').value.trim()) body.endpoint = $('endpoint').value.trim();
      const result = await request('/api/profiles',{method:'POST',body});
      state.config.profile_id = result.perfil_id || result.profile?.perfil_id || state.config.profile_id;
      await refreshStatus({quiet:true}); selectModelProfile(state.config.profile_id);
      await refreshModels(); message('Perfil cadastrado. O modelo deve estar disponível no serviço para geração e consulta.', 'success');
    } catch(error) { message(error.message,'error'); }
    finally { button.disabled = false; }
  }
  function modelKey(row) { return row.provedor + '/' + row.modelo; }
  function modelProfiles(row) {
    const ids = list(row.profile_ids);
    return list(state.status?.profiles).filter(profile => ids.includes(profile.perfil_id));
  }
  function selectModelProfile(id) {
    if (!id) return;
    state.config.profile_id = id;
    document.querySelectorAll('.profile-select').forEach(select => select.value = id);
    state.lab?.selectProfile(id);
    $('lab-state').textContent = 'Perfil selecionado para geração e para o experimento em edição.';
  }
  function renderModels() {
    $('local-models').replaceChildren(); $('api-models').replaceChildren();
    const names = {'bge-m3':'BGE-M3','embeddinggemma:300m-qat-q4_0':'EmbeddingGemma 300M · Q4','intfloat/multilingual-e5-small':'Multilingual E5 Small','sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2':'Multilingual MiniLM · L12'};
    list(state.models).forEach(row => {
      const local = ['ollama','sentence_transformers'].includes(row.provedor);
      const profiles = modelProfiles(row);
      const profile = profiles.find(item => item.perfil_id === state.config.profile_id) || profiles.find(item => item.dimensoes === row.dimensoes) || profiles[0];
      const busy = state.modelBusy?.has(modelKey(row));
      const card = document.createElement('article'); card.className = 'model-card';
      const heading = document.createElement('div'); heading.className = 'model-card-heading';
      const title = document.createElement('h4'); title.textContent = names[row.modelo.replace(/:latest$/,'')] || row.modelo;
      const badge = document.createElement('span'); badge.className = 'pill ' + (row.disponivel ? '' : 'subtle');
      badge.textContent = row.disponivel ? (local ? 'INSTALADO' : 'API CONFIGURADA') : (local ? 'INDISPONÍVEL' : 'SEM CREDENCIAL');
      heading.append(title,badge); card.append(heading);
      const identifier = document.createElement('p'); identifier.className = 'model-identifier'; identifier.textContent = row.modelo; card.append(identifier);
      const description = document.createElement('p'); description.className = 'model-description'; description.textContent = row.descricao || ''; card.append(description);
      const dimensions = document.createElement('p'); dimensions.className = 'model-dimensions';
      dimensions.textContent = [row.dimensoes ? row.dimensoes + ' dimensões' : 'Dimensão a verificar',row.dimensoes_nativas && row.dimensoes_nativas !== row.dimensoes ? 'nativa: ' + row.dimensoes_nativas : '',row.provedor,profiles.length ? profiles.length + ' perfil cadastrado' + (profiles.length > 1 ? 's' : '') : ''].filter(Boolean).join(' · '); card.append(dimensions);
      const availability = document.createElement('p'); availability.className = 'field-help'; availability.textContent = row.detalhe || ''; card.append(availability);
      const actions = document.createElement('div'); actions.className = 'model-actions';
      const primary = document.createElement('button'); primary.type = 'button'; primary.className = 'button secondary small';
      primary.disabled = Boolean(busy);
      if (!row.disponivel) {
        primary.textContent = local ? 'Ver orientação' : 'Como configurar a API';
        primary.addEventListener('click',() => message(local ? 'Este modelo ainda não está disponível no servidor de processamento. A instalação precisa ser concluída no servidor antes de cadastrar e gerar embeddings.' : 'Configure a credencial de ' + row.provedor + ' no servidor de processamento e clique em Verificar disponibilidade. A consulta do catálogo não chama o serviço de embeddings.', 'warning','models-message'));
      } else if (profile) {
        primary.textContent = profile.perfil_id === state.config.profile_id ? 'Perfil selecionado ✓' : 'Usar perfil';
        primary.addEventListener('click',() => {selectModelProfile(profile.perfil_id);renderModels();message('Perfil selecionado. As categorias e os parâmetros do laboratório foram preservados.','success','models-message');});
      } else {
        primary.textContent = busy ? 'Cadastrando…' : 'Cadastrar perfil';
        primary.addEventListener('click',() => registerCatalogModel(row));
      }
      actions.append(primary);
      if (local && row.disponivel && profile) {
        const test = document.createElement('button'); test.type = 'button'; test.className = 'text-button'; test.disabled = Boolean(busy); test.textContent = busy ? 'Verificando…' : 'Testar modelo';
        test.setAttribute('aria-label','Testar modelo ' + title.textContent); test.addEventListener('click',() => testCatalogModel(row,profile.perfil_id)); actions.append(test);
      }
      card.append(actions); $(local ? 'local-models' : 'api-models').append(card);
    });
    if (!$('local-models').children.length) $('local-models').innerHTML = '<p class="empty-inline">Nenhum modelo local encontrado no catálogo.</p>';
  }
  async function refreshModels() {
    if (state.modelsLoading) return;
    state.modelsLoading = true; $('refresh-models').disabled = true;
    try { const data = await request('/api/models'); state.models = list(data.models); renderModels(); }
    catch(error) { message(error.message,'error','models-message'); if (!state.models) $('local-models').innerHTML = '<p class="empty-inline">Não foi possível verificar os modelos. Confira a conexão e tente novamente.</p>'; }
    finally { state.modelsLoading = false; $('refresh-models').disabled = false; }
  }
  async function registerCatalogModel(row) {
    state.modelBusy ||= new Set(); state.modelBusy.add(modelKey(row)); renderModels();
    message('Validando a instalação e cadastrando o perfil…','','models-message');
    try {
      const profile = await request('/api/profiles',{method:'POST',body:{provider:row.provedor,model:row.modelo,dimensions:row.dimensoes,activate:false}});
      state.config.profile_id = profile.perfil_id;
      await refreshStatus({quiet:true}); selectModelProfile(profile.perfil_id); await refreshModels();
      message('Perfil cadastrado e selecionado. Você já pode gerar embeddings das categorias desejadas.','success','models-message');
    } catch(error) { message(error.message,'error','models-message'); }
    finally { state.modelBusy.delete(modelKey(row)); renderModels(); }
  }
  async function testCatalogModel(row,profileId) {
    state.modelBusy ||= new Set(); state.modelBusy.add(modelKey(row)); renderModels();
    message('Testando duas entradas sintéticas no modelo local…','','models-message');
    try {
      const result = await request('/api/models/check',{method:'POST',body:{profile_id:profileId},progress:text => message(text,'','models-message')});
      if (result.finite !== true || result.quantity !== 2) throw new Error('O serviço não confirmou a integridade dos dois embeddings de teste.');
      message('Teste concluído: ' + result.quantity + ' vetores de ' + result.dimensions + ' dimensões, coordenadas finitas. Normas: consulta ' + number(result.norms?.consulta) + ', documento ' + number(result.norms?.documento) + '. Tempo: ' + new Intl.NumberFormat('pt-BR',{maximumFractionDigits:0}).format(result.latency_ms) + ' ms. A base foi preservada.','success','models-message');
    } catch(error) { message(error.message,'error','models-message'); }
    finally { state.modelBusy.delete(modelKey(row)); renderModels(); }
  }
  function date(value) { const time = new Date(value); return Number.isNaN(time.getTime()) ? String(value || '') : time.toLocaleString('pt-BR'); }
  async function refreshHistory() {
    state.lab?.refreshHistory();
    $('refresh-history').disabled = true;
    try {
      const data = await request('/api/experiments'); const rows = list(data.experiments || data.execucoes || data);
      $('history-list').replaceChildren();
      if (!rows.length) { $('history-list').innerHTML = '<p class="empty-inline">Os experimentos salvos aparecerão aqui após a primeira consulta.</p>'; return; }
      rows.forEach(row => {
        const id = row.id || row.execution_id || row.execucao_id || row.experiment_id; if (!id) return;
        const wrapper = document.createElement('div'); wrapper.className = 'history-row';
        const content = document.createElement('div'); const title = document.createElement('strong'); title.textContent = row.query || row.texto_original_entrada || id;
        const meta = document.createElement('p'); meta.textContent = [date(row.created_at || row.criado_em || row.date || row.data || row.timestamp), row.perfil?.modelo || row.model || '', row.status || ''].filter(Boolean).join(' · ');
        content.append(title,meta); const button = document.createElement('button'); button.type = 'button'; button.className = 'text-button'; button.textContent = 'Abrir ↗'; button.addEventListener('click',() => loadExperiment(id)); wrapper.append(content,button); $('history-list').append(wrapper);
      });
    } catch(error) { $('history-list').innerHTML = '<p class="empty-inline">' + esc(error.message) + '</p>'; }
    finally { $('refresh-history').disabled = false; }
  }
  function experimentResult(data) {
    const result = data.result || data.resultado || data;
    if (Array.isArray(result.semantic_results) || Array.isArray(result.baseline_results)) return result;
    if (!Array.isArray(result.resultados)) return result;
    return {...result, mode:'laboratory', legacy:true, profile:result.perfil,
      parameters:{...result.parametros,metric:'cosine'},
      warnings:['Execução do laboratório original. A recuperação direta de conteúdo-fonte não foi registrada neste experimento.'],
      semantic_results:result.resultados.map(row => ({...row,topic_id:row.topico_id,title:row.titulo,
        text:row.texto_fonte_recuperado || list(row.comparacoes).map(item => label(item.categoria) + ': ' + (item.texto_fonte_recuperado || '')).join('\n\n'),
        distance:row.distancia_cosseno ?? row.distancia_cosseno_media,
        categories:row.categoria ? [row.categoria] : [...new Set(list(row.comparacoes).map(item => item.categoria))],source:row.fonte})),
      baseline_results:[], projection:[], cluster:{}
    };
  }
  async function loadExperiment(id) {
    try {
      const data = await request('/api/experiments/' + encodeURIComponent(id)); state.selectedExperiment = data;
      $('experiment-title').textContent = 'Experimento ' + id;
      $('experiment-summary').textContent = data.query || data.texto_original_entrada || 'Dados e parâmetros da execução salva.';
      $('experiment-json').textContent = JSON.stringify(data,null,2);
      const result = experimentResult(data);
      $('open-experiment').hidden = !Array.isArray(result.semantic_results) && !Array.isArray(result.baseline_results);
      $('experiment-dialog').showModal();
    } catch(error) { message(error.message,'error'); }
  }
  function exportJSON(data, name) {
    if (!data) return; const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json;charset=utf-8'}));
    const link = document.createElement('a'); link.href = url; link.download = name; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url),1000);
  }
  async function downloadAuthenticated(path, filename) {
    const headers = {}; if (token) headers.Authorization = 'Bearer ' + token;
    const response = await fetch(apiUrl(path),{headers,credentials:'same-origin',redirect:'error'});
    if (!response.ok) { const data = await response.json().catch(() => ({})); throw new Error(typeof data.detail === 'string' ? data.detail : 'Não foi possível baixar o arquivo (' + response.status + ').'); }
    const url = URL.createObjectURL(await response.blob()); const link = document.createElement('a'); link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url),1000);
  }
  function settings() { $('backend-url').value = backend; $('access-token').value = token; $('settings-dialog').showModal(); }
  function saveSettings(event) {
    event.preventDefault(); const raw = $('backend-url').value.trim().replace(/\/+$/,'');
    if (raw) { const url = new URL(raw); if (!['https:','http:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) { message('Informe uma URL HTTP ou HTTPS, sem credenciais, parâmetros ou fragmentos.', 'error'); return; } }
    backend = raw; token = $('access-token').value.trim(); session.set('autorag_backend',backend); session.set('autorag_token',token);
    state.models = null; state.llms = null;
    $('settings-dialog').close(); state.assistance?.resetConnection(); refreshStatus(); state.lab?.refresh();
    if (state.activePage === 'laboratorio') refreshModels();
    if (state.activePage === 'preparar') refreshLLMs();
  }
  document.querySelectorAll('[data-page]').forEach(button => button.addEventListener('click',() => navigate(button.dataset.page)));
  document.querySelectorAll('[data-goto]').forEach(button => button.addEventListener('click',() => navigate(button.dataset.goto)));
  document.querySelector('.brand').addEventListener('click', event => {event.preventDefault();navigate('consulta');});
  $('refresh-status').addEventListener('click',() => refreshStatus()); $('search-form').addEventListener('submit',search);
  $('cancel-search').addEventListener('click',() => state.searchController?.abort()); $('lab-form').addEventListener('submit',applyLab); $('reset-lab').addEventListener('click',resetLab);
  $('ingest-form').addEventListener('submit',preview); $('ingest-form').addEventListener('input',invalidatePreview); $('commit-button').addEventListener('click',commit);
  $('base-import-form').addEventListener('submit',importPreparedBase); $('export-base').addEventListener('click',exportBase);
  $('files').addEventListener('change',() => {state.selectedFiles.push(...$('files').files);$('files').value = '';renderFiles();invalidatePreview();});
  const zone = document.querySelector('.upload-zone');
  ['dragenter','dragover'].forEach(name => zone.addEventListener(name,event => {event.preventDefault();zone.classList.add('dragging');}));
  ['dragleave','drop'].forEach(name => zone.addEventListener(name,event => {event.preventDefault();zone.classList.remove('dragging');}));
  zone.addEventListener('drop',event => {state.selectedFiles.push(...event.dataTransfer.files);renderFiles();invalidatePreview();});
  $('generate-form').addEventListener('submit',generate); $('profile-form').addEventListener('submit',registerProfile); $('refresh-history').addEventListener('click',refreshHistory);
  $('refresh-models').addEventListener('click',refreshModels);
  $('llm-enabled').addEventListener('change',() => {updateLLMFields();invalidatePreview();});
  $('llm-provider').addEventListener('change',() => {renderLLMOptions();invalidatePreview();});
  $('llm-model').addEventListener('change',() => {updateLLMFields();invalidatePreview();});
  $('llm-custom-model').addEventListener('input',invalidatePreview);
  $('open-settings').addEventListener('click',settings); $('close-settings').addEventListener('click',() => $('settings-dialog').close()); $('settings-form').addEventListener('submit',saveSettings);
  $('clear-token').addEventListener('click',() => {$('access-token').value = '';});
  $('close-experiment').addEventListener('click',() => $('experiment-dialog').close());
  $('export-results').addEventListener('click',() => exportJSON({...state.results, query:state.resultQuery},'autorag-consulta.json'));
  $('export-experiment').addEventListener('click',() => exportJSON(state.selectedExperiment,'autorag-experimento.json'));
  $('open-experiment').addEventListener('click',() => {const data = state.selectedExperiment;renderResults(experimentResult(data),data.query || data.texto_original_entrada || 'Experimento salvo');$('experiment-dialog').close();navigate('consulta');});
  window.addEventListener('hashchange',() => navigate(location.hash.slice(1),false));
  renderCategories();
  state.assistance = AutoRAGAssistance.init({$,request,message,state,esc,number,refreshStatus});
  state.lab = AutoRAGLaboratory.init({$,request,message,state,esc,label,number,date,navigate,resultCard,renderProjection,renderDendrogram,download:downloadAuthenticated});
  updateActiveDescription(); navigate(location.hash.slice(1),false); refreshStatus();
})();
