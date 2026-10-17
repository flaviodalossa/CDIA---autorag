'use strict';
(() => {
  const METHODS = {vector: 'Similaridade vetorial', semantic: 'UMAP + HDBSCAN', laboratory: 'Laboratório de vetores'};
  const FIELDS = ['id','name','mode','profile_ids','model_categories','categories','metric','top_k','minimum_similarity','maximum_distance','fusion_depth','n_neighbors','n_components','min_dist','min_cluster_size','min_samples','random_state','umap_metric','cluster_metric','cluster_selection_method','allow_single_cluster','dendrogram','partial','individual_limits','individual_rule'];
  const defaults = () => ({name: 'Experimento 1', mode: 'vector', profile_ids: [], categories: ['conteudo_fonte'], metric: 'cosine', top_k: 6, n_neighbors: 15, n_components: 5, min_dist: 0.0, min_cluster_size: 5, min_samples: 2, random_state: 42, dendrogram: false, fusion_depth: 50, umap_metric: 'cosine', cluster_metric: 'euclidean', cluster_selection_method: 'eom', allow_single_cluster: false, partial: false, individual_rule: 'todos'});
  const clone = value => JSON.parse(JSON.stringify(value));
  function normalizeBundle(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('O arquivo deve conter uma configuração JSON de experimentos.');
    const rows = value.experiments || (value.config ? [value.config] : value.categories ? [value] : null);
    if (!Array.isArray(rows) || rows.length < 1 || rows.length > 3) throw new Error('Escolha entre um e três experimentos.');
    const experiments = rows.map((row, index) => {
      row = row?.config || row;
      if (!row || typeof row !== 'object' || Array.isArray(row)) throw new Error('Configuração de experimento inválida.');
      const config = Object.fromEntries(FIELDS.filter(key => Object.hasOwn(row, key)).map(key => [key, row[key] !== null && typeof row[key] === 'object' ? clone(row[key]) : row[key]]));
      config.name = String(config.name || 'Experimento ' + (index + 1)).trim();
      config.profile_ids ||= row.profile_id ? [row.profile_id] : [];
      if (!config.name || config.name.length > 160) throw new Error('Dê um nome de até 160 caracteres a cada experimento.');
      if (!Object.hasOwn(METHODS, config.mode)) throw new Error('Método de recuperação desconhecido.');
      if (!Array.isArray(config.profile_ids) || !config.profile_ids.length || config.profile_ids.length > 12 || config.profile_ids.some(id => typeof id !== 'string' || !id || id.length > 100)) throw new Error('Selecione de um a doze perfis de embeddings por experimento.');
      if (new Set(config.profile_ids).size !== config.profile_ids.length) throw new Error('Um perfil não pode aparecer duas vezes no mesmo experimento.');
      if (!Array.isArray(config.categories) || !config.categories.length || config.categories.some(id => typeof id !== 'string' || !id || id.length > 100)) throw new Error('Selecione categorias de vetores para cada experimento.');
      if (Object.hasOwn(config,'model_categories')) {
        const mapping = config.model_categories;
        if (!mapping || typeof mapping !== 'object' || Array.isArray(mapping) || Object.keys(mapping).length !== config.profile_ids.length || config.profile_ids.some(id => !Object.hasOwn(mapping,id))) throw new Error('Defina categorias para cada modelo selecionado.');
        for (const categories of Object.values(mapping)) if (!Array.isArray(categories) || !categories.length || categories.some(value => typeof value !== 'string' || !value)) throw new Error('Selecione pelo menos uma categoria para cada modelo.');
        config.categories = [...new Set(Object.values(mapping).flat())];
      }
      if (!['cosine','euclidean','manhattan'].includes(config.metric)) throw new Error('Escolha a distância cosseno, euclidiana ou Manhattan.');
      if (!Number.isInteger(config.top_k) || config.top_k < 1 || config.top_k > 100) throw new Error('Top K deve ficar entre 1 e 100.');
      for (const key of ['minimum_similarity','maximum_distance','minimum','maximum','min_dist']) if (config[key] != null && (typeof config[key] !== 'number' || !Number.isFinite(config[key]))) throw new Error('Os limites precisam ser números finitos.');
      if (config.minimum_similarity != null && (config.minimum_similarity < -1 || config.minimum_similarity > 1)) throw new Error('A similaridade mínima deve ficar entre -1 e 1.');
      if (config.maximum_distance != null && config.maximum_distance < 0) throw new Error('A distância máxima não pode ser negativa.');
      if (config.minimum != null && config.maximum != null && config.minimum > config.maximum) throw new Error('O escore mínimo não pode superar o máximo.');
      return config;
    });
    const supplied = value.analysis_config || value.analysis || {};
    const analysis = {enabled: supplied.enabled === true, provider: supplied.provider || 'openai', model: supplied.model === undefined ? 'gpt-6-luna' : supplied.model};
    if (!['openai','ollama','openai_compatible'].includes(analysis.provider) || typeof analysis.model !== 'string' || analysis.model.length > 200 || (analysis.enabled && !analysis.model.trim())) throw new Error('Configuração de análise automática inválida.');
    return {name: typeof value.name === 'string' ? value.name.slice(0,160) : '', experiments, analysis};
  }
  function analysisLabel(value) {
    if (!value || value.status === 'not_requested' || value.status === 'disabled') return '';
    if (value.status === 'failed' || value.status === 'error') return 'Análise não concluída';
    if (value.relevant === true) return 'Pertinente';
    if (value.relevant === false) return 'Não pertinente';
    return '';
  }
  function init(deps) {
    const {$, request, message, state, esc, label, number, date, navigate, resultCard, renderProjection, renderDendrogram} = deps;
    const ui = {drafts: [], editing: -1, presets: [], active: {experiments: [], analysis: {enabled: false, provider: 'openai', model: 'gpt-6-luna'}}, latest: null, loaded: false, dirty: false};
    const ids = {'top_k':'top-k','metric':'metric','n_neighbors':'n-neighbors','n_components':'n-components','min_cluster_size':'min-cluster-size','min_samples':'min-samples','min_dist':'umap-min-dist','random_state':'random-state','minimum_similarity':'minimum-similarity','maximum_distance':'maximum-distance','fusion_depth':'fusion-depth','umap_metric':'umap-metric','cluster_metric':'cluster-metric','cluster_selection_method':'cluster-selection-method'};
    const inform = (text, type = '') => message(text, type, 'comparison-message');
    const profiles = () => state.status?.profiles || [];
    const profileName = id => { const p = profiles().find(row => row.perfil_id === id); return p ? p.modelo + ' · ' + p.dimensoes + 'd' : id; };
    const selectedProfiles = () => Array.from(document.querySelectorAll('input[name="experiment-profile"]:checked'), input => input.value);
    const analysisConfig = () => ({enabled:$('analysis-enabled').checked, provider:$('analysis-provider').value, model:($('analysis-model').value === '__custom__' ? $('analysis-custom-model').value : $('analysis-model').value).trim()});
    function setAnalysis(config = {}) {
      $('analysis-enabled').checked = config.enabled === true;
      $('analysis-provider').value = config.provider || 'openai';
      renderAnalysisModels(config.model || 'gpt-6-luna');
    }
    function renderAnalysisModels(selected) {
      const provider = $('analysis-provider').value;
      const row = state.llms?.providers?.find(item => item.provider === provider);
      const models = row?.models || (provider === 'openai' ? [{id:'gpt-6-luna',label:'GPT-6 Luna'},{id:'gpt-6.1-sol',label:'GPT-6.1 Sol'},{id:'gpt-6-astra',label:'GPT-6 Astra'}] : []);
      const previous = selected || ($('analysis-model').value === '__custom__' ? $('analysis-custom-model').value : $('analysis-model').value);
      $('analysis-model').replaceChildren();
      models.forEach(model => $('analysis-model').add(new Option(model.label || model.id, model.id)));
      $('analysis-model').add(new Option('Outro modelo…', '__custom__'));
      $('analysis-model').value = models.some(item => item.id === previous) ? previous : '__custom__';
      if ($('analysis-model').value === '__custom__') $('analysis-custom-model').value = previous || '';
      updateAnalysis();
      $('analysis-availability').textContent = row?.detail || 'As credenciais e os serviços são configurados no servidor.';
    }
    function updateAnalysis() {
      $('analysis-fields').disabled = !$('analysis-enabled').checked;
      $('analysis-custom-choice').hidden = $('analysis-model').value !== '__custom__';
    }
    function describe() {
      const active = ui.active.experiments || [];
      $('active-method').textContent = active.length ? active.length + (active.length === 1 ? ' experimento salvo' : ' experimentos salvos') : 'Similaridade vetorial do conteúdo original';
      $('active-selection').textContent = active.length ? active.map(e => e.name).join(' · ') : 'Cosseno · conteúdo-fonte · Top K 6';
      $('method-description').textContent = active.length ? 'A consulta reproduz as configurações salvas no laboratório para este projeto.' + (ui.active.analysis?.enabled ? ' Inclui análise automática por ' + ui.active.analysis.model + '.' : '') : 'Sem configurações do laboratório, a consulta ordena os tópicos pela distância cosseno dos vetores do conteúdo original.';
      $('consultation-config-state').textContent = active.length ? 'Consulta do projeto: ' + active.map(e => e.name).join(' · ') : 'Consulta do projeto: similaridade vetorial do conteúdo original (padrão).';
    }
    function renderProfiles() {
      if (!ui.drafts.length && !ui.loaded) {
        const config = defaults(); config.profile_ids = state.status?.active_profile ? [state.status.active_profile] : profiles()[0] ? [profiles()[0].perfil_id] : [];
        ui.drafts = [config]; ui.editing = 0;
      }
      const existing = selectedProfiles();
      const selected = existing.length ? existing : ui.drafts[ui.editing]?.profile_ids || [];
      if (!selected.length && state.status?.active_profile) selected.push(state.status.active_profile);
      const rows = profiles().map(profile => ({id:profile.perfil_id,name:profileName(profile.perfil_id)}));
      selected.filter(id => !rows.some(row => row.id === id)).forEach(id => rows.push({id,name:id + ' · perfil ausente neste projeto'}));
      $('experiment-profiles').innerHTML = rows.length ? rows.map(row => '<label class="category-option"><input type="checkbox" name="experiment-profile" value="' + esc(row.id) + '"' + (selected.includes(row.id) ? ' checked' : '') + '><span>' + esc(row.name) + '</span></label>').join('') : '<p class="field-help">Cadastre um perfil de embeddings para preparar os experimentos.</p>';
      if (ui.editing >= 0 && !ui.drafts[ui.editing].profile_ids.length && selected.length) ui.drafts[ui.editing].profile_ids = [...selected];
      renderModelCategories(); renderSlots();
    }
    function renderModelCategories() {
      const enabled = $('separate-model-categories').checked; const host = $('model-categories'); host.hidden = !enabled; $('common-categories').hidden = enabled;
      const old = Array.from(host.querySelectorAll('input[name="experiment-model-category"]'));
      const ids = selectedProfiles(); const fallback = Array.from(document.querySelectorAll('input[name="search-category"]:checked'),input => input.value);
      const mapping = ui.drafts[ui.editing]?.model_categories || {};
      host.innerHTML = ids.map(id => { const previous = old.filter(input => input.dataset.profile === id); const selected = previous.length ? previous.filter(input => input.checked).map(input => input.value) : mapping[id] || fallback; return '<fieldset class="model-category-group"><legend>' + esc(profileName(id)) + '</legend><div class="category-grid">' + state.categories.map(category => '<label class="category-option"><input type="checkbox" name="experiment-model-category" data-profile="' + esc(id) + '" value="' + esc(category.id) + '"' + (selected.includes(category.id) ? ' checked' : '') + '><span>' + esc(category.label) + '</span></label>').join('') + '</div></fieldset>'; }).join('');
    }
    function renderSlots() {
      $('experiment-slots').innerHTML = ui.drafts.map((config, index) => '<article class="experiment-slot' + (ui.editing === index ? ' selected' : '') + '"><p class="eyebrow">EXPERIMENTO ' + (index + 1) + '</p><h3>' + esc(config.name) + '</h3><p>' + esc(METHODS[config.mode]) + ' · ' + esc(({cosine:'cosseno',euclidean:'euclidiana',manhattan:'Manhattan'}[config.metric])) + ' · Top K ' + esc(config.top_k) + '</p><p class="field-help">' + esc(config.categories.map(label).join(' · ')) + '</p><div class="slot-actions"><button type="button" class="text-button" data-edit-experiment="' + index + '">' + (ui.editing === index ? 'Em edição' : 'Editar') + '</button><button type="button" class="text-button" data-remove-experiment="' + index + '" aria-label="Remover experimento ' + (index + 1) + '">Remover</button></div></article>').join('');
      $('add-experiment').disabled = ui.drafts.length >= 3;
      $('compare-button').disabled = !ui.drafts.length;
      $('lab-form').hidden = ui.editing < 0;
      $('editor-title').textContent = ui.editing < 0 ? 'Configuração do experimento' : 'Editar experimento ' + (ui.editing + 1);
      $('draft-count').textContent = ui.drafts.length + ' / 3 experimentos';
    }
    function writeEditor(config, index) {
      ui.editing = index;
      state.config = {...clone(config), profile_id:config.profile_ids[0] || ''};
      $('experiment-name').value = config.name;
      Object.entries(ids).forEach(([key,id]) => { $(id).value = config[key] ?? ''; });
      document.querySelector('input[name="mode"][value="' + config.mode + '"]').checked = true;
      document.querySelectorAll('input[name="search-category"]').forEach(input => { input.checked = config.categories.includes(input.value); });
      $('lab-view').value = 'topicos'; $('score-min').value = ''; $('score-max').value = ''; $('partial-coverage').checked = config.partial === true; $('allow-single-cluster').checked = config.allow_single_cluster === true;
      $('dendrogram').checked = config.dendrogram === true; $('individual-rule').value = config.individual_rule || 'todos';
      $('individual-limits').value = config.individual_limits ? JSON.stringify(config.individual_limits) : '';
      $('experiment-profiles').replaceChildren(); $('model-categories').replaceChildren(); $('separate-model-categories').checked = Boolean(config.model_categories);
      renderProfiles(); updateMode();
    }
    function readEditor() {
      const config = {name:$('experiment-name').value.trim(),mode:document.querySelector('input[name="mode"]:checked').value,profile_ids:selectedProfiles(),categories:Array.from(document.querySelectorAll('input[name="search-category"]:checked'),input => input.value),dendrogram:$('dendrogram').checked,allow_single_cluster:$('allow-single-cluster').checked,partial:$('partial-coverage').checked,individual_rule:$('individual-rule').value};
      if ($('separate-model-categories').checked) {
        const choices = Array.from(document.querySelectorAll('input[name="experiment-model-category"]:checked'));
        config.model_categories = Object.fromEntries(config.profile_ids.map(id => [id,choices.filter(input => input.dataset.profile === id).map(input => input.value)]));
        config.categories = [...new Set(Object.values(config.model_categories).flat())];
      }
      for (const [key,id] of Object.entries(ids)) {
        if (['metric','umap_metric','cluster_metric','cluster_selection_method'].includes(key)) config[key] = $(id).value;
        else if ($(id).value.trim()) config[key] = Number($(id).value);
      }
      if (config.mode === 'laboratory' && $('individual-limits').value.trim()) {
        try { config.individual_limits = JSON.parse($('individual-limits').value); } catch { throw new Error('Os mínimos por categoria precisam ser um objeto JSON.'); }
      }
      return normalizeBundle({experiments:[config]}).experiments[0];
    }
    function capture() {
      if (ui.editing >= 0) ui.drafts[ui.editing] = readEditor();
      return normalizeBundle({experiments:ui.drafts,analysis:analysisConfig(),name:$('preset-name').value.trim()});
    }
    function saveEditor(event) {
      event?.preventDefault();
      try { capture(); renderSlots(); $('lab-state').textContent = 'Alterações guardadas no rascunho. Salve a configuração ou use na consulta para reutilizá-la.'; inform('Experimento atualizado no rascunho.', 'success'); }
      catch(error) { inform(error.message, 'warning'); }
    }
    function addExperiment() {
      ui.dirty = true;
      try {
        if (ui.editing >= 0) ui.drafts[ui.editing] = readEditor();
        if (ui.drafts.length >= 3) throw new Error('É possível comparar até três experimentos por execução.');
        const config = defaults(); config.name = 'Experimento ' + (ui.drafts.length + 1);
        config.profile_ids = state.status?.active_profile ? [state.status.active_profile] : profiles()[0] ? [profiles()[0].perfil_id] : [];
        ui.drafts.push(config); writeEditor(config, ui.drafts.length - 1); $('experiment-name').focus();
      } catch(error) { inform(error.message, 'warning'); }
    }
    function updateMode() {
      const mode = document.querySelector('input[name="mode"]:checked')?.value;
      const semantic = mode === 'semantic'; $('semantic-options').hidden = !semantic; document.querySelector('.advanced-lab').hidden = mode !== 'laboratory';
      document.querySelectorAll('[data-semantic-parameter]').forEach(node => { node.disabled = !semantic; });
    }
    async function refreshPresets() {
      const data = await request('/api/experiment-configs'); ui.presets = data.configs || [];
      const previous = $('saved-config').value;
      $('saved-config').replaceChildren(new Option('Selecione uma configuração salva', ''));
      ui.presets.forEach(preset => $('saved-config').add(new Option(preset.name, preset.id || preset.preset_id)));
      if (ui.presets.some(p => (p.id || p.preset_id) === previous)) $('saved-config').value = previous;
    }
    function reuse(value) {
      const bundle = normalizeBundle(value);
      ui.loaded = true; ui.drafts = bundle.experiments; $('preset-name').value = bundle.name;
      setAnalysis(bundle.analysis); writeEditor(ui.drafts[0], 0);
      inform('Configuração carregada no rascunho. Revise os parâmetros antes de executar ou usar na consulta.', 'success');
    }
    async function savePreset() {
      const button = $('save-config'); button.disabled = true;
      try {
        const body = capture(); if (!body.name) throw new Error('Informe um nome para salvar esta configuração.');
        const result = await request('/api/experiment-configs',{method:'POST',body});
        await refreshPresets(); $('saved-config').value = result.id || result.preset_id || '';
        inform('Configuração salva no projeto para reprodução e reuso.', 'success');
      } catch(error) { inform(error.message, 'error'); } finally { button.disabled = false; }
    }
    async function applyConsultation(useDefault = false) {
      const button = $(useDefault ? 'use-default-search' : 'use-experiments'); button.disabled = true;
      try {
        const body = useDefault ? {experiments:[],analysis:{enabled:false,provider:'openai',model:'gpt-6-luna'}} : capture();
        ui.active = await request('/api/consultation-config',{method:'PUT',body:{experiments:body.experiments,analysis:body.analysis}});
        describe(); inform(useDefault ? 'Consulta padrão restaurada: similaridade vetorial do conteúdo original.' : 'Experimentos salvos como parâmetros da consulta deste projeto.', 'success');
      } catch(error) { inform(error.message, 'error'); } finally { button.disabled = false; }
    }
    function downloadText(value, filename, type) {
      const url = URL.createObjectURL(new Blob([value],{type})); const a = document.createElement('a'); a.href = url; a.download = filename; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url),1000);
    }
    async function download(id, format) {
      try { await deps.download('/api/comparisons/' + encodeURIComponent(id) + '/export?format=' + format, 'autorag-' + id + (format === 'config' ? '-config.json' : format === 'vectors' ? '-vetores.zip' : '.' + format)); }
      catch(error) { inform(error.message, 'error'); message(error.message, 'error'); }
    }
    function analysisMarkup(value) {
      const title = analysisLabel(value); if (!title) return '';
      const failed = value.status === 'failed' || value.status === 'error';
      const provenance = value.provenance || {};
      return '<aside class="evaluation-note' + (failed ? ' evaluation-error' : value.relevant ? ' relevant' : ' irrelevant') + '"><strong>' + esc(title) + (value.utility_rank ? ' · utilidade ' + esc(value.utility_rank) : '') + '</strong><p>' + esc(value.comment || value.error || '') + '</p><small>' + esc([provenance.provider,provenance.model].filter(Boolean).join(' · ')) + '</small></aside>';
    }
    function render(data, query, target = 'lab') {
      ui.latest = data; state.results = data; state.resultQuery = query || data.query || '';
      const node = $(target === 'consultation' ? 'consultation-comparisons' : 'lab-comparison-results'); node.hidden = false;
      if (target === 'consultation') { $('results').hidden = true; $('search-empty').hidden = true; $('search-warnings').hidden = true; }
      const records = data.experiments || [];
      if (data.reproduced_from) inform('Reprodução da recuperação ' + data.reproduced_from + '. A análise automática anterior não foi reexecutada.','success');
      node.innerHTML = '<div class="section-heading"><div><p class="eyebrow">COMPARAÇÃO SALVA</p><h2>Resultados dos experimentos</h2><p class="result-query">' + esc(data.query || query) + '</p><p class="field-help">' + esc(date(data.date)) + ' · ' + esc(data.comparison_id) + '</p></div></div><div class="comparison-downloads">' + [['json','Baixar JSON'],['csv','Baixar CSV'],['jsonl','Baixar registros JSONL'],['config','Baixar configuração'],['vectors','Baixar vetores e métricas (ZIP)']].map(([format,text]) => '<button type="button" class="button secondary small" data-comparison-export="' + esc(data.comparison_id) + '" data-format="' + format + '">↓ ' + text + '</button>').join('') + '<button type="button" class="text-button" data-reuse-comparison="true">Reusar configuração</button><button type="button" class="button secondary small" data-reproduce-comparison="true">Reproduzir recuperação salva</button></div><div class="comparison-grid" data-count="' + records.length + '">' + records.map((experiment,index) => {
        const config = experiment.config || {}; const rows = experiment.results || [];
        const error = experiment.status === 'error' || experiment.status === 'failed';
        const params = {...config}; delete params.name;
        const categoriesDescription = config.model_categories ? Object.entries(config.model_categories).map(([id,categories]) => profileName(id) + ': ' + categories.map(label).join(' + ')).join(' / ') : (config.categories || []).map(label).join(' · ');
        return '<section class="comparison-column"><div class="comparison-column-heading"><p class="eyebrow">EXPERIMENTO ' + (index + 1) + '</p><h3>' + esc(config.name || experiment.experiment_id) + '</h3><p>' + esc(METHODS[config.mode] || config.mode) + ' · ' + esc(({cosine:'cosseno',euclidean:'euclidiana',manhattan:'Manhattan'}[config.metric])) + ' · Top K ' + esc(config.top_k) + '</p><p class="field-help">' + esc((config.profile_ids || []).map(profileName).join(' + ')) + '</p><p class="field-help">' + esc(categoriesDescription) + '</p></div>' + (error ? '<p class="message error">' + esc(experiment.error || 'Este experimento não foi concluído.') + '</p>' : '') + (experiment.warnings || []).map(w => '<p class="message warning">' + esc(w) + '</p>').join('') + (experiment.analysis?.status === 'failed' ? '<p class="message warning">Análise automática não concluída: ' + esc(experiment.analysis.error || '') + '</p>' : '') + '<div class="result-list">' + (rows.length ? rows.map((row,i) => {
          let card = resultCard(row,i,config.mode === 'semantic');
          if (row.fusion_score != null || row.rrf_score != null) card = card.replace(/Distância <b>[^<]*<\/b>/, 'Fusão RRF <b>' + number(row.fusion_score ?? row.rrf_score) + '</b>');
          const evidence = row.model_evidence || row.evidence || row.evidences || row.spaces || [];
          return card + analysisMarkup(row.analysis) + (evidence.length ? '<details class="evidence-detail"><summary>Pontuações por modelo</summary><pre>' + esc(JSON.stringify(evidence,null,2)) + '</pre></details>' : '');
        }).join('') : '<p class="empty-inline">' + (error ? 'Os outros experimentos continuam disponíveis.' : 'Nenhum tópico atende a esta configuração e aos limites informados.') + '</p>') + '</div><details class="experiment-parameters"><summary>Parâmetros e reprodução</summary><pre>' + esc(JSON.stringify(params,null,2)) + '</pre><p class="field-help">As revisões dos perfis e índices utilizados acompanham os arquivos exportados.</p></details><div data-comparison-visual="' + index + '"></div>' + (experiment.baseline_results?.length && (config.mode !== 'vector' || (config.categories || []).some(category => category !== 'conteudo_fonte')) ? '<details class="baseline-detail"><summary>Comparar com o conteúdo original</summary>' + experiment.baseline_results.map((row,i) => resultCard(row,i,false)).join('') + '</details>' : '') + '</section>';
      }).join('') + '</div>';
      records.forEach((experiment,index) => (experiment.spaces || []).forEach(space => {
        const host = node.querySelector('[data-comparison-visual="' + index + '"]');
        if (space.baseline_results?.length && ((experiment.config?.profile_ids?.length || 0) > 1 || experiment.status === 'error')) { const details = document.createElement('details'); details.className = 'baseline-detail'; const summary = document.createElement('summary'); summary.textContent = 'Conteúdo original · ' + profileName(space.profile_id); const body = document.createElement('div'); body.innerHTML = space.baseline_results.map((row,i) => resultCard(row,i,false)).join(''); details.append(summary,body); host.append(details); }
        const points = space.projection || space.cluster?.points;
        if (points?.length) {
          const details = document.createElement('details'); details.className = 'experiment-visual'; const summary = document.createElement('summary'); summary.textContent = 'Mapa semântico · ' + profileName(space.profile_id); const chart = document.createElement('div'); chart.className = 'chart-wrap'; details.append(summary,chart); host.append(details); renderProjection(points,space.cluster?.query_label,chart,details);
        }
        if (space.dendrogram?.svg) {
          const details = document.createElement('details'); details.className = 'experiment-visual'; const summary = document.createElement('summary'); summary.textContent = 'Dendrograma · ' + profileName(space.profile_id); const chart = document.createElement('div'); chart.className = 'chart-wrap'; details.append(summary,chart); host.append(details); renderDendrogram(space.dendrogram,chart,details);
        }
      }));
      node.querySelectorAll('[data-comparison-export]').forEach(button => button.addEventListener('click',() => download(button.dataset.comparisonExport,button.dataset.format)));
      node.querySelector('[data-reproduce-comparison]').addEventListener('click',async event => { event.currentTarget.disabled = true; try { const result = await request('/api/comparisons/' + encodeURIComponent(data.comparison_id) + '/reproduce',{method:'POST',body:{}}); render(result,result.query,target); await refreshHistory(); inform('Recuperação reproduzida com os vetores e índices registrados, sem novas chamadas aos modelos.','success'); } catch(error) { inform(error.message,'error'); message(error.message,'error'); } finally { const button = node.querySelector('[data-reproduce-comparison]'); if(button) button.disabled = false; } });
      node.querySelector('[data-reuse-comparison]').addEventListener('click',() => { reuse(data); navigate('laboratorio'); });
      if (data.warnings?.length) { const box = document.createElement('div'); box.className = 'message warning'; box.textContent = data.warnings.join(' '); node.prepend(box); }
      const analysis = data.analysis || {};
      if (analysis.status === 'failed') message('A recuperação foi salva, mas a análise automática não foi concluída: ' + (analysis.error || ''), 'warning');
    }
    async function compare(event) {
      event.preventDefault(); const button = $('compare-button'); const query = $('comparison-query').value.trim();
      if (!query) { $('comparison-query').focus(); return; }
      button.disabled = true; $('comparison-progress').hidden = false; inform('');
      try {
        const bundle = capture();
        const result = await request('/api/comparisons',{method:'POST',body:{query,experiments:bundle.experiments,analysis:bundle.analysis},progress:text => { $('comparison-progress-label').textContent = text; }});
        render(result,query); await refreshHistory(); inform('Comparação salva com configurações, resultados e registros de análise disponíveis para download.', 'success');
      } catch(error) { inform(error.message, 'error'); }
      finally { button.disabled = !ui.drafts.length; $('comparison-progress').hidden = true; }
    }
    async function refreshHistory() {
      try {
        const data = await request('/api/comparisons'); const rows = data.comparisons || [];
        $('comparison-history').replaceChildren();
        if (!rows.length) { $('comparison-history').innerHTML = '<p class="empty-inline">As comparações executadas ficarão salvas aqui.</p>'; return; }
        rows.forEach(row => {
          const wrapper = document.createElement('div'); wrapper.className = 'history-row'; const info = document.createElement('div'); const title = document.createElement('strong'); title.textContent = row.query || row.comparison_id; const meta = document.createElement('p'); meta.textContent = date(row.date) + (row.names ? ' · ' + row.names.join(' · ') : ''); info.append(title,meta); const open = document.createElement('button'); open.type = 'button'; open.className = 'text-button'; open.textContent = 'Abrir comparação ↗';
          open.addEventListener('click',async () => { try { const result = await request('/api/comparisons/' + encodeURIComponent(row.comparison_id)); $('comparison-query').value = result.query; render(result,result.query); } catch(error) { inform(error.message,'error'); } }); wrapper.append(info,open); $('comparison-history').append(wrapper);
        });
      } catch(error) { $('comparison-history').textContent = error.message; }
    }
    async function refresh() {
      try {
        const [active] = await Promise.all([request('/api/consultation-config'),refreshPresets()]); ui.active = active; if (!ui.loaded && !ui.dirty && active.experiments?.length) reuse({experiments:active.experiments,analysis:active.analysis}); describe();
        if (!state.llms) state.llms = await request('/api/llms'); renderAnalysisModels();
      } catch(error) { inform(error.message, 'error'); }
    }
    $('experiment-profiles').addEventListener('change',renderModelCategories);
    $('separate-model-categories').addEventListener('change',() => {ui.dirty = true; renderModelCategories();});
    $('add-experiment').addEventListener('click',addExperiment);
    $('experiment-slots').addEventListener('click',event => {
      const edit = event.target.closest('[data-edit-experiment]'); const remove = event.target.closest('[data-remove-experiment]');
      try {
        if (edit) { if (ui.editing >= 0) ui.drafts[ui.editing] = readEditor(); const index = Number(edit.dataset.editExperiment); writeEditor(ui.drafts[index],index); $('experiment-name').focus(); }
        if (remove) { const index = Number(remove.dataset.removeExperiment); const previous = ui.editing; if (previous >= 0 && previous !== index) ui.drafts[previous] = readEditor(); ui.drafts.splice(index,1); ui.loaded = true; ui.dirty = true; if (ui.drafts.length) { const next = previous === index ? Math.min(index,ui.drafts.length - 1) : previous - (index < previous ? 1 : 0); writeEditor(ui.drafts[next],next); } else {ui.editing = -1;renderSlots();} }
      } catch(error) { inform(error.message,'warning'); }
    });
    $('save-config').addEventListener('click',savePreset);
    $('load-config').addEventListener('click',() => { try { const preset = ui.presets.find(row => (row.id || row.preset_id) === $('saved-config').value); if (!preset) throw new Error('Selecione uma configuração salva.'); reuse(preset); } catch(error) { inform(error.message,'warning'); } });
    $('download-config').addEventListener('click',() => { try { const bundle = capture(); downloadText(JSON.stringify({...bundle,schema_version:'autorag.experiments.v2'},null,2),'autorag-configuracao.json','application/json;charset=utf-8'); } catch(error) { inform(error.message,'warning'); } });
    $('import-config').addEventListener('change',async () => { try { const file = $('import-config').files[0]; if (!file) return; if (file.size > 1024 * 1024) throw new Error('A configuração deve ter no máximo 1 MiB.'); reuse(JSON.parse(await file.text())); } catch(error) { inform(error instanceof SyntaxError ? 'O arquivo não contém JSON válido.' : error.message,'error'); } finally { $('import-config').value = ''; } });
    $('use-experiments').addEventListener('click',() => applyConsultation()); $('use-default-search').addEventListener('click',() => applyConsultation(true));
    $('comparison-form').addEventListener('submit',compare); $('refresh-comparisons').addEventListener('click',refreshHistory);
    $('analysis-enabled').addEventListener('change',updateAnalysis); $('analysis-provider').addEventListener('change',() => { $('analysis-model').value = ''; $('analysis-custom-model').value = ''; renderAnalysisModels(''); }); $('analysis-model').addEventListener('change',updateAnalysis);
    $('top-k').addEventListener('input',() => { $('fusion-depth').min = $('top-k').value; if (Number($('fusion-depth').value) < Number($('top-k').value)) $('fusion-depth').value = $('top-k').value; });
    document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener('change',updateMode));
    $('lab-form').addEventListener('input',() => {ui.dirty = true;}); $('lab-form').addEventListener('change',() => {ui.dirty = true;});
    setAnalysis(); renderProfiles(); if (ui.drafts.length) writeEditor(ui.drafts[0],0); describe(); refresh();
    return {saveEditor,describe,render,renderProfiles,refreshHistory,refresh,isComparison:data => Boolean(data?.comparison_id && Array.isArray(data.experiments)),hideConsultation:() => {$('consultation-comparisons').hidden = true;},resetEditor:() => {if(ui.editing >= 0){const config=defaults();config.name=ui.drafts[ui.editing].name;config.profile_ids=ui.drafts[ui.editing].profile_ids;writeEditor(config,ui.editing);}},navigate:page => {if(page === 'laboratorio'){refreshHistory();refresh();}},selectProfile:id => {document.querySelectorAll('input[name="experiment-profile"]').forEach(input => {input.checked = input.value === id;});}};
  }
  globalThis.AutoRAGLaboratory = Object.freeze({normalizeBundle,analysisLabel,defaults,init});
})();
