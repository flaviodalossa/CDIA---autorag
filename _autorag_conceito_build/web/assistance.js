'use strict';
(() => {
  function init({$, request, message, state, esc, number, refreshStatus}) {
    let offset = 0, total = 0, topicQuery = '', loading = false, loaded = false, settingsEpoch = 0;
    let pendingSettings = Promise.resolve();
    const selectedTopics = new Set();
    const settings = () => ({full_document:$('full-document').checked,support:{enabled:$('support-enabled').checked, provider:$('support-provider').value, model:$('support-model').value.trim()}, noise_filter:{enabled:$('noise-enabled').checked, text:$('noise-text').value.trim(), margin:Number($('noise-margin').value)}});
    function setSettings(value) {
      $('full-document').checked = value.full_document === true;
      const support = value.support || {enabled:true,provider:'openai',model:'gpt-6-luna'};
      $('support-enabled').checked = support.enabled;
      $('support-provider').value = support.provider;
      $('support-model').value = support.model;
      if (value.noise_filter) {
        $('noise-enabled').checked = value.noise_filter.enabled;
        $('noise-text').value = value.noise_filter.text;
        $('noise-margin').value = value.noise_filter.margin;
      }
      $('noise-fields').disabled = !$('noise-enabled').checked;
    }
    function render(data) {
      const panel = $('support-answer'); const support = data.support;
      panel.hidden = !support || support.status === 'disabled'; panel.replaceChildren();
      if (!support || support.status === 'disabled') return;
      const title = document.createElement('h2'); title.textContent = support.status === 'noise' ? 'Consulta classificada como ruído' : 'Resposta do LLM de apoio';
      const body = document.createElement('p'); body.className = 'answer-body'; body.textContent = support.answer || support.error || 'Resposta de apoio indisponível.';
      panel.append(title, body);
      if (support.provenance) { const note = document.createElement('p'); note.className = 'field-help'; note.textContent = support.provenance.provider + ' · ' + support.provenance.model + (support.topic_ids?.length ? ' · Tópicos: ' + support.topic_ids.join(', ') : ''); panel.append(note); }
      const refs = (data.experiments || []).flatMap(e => (e.spaces || []).filter(s => s.noise_reference).map(s => ({profile:s.profile_id,...s.noise_reference})));
      if (refs.length) {
        const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = 'Referência de ruído usada nesta busca'; details.append(summary);
        for (const row of refs) { const p = document.createElement('p'); p.textContent = row.profile + ' · cosseno da referência: ' + number(row.cosine_similarity) + ' · margem: ' + number(row.margin) + ' · aceitar somente acima de ' + number(row.threshold); details.append(p); }
        const text = document.createElement('p'); text.textContent = refs[0].text; details.append(text); panel.append(details);
      }
    }
    function renderProfiles() {
      const host = $('enrichment-profiles'); const old = [...host.querySelectorAll('input:checked')].map(n => n.value);
      const profiles = state.status?.profiles || [];
      host.innerHTML = profiles.map(p => '<label class="category-option"><input type="checkbox" name="enrichment-profile" value="' + esc(p.perfil_id) + '"' + ((old.length ? old.includes(p.perfil_id) : p.perfil_id === state.status.active_profile) ? ' checked' : '') + '><span>' + esc(p.modelo + ' · ' + p.dimensoes + 'd') + '</span></label>').join('') || '<p class="field-help">Cadastre um perfil de embeddings.</p>';
    }
    async function loadTopics() {
      const epoch = settingsEpoch;
      try {
        const result = await request('/api/enrichment/topics?query=' + encodeURIComponent(topicQuery) + '&offset=' + offset + '&limit=100'); if (epoch !== settingsEpoch) return; total = result.total;
        $('enrichment-topics').innerHTML = result.topics.map(t => '<label><input type="checkbox" value="' + esc(t.topic_id) + '"' + (selectedTopics.has(t.topic_id) ? ' checked' : '') + '><span><strong>' + esc(t.title || t.topic_id) + '</strong><small class="field-help"> ' + esc(t.topic_id + ' · ' + t.document) + '</small></span></label>').join('') || '<p class="field-help">Nenhum tópico encontrado.</p>';
        $('enrichment-topic-page').textContent = total ? (offset + 1) + '–' + Math.min(offset + 100,total) + ' de ' + total : '0 tópicos';
        $('enrichment-topic-prev').disabled = offset === 0; $('enrichment-topic-next').disabled = offset + 100 >= total;
      } catch(error) { message(error.message,'error','enrichment-message'); }
    }
    function updateMode() {
      const mode = $('enrichment-mode').value;
      $('enrichment-iterations-field').hidden = mode !== 'loop';
      $('enrichment-iterations').disabled = mode !== 'loop';
      $('enrichment-text-label').textContent = mode === 'loop' ? 'Objetivo para o LLM gerar entradas' : 'Entrada útil para recuperação';
      $('enrichment-topic-help').textContent = mode === 'manual' ? 'Selecione de 1 a 30 tópicos que receberão o embedding.' : 'Opcional: limite os candidatos aos tópicos selecionados. Sem seleção, a aplicação recupera candidatos no conteúdo-fonte da base.';
    }
    async function refresh() {
      if (loading) return;
      loading = true;
      const epoch = settingsEpoch;
      try {
        const [entriesData, runsData] = await Promise.all([request('/api/enrichments'),request('/api/enrichment-runs')]);
        if (epoch !== settingsEpoch) return;
        const rows = entriesData.entries || [];
        $('enrichment-entries').innerHTML = rows.length ? '<table><thead><tr><th>Selecionar</th><th>Entrada</th><th>Perfil / tópicos</th><th>Estado</th></tr></thead><tbody>' + rows.map(r => '<tr><td><input type="checkbox" data-enrichment-id="' + esc(r.id) + '" aria-label="Selecionar entrada ' + esc(r.text.slice(0,80)) + '"></td><td>' + esc(r.text) + '<p class="field-help">' + esc(r.provenance?.kind || 'manual') + '</p></td><td>' + esc(r.profile_id) + '<p class="field-help">' + esc(r.topic_ids.join(', ')) + '</p></td><td>' + (r.active ? 'Ativo para recuperação' : 'Rascunho / inativo') + (r.stale_topic_ids?.length ? '<p class="message warning">Conteúdo alterado ou tópico inativo: ' + esc(r.stale_topic_ids.join(', ')) + '</p>' : '') + '</td></tr>').join('') + '</tbody></table>' : '<p class="empty-inline">Nenhum enriquecimento armazenado.</p>';
        $('enrichment-runs').replaceChildren();
        for (const run of runsData.runs || []) {
          const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = run.mode + ' · ' + run.status + ' · ' + run.iterations.length + '/' + run.requested_iterations + ' iterações · ' + new Date(run.date).toLocaleString('pt-BR');
          const pre = document.createElement('pre'); pre.textContent = JSON.stringify(run,null,2); details.append(summary,pre); $('enrichment-runs').append(details);
        }
      } catch(error) { message(error.message,'error','enrichment-message'); }
      finally { if (epoch === settingsEpoch) loading = false; }
    }
    async function generate(event) {
      event.preventDefault();
      const profile_ids = [...document.querySelectorAll('input[name="enrichment-profile"]:checked')].map(n => n.value);
      if (!profile_ids.length) {message('Selecione um perfil de embeddings.','warning','enrichment-message');return;}
      const mode = $('enrichment-mode').value;
      if (mode === 'manual' && !selectedTopics.size) {message('Selecione ao menos um tópico.','warning','enrichment-message');return;}
      $('enrichment-submit').disabled = true; $('enrichment-progress').hidden = false;
      message('','','enrichment-message');
      try {
        const result = await request('/api/enrichments',{method:'POST',body:{text:$('enrichment-text').value.trim(),profile_ids,topic_ids:[...selectedTopics],mode,iterations:mode === 'loop' ? Number($('enrichment-iterations').value) : 1,llm:{...settings().support,enabled:true}},progress:text => {$('enrichment-progress-label').textContent = text;}});
        message(result.iterations.length + '/' + result.requested_iterations + ' iterações concluídas · ' + result.entry_ids.length + ' enriquecimentos guardados. ' + (result.error || 'Revise e ative as entradas que deseja usar nas buscas.'),result.status === 'completed' ? 'success' : 'warning','enrichment-message');
        await refresh();
      } catch(error) {message(error.message,'error','enrichment-message');}
      finally {$('enrichment-submit').disabled = false; $('enrichment-progress').hidden = true;}
    }
    async function activate(active) {
      const entry_ids = [...$('enrichment-entries').querySelectorAll('input[data-enrichment-id]:checked')].map(n => n.dataset.enrichmentId);
      if (!entry_ids.length) {message('Selecione as entradas armazenadas.','warning','enrichment-message');return;}
      $('enrichment-activate').disabled = true; $('enrichment-deactivate').disabled = true;
      try {
        await request('/api/enrichments/activation',{method:'PATCH',body:{entry_ids,active}});
        message(active ? 'Vetores publicados. Escolha a categoria Enriquecimento de recuperação no experimento usado pela Consulta.' : 'Entradas retiradas da produção; os vetores continuam armazenados.','success','enrichment-message');
        await refresh(); await refreshStatus({quiet:true});
      } catch(error) {message(error.message,'error','enrichment-message');}
      finally {$('enrichment-activate').disabled = false; $('enrichment-deactivate').disabled = false;}
    }
    $('noise-enabled').addEventListener('change',() => {$('noise-fields').disabled = !$('noise-enabled').checked;});
    $('save-assistance').addEventListener('click',async () => {
      $('save-assistance').disabled = true;
      try {const current = await request('/api/consultation-config'); const result = await request('/api/consultation-config',{method:'PUT',body:{experiments:current.experiments,analysis:current.analysis,...settings()}}); setSettings(result); message('Opções salvas para as próximas consultas.','success','assistance-settings-message');}
      catch(error) {message(error.message,'error','assistance-settings-message');}
      finally {$('save-assistance').disabled = false;}
    });
    $('enrichment-form').addEventListener('submit',generate); $('enrichment-mode').addEventListener('change',updateMode);
    $('enrichment-topics').addEventListener('change',event => {
      const input = event.target; if (input.type !== 'checkbox') return;
      if (input.checked && selectedTopics.size >= 30) {input.checked = false;message('Selecione no máximo 30 tópicos.','warning','enrichment-message');return;}
      if (input.checked) selectedTopics.add(input.value); else selectedTopics.delete(input.value);
      $('enrichment-topic-count').textContent = selectedTopics.size;
    });
    $('enrichment-topic-search').addEventListener('click',() => {offset = 0;topicQuery = $('enrichment-topic-query').value.trim();loadTopics();});
    $('enrichment-topic-query').addEventListener('keydown',event => {if(event.key === 'Enter'){event.preventDefault();$('enrichment-topic-search').click();}});
    $('enrichment-topic-clear').addEventListener('click',() => {selectedTopics.clear();$('enrichment-topic-count').textContent = '0';loadTopics();});
    $('enrichment-topic-prev').addEventListener('click',() => {offset = Math.max(0,offset-100);loadTopics();});
    $('enrichment-topic-next').addEventListener('click',() => {if(offset+100 < total){offset+=100;loadTopics();}});
    $('enrichment-refresh').addEventListener('click',refresh);
    $('enrichment-activate').addEventListener('click',() => activate(true)); $('enrichment-deactivate').addEventListener('click',() => activate(false));
    let edited = false;
    $('full-document').addEventListener('change',() => {edited = true;});
    document.querySelector('.assistance-settings').addEventListener('input',() => {edited = true;});
    function refreshSettings(reset = false) {
      const epoch = ++settingsEpoch;
      if (reset) {
        edited = false; selectedTopics.clear(); offset = 0; total = 0; topicQuery = ''; loaded = false; loading = false;
        $('enrichment-topic-count').textContent = '0'; $('enrichment-topic-query').value = '';
        $('enrichment-profiles').replaceChildren();
        $('enrichment-topics').replaceChildren(); $('enrichment-entries').replaceChildren(); $('enrichment-runs').replaceChildren();
        $('support-answer').hidden = true;
        setSettings({support:{enabled:true,provider:'openai',model:'gpt-6-luna'},noise_filter:{enabled:false,text:'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',margin:0}});
      }
      pendingSettings = request('/api/consultation-config').then(value => {if(epoch === settingsEpoch && !edited)setSettings(value);}).catch(error => {if(epoch === settingsEpoch)message(error.message,'warning','assistance-settings-message');});
      return pendingSettings;
    }
    refreshSettings();
    updateMode();
    return {settings,render,renderProfiles,setFullDocument:value => {$('full-document').checked = value === true;edited = true;},ready:() => pendingSettings,resetConnection:() => refreshSettings(true),navigate:page => {if(page === 'laboratorio'){renderProfiles();refresh();if(!loaded){loaded = true;loadTopics();}}}};
  }
  globalThis.AutoRAGAssistance = Object.freeze({init});
})();
