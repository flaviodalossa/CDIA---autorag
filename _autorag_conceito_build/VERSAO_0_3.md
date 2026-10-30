# AutoRAG Web v0.3

Branch: feature/autorag-v0.3-apoio-ruido-enriquecimento.

## Resposta na Consulta

A Consulta gera uma resposta de apoio usando os conteúdos originais dos tópicos recuperados. O padrão é OpenAI gpt-6-luna. Em “Resposta de apoio e limite por ruído”, você pode desativar a resposta, mudar o provedor/modelo e salvar as opções do projeto. As opções visíveis também acompanham cada busca, mesmo sem salvar. Credenciais e URLs são configuradas somente no servidor, como na versão anterior. As chamadas aos provedores podem ter cobrança.

A resposta cita IDs de tópicos e fica salva no histórico e no JSON exportado. A avaliação de pertinência existente é independente da resposta. Se o LLM falhar ou o contexto exceder os limites, os resultados vetoriais são preservados e a interface informa a falha. São aceitos até 60 tópicos, 20.000 caracteres por tópico e 180.000 caracteres de contexto e consulta; conteúdos não são truncados silenciosamente.

## Referência de ruído

O filtro é opcional e começa desativado. O texto padrão é Lorem ipsum, editável. A cada busca a referência é recuperada no perfil de embeddings utilizado, com reaproveitamento de seu vetor persistido. A referência usa o papel de documento do modelo, como os candidatos da base, inclusive em modelos com preparação distinta para consulta e documento. Para cada modelo:

    limite = cosseno(consulta, referência) + margem
    aceitar retorno somente se cosseno_original(consulta, retorno) > limite

Igualdade é rejeitada. A margem é aditiva em unidades de cosseno, de 0 a 2. Se o limite superar 1, nenhum retorno é aceito. Os limites atuam antes de Top K e da fusão RRF, em cada espaço de modelo. UMAP usa o cosseno original para este filtro; a média de vetores usa o cosseno médio do tópico. O baseline também é filtrado.

Sem tópicos aceitos, a aplicação retorna “Texto classificado como ruído no contexto da base.” e não chama o LLM de apoio nem o avaliador de pertinência. Falhas de recuperação continuam identificadas como falhas. O texto, margem, escore, vetor e checksum da referência são registrados por modelo. A reprodução usa esses artefatos, sem novas inferências ou respostas LLM.

## Enriquecer no Laboratório

1. Escolha os perfis de embeddings e informe uma entrada útil.
2. No modo manual, selecione os tópicos associados. A lista tem busca e paginação, com seleção de até 30 tópicos entre páginas.
3. No modo automático, o LLM de apoio seleciona as associações. Com tópicos selecionados, eles delimitam os candidatos; sem seleção, a aplicação recupera até oito candidatos por perfil pelo conteúdo-fonte e apresenta até 30 tópicos distintos ao LLM. A seleção pode ficar vazia.
4. No modo loop, informe um objetivo e de 1 a 30 iterações. O LLM gera uma entrada diferente por iteração, recebe os candidatos e escolhe as associações. O histórico registra cada entrada, motivo, modelo e associação. Iterações concluídas permanecem salvas se uma posterior falhar. A geração usa os mesmos provedores/modelos de apoio da Consulta.
5. As novas entradas ficam em rascunho. Selecione as entradas armazenadas e use “Ativar selecionados”. Para usá-las nas consultas, adicione a categoria **Enriquecimento de recuperação** ao experimento e use esse experimento na Consulta. A recuperação padrão continua usando conteúdo-fonte até essa escolha.
6. “Retirar selecionados da produção” desativa as entradas e recalcula a categoria, preservando os vetores individuais e o histórico.

Cada entrada tem um vetor persistido por perfil, com o papel de consulta do modelo. Para cada tópico/perfil, os vetores ativos normalizados são agregados pela média aritmética, seguida de normalização L2. Um único enriquecimento resulta em um vetor equivalente. A média é publicada em índices NPZ imutáveis da nova categoria; dimensões de modelos diferentes permanecem separadas. Conteúdos alterados e tópicos inativos invalidam as associações antigas e impedem sua ativação. Não há treinamento de pesos do modelo.

O arquivo privado dados/enriquecimentos.json guarda os registros e vetores individuais; estudos/enrichment_runs guarda os loops. Dados, índices e históricos participam dos snapshots privados R2 existentes, e não entram no Git ou no pacote de código. A ativação é executada como job com confirmação de persistência no R2. Os proxies permitem PATCH somente para a rota de ativação.

## API

- GET/PUT /api/consultation-config: campos adicionais support (enabled, provider, model) e noise_filter (enabled, text, margin). Atualizações antigas do laboratório preservam essas opções.
- POST /api/search: query, com support e noise_filter opcionais. Também disponíveis na busca explícita por profile_id.
- POST /api/comparisons: noise_filter opcional, support opcional; comparações de laboratório não geram resposta por padrão.
- GET /api/enrichment/topics?query=&offset=0&limit=100: tópicos ativos pesquisáveis.
- GET/POST /api/enrichments: listar registros ou criar job de enriquecimento com text, profile_ids, topic_ids, mode (manual/automatic/loop), iterations e llm opcionais.
- PATCH /api/enrichments/activation: entry_ids e active booleano; retorna job.
- GET /api/enrichment-runs: histórico das últimas 100 execuções.

## Execução e verificação

Abra 09_abrir_web.bat como nas versões anteriores. Para testes offline:

    python -m unittest discover -s tests -p "test_*.py"
    python -m unittest discover -s cloudflare/tests -p "test_*.py"
    npm run check
    node --test tests/*.test.mjs cloudflare/tests/*.test.mjs

Os testes de apoio e enriquecimento usam embeddings e respostas simulados; não consomem APIs pagas. A validação de credenciais/modelos reais ocorre no uso da aplicação.
