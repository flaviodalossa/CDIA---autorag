# AutoRAG Web v0.4 - documento completo

Branch: feature/autorag-v0.4-documento-completo.

Na aba Consulta, ative **Trazer documento completo** antes de pesquisar. A opção começa desativada. Em **Resposta de apoio e limite por ruído**, use **Salvar opções da consulta** se quiser guardar essa preferência no projeto. A escolha visível também acompanha a busca atual, mesmo sem salvar, e as comparações feitas no Laboratório.

A busca continua aplicando categorias, Top K, filtros e fusão por modelo. Depois, cada documento com um tópico aceito fornece seus demais tópicos ativos, inclusive os que ainda não têm embeddings. Tópicos de documentos sem resultado aceito não são incluídos. Tópicos repetidos são deduplicados. Tópicos inativos ficam excluídos.

O Top K limita os tópicos da busca inicial. A expansão pode ultrapassá-lo. Os resultados diretos mantêm o ranking original; os demais vêm em seguida, agrupados pela ordem dos documentos encontrados e pela ordem dos tópicos na base. O rótulo **Contexto do documento** identifica os acréscimos. Eles não recebem distância, cosseno ou pontuação de fusão artificial.

O limite por ruído decide quais documentos podem originar expansão: se nenhum tópico passar, não há expansão nem chamada ao LLM. Os tópicos adicionais do documento podem ficar abaixo do limite de ruído, pois entram como contexto do documento aceito e não como novos resultados vetoriais.

Todos os tópicos retornados participam do contexto da resposta de apoio. Com documento completo, são aceitos até 600 tópicos, mantendo 20.000 caracteres por tópico e 180.000 caracteres de texto e consulta no total. Se o contexto exceder esses limites, os resultados continuam disponíveis e a aplicação informa que não gerou a resposta; não há truncamento silencioso. A análise opcional de pertinência mantém seus próprios limites anteriores.

A opção, os tópicos de origem e os acréscimos ficam registrados no histórico, no JSON/JSONL/CSV e na configuração exportada. O ZIP de vetores informa quais tópicos são apenas contexto; não cria vetores fictícios para eles. A reprodução usa a base congelada da execução original, sem novas inferências, mesmo após alterações da base corrente.

API: full_document booleano opcional em POST /api/search, POST /api/comparisons, GET/PUT /api/consultation-config e configurações nomeadas. Na busca por consulta salva, a omissão usa a opção do projeto; false explícito desativa apenas naquela requisição. Configurações antigas preservam o comportamento anterior.
