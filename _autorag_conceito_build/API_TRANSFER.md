# API de transferência nativa

Estas rotas transportam uma base já preparada ou índices já calculados. Não executam extração, LLM nem geração de embeddings.

Todas exigem AUTORAG_API_TOKEN configurado no servidor e o cabeçalho Authorization: Bearer <token>, inclusive em loopback. Não coloque tokens na URL. A resposta não inclui credenciais.

## Importar base

POST /api/base/import, multipart:

| Campo | Conteúdo |
| --- | --- |
| base_file | Um arquivo JSON UTF-8 obrigatório, com documentos. |
| legacy_npz | Um NPZ legado opcional, com vetores OpenAI já calculados. |

O limite total da requisição, incluindo multipart, é 25 MiB. O JSON é interpretado como estrutura, mantendo os IDs, textos, representações, metadados e flags de atividade. A importação atualiza documentos por documento_id e mantém documentos não enviados. IDs de tópicos devem ser únicos na base resultante.

Controles de embeddings vindos do JSON são descartados e reconstruídos pelos índices do destino. Perfis e o perfil ativo do destino são preservados. Caminhos registrados como proveniência permanecem dados; não são abertos ou executados.

Na primeira importação em uma base vazia, a whitelist validada preserva base_id, ordem_documentos, eixos e execucoes_processamento. As estatísticas são recalculadas. Em destino não vazio, conserva a identidade, classificação e proveniência existentes, acrescentando referências de documentos novos.

O NPZ legado deve conter vetores, topico_ids, hashes_texto, modelo, dimensoes e, opcionalmente, normalizacao=l2. Os IDs devem cobrir exatamente os tópicos ativos enviados, e cada hash deve corresponder ao texto composto atual. São aceitos os modelos OpenAI text-embedding-3-small, text-embedding-3-large e text-embedding-ada-002. Os vetores são publicados em um perfil legado separado; o perfil ativo anterior permanece selecionado.

Antes de escrever a base, a validação verifica tipos, identificadores, credenciais indevidas, valores finitos, dimensões, unicidade, hashes, integridade dos índices existentes e colisões com documentos do destino. O NPZ passa por preflight ZIP/NPY, sem pickle, com até 64 MiB descompactados e checagem de forma antes de alocar arrays. Em falha após a publicação começar, os ponteiros ativos, a base e a configuração são restaurados; revisões imutáveis não referenciadas podem permanecer no disco.

A resposta de aceitação é HTTP 202 com job_id. Consulte GET /api/jobs/{job_id} até succeeded ou failed. No adaptador Cloudflare, o checkpoint é confirmado antes de publicar succeeded.

O resultado inclui:

- documents, topics, active_topics, inactive_topics: contagens do lote recebido.
- total_documents, total_topics: contagens após a união.
- legacy_embeddings, legacy_profile_id, active_profile_id.
- root_metadata_preserved, root_metadata_fields.
- canonical_digest: hash dos documentos recebidos, sem controles de embeddings.
- base_canonical_digest: o mesmo hash aplicado à base resultante completa.
- digest_algorithm: sha256.documents_without_embedding_controls.v1.

## Exportar base

GET /api/base/export devolve o JSON completo da base, com Content-Disposition: attachment. Os cabeçalhos X-AutoRAG-Canonical-Digest e X-AutoRAG-Digest-Algorithm permitem comparar o conteúdo após a transferência. O hash usa documentos ordenados por ID e preserva os demais dados de cada documento, removendo apenas controles de embeddings do documento/tópico. Revisões e timestamps da raiz ficam fora desse digest.

A exportação pode incluir caminhos e URLs de proveniência. Guarde o arquivo em local privado; não o publique em um repositório aberto.

## Importar índice nativo

POST /api/indices/import, multipart index_file: um bundle ZIP gerado por autorag.index_transfer.export_index_bundle.

O perfil correspondente precisa estar cadastrado no destino, com identidade compatível. A validação ocorre antes da fila e é repetida sob a trava na publicação. Não há alteração do perfil ativo nem inferência. Limites: 25 MiB compactados e 64 MiB descompactados. O resultado utiliza a mesma fila e confirmação de checkpoint.

JSON preparado não deve ser enviado a /api/ingest: essa rota trata arquivos como fontes documentais para extração e não preserva o esquema original.
