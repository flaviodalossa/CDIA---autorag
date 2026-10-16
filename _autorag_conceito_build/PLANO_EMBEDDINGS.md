# Plano de implementação — modelos e modos de embeddings

Data: 30/09/2026. Escopo: planejamento; funcionalidades ainda não implementadas.

## 1. Requisitos e decisões

O usuário poderá selecionar um modelo de embeddings disponível localmente ou um serviço online com modalidade gratuita. Cada um dos oito tipos de vetor poderá ser gerado, atualizado ou regenerado independentemente para cada modelo e cada revisão da base, somente conforme a seleção do usuário. Os três modos de geração serão atalhos opcionais de seleção, sem obrigar a geração conjunta dos tipos; na recuperação, o usuário poderá escolher qualquer combinação dos tipos disponíveis, inclusive entre modos:

| Modo | Texto usado | Vetores por tópico antes de eventual divisão por tamanho |
| --- | --- | --- |
| `topico_completo` | Título, hierarquia, resumo, conteúdo-fonte, perguntas, resposta-síntese e base normativa, com ou sem palavras-chave, em unidades distintas | Até 2, gerados independentemente |
| `campos_separados` | `resumo`, `conteudo_fonte`, `perguntas_sugeridas`, `resposta_sintese`, `palavras_chave`, cada campo em um texto independente | Até 5 |
| `metadados` | Apenas metadados documentais descritivos vinculados ao tópico | 1 |

O terceiro modo foi confirmado pelo usuário como somente metadados. Para planejamento, “metadados” significa os dados documentais já importados, inclusive os armazenados em `identificacao` e `vigencia`, além de `metadados`; não significa exclusivamente o objeto JSON chamado `metadados`.

Nos campos separados, cada lista de perguntas ou palavras-chave forma um texto por campo; não haverá um vetor por pergunta ou palavra nesta versão. Campos vazios não geram embeddings. Cada vetor continua associado ao mesmo `topico_id` de origem; esses campos não se tornam novos tópicos da base.

A ausência de palavras-chave no texto completo atual foi intencional, para comparar os modelos sem sua influência. Preservar essa composição como referência. Gerar um vetor independente somente de palavras-chave e oferecer adicionalmente a variante completa com palavras-chave não autoriza sua inclusão automática nas buscas: ambos estarão desmarcados por padrão e só participarão quando escolhidos pelo usuário.

Tipos selecionáveis: `topico_total_sem_palavras_chave`, `topico_total_com_palavras_chave`, `resumo`, `conteudo_fonte`, `perguntas_sugeridas`, `resposta_sintese`, `palavras_chave` e `metadados`. Há oito tipos possíveis por tópico antes da segmentação. O tópico total sem palavras-chave continuará como representação de referência e opção disponível; não será um pré-requisito para gerar qualquer outro tipo. A versão com palavras-chave será adicional, nunca uma substituição automática da versão sem palavras-chave.

Manter uma matriz de controle **modelo × tipo de vetor**, apresentando disponíveis/atuais, desatualizados, pendentes e nunca gerados. Atualizar a base não gera nem apaga embeddings automaticamente. Os vetores já gerados continuam fisicamente armazenados até que uma atualização ou regeneração escolhida para aquele modelo/tipo/tópico publique seus substitutos; entradas não selecionadas permanecem intactas.

Para cada tópico recuperado, mostrar a similaridade individual de cada vetor utilizado e a média aritmética dessas similaridades. A ordenação será pela média dos vetores escolhidos. Selecionar ou combinar vetores já gerados não exige gerar embeddings documentais novamente.

Requisito central da exploração: selecionar explicitamente **modelo gerador × categoria de vetor × consulta**, recuperar resultados separados por esse recorte e formar conjuntos por seus escores. Categoria significa um dos oito tipos de vetor. O mesmo texto de consulta poderá ser avaliado por vários modelos em execuções separadas e apresentadas lado a lado; cada modelo terá seu próprio vetor da consulta. Grupos por escore pertencem à consulta/execução, não são propriedades permanentes do vetor.

A escolha do modelo de embeddings é independente do modelo que extrai os tópicos. Trocar o modelo de embeddings não exige reextrair documentos.

A aplicação será um ambiente de estudo dos embeddings associados aos conteúdos. Na primeira entrega, priorizar exploração da recuperação e persistência das entradas/resultados; estruturar os dados e serviços para permitir comparação de vetores, cálculos, manipulação de conjuntos e clusterização por vários métodos posteriormente. Todo embedding gerado para um texto de entrada será persistido em NPZ, acompanhado de JSON específico com o texto original e a rastreabilidade das recuperações. A referência aos vetores recuperados nesse JSON inclui seus IDs, arquivos, revisões, linhas e distâncias; os valores numéricos dos vetores continuam exclusivamente nos NPZ, conforme a regra já definida pelo usuário.

## 2. Diagnóstico do projeto

- `pipeline_etica.py`: extrai tópicos, controla alterações e também tem uma geração OpenAI que grava o vetor dentro do JSON.
- `gerar_embeddings.py`: gera incrementos em NPZ e manifesto, mantendo `embedding.vetor` como `null` na base; já normaliza vetores e salva progresso por lote.
- `buscar_topicos.py`: lê o NPZ, gera consultas com OpenAI e calcula similaridade por produto interno. Rejeita IDs de tópicos repetidos e pressupõe um vetor por tópico.
- `executar_embeddings.ps1`: fixa `text-embedding-3-small`, 1536 dimensões e dependências OpenAI.
- `executar_busca.ps1`: fixa os caminhos do índice e exige OpenAI.
- `README.md`: descreve principalmente o caminho que grava embeddings no JSON, diferente do executado pelos atalhos atuais.
- `app_1.py` e `app_2.py`: contêm versões da pipeline por linha de comando; os arquivos examinados não constituem uma interface gráfica de escolha de modelos. Os atalhos identificados usam `pipeline_etica.py`, `gerar_embeddings.py` e `buscar_topicos.py`.
- `requirements.txt`: não declara todas as dependências usadas pelo gerador e buscador, como `numpy` e `tiktoken`.
- `tests/test_pipeline.py`: cobre estados e importação, mas precisa ser complementado para os novos índices e recuperação.
- O fluxo atual de `buscar_topicos.py` mantém o embedding da consulta apenas em memória. `save_result` exporta consulta, modelo e resultados, sem persistir o vetor de entrada nem capturar todas as referências/distâncias necessárias para reproduzir o experimento.

A base piloto examinada contém 12 documentos e 282 tópicos ativos. Seu schema textual é 2.1; o manifesto de embeddings usa schema 1.0. O texto canônico atual não inclui `palavras_chave` por decisão do usuário: modificar apenas esse campo não deve invalidar o tópico total sem palavras-chave, mas deverá invalidar o vetor exclusivo de palavras-chave e a variante completa que as inclui.

A enumeração dos caminhos padrão e do comando `ollama` não confirmou modelos instalados neste ambiente. Isso não prova ausência de modelos em outros caminhos ou servidores. A implementação deverá descobrir a disponibilidade real antes de oferecer opções locais.

## 3. Arquitetura proposta

Manter JSON textual e metadados dos embeddings + NPZ exclusivo para vetores + manifesto; o volume atual não justifica introduzir um banco vetorial. O usuário confirmou que os vetores ficam apenas no NPZ e que o JSON deve ser reestruturado para acomodar os metadados das várias unidades de embedding. Nenhum caminho de geração, checkpoint, exportação ou compatibilidade deverá gravar arrays de vetores no JSON. Extrair uma implementação compartilhada para que os comandos existentes usem as mesmas regras:

| Componente proposto | Responsabilidade |
| --- | --- |
| `embedding_providers.py` | Adaptadores de modelos, disponibilidade, preparação específica, documentos e consultas |
| `embedding_config.py` | Perfis, catálogo local/online e configuração sem segredos |
| `embedding_credentials.py` | Ponto de configuração de chave direta por provedor e resolução por variável de ambiente/provisão segura |
| `embedding_records.py` | Construção dos textos, IDs, hashes, campos e segmentos |
| `embedding_store.py` | Leitura, escrita, migração, checkpoints e validação dos índices |
| `embedding_search.py` | Similaridade, filtros, agrupamento e montagem dos resultados |
| `embedding_inputs.py` | Textos de entrada, embeddings persistentes por modelo/papel e reaproveitamento |
| `embedding_experiments.py` | Execuções, parâmetros, resultados, distâncias e revisões reproduzíveis |
| `embedding_analysis.py` | Contratos para comparação, operações vetoriais e conjuntos derivados |
| `embedding_clustering.py` | Interface extensível de métodos de clusterização e artefatos experimentais |

Interface comum: `list_models()`, `describe_model()`, `embed_documents(texts)` e `embed_query(text)`. O descritor informa provedor, nome, revisão/digest quando disponível, dimensão, limite de entrada, tamanho de lote e regras de preparação.

Importar dependências opcionais somente ao usar seu provedor. Executar geração e busca locais sem chave OpenAI. Não trocar de provedor automaticamente quando ocorrer uma falha.

### Provedores iniciais

1. **Ollama local:** descobrir modelos instalados com `GET /api/tags`; verificar capacidade de embeddings com os metadados de capacidade quando disponíveis e uma chamada de diagnóstico quando necessário; gerar via `POST /api/embed`. Oferecer somente modelos de embeddings utilizáveis. Nunca executar download automático ao selecionar um modelo.
2. **Sentence Transformers local:** localizar snapshots completos no cache Hugging Face e modelos em pastas configuradas, validando carregamento com `local_files_only=True`. `intfloat/multilingual-e5-small` é um candidato para avaliação em português se instalado, não uma instalação presumida. CPU como opção inicial; GPU quando disponível e selecionada.
3. **Gemini online:** oferecer modelos de embeddings que tenham camada gratuita documentada e estejam disponíveis à chave/conta. A tabela oficial consultada oferece texto gratuito para `gemini-embedding-2` no nível gratuito; disponibilidade e cotas precisam ser verificadas na implementação. Usar chamadas padrão, pois gratuidade do endpoint padrão não implica gratuidade da Batch API. Cota esgotada deve interromper e preservar progresso, sem migração automática para pagamento ou outro modelo.
4. **OpenAI existente:** preservar compatibilidade como opção identificada como paga; não apresentá-la entre serviços gratuitos.

“Modelo aberto para baixar” e “inferência online gratuita” são disponibilidades diferentes. O catálogo mostrará local instalado, online gratuito sujeito a cota, pago ou indisponível, incluindo necessidade de chave. Não há um mecanismo universal para descobrir todos os modelos no computador: a descoberta será limitada aos provedores, caches e pastas configurados.

### Opções leves pesquisadas: menos de 400 dimensões

Pesquisa em documentação e fichas oficiais consultadas em 30/09/2026:

| Opção | Dimensão usada | Contexto / observação de tamanho | Execução e avaliação |
| --- | --- | --- | --- |
| `intfloat/multilingual-e5-small` | **384 nativas** | 512 tokens; arquitetura de 12 camadas, menor que BGE-M3 | Sentence Transformers local; candidato recomendado para recuperação multilíngue com dimensão nativa abaixo de 400 |
| `embeddinggemma:300m-qat-q4_0` | **256 ou 128**, por redução Matryoshka de uma saída nativa de 768 | 2048 tokens; pacote Ollama de aproximadamente **239 MB** | Ollama local; opção compacta no mesmo servidor que BGE-M3, com preparação e redução de dimensão validadas no adaptador |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | **384 nativas** | 128 tokens no wrapper oficial; suporte a 50 idiomas | Sentence Transformers local; alternativa para campos curtos, exigindo maior segmentação nos tópicos extensos |
| `all-minilm:l6` / `all-MiniLM-L6-v2` | **384 nativas** | Pacote Ollama de aproximadamente **46 MB**; versão L6 de 22M parâmetros | Muito leve, mas o modelo original é identificado como inglês; candidato experimental, sem ser a escolha principal para esta base em português |

Incluir E5 Small (384) no catálogo como opção nativa menor que 400 dimensões e incluir EmbeddingGemma QAT Q4 (256) como opção leve para o Ollama, além de BGE-M3 (1024). Recomenda-se iniciar a avaliação no Ollama com BGE-M3 e EmbeddingGemma Q4 em 256 dimensões; E5 Small fica como alternativa local de dimensão nativa reduzida pelo adaptador Sentence Transformers, sem presumir disponibilidade oficial desse modelo no catálogo Ollama.

Dimensão do vetor, quantidade de parâmetros, tamanho do download, memória e tempo de inferência são medidas diferentes. Reduzir EmbeddingGemma de 768 para 256/128 reduz armazenamento e custo da comparação; não transforma a rede em um modelo de 256 parâmetros nem garante redução proporcional do tempo para gerar embeddings. A quantização Q4 reduz o pacote/pesos. Confirmar recursos e qualidade em português com o piloto previsto.

Para EmbeddingGemma, solicitar a dimensão pela API quando a versão instalada oferecer o comportamento necessário e validar o resultado; alternativamente aplicar a redução Matryoshka documentada no adaptador, seguida de normalização L2, de modo idêntico para documentos e consultas. Registrar dimensão nativa/efetiva, método de redução, quantização e preparação no perfil. Não cortar dimensões de modelos arbitrários nem usar prefixos do E5 no EmbeddingGemma: seguir o formato de consulta/documento próprio do modelo. Um perfil 256 e um perfil 128 são configurações distintas.

O suporte multilíngue torna E5/EmbeddingGemma/MiniLM multilíngue candidatos para português; não substitui medição de qualidade nos documentos normativos do projeto. Comparar com BGE-M3 usando a mesma seleção de tipos, perguntas e métricas. Pesos menores e limite curto podem aumentar a segmentação; considerar isso ao comparar as médias e a qualidade.

## 4. Construção dos textos

### Tópico completo: duas representações distintas

Compor `topico_total_sem_palavras_chave` com a composição canônica atual: título, hierarquia, resumo, conteúdo-fonte, perguntas, resposta-síntese e base normativa. Preservar texto e hash quando nada nesses campos mudar, permitindo reutilizar os vetores atuais. Compor separadamente `topico_total_com_palavras_chave` a partir desse texto mais o campo de palavras-chave, com identidade e hash próprios. Excluir IDs técnicos, status, datas de processamento e vetores de ambas as representações. “Completo” refere-se ao conteúdo semântico do tópico, não ao JSON bruto com controles operacionais.

### Campos separados

Gerar uma unidade por campo não vazio, usando apenas seu conteúdo, sem acrescentar título/hierarquia nem concatenar outros campos. O vetor `palavras_chave` conterá exclusivamente as palavras-chave, além da preparação técnica exigida pelo modelo. Essa regra ficará explícita e versionada. Normalizar listas e espaços de maneira determinística, preservando o conteúdo. O vetor de palavras-chave poderá ser usado sozinho ou junto de qualquer conjunto escolhido, mas somente por seleção explícita.

### Somente metadados

Texto inicial: `nome_documento`; `identificacao.tipo_documento`, `identificacao.identificacao_documento`, `identificacao.data_documento`, `identificacao.descricao_documento`; `metadados.eixos`, `metadados.idioma`; `vigencia.inicio_vigencia` e `vigencia.fim_vigencia`.

Não incluir resumo, conteúdo-fonte, perguntas, resposta, palavras-chave, título/hierarquia de tópico nem texto extraído. URL e páginas permanecem na referência da fonte; caminhos locais, hashes de arquivo e controles de execução ficam fora do texto vetorizado.

Como tópicos do mesmo documento compartilham esse texto, gerar uma vez por texto/configuração e reaproveitar o vetor nas linhas correspondentes a cada tópico. A primeira versão mantém uma linha por tópico para simplificar a compatibilidade. Não usar `topico_id` como texto artificial para diferenciar vetores iguais.

### Limites de entrada

Substituir o limite OpenAI fixo e o uso universal de `tiktoken` por limites/tokenização do provedor. Reservar espaço para prefixos e contexto. Evitar truncamento silencioso: dividir textos longos em segmentos ordenados, mantendo vínculo ao tópico/campo. Assim, um tópico completo pode ter mais de um vetor quando necessário; a tabela mostra a contagem sem essa divisão.

Registrar política, limites e hashes do texto original e do texto efetivamente enviado. Na carga de modelos legados, respeitar a política antiga para que a validação não invalide seus hashes indevidamente. A documentação do E5 exige prefixos distintos para consulta/documento e limita entradas a 512 tokens; esses detalhes pertencem ao adaptador, sem duplicação acidental de prefixos.

## 5. Índices e compatibilidade

Criar schema 2.0 dos artefatos de embeddings e migrar o schema da base textual de 2.1 para 3.0, pois o objeto único `embedding` de cada tópico será substituído por uma coleção de metadados organizada por perfil. A matriz numérica continuará exclusivamente no NPZ. Cada linha do NPZ terá:

- `embedding_id` único e estável, por exemplo `tp_x::resumo::0000`;
- `topico_id`, `documento_id`, `campo_origem` e `segmento`;
- hash do texto fonte e do texto preparado;
- vetor normalizado L2 e dimensão validada.

No NPZ: matriz `vetores`, `embedding_ids`, `topico_ids`, `campos`, `segmentos`, hashes e identificação do perfil. `topico_ids` poderá repetir; `embedding_ids` não. Separar os NPZ por **perfil de modelo × tipo de vetor**, para que gerar resumo não regrave o arquivo de palavras-chave nem exija atualizar os outros sete tipos. No manifesto JSON: identificação do perfil/tipo, totais distintos de tópicos e vetores, status de geração, itens alinhados às linhas, revisão e checksum, sempre sem valores numéricos dos vetores. O vínculo principal será `perfil_id` + `embedding_id`; posição da linha é válida somente dentro da revisão do NPZ daquele tipo.

Para estudo, conservar nos novos NPZ os embeddings tal como retornados pelo provedor, além da representação usada na busca normalizada L2. Guardar `vetores_originais` quando diferirem de `vetores`, com dtype/precisão, norma e transformações registrados; se o provedor já devolver vetores normalizados, registrar isso e evitar cópia numérica desnecessária. “Original” aqui significa retorno do provedor, não estados internos da rede. Não converter a precisão silenciosamente antes de preservar esse retorno. Índices legados que só guardam versões normalizadas continuam utilizáveis, identificados como tais; não tentar reconstruir suas magnitudes originais.

### Nova estrutura de metadados na base JSON

Na raiz, acrescentar `controle_embeddings` com versão do formato, revisão da base, perfil ativo e catálogo `perfis`. Registrar provedor, modelo, revisão/digest, dimensão, tipos de vetor disponíveis, versões da composição por tipo, preparação, normalização e política de segmentação uma única vez por perfil. Em `indices_por_tipo`, cada perfil aponta para os NPZ/manifestos e suas revisões independentes; não repetir essas configurações em cada campo de cada tópico. No catálogo de tipos, identificar explicitamente quais dependem de palavras-chave. Registrar `revisao_base_geracao` por unidade para rastrear sua origem, sem marcar como desatualizada uma unidade cujo hash fonte permanece igual em uma revisão posterior da base.

Em cada tópico, substituir `embedding` por `embeddings.perfis`, com uma entrada por perfil e uma lista `unidades` contendo somente metadados. Exemplo ilustrativo, mostrando apenas uma unidade do modo separado:

```json
{
  "schema_version": "3.0",
  "controle_embeddings": {
    "schema_version": "2.0",
    "revisao_base": "base_rev_001",
    "perfil_ativo_id": "pf_exemplo",
    "perfis": {
      "pf_exemplo": {
        "provedor": "sentence_transformers",
        "modelo": "intfloat/multilingual-e5-small",
        "revisao_modelo": "revisao_identificada_no_carregamento",
        "dimensoes": 384,
        "tipos_vetor": ["resumo"],
        "versoes_composicao": {"resumo": "1"},
        "normalizacao": "l2",
        "preparacao": {
          "prefixo_documento": "passage: ",
          "prefixo_consulta": "query: "
        },
        "segmentacao": {"politica": "por_limite_do_modelo"},
        "indices_por_tipo": {
          "resumo": {
            "revisao_indice": "rev_001",
            "arquivo_npz": "indices/pf_exemplo/resumo/rev_001/vetores.npz",
            "arquivo_manifesto": "indices/pf_exemplo/resumo/rev_001/manifesto.json"
          }
        }
      }
    }
  },
  "documentos": [
    {
      "documento_id": "doc_exemplo",
      "estrutura": [
        {
          "topico_id": "tp_exemplo",
          "embeddings": {
            "perfis": {
              "pf_exemplo": {
                "status": "atualizado",
                "unidades_esperadas": 1,
                "unidades": [
                  {
                    "embedding_id": "tp_exemplo::resumo::0000",
                    "campo_origem": "resumo",
                    "segmento": 0,
                    "status": "atualizado",
                    "hash_texto_fonte": "sha256_do_texto_fonte",
                    "hash_texto_embeddado": "sha256_do_texto_preparado",
                    "revisao_base_geracao": "base_rev_001",
                    "revisao_indice": "rev_001",
                    "linha_npz": 0,
                    "tokens_enviados": 80,
                    "texto_truncado": false,
                    "data_geracao": "2026-09-30T12:00:00Z",
                    "erro": null
                  }
                ]
              }
            }
          }
        }
      ]
    }
  ]
}
```

O exemplo é um recorte: os demais campos atuais de documentos e tópicos serão preservados. `unidades_esperadas: 1` representa um perfil que indexa apenas resumo; cinco campos escolhidos e preenchidos terão ao menos cinco unidades. Identificar os vetores completos por `campo_origem: topico_total_sem_palavras_chave` e `campo_origem: topico_total_com_palavras_chave`; usar `campo_origem: palavras_chave` para o vetor exclusivo e `campo_origem: metadados` para o terceiro modo. Unidade pendente ainda sem vetor terá referência de linha/revisão ausente ou nula, nunca um array numérico. Acrescentar início/fim de segmento quando necessário para rastrear o trecho de origem, sem duplicar o texto completo na coleção de metadados.

O JSON textual controla os vínculos e os estados esperados por tópico/perfil/unidade. O manifesto descreve a revisão efetivamente publicada do NPZ. Validar correspondência de IDs, hashes, dimensão, revisão, linhas e checksum antes de considerar uma unidade disponível. Após interrupção, reconciliar referências com o manifesto publicado; não aceitar `status: atualizado` isoladamente como prova da existência do vetor.

Recalcular estatísticas por perfil, distinguindo tópicos de unidades/vetores: cinco embeddings de um tópico continuam contando como um tópico. As estatísticas legadas poderão resumir o perfil ativo durante a transição.

O perfil terá identidade calculada a partir de provedor, modelo/revisão, dimensão, preparação, redução de dimensão quando houver, quantização e normalização. A identidade de cada unidade inclui tipo de vetor, versão da composição e segmentação. Um mesmo perfil de modelo poderá armazenar todos os tipos em NPZ independentes, permitindo combinar vetores de modos diferentes na busca. Acrescentar ou regenerar um tipo publica somente sua revisão e preserva as unidades/tipos não selecionados. Perfis de modelos incompatíveis não compartilham um índice mesmo que tenham a mesma dimensão. A seleção de tipos para busca é uma configuração de recuperação, separada da identidade dos embeddings.

Organizar `indices/<perfil_id>/<tipo_vetor>/<revisao>/`, com um ponteiro `atual.json` por perfil/tipo. Gravar NPZ e manifesto de uma nova revisão daquele tipo, validar sua correspondência e somente então trocar atomicamente seu ponteiro. Arquivos individualmente atômicos não garantem consistência entre vários artefatos. Checkpoints parciais terão indicação explícita e poderão ser retomados; a busca usará por padrão a última revisão publicada completa de cada tipo selecionado. Capturar e registrar esse conjunto de revisões ao iniciar cada consulta, para que uma atualização concorrente não misture revisões durante o cálculo.

Não eliminar fisicamente o vetor anterior ao iniciar sua regeneração: enquanto o substituto não estiver validado/publicado, a referência anterior continua armazenada, com seu estado de atualidade explícito. Falha no novo lote não apaga a última geração concluída. Após publicação, trocar apenas as referências das unidades selecionadas. Manter snapshots anteriores para recuperação; limpeza de histórico é uma ação explícita e separada, sem remoção automática nesta implementação.

Permitir perfis coexistentes e ativar um perfil já pronto sem recalcular. Trocar modelo/revisão cria ou retoma seu próprio índice; exige novos embeddings se não houver índice compatível. Os modos de geração servem como atalhos para escolher os tipos a gerar. Trocar a combinação usada na busca reutiliza os vetores existentes; tipos ausentes deverão ser gerados explicitamente antes de serem selecionados, sem geração automática durante uma busca.

Compatibilidade: ler schema 1.0 do índice como perfil OpenAI legado com tipo `topico_total_sem_palavras_chave`, mantendo a composição atual sem palavras-chave. Migrar referências sem reextrair tópicos nem apagar os artefatos anteriores. O vetor geral com palavras-chave é uma unidade adicional; seu acréscimo não invalida o vetor completo sem palavras-chave.

A migração do JSON converterá os metadados do objeto legado `embedding` em uma unidade do perfil legado, conciliando hash, data, modelo e dimensão com o NPZ/manifesto. Remover o campo legado `vetor`, inclusive quando estiver `null`, da base migrada. Para leitura de arquivos antigos, oferecer adaptação em memória, sem manter duas estruturas persistentes com informações duplicadas. Se um arquivo legado excepcional contiver um array de vetor, a migração deverá transferi-lo e validar sua presença no NPZ antes de removê-lo do JSON, sem perder embeddings.

Eliminar a escrita de vetores em JSON da rotina `pipeline_etica.py embeddings` e de qualquer entrada histórica ainda utilizada: todas deverão encaminhar a geração ao armazenamento NPZ. Não substituir o hash histórico `controle_topico.hash_conteudo` diretamente por uma regra nova: manter verificação legada e acrescentar/versionar hashes semânticos quando necessário. Adaptar `merge_topics` para detectar mudanças em palavras-chave e invalidar os perfis afetados.

## 6. Geração incremental

O gerador comparará hashes e configuração por `perfil_id` + tipo + `embedding_id`, e não apenas por tópico. Em cada execução, selecionar pares **modelo × tipo**, escopo de tópicos/documentos e operação: gerar faltantes; atualizar alterados; ou regenerar sob demanda mesmo com hash igual. Nenhum par será selecionado automaticamente por existir na base. A tabela abaixo indica quais unidades ficam desatualizadas quando a fonte muda; sua regeneração só ocorre quando elas forem escolhidas:

| Alteração | Regeneração necessária |
| --- | --- |
| Somente `resumo` | As duas variantes de tópico completo, se geradas, e a unidade de resumo; metadados preservados |
| Somente `palavras_chave` | Apenas unidade de palavras-chave e tópico total com palavras-chave; preservar tópico total sem palavras-chave e demais campos |
| Somente metadado incluído no texto | Modo metadados dos tópicos vinculados; demais modos preservados |
| Título/hierarquia do tópico | As variantes de tópico completo; campos independentes preservados |
| Modelo/revisão/dimensão/preparação | Criar configuração independente; preservar o perfil anterior e gerar somente os tipos escolhidos para o novo |
| Campo removido ou vazio | Marcar referências sem fonte utilizável; omiti-las da próxima publicação escolhida para esse tipo, preservando o histórico físico |
| Tópico desativado | Excluir da recuperação; preservar vetores armazenados, retirando referências correntes na próxima atualização escolhida |

Preservar seleção explícita de IDs, simulação e retomada. Documentar a diferença atual: a pipeline só seleciona pendentes sem IDs, enquanto o gerador NPZ atual inclui alterados. A implementação compartilhada terá seleção explícita (`pendentes`, `novos-e-alterados`, `ids`) e `--regenerar` para forçar novos vetores apenas nos pares/IDs escolhidos. A simulação mostrará a matriz de pares selecionados e o efeito em cada um antes de executar.

Controlar status primariamente por tópico/modelo/tipo/unidade. Os oito tipos não precisam estar gerados nem atualizados conjuntamente para que um tipo disponível seja usado. O estado agregado do perfil é apenas resumo da cobertura; pendência em um tipo não escolhido não bloqueia outro. Um campo vazio será omitido; tópico sem nenhuma unidade utilizável terá indicação de não indexável, nunca um vetor nulo. Simulação mostrará tópicos, unidades, textos únicos, segmentos, reaproveitamento e chamadas estimadas.

Após cada atualização da base, recalcular a atualidade pelos hashes dos textos específicos e preservar todos os NPZ existentes. A matriz informará, por par modelo/tipo, quantos tópicos estão atuais, alterados, novos sem vetor e inativos. Por padrão a busca seleciona somente unidades atuais; vetores desatualizados continuam armazenados e visíveis no inventário, mas não são usados silenciosamente. Caso seja implementada consulta histórica, ela deve escolher explicitamente uma revisão compatível da base e seus índices, sem associar vetor antigo ao texto novo.

Exemplo: gerar apenas resumo para BGE-M3 e apenas palavras-chave para E5; depois gerar metadados para BGE-M3. As duas gerações anteriores permanecem armazenadas. Se a base mudar somente no resumo, marcar somente representações dependentes desse resumo como desatualizadas; atualizar resumo em BGE-M3 não gera resumo em E5 nem substitui palavras-chave ou metadados de qualquer modelo.

## 7. Recuperação

Carregar o perfil do manifesto e usar seu adaptador para a consulta. Impedir consulta com outro modelo/dimensão/revisão/preparação. Conferir hashes contra a base atual; vetores de conteúdo alterado ou tópico inativo não participam da recuperação. Em modo estrito, divergência bloqueia; no modo tolerante, unidades inconsistentes são excluídas em memória e a cobertura incompleta é informada.

### Seleção livre e média das similaridades

Oferecer seleção múltipla dos oito tipos de vetor, com liberdade para qualquer conjunto não vazio disponível no modelo escolhido. Exemplos: apenas palavras-chave; resumo + resposta; tópico total sem palavras-chave + metadados; ou todos os tipos. Nenhum tipo não selecionado participa da média ou da ordenação. Exibir a disponibilidade/atualidade de cada tipo por modelo. Sugerir como seleção inicial apenas `topico_total_sem_palavras_chave` quando estiver disponível; se não tiver sido gerado, solicitar a escolha entre os tipos disponíveis, sem gerá-lo automaticamente. Reutilizar uma seleção anteriormente salva com palavras-chave somente quando essa escolha tiver sido feita explicitamente pelo usuário.

1. Gerar o vetor da consulta com o perfil escolhido e calcular a similaridade cosseno individual de cada vetor selecionado e válido.
2. Agrupar por `topico_id`, conservando as similaridades individuais e os IDs dos vetores efetivamente utilizados.
3. Calcular `similaridade_media = soma(similaridades_utilizadas) / quantidade_vetores_utilizados`, sem pesos e sem substituir a média pelo máximo. Com um único vetor, a média é sua própria similaridade. A média é das pontuações, não uma média de vetores seguida de novo cálculo de cosseno.
4. Ordenar todos os tópicos candidatos pela média e somente então aplicar limiar e `top_k`. Não pré-selecionar apenas os melhores resultados de um campo, pois isso pode perder tópicos com a melhor média da combinação.
5. Mostrar cada tópico uma vez, com fonte, similaridade individual por vetor, média, quantidade de vetores utilizados e seleção aplicada. Valores arredondados são apenas de apresentação; ordenar e calcular com valores não arredondados.

Exemplo de um tópico com similaridades ilustrativas: resumo = 0,90; resposta-síntese = 0,60; palavras-chave = 0,30.

| Seleção explícita | Similaridades exibidas | Média para ordenar |
| --- | --- | --- |
| Resumo + resposta-síntese | 0,90; 0,60 | 0,75 |
| Somente palavras-chave | 0,30 | 0,30 |
| Resumo + resposta-síntese + palavras-chave | 0,90; 0,60; 0,30 | 0,60 |

Gerar palavras-chave no índice não muda a média da primeira seleção: somente a seleção da terceira linha autoriza sua participação naquela recuperação.

Textos longos: mostrar a similaridade de cada segmento separadamente, identificado por `embedding_id`. Ao selecionar um tipo, incluir por padrão todos os seus segmentos válidos, cada um contando uma vez na média; informar o número de segmentos para que o denominador fique explícito. Permitir seleção detalhada de segmentos/IDs existentes quando o usuário precisar restringir o conjunto. Não reduzir segmentos ao máximo silenciosamente.

Campos ausentes: por padrão, exigir que todos os tipos escolhidos tenham unidades utilizáveis no tópico; excluir tópicos sem cobertura completa e informar essa quantidade. Não atribuir zero a campo ausente nem mudar o denominador silenciosamente. Oferecer opção explícita de cobertura parcial, exibindo campos ausentes e o denominador efetivo da média. Nenhum vetor válido significa tópico não elegível para aquela busca. Quando há um tipo com segmentos pendentes/desatualizados, considerar esse tipo incompleto até sua regeneração.

### Recuperação por modelo, categoria e conjuntos de escore

O seletor de recuperação deverá apresentar a matriz modelo/perfil × oito categorias, com cobertura e disponibilidade, permitindo escolher os pares exatos. Uma mesma consulta pode executar BGE-M3 em resumo/resposta e E5 em palavras-chave, por exemplo, sem exigir que todas as categorias estejam geradas em todos os modelos. Um plano de comparação reúne as execuções pelo `entrada_id` e `comparacao_id`, mantendo configurações, vetores de consulta e resultados próprios de cada modelo.

Disponibilizar duas visões com unidade de resultado explícita:

1. **Vetores individuais:** linhas por `embedding_id`, separadas por modelo e categoria, com texto/fonte, segmento, escore individual, distâncias e posição no respectivo grupo. Não deduplicar por tópico nesta visão, pois o estudo deve permitir inspecionar cada vetor.
2. **Tópicos recuperados:** linhas por tópico/modelo, com os escores individuais das categorias selecionadas e sua média; usar o agrupamento e a ordenação já definidos. A comparação entre modelos continua em resultados distintos, sem média cruzada de seus escores.

`top_k` e limiar respeitam a visão escolhida: na visão de vetores, limitam vetores por modelo/categoria; na visão de tópicos, limitam tópicos por modelo/combinação. Registrar `unidade_resultado` e `escopo_limites` no JSON. Não dividir um top-k único entre modelos/categorias silenciosamente. A visão de tópicos permanece como padrão compatível com o buscador atual.

Permitir ordenar, filtrar e separar os vetores por escore individual de uma execução. Na visão agregada, permitir separar tópicos por sua média, conservando os escores dos vetores participantes. Selecionar explicitamente a base do agrupamento (`similaridade_individual` ou `similaridade_media`) e, para distância, informar a métrica e a direção de ordenação.

Faixas configuráveis pelo usuário, por exemplo: [-1, 0,40), [0,40, 0,70), [0,70, 1]. Esses limites são ilustrativos; não atribuir nomes como “relevante” ou “irrelevante” sem critério escolhido. Usar intervalos inferiores inclusivos/superiores exclusivos, fechando o último em 1, para impedir duplicação no limite. Validar valores finitos, ordem e ausência de sobreposição nas partições; permitir faixas incompletas com indicação de itens fora delas. Filtros simples por intervalo mínimo/máximo também serão aceitos, sem obrigar criar uma partição inteira.

Cada conjunto produzido registra consulta/execução, modelo/perfil, categoria(s), base do escore, limites, IDs membros, seus escores/distâncias e revisões dos NPZ. Salvar descrições/referências no JSON e permitir carregar seus vetores originais nos NPZ para cálculos/clusterização posteriores. Separar por escore não copia nem modifica embeddings; é uma seleção persistente associada à execução.

Preservar os escores calculados de **todos os vetores elegíveis das categorias selecionadas**, não somente os que entraram no top-k ou passaram no limiar final. Gravar esse universo em `estudos/execucoes/<execucao_id>/escores.npz`, com IDs alinhados, escores, distâncias e referências de origem descritas em seu manifesto/JSON. No JSON principal, guardar os resultados finais com suas medidas escalares, contagens por categoria/faixa e referência ao universo avaliado. Assim, é possível inspecionar faixas abaixo do corte, alterar faixas/limites e reapresentar resultados sem regerar a consulta nem os vetores da base. A seleção de novas categorias/modelos fora do universo registrado exige uma nova execução compatível.

Não aplicar o top-k individual de uma categoria antes de calcular as médias de tópicos: a visão individual é uma apresentação do mesmo universo avaliado, não um filtro implícito que altera a combinação. Uma mudança da média ou da seleção deverá criar um novo resultado/execução derivada, referenciando a avaliação original quando puder reaproveitar seus escores.

### Escore mínimo e top-k como critérios de término dos resultados

Verificação do código atual: `buscar_topicos.py` já oferece `--limiar` (sem filtro por padrão) e `--top-k` (padrão 5), valida limiar no intervalo [-1, 1] e top-k positivo. Hoje esses parâmetros atuam sobre um vetor por tópico. Preservar os nomes e adaptá-los ao escore médio calculado com a seleção do usuário:

A tabela seguinte descreve a visão de tópicos. Na visão de vetores individuais, aplicar as mesmas regras ao escore individual e à quantidade de vetores de cada grupo modelo/categoria, conforme a seção anterior.

| Configuração | Comportamento planejado |
| --- | --- |
| Somente top-k | Retornar os k tópicos únicos de maior média, sem escore mínimo |
| Somente escore mínimo | Com `--sem-top-k`, retornar todos os tópicos cuja média seja maior ou igual ao limiar |
| Escore mínimo + top-k | Retornar até k tópicos, todos com média maior ou igual ao limiar; pode haver menos de k ou zero |
| Sem os dois limites | Com `--sem-top-k` e sem limiar, retornar todos os tópicos elegíveis em ordem decrescente de média |

Por padrão manter `top_k=5` e `limiar=None`, sem impor um valor de similaridade ainda não calibrado. Validar top-k inteiro positivo, limiar numérico finito entre -1 e 1 inclusive, e conflito entre `--top-k` explicitamente fornecido e `--sem-top-k`. Tratar limiar 0 como filtro válido, não como ausência de filtro. Similaridade cosseno não é probabilidade nem porcentagem de acerto; exibir o escore decimal.

Calcular todas as similaridades elegíveis e médias antes de decidir os resultados finais. Depois de ordenar por média, encerrar a emissão ao atingir k tópicos ou ao encontrar a primeira média abaixo do limiar, o que ocorrer primeiro; se nenhum limite se aplicar, encerrar ao esgotar candidatos. Não interromper a varredura de uma matriz não ordenada ao encontrar um escore baixo. Esses parâmetros limitam a saída; não autorizam uma aproximação que deixe de avaliar candidatos com melhor média.

Exemplo: médias 0,82, 0,76 e 0,61, `top_k=5` e `limiar=0,70` produzem dois resultados, não cinco. Uma média exatamente 0,70 passa no filtro. Em empate no limite k, retornar no máximo k, com desempate estável por ordem de documento/tópico e indicação de empate, sem expandir silenciosamente além de k.

O limiar principal é aplicado à média. Similaridades individuais abaixo dele não excluem um tópico que atinja a média solicitada. Como opção adicional, permitir filtros mínimos individuais por tipo selecionado, com combinação explícita `todos` ou `qualquer`; esses filtros são separados do limiar da média e não mudam sua fórmula. Quando houver segmentos, validar os escores individuais das unidades daquele tipo segundo o critério escolhido e mostrar a regra usada. Nenhum filtro individual será ativado por padrão.

Configurar/salvar limites por modelo/dimensão e combinação de tipos, com exportação de parâmetros efetivos, cobertura, revisões dos NPZ usados, quantidade de candidatos avaliados/elegíveis, quantidade retornada e motivo de término (`top_k`, `limiar`, `candidatos_esgotados` ou `sem_resultados`). Calibrar limites com perguntas reais; escores do E5, BGE-M3 e EmbeddingGemma não têm a mesma distribuição. Cada consulta constitui uma execução persistente, ligada ao texto de entrada e a seu embedding armazenado, conforme a seção 11.

Comparar modelos em resultados separados, aplicando a mesma seleção e a mesma regra de média a cada modelo. Não calcular automaticamente médias cruzadas entre modelos diferentes nem reutilizar um limiar universal. Registrar consulta, modelo/revisão, tipos e IDs selecionados, política de cobertura, similaridades individuais e média no resultado exportado, sem incluir vetores numéricos.

Quando a seleção contiver apenas metadados, tópicos do mesmo documento terão pontuações iguais. A saída identificará que a correspondência vem dos metadados documentais e exibirá os IDs dos demais tópicos relacionados. Com metadados combinados a outros vetores, mostrar sua similaridade individual e incluí-la na mesma média aritmética solicitada. Manter `top_k` como quantidade de tópicos únicos, desempatar por ordem documental e ordem do tópico e informar quando o limite corta um grupo empatado. Oferecer visualização agrupada por documento sem atribuir ao conteúdo dos tópicos uma relevância que não foi calculada.

Filtros exatos de tipo, eixo, identificação ou datas, quando usados, serão aplicados aos metadados estruturados, não inferidos da similaridade. Não acrescentar regras de vigência automática sem definição do usuário.

## 8. Escolha do usuário e arquivos afetados

O projeto atual usa terminal e atalhos Windows. Acrescentar menu aos scripts PowerShell e argumentos equivalentes; não introduzir uma aplicação web para esta alteração.

Fluxo de geração: escolher pares modelo/tipo na matriz → escolher escopo da base e operação (faltantes, alterados ou regeneração) → exibir simulação → executar e publicar somente as revisões selecionadas. Nenhum tipo exige gerar outro. Na busca: escolher pares modelo/categoria disponíveis → escolher visão (vetores ou tópicos) → configurar limites/faixas → executar avaliações separadas por modelo → mostrar escores individuais/médias e grupos por faixa. Exibir escolha explícita para palavras-chave isoladas e para a variante total com palavras-chave. O menu não solicitará chave de um provedor que não foi escolhido.

Argumentos propostos: `--listar-modelos`, `--listar-vetores`, `--provedor`, `--modelo`, `--modo` (atalho opcional), `--tipos-gerar`, `--regenerar`, `--vetores` (seleção múltipla para busca), `--embedding-ids` (seleção detalhada), `--permitir-cobertura-parcial`, `--perfil`, `--modelos-dir`, `--dimensoes` opcional, `--selecionar`, `--ids`, `--ids-arquivo`, `--simular`, `--validar` e `--ativar-perfil`. Preservar `--limiar` e `--top-k`, acrescentar `--sem-top-k` e filtros individuais opcionais `--limiar-por-vetor`/`--criterio-limiar-vetores`. Para executar pares diferentes em um único trabalho, usar plano de geração explícito (arquivo de configuração ou menu), evitando interpretar uma lista de modelos e tipos como produto cartesiano obrigatório. Validar seleção não vazia e nomes/tipos realmente disponíveis. Não impor 1536 dimensões a todos os modelos. Os exemplos no README só serão apresentados como comandos funcionais após a implementação.

Acrescentar opções de exploração: `--plano-busca` para pares/modelos e categorias específicos, `--visao vetores|topicos`, `--escore-maximo`, `--faixas-escore`, `--agrupar-por modelo,categoria,faixa`, `--execucao` para reabrir um resultado, e exportação de conjuntos por faixa. As faixas do estudo poderão ser reconfiguradas sobre o universo de escores persistido, sem alterar a execução fonte.

| Arquivo | Alteração planejada |
| --- | --- |
| `gerar_embeddings.py` | Geração e regeneração independentes por modelo/tipo/tópico, preservação de vetores anteriores e publicação de NPZ por tipo |
| `buscar_topicos.py` | Seleção de tipos disponíveis por modelo, similaridades individuais/média, limiar e top-k, agrupamento por tópico e conferência de hashes |
| `buscar_topicos.py` / `save_result` | Persistir embedding de entrada e JSON de metadados/execução, com texto original, referências aos vetores recuperados, textos avaliados, distâncias e parâmetros |
| `pipeline_etica.py` | Migrar o JSON para schema 3.0, delegar embeddings ao NPZ, detectar palavras-chave e sincronizar os metadados das unidades/validação |
| `executar_embeddings.ps1` e `.bat` | Menu e encaminhamento de configuração; eliminar modelo/dimensão fixos |
| `executar_busca.ps1` e `.bat` | Seleção de perfil e combinação de vetores, palavras-chave desmarcadas por padrão e dependências condicionais |
| `01_configurar_ambiente.bat`, `requirements.txt` | Dependências básicas completas e instalação opcional por provedor |
| `app_1.py`, `app_2.py` | Documentar como versões históricas; se forem entradas usadas, delegar a implementação compartilhada |
| `README.md`, `LEIA_PRIMEIRO.txt` | Unificar instruções com o caminho NPZ, explicar modos, perfis e limitações |
| `tests/` | Testes de registros, provedores, índice, migração e recuperação |

Dependências locais pesadas como Sentence Transformers/PyTorch serão opcionais; `google-genai` para Gemini também. Corrigir verificações dos atalhos para não instalar OpenAI em uma execução exclusivamente local.

Conforme a instrução mais recente do usuário, tratamento de segurança/exposição de chaves não fará parte do escopo atual. Manter a opção de alimentação direta da chave no código, como existe hoje, e reservar um ponto de integração para provisão segura. Centralizar esse contrato em `embedding_credentials.py`, com campos `OPENAI_API_KEY_DIRETA` e `GEMINI_API_KEY_DIRETA` e função `obter_chave(provedor)`. Preservar a leitura já existente por variável de ambiente/configuração e aceitar a chave direta como alternativa; prever um resolvedor configurável para entrada protegida ou outro mecanismo seguro posteriormente. Manter esses campos e a precedência documentados, sem condicionar a exploração a uma mudança de estratégia de credenciais. Provedores locais não precisam de chave.

O [ROTEIRO_IMPLEMENTACAO.md](ROTEIRO_IMPLEMENTACAO.md) resume a ordem de execução e os critérios de conclusão. O presente plano mantém os detalhes dos contratos, schemas e comportamentos.

## 9. Sequência de entrega e critérios de aceite

1. **Consolidar a arquitetura:** extrair contratos compartilhados e preservar a leitura das composições/artefatos legados.
2. **Reestruturar JSON e NPZ:** implementar schemas, referências e publicação por perfil/categoria, com migração sem perda.
3. **Implementar provedores e chaves:** manter OpenAI; acrescentar Ollama/Sentence Transformers e Gemini gratuito disponível, com chave direta e alternativa de provisão centralizadas.
4. **Gerar os oito tipos independentemente:** preservar o total sem palavras-chave e implementar textos, segmentação, hashes, atualização/regeneração por pares/IDs e armazenamento exclusivo em NPZ.
5. **Persistir entradas e experimentos:** salvar embeddings/textos de entrada, parâmetros, revisões, referências, textos recuperados e distâncias.
6. **Adaptar busca e exploração:** seleção por modelo/categoria, escores individuais/médias, limiar/top-k, cobertura, persistência do universo avaliado e faixas de escore.
7. **Integrar menus/exportação e validar offline:** atualizar atalhos, parâmetros, dependências, documentação e reabertura dos experimentos; testar contratos/migração com provedores simulados antes de instalar os modelos.
8. **Instalar modelos e validar operação real:** instalar Ollama no D:, baixar BGE-M3 e EmbeddingGemma Q4, configurar perfil 256 e oferecer E5 Small 384 como alternativa local; executar piloto em português antes das gerações completas escolhidas. Esta instalação permanece posterior à implementação, conforme solicitado.
9. **Ampliar a bancada analítica:** implementar cálculos/manipulação e métodos de clusterização após explorar a recuperação; seus contratos/proveniência já estarão preparados nas etapas iniciais.

Critérios verificáveis:

- Listar apenas modelos locais utilizáveis nos mecanismos configurados e distinguir serviços gratuitos sujeitos a cota.
- Oferecer pelo menos uma opção local com dimensão efetiva abaixo de 400, distinguindo dimensão nativa de redução Matryoshka; verificar tamanho efetivo dos vetores de documentos e consultas.
- Gerar qualquer um dos oito tipos para qualquer modelo sem exigir a geração dos demais, inclusive gerar apenas palavras-chave ou apenas metadados em um perfil novo.
- Atualizar um par modelo/tipo sem substituir outros pares; manter suas revisões, hashes e dados numéricos intactos.
- Atualizar a base e marcar somente unidades afetadas sem apagar seu NPZ nem iniciar geração automática; falha ao regenerar não elimina o último vetor armazenado.
- Forçar regeneração dos pares/IDs escolhidos com hash inalterado, preservando os não selecionados.
- Gerar e consultar localmente sem OpenAI nem sua chave.
- Selecionar os três modos e manter o vínculo dos vetores ao tópico original.
- Preservar texto, hash e vetor do tópico total sem palavras-chave quando apenas palavras-chave mudarem.
- Manter palavras-chave e total com palavras-chave fora da recuperação padrão; permitir seu uso isolado ou em qualquer combinação explicitamente escolhida.
- Mostrar todas as similaridades dos vetores utilizados e sua média aritmética; garantir que vetores desmarcados não influenciem o resultado.
- Testar ordenação pela média com casos em que ela difere da ordenação pelo máximo, incluindo cálculo com um vetor, vários vetores e segmentos.
- Validar seleção vazia, tipos ausentes, cobertura incompleta e opção explícita de cobertura parcial, mostrando o denominador da média.
- Testar somente top-k, somente limiar, ambos e nenhum; limiar inclusivo, limiar zero, k maior que a cobertura, zero resultados, empates e rejeição de NaN/infinito.
- Confirmar término pela média dos vetores selecionados, e não pelo escore máximo de um campo; registrar parâmetros e motivo de término.
- Executar a mesma consulta em pares modelo/categoria escolhidos sem incluir pares não selecionados; exibir resultados separados e preservar o vetor de consulta correto para cada modelo.
- Inspecionar vetores individualmente e tópicos agregados, distinguindo top-k por vetor/categoria de top-k por tópico/combinação.
- Criar e salvar faixas por escore individual ou média, validar suas fronteiras e reabrir os conjuntos por consulta/modelo/categoria sem regenerar embeddings.
- Persistir todos os escores elegíveis avaliados, inclusive abaixo do limiar e fora do top-k, permitindo explorar outras faixas posteriormente.
- Conferir que faixas de uma consulta não alteram nem se confundem com faixas obtidas pelo mesmo vetor em outra consulta.
- Persistir em NPZ cada embedding de entrada antes de executar a recuperação e registrar o texto original, papel consulta/documento, hashes e configuração de geração no JSON próprio.
- Recuperar uma execução salva e localizar os vetores de entrada e de todos os textos recuperados, nas revisões exatas, sem chamar novamente o modelo.
- Registrar similaridade/distância individual por vetor e médias; verificar `distancia_cosseno = 1 - similaridade_cosseno` e correspondência da distância euclidiana com os vetores normalizados.
- Reutilizar embedding de entrada compatível em buscas com seleções/limites diferentes, criando novas execuções sem sobrescrever resultados anteriores.
- Atualizar a base e preservar a reprodução dos experimentos antigos por snapshots/referências/textos registrados; limpeza não deve remover artefatos ainda referenciados.
- Conservar vetores como retornados pelo provedor e rastrear normalização/redução/precisão, sem inventar versões originais para o legado normalizado.
- Validar operações entre vetores de espaços compatíveis e salvar qualquer vetor derivado em NPZ separado com identificação dos operandos e parâmetros.
- Garantir que base JSON, manifesto, configuração e checkpoints JSON não contenham arrays de vetores em nenhum caminho de execução.
- Verificar que cada unidade marcada como atualizada no JSON referencia um `embedding_id` existente no NPZ da revisão correta, com hash e dimensão compatíveis.
- Migrar `embedding` único para `embeddings.perfis` sem perda de referências e sem manter o campo `vetor` no novo schema.
- Retornar `top_k` tópicos distintos quando houver resultados suficientes, mesmo que os melhores vetores pertençam ao mesmo tópico.
- Alterar um campo e regenerar apenas suas unidades afetadas; alterar metadados sem recalcular modos que não os usam.
- Reutilizar um perfil existente; bloquear mistura de modelos e reconstruir somente quando necessário.
- Retomar após falha de lote e impedir leitura de NPZ/manifesto de revisões diferentes.
- Ler índices legados, preservando suas regras e seus artefatos.
- Tratar campos vazios, texto longo, dimensão incorreta, vetor nulo/não finito, modelo ausente, cota esgotada e tópicos inativos.
- Conferir a estrutura com exemplos reais. Para os 282 tópicos: 282 unidades completas sem palavras-chave, até 282 unidades completas adicionais com palavras-chave, até 1.410 unidades separadas (incluindo 282 de palavras-chave) e 282 referências de metadados com aproximadamente 12 textos documentais únicos; até 2.256 unidades no conjunto dos oito tipos, antes da segmentação e conforme disponibilidade dos campos.

Comparar recuperação usando perguntas reais em português, com tópicos esperados definidos e revisados por uma pessoa: Recall@5, MRR, tempo de geração/consulta e tamanho do índice. A referência inicial de comparação entre modelos será somente tópico total sem palavras-chave. Experimentos com palavras-chave terão seleção explicitamente registrada, mantendo as mesmas combinações entre modelos e resultados separados da referência. A qualidade do modo metadados deve ser avaliada como busca documental; não pressupor que ele distingue assuntos internos do mesmo documento.

## 10. Etapa posterior: instalar e configurar modelos locais maior e leve

Depois de implementar e validar a nova estrutura, instalar e configurar o Ollama neste Windows com **BGE-M3 (`bge-m3`, 1024 dimensões)** e a opção menor **EmbeddingGemma QAT Q4 (`embeddinggemma:300m-qat-q4_0`, perfil de 256 dimensões)**, ambos locais e sem cobrança por inferência. Incluir **E5 Small (384 dimensões nativas)** no catálogo como alternativa local pelo adaptador Sentence Transformers. Não antecipar esta instalação durante a fase de planejamento.

O BGE-M3 tem licença MIT, suporte multilíngue para mais de 100 idiomas, saída densa de 1024 dimensões e limite de 8192 tokens. O pacote publicado no Ollama tem aproximadamente 1,2 GB. Esses atributos justificam escolhê-lo como candidato inicial para recuperação em português com tópicos normativos extensos; a qualidade específica em português nesta base será medida, não presumida a partir do suporte multilíngue. A integração usará os vetores densos de `/api/embed`; não pressupor que essa API oferece as saídas sparse/ColBERT disponíveis em outras implementações do modelo.

A consulta de disco realizada em 30/09/2026 encontrou aproximadamente 2,5 GB livres no C: e 552,8 GB no D:. A documentação do Ollama informa necessidade de pelo menos 4 GB para os binários Windows, além dos modelos. Por isso, usar o D: para download temporário, instalação e armazenamento de modelos. RAM/GPU não foram confirmadas nesta fase: a consulta ao inventário de hardware não teve acesso. Conferir esses recursos antes de dimensionar lotes e medir desempenho.

Procedimento previsto:

1. Verificar instalação/servidor Ollama existente, versão do Windows, espaço livre atualizado, memória e disponibilidade de aceleração. Reutilizar uma instalação compatível quando encontrada.
2. Baixar o instalador oficial para o D: e instalar os binários em `D:\Ollama`, usando a opção oficial `/DIR` quando aplicável. Definir o destino de arquivos temporários do instalador no D: e conferir se a instalação ainda exige espaço adicional no C:.
3. Configurar `OLLAMA_MODELS=D:\Ollama\models` antes de baixar o modelo. Preservar modelos e configurações de qualquer instalação anterior; não mover/apagar pastas automaticamente.
4. Configurar `OLLAMA_HOST=127.0.0.1:11434` e `OLLAMA_NO_CLOUD=1`, iniciar/reiniciar o servidor e verificar a API local. Downloads do instalador/modelo exigem internet; geração e consulta serão locais após o download.
5. Executar `ollama pull bge-m3` e `ollama pull embeddinggemma:300m-qat-q4_0`, conferir a presença com a listagem de modelos e registrar digests. Verificar versão compatível do Ollama; a ficha EmbeddingGemma indica mínimo 0.11.10, devendo-se confirmar a versão estável no momento da instalação. Não exigir conta/chave de um provedor de inferência pago. Para a alternativa E5, oferecer download/instalação local explícitos pelo adaptador correspondente, com cache no D:.
6. Testar a inferência com textos em português, confirmando número de respostas, dimensões efetivas de 1024 para BGE-M3 e 256 para EmbeddingGemma e vetores finitos/não nulos. Validar redução/normalização e preparação correta do EmbeddingGemma; para E5, 384 dimensões e prefixos próprios. Usar a mesma configuração para documentos e consultas de cada perfil.
7. Cadastrar perfis independentes para os modelos/dimensões; publicar NPZ independentes para cada tipo que o usuário escolher gerar, sem exigir todos os oito. A seleção para recuperação será independente dos tipos gerados. Os vetores ficarão exclusivamente nos NPZ; metadados/referências ficarão no JSON reestruturado.
8. Gerar um pequeno conjunto piloto com pares modelo/tipo escolhidos e validar hashes, agrupamento, preservação de outros pares, atualização incremental, escore mínimo e top-k. Só então gerar os conjuntos completos efetivamente selecionados, mantendo os índices anteriores disponíveis.
9. Avaliar perguntas em português sobre os documentos, com tópicos esperados revisados, medindo Recall@5, MRR, tempo de geração, latência da consulta e consumo de memória. Ajustar lote e limites ao hardware real; CPU é uma opção caso não haja GPU utilizável.
10. Registrar versão do Ollama, digests dos modelos, dimensões/preparação, configuração, resultados e instruções para iniciar o servidor, escolher perfis e selecionar tipos. A etapa termina com geração independente e busca por combinações funcionais para modelo maior e leve, inclusive sem chave OpenAI.

Se a avaliação mostrar qualidade ou tempo inadequados, comparar outro modelo multilíngue local antes de adotá-lo, mantendo a escolha do usuário e um novo perfil para cada modelo. Esta etapa fica autorizada como parte da sequência solicitada; eventuais acessos de instalação, rede ou escrita fora do workspace obedecerão às permissões do ambiente no momento da execução.

## 11. Persistência das entradas e dos resultados de exploração

### Separação dos conjuntos

Manter quatro famílias de dados, com IDs estáveis e metadados JSON vinculados aos NPZ:

1. **Base de conteúdos:** documentos, tópicos e oito tipos de embedding por modelo, conforme as seções anteriores.
2. **Entradas:** textos que o usuário informa para recuperação/estudo e embeddings efetivamente gerados para cada modelo/configuração/papel.
3. **Execuções:** cada recuperação, com seleção dos vetores, limites, resultados, métricas e revisões utilizadas. Uma mesma entrada pode originar várias execuções.
4. **Análises/derivados:** conjuntos filtrados, comparações, cálculos, projeções e clusterizações futuras, com suas origens registradas.

Não inserir as entradas como tópicos da base documental nem incluí-las automaticamente na recuperação/clusterização dos conteúdos. Sua participação em análises é uma seleção explícita do usuário.

Organização sugerida: `estudos/entradas/<entrada_id>/entrada.json`; embeddings em `estudos/entradas/<entrada_id>/<perfil_id>/<papel>/<geracao_id>/vetores.npz`; resultados em `estudos/execucoes/<execucao_id>/resultado.json`; derivados em `estudos/analises/<analise_id>/`. O JSON da entrada lista suas gerações e execuções; o JSON de cada execução também contém o texto original, para permanecer compreensível quando aberto isoladamente. Vetores recuperados são referenciados nos NPZ versionados da base, sem duplicação obrigatória em outro arquivo.

### Geração, reaproveitamento e retenção

Salvar o texto original exatamente como recebido, inclusive acentos, caixa e quebras de linha, separado do texto eventualmente preparado para o modelo. Gerar e persistir o embedding da entrada antes de calcular a recuperação. Registrar modelo/digest, dimensão, papel (`consulta` por padrão), hash original/preparado, normalização, segmentação quando aplicável, dtype, data e referência ao NPZ. Falha na persistência deve impedir apresentar a execução como concluída/registrada.

A chave de reaproveitamento inclui hash do texto preparado, perfil do modelo e papel/preparação. Repetir uma entrada com parâmetros de busca diferentes reutiliza o vetor compatível, mas gera outra execução. Não reutilizar automaticamente vetores entre modelos, dimensões, revisões ou papéis diferentes. Preparação de consulta e documento pode diferir; se o estudo requerer embedding do mesmo texto como documento, gerar essa representação separadamente e somente quando solicitada.

Regeneração sob demanda cria uma nova geração da entrada, preservando a anterior enquanto houver execução referenciada. Alterar o texto cria uma nova entrada/versão, sem substituir silenciosamente o texto de uma execução antiga. Registrar revisão da base e referências de arquivos imutáveis/checksums. Como textos da base podem mudar, registrar o texto fonte de cada unidade recuperada no resultado ou em um snapshot textual do experimento por ele referenciado; manter o texto efetivamente embeddado/preparação rastreáveis. Isso permite entender a recuperação original sem associar um vetor antigo ao conteúdo atual.

Não permitir limpeza de um NPZ referenciado por uma execução/analise conservada sem primeiro preservar seus dados em um pacote do experimento ou excluir explicitamente o experimento relacionado. Oferecer exportação futura de um pacote JSON + NPZ autossuficiente, sem arrays de vetores no JSON.

### Informações obrigatórias no JSON da execução

- `entrada_id`, `execucao_id`, texto original, hashes, data, status, duração e erro quando houver.
- Perfil do modelo, revisão/digest, dimensão, papel da entrada, referências ao seu embedding e revisões/checksums dos índices usados.
- Tipos/IDs de vetor selecionados, política de cobertura/segmentação, limiar, top-k e filtros individuais efetivos.
- `comparacao_id` quando houver avaliação conjunta de modelos, visão/unidade de resultado, escopo dos limites, agrupamento por modelo/categoria, faixas de escore e referência ao universo completo de escores calculados.
- Para cada resultado: tópico/documento, posição, fonte, texto fonte recuperado ou referência ao snapshot, similaridades individuais, distâncias individuais, média, denominador e motivo de término.
- Para cada vetor recuperado utilizado: `embedding_id`, tipo/campo, segmento, hashes, arquivo NPZ, revisão, linha, representação original/normalizada e vínculo com o vetor de entrada comparado.

Os nomes `vetor_entrada` e `vetor_recuperado` nos JSON significam **referências de armazenamento**, nunca listas de coordenadas. Guardar nos NPZ também os embeddings das entradas que não produziram resultados; registrar `resultados: []` e a razão.

Grupos por faixa terão seus próprios IDs de conjunto e metadados vinculados à execução, com definição de limites e IDs dos vetores/tópicos membros. Não gravar uma categoria de escore estática no cadastro global do embedding: sua associação depende da consulta, modelo, métrica e combinação aplicada.

### Similaridade e distâncias

Na exploração inicial, registrar para cada par entrada/unidade recuperada:

- Similaridade cosseno `s` usada na recuperação.
- Distância cosseno `1 - s`, indicando que valores menores são mais próximos.
- Distância euclidiana L2 entre as representações normalizadas, calculada diretamente ou conferida por `sqrt(max(0, 2 - 2*s))` quando ambos os vetores têm norma 1.

Por tópico, salvar a média das similaridades e a média das distâncias calculadas individualmente. `distancia_cosseno_media = 1 - similaridade_media` para o mesmo conjunto de unidades. A média das distâncias euclidianas não é, em geral, `sqrt(2 - 2*similaridade_media)`; calcular a média de suas distâncias individuais, sem confundir as duas expressões. A ordenação inicial permanece por média da similaridade cosseno, conforme solicitado; outras métricas de análise não alteram silenciosamente o ranking.

Usar IDs dos dois operandos e indicar a representação/normalização em cada cálculo. Futuramente permitir métricas adicionais (produto interno, distância angular, L1 etc.) por seleção, com definição/formulação e parâmetros registrados. Distâncias sobre vetores originais não normalizados são resultados distintos dos cálculos normalizados. Uma similaridade agregada de vários campos não representa a distância a um suposto único vetor de tópico.

### Exemplo ilustrativo de resultado persistente

```json
{
  "schema_version": "1.0",
  "entrada_id": "ent_exemplo",
  "execucao_id": "exp_exemplo",
  "texto_original_entrada": "Quais são as condições para o consentimento?",
  "perfil_id": "pf_exemplo_384",
  "revisao_base": "base_rev_001",
  "vetor_entrada": {
    "embedding_id": "ent_exemplo::consulta::ger_001",
    "arquivo_npz": "estudos/entradas/ent_exemplo/pf_exemplo_384/consulta/ger_001/vetores.npz",
    "linha": 0,
    "dimensoes": 384,
    "normalizacao": "l2"
  },
  "parametros": {
    "tipos_vetor": ["resumo"],
    "top_k": 5,
    "limiar": null,
    "cobertura": "completa",
    "metrica_ranking": "media_similaridade_cosseno"
  },
  "resultados": [
    {
      "topico_id": "tp_exemplo",
      "documento_id": "doc_exemplo",
      "posicao": 1,
      "similaridade_media": 0.8,
      "distancia_cosseno_media": 0.2,
      "distancia_euclidiana_media": 0.632455532,
      "quantidade_vetores": 1,
      "comparacoes": [
        {
          "tipo_vetor": "resumo",
          "texto_fonte_recuperado": "Regras de consentimento descritas no documento.",
          "vetor_recuperado": {
            "embedding_id": "tp_exemplo::resumo::0000",
            "arquivo_npz": "indices/pf_exemplo_384/resumo/rev_001/vetores.npz",
            "revisao_indice": "rev_001",
            "linha": 0
          },
          "similaridade_cosseno": 0.8,
          "distancia_cosseno": 0.2,
          "distancia_euclidiana_normalizada": 0.632455532
        }
      ]
    }
  ],
  "status": "concluido",
  "motivo_termino": "candidatos_esgotados"
}
```

Os números e IDs são ilustrativos; o exemplo omite hashes/checksums, datas, fontes e configuração completa para facilitar leitura. A implementação deverá incluir os campos obrigatórios acima. Resultados finais preservam o vínculo de todos os vetores escolhidos que participaram de suas médias, não somente o vetor de maior escore.

## 12. Preparação para manipulação, cálculos e clusterização

### Contrato de conjuntos e operações

Oferecer um serviço de acesso que carregue conjuntos por seleção explícita de modelos/perfis, categorias/tipos, tópicos, documentos, entradas, execuções ou faixas de escore de uma consulta, devolvendo matriz + IDs + metadados + revisões/representação. Operações básicas previstas: filtrar, selecionar/reordenar, exportar/importar conjuntos compatíveis, comparar pares, produzir matrizes de similaridades/distâncias, calcular normas, somas, diferenças, médias, centroides, combinações ponderadas e normalizações.

Salvar matrizes grandes, vetores derivados, centroides e coordenadas de projeção em NPZ; JSON contém descrições, referências, parâmetros, IDs e medidas escalares. Todo derivado registra operandos, coeficientes, método, perfil/espaço, transformações e data. Vetores derivados não passam a ser embeddings de textos originais e não substituem os oito tipos. A média vetorial ou um centroide de estudo é uma operação distinta da média de similaridades usada na recuperação.

Validar identidade do espaço, dimensão, revisão do modelo e preparação antes de operações coordenada a coordenada. Ter o mesmo número de dimensões não torna dois modelos compatíveis. Comparação entre modelos usará alinhamento por textos/IDs, rankings, matrizes de relações e métricas de avaliação separadas; não somar coordenadas de modelos diferentes. Alterações manuais/derivadas criam novos conjuntos, preservando a base fonte. Substituir embeddings correntes exige operação explícita de atualização/regeneração, conforme o controle independente já definido.

Planejar matrizes par a par em blocos quando necessário: armazenamento/cálculo completo cresce quadraticamente no número de vetores. Permitir seleção/amostragem e registrar quais IDs participaram. Não concatenar oito tipos nem fazer sua média vetorial automaticamente; o usuário escolhe a representação de cada experimento.

### Métodos de clusterização previstos

Preparar adaptadores para K-Means/MiniBatchKMeans, agrupamento hierárquico aglomerativo, DBSCAN, HDBSCAN e agrupamento espectral. Esses métodos constam da documentação oficial scikit-learn; suas hipóteses, métricas e parâmetros diferem. O planejamento de extensibilidade não exige implementar todos na fase inicial de exploração da recuperação.

Cada execução de clusterização deverá registrar: conjunto/IDs de entrada, modelo/dimensão, tipos/papéis, representação, normalização, eventual redução de dimensão, algoritmo, parâmetros, métrica, semente aleatória quando aplicável, versões das bibliotecas e duração. Salvar rótulos por ID, ruído/outliers quando houver, centroides quando pertinentes, medidas de avaliação e referência aos vetores originais.

Configurações específicas: número de grupos para K-Means; número de grupos ou corte por distância e linkage na hierarquia; `eps`/`min_samples` no DBSCAN; tamanho mínimo e demais parâmetros de densidade no HDBSCAN; afinidade/vizinhança e número de grupos no espectral. Não presumir que K-Means ou Ward aceitam distância cosseno arbitrária: usar formulação euclidiana apropriada ou outro linkage/método que suporte a métrica escolhida.

Evitar que o mesmo tópico com mais campos/segmentos conte como várias observações sem essa escolha estar explícita. Definir a unidade da análise: vetor/campo/segmento, tópico com representação escolhida ou documento. Em metadados, considerar vetores idênticos de tópicos do mesmo documento; oferecer deduplicação/pesos explícitos para não confundir repetição com densidade semântica.

Projetar inicialmente PCA para inspeção em 2D/3D, com possibilidade futura de UMAP/t-SNE por adaptadores. Coordenadas projetadas são dados derivados, não substitutos dos embeddings originais. Se a clusterização for executada no espaço projetado, registrar essa escolha: clusters/proximidades em 2D não garantem a mesma estrutura no espaço completo. Resultados serão hipóteses de agrupamento semântico para exploração/revisão, não categorias normativas automaticamente validadas.

### Escopo da primeira entrega

Implementar primeiro armazenamento de entradas, geração/reaproveitamento por modelo, recuperação por modelo/categoria, visões individual/agregada, persistência do universo de escores, grupos por faixa, JSON com textos/referências/distâncias, listagem de experimentos e reabertura/exportação dos resultados. Preparar contratos de conjuntos/proveniência e módulos de análise para receber operações/clusterizações posteriores, com dependências analíticas opcionais. A instalação de bibliotecas extras e a implementação completa dos métodos analíticos ficam em uma etapa específica após a exploração inicial.

## 13. Referências técnicas consultadas

- [Ollama: listar modelos instalados](https://docs.ollama.com/api/tags) e [gerar embeddings](https://docs.ollama.com/api/embed).
- [Sentence Transformers: modelo local, preparação de consulta e documento](https://sbert.net/docs/package_reference/sentence_transformer/model.html).
- [Modelo E5: prefixos e limites](https://huggingface.co/intfloat/multilingual-e5-small).
- [Gemini: embeddings](https://ai.google.dev/gemini-api/docs/embeddings) e [preços e camada gratuita](https://ai.google.dev/gemini-api/docs/pricing).
- [BGE-M3 no Ollama: download e API](https://ollama.com/library/bge-m3) e [ficha do modelo BAAI: licença, dimensões e limites](https://huggingface.co/BAAI/bge-m3).
- [Ollama no Windows: instalação e local de armazenamento](https://docs.ollama.com/windows) e [configuração do servidor, modelos e modo local](https://docs.ollama.com/faq).
- [E5 Small: dimensão nativa e limites](https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md) e [configuração do modelo](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/config.json).
- [EmbeddingGemma: opções quantizadas e tamanhos no Ollama](https://ollama.com/library/embeddinggemma/tags), [redução de dimensões Matryoshka](https://ai.google.dev/gemma/docs/embeddinggemma) e [exemplo oficial com 256 dimensões](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers).
- [MiniLM multilíngue: 384 dimensões e contexto](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [All-MiniLM no Ollama: tamanhos](https://ollama.com/library/all-minilm/tags) e [ficha original All-MiniLM L6: modelo em inglês](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- [scikit-learn: métodos de clusterização e suas configurações](https://scikit-learn.org/stable/modules/clustering.html), [métricas entre pares](https://scikit-learn.org/stable/modules/metrics.html) e [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

O catálogo online deverá ser revalidado na implementação: disponibilidade, modelos e condições gratuitas podem mudar. Este planejamento não instalou modelos, não regenerou embeddings e não alterou as bases existentes.
