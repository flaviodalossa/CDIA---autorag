# AutoRAG Web v0.1

Execute `09_abrir_web.bat` para abrir o serviço local na porta 8766. Veja [guia da aplicação web](../APLICACAO_WEB.md) para preparar bases normativas e publicar a interface no Cloudflare com backend Python persistente.

O aplicativo desktop e os recursos analíticos originais continuam abaixo.

# AutoRAG Conceito

Aplicação desktop para estudar embeddings de tópicos em português. A interface usa Tkinter, com geração em segundo plano, busca por pares de modelo/categoria e histórico de experimentos. Também há uma interface de linha de comando e módulos Python para operações analíticas.

## Começar no Windows

1. Execute **01_configurar_ambiente.bat**. Cria `.venv` e instala NumPy. Requer Python 3.10 ou superior com Tcl/Tk funcional. Detecta e verifica Python instalado. Para um Python específico: `01_configurar_ambiente.bat -PythonExe "D:\Python\python.exe"`. Se não houver Python compatível, use `-InstalarPython` para instalar o Python oficial 3.13.16 na subpasta `.python`, na unidade D:, sem depender de winget. O parâmetro `-PythonVersion` permite escolher outra versão publicada em python.org.
2. Execute **04_configurar_ollama.bat**. Instala a CLI oficial em `D:\Ollama`, inicia o servidor local, baixa BGE-M3 e EmbeddingGemma QAT Q4 e cadastra seus perfis. Os modelos ficam em `D:\Ollama\models` quando o servidor é iniciado por este projeto. Para instalar somente o leve: `04_configurar_ollama.bat -SomenteLeve`.
3. Execute **02_abrir_aplicacao.bat**. Na aba **Base e geração**, selecione perfil e categorias, simule a seleção e gere as categorias escolhidas. Nada é gerado automaticamente ao importar ou instalar modelos.
4. Na aba **Recuperação e comparação**, selecione pares de modelo/categoria com Ctrl ou Shift, digite a consulta e escolha os parâmetros de recuperação.

O índice anterior já foi importado: 12 documentos, 306 tópicos, 282 ativos e 282 embeddings de `text-embedding-3-small`, na categoria `topico_total_sem_palavras_chave`. Os arquivos do projeto original permanecem na pasta anterior. Não houve chamadas pagas durante a migração. A chave existente não foi copiada: há campos para preenchimento direto em `embedding_credentials.py`.

O arquivo `ENTREGA.md` registra quais instalações e verificações foram efetivamente executadas nesta máquina. Os scripts de configuração são repetíveis; baixar os modelos pode levar alguns minutos. A CLI portátil do Ollama é utilizada para manter seus binários na unidade D:. Se outro servidor Ollama já estiver ativo, os scripts o reutilizam, com suas configurações existentes.

## Oito categorias independentes

| Categoria | Texto embeddado |
|---|---|
| `topico_total_sem_palavras_chave` | Composição original: título, caminho, resumo, conteúdo-fonte, perguntas, resposta e base normativa |
| `topico_total_com_palavras_chave` | Composição anterior acrescida das palavras-chave |
| `resumo` | Apenas o resumo |
| `conteudo_fonte` | Apenas o conteúdo-fonte |
| `perguntas_sugeridas` | Todas as perguntas do tópico, separadas por quebras de linha |
| `resposta_sintese` | Apenas a resposta-síntese |
| `palavras_chave` | Apenas as palavras-chave |
| `metadados` | Identificação documental, nome, eixos, idioma e vigência; sem os campos de conteúdo |

Cada categoria é gerada e atualizada separadamente, para cada perfil de modelo e conjunto de tópicos/documentos. Todas suas unidades apontam para o mesmo tópico. Campos vazios não geram vetores. Metadados iguais são calculados uma vez no lote e associados aos tópicos correspondentes.

Textos que excedem o orçamento conservador de entrada são segmentados sem descartar caracteres. Há, portanto, oito **categorias**; um tópico longo pode ter mais de oito unidades vetoriais. Manifestos registram segmento, offsets e preparação. E5 usa `query:`/`passage:`; EmbeddingGemma usa os prefixos recomendados de consulta e documento. Ollama recebe `truncate=false`. O limite conservador por bytes UTF-8 favorece a integridade do conteúdo e pode produzir mais segmentos que um limite calculado por tokens.

## Modelos

| Opção | Dimensão efetiva | Execução | Observação |
|---|---:|---|---|
| BGE-M3 | 1024 | Ollama local | Referência multilíngue, pesos de aproximadamente 1,2 GB |
| EmbeddingGemma QAT Q4 | 256 | Ollama local | Aproximadamente 239 MB; nativa 768, redução Matryoshka explícita para 256 |
| multilingual-e5-small | 384 | Sentence Transformers local | Dimensão nativa; instalação opcional pelo script 05 |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | Sentence Transformers local | Alternativa para campos curtos; instale seus arquivos no cache ou pasta configurada |
| Gemini Embedding 2 | 768 configurável | Online | Camada gratuita sujeita à disponibilidade da conta, região e cotas |
| text-embedding-3-small | 1536 configurável | Online | Compatibilidade com a base anterior; serviço pago |

Os modelos multilíngues são candidatos para português. O desempenho na sua base deve ser avaliado por experimentos; redução de dimensão e quantização podem alterar os resultados. Descoberta também lista outros modelos Ollama instalados cuja capacidade de embeddings é confirmada. A CLI permite cadastrar modelos locais por nome/caminho.

Um perfil distingue provedor, modelo, digest/versão dos pesos, dimensão, normalização e preparação. Atualizar os pesos do Ollama exige cadastrar um novo perfil; índices antigos continuam acessíveis. Serviços online sem digest registram o nome público do modelo. Dimensão menor, isoladamente, não garante menor custo de inferência.

Fontes: [Ollama Windows](https://docs.ollama.com/windows), [BGE-M3](https://ollama.com/library/bge-m3), [EmbeddingGemma e quantizações](https://ollama.com/library/embeddinggemma/tags), [Matryoshka e prefixos do EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma), [E5-small](https://huggingface.co/intfloat/multilingual-e5-small), [MiniLM multilíngue](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings), [cotas e preços do Gemini](https://ai.google.dev/gemini-api/docs/pricing).

## Atualizações e armazenamento

- `dados/base_etica.json`: conteúdo e metadados dos vetores; nunca coordenadas. `dados/projeto.json`: perfis cadastrados.
- `indices/<perfil>/<categoria>/<revisao>/vetores.npz`: coordenadas, originais efetivos quando disponíveis e IDs. `manifesto.json`: hashes, textos dos segmentos, IDs e referências às linhas. `atual.json`: ponteiro para a revisão publicada.
- `estudos/entradas/<entrada>/entrada.json`: texto original exato, hashes, gerações por perfil e execuções. Os vetores de consulta e seus segmentos ficam em NPZ.
- `estudos/execucoes/<execucao>/resultado.json`: parâmetros, todos os escores, resultados, textos recuperados, distâncias e referências a vetores/índices utilizados. `escores.npz`: arrays de escores do universo completo.
- `estudos/conjuntos`: faixas de escore com membros e referências. `estudos/derivados` e `estudos/analises`: resultados numéricos NPZ e proveniência JSON.
- `historico/bases`: versões anteriores da base de conteúdo.

As revisões publicadas são imutáveis e retidas. Só uma geração selecionada substitui o ponteiro corrente da categoria. Uma nova base marca unidades desatualizadas por hash, sem excluí-las nem regenerá-las. Importar um arquivo somente de metadados mantém a estrutura dos documentos já conhecidos. Vetores de tópicos inativos, removidos ou desatualizados ficam fora de novas buscas; experimentos anteriores continuam apontando para suas revisões originais.

Operações de geração: **faltantes** gera unidades ausentes; **atualizar** gera ausentes e textos modificados; **regenerar** refaz os tópicos selecionados mesmo com texto igual. IDs de tópicos/documentos restringem o alcance. Cada categoria publica independentemente; uma falha posterior não desfaz categorias anteriores concluídas. Checkpoints guardam lotes concluídos para retomada da mesma operação/revisão. O índice anterior permanece utilizável se a geração falhar.

Vetores armazenados são normalizados L2. Originais preservados representam a dimensão **efetiva** do perfil, após eventual redução Matryoshka. Vetores antigos já normalizados não permitem reconstruir seus originais; `original_disponivel=false` registra essa limitação. Os metadados dos resultados usam referências aos vetores recuperados, jamais listas de coordenadas.

## Busca e experimentos

Selecione qualquer combinação de categorias disponíveis para cada modelo. Palavras-chave e tópico com palavras-chave exigem seleção explícita. Comparações entre modelos mantêm rankings separados: vetores de espaços diferentes não são somados nem suas similaridades fundidas em uma média global.

Na visão **vetores**, cada unidade tem seu escore; top-k e intervalo de escore são aplicados por modelo/categoria. Na visão **tópicos**, todas as unidades das categorias selecionadas contribuem para a média aritmética; filtros e top-k são aplicados depois dessa média. Se houver segmentação, cada segmento conta como uma unidade. O detalhe do resultado mostra cada similaridade. Cobertura completa é exigida por padrão; habilite cobertura parcial para aceitar tópicos sem alguma categoria e mostrar quais estão ausentes.

Top-k padrão: 5. Escore mínimo/máximo são opcionais, inclusivos, no intervalo [-1,1]. **Sem top-k** retorna todos os candidatos aprovados pelos filtros. Mínimos individuais por categoria e regra `todos` ou `qualquer` permitem exigir escores individuais além do filtro sobre a média. Não há parada antecipada: o universo elegível é avaliado integralmente, e então os critérios limitam os resultados.

Distância cosseno = `1 - similaridade`. Distância euclidiana dos vetores normalizados = `sqrt(2 - 2*similaridade)`. A distância euclidiana média é a média das distâncias individuais, não a transformação da similaridade média. Uma consulta longa usa a média dos vetores originais dos segmentos, seguida de normalização; seus segmentos também ficam guardados.

Entradas iguais reutilizam o vetor do mesmo perfil; a CLI oferece regeneração explícita. Consultas sem resultados também são salvas. Na aba **Experimentos**, reabra resultados, veja todos os escores, crie faixas por vetor/categoria ou pela média do tópico e exporte JSON/NPZ em ZIP. Faixas usam mínimo inclusivo e máximo exclusivo, exceto a última, que inclui ambos. Reordenar usa os controles da aba Recuperação e apenas categorias já avaliadas na execução, sem nova inferência.

## Terminal e operações analíticas

Abra `03_abrir_terminal.bat`. Exemplos (troque `ID` pelo perfil do inventário):

```bat
python autorag_cli.py modelos
python autorag_cli.py inventario
python autorag_cli.py cadastrar --provedor ollama --modelo bge-m3 --dimensoes 1024
python autorag_cli.py gerar --perfil ID --categorias resumo metadados --simular
python autorag_cli.py gerar --perfil ID --categorias resumo conteudo_fonte --operacao atualizar
python autorag_cli.py buscar --perfil ID --categorias resumo palavras_chave --consulta "Consentimento em pesquisas" --top-k 10 --minimo 0.4
python autorag_cli.py buscar --perfil ID --categorias resumo --consulta "Consentimento" --visao vetores --sem-top-k
python autorag_cli.py faixas EXECUCAO --limites -1 0.4 0.7 1
python autorag_cli.py reordenar EXECUCAO --categorias resumo --visao vetores --sem-top-k
python autorag_cli.py exportar EXECUCAO --destino "D:\estudo.zip"
python autorag_cli.py analisar EXECUCAO --operacao centroide --categorias resumo
python autorag_cli.py analisar EXECUCAO --operacao cluster --metodo kmeans --parametros "{\"n_clusters\":5}"
python autorag_cli.py validar
```

`--arquivo-consulta` lê o texto exato de um arquivo UTF-8. `--projeto` é uma opção global e deve vir antes do comando. `gerar_embeddings.py` e `buscar_topicos.py` são atalhos para os comandos correspondentes. Para passar JSON com facilidade no PowerShell, use aspas simples: `--parametros '{"n_clusters":5}'`.

Instale `07_instalar_analise.bat` para clusterização. Métodos: KMeans, MiniBatchKMeans, aglomerativo, DBSCAN, HDBSCAN e espectral; os parâmetros são explícitos e ficam registrados. A interface opera sobre o universo da execução filtrado pelo intervalo de escore configurado na aba Recuperação; o top-k não restringe o conjunto analítico. A API `autorag.analysis` oferece carregamento por referências, comparação, centroides, soma, diferença, combinação linear, matrizes entre pares e clusterização. Operações exigem o mesmo perfil, registram proveniência e não alteram índices originais. Matriz entre pares tem limite padrão de 5.000 unidades devido ao custo quadrático de memória.

## Extração de novos tópicos

O extrator original foi preservado em `pipeline_etica.py`, com credenciais centralizadas e geração legada de embeddings delegada ao novo armazenamento NPZ. Use o wrapper para publicar atualizações:

```bat
python extrair_topicos.py --metadados "D:\planilha.xlsx"
python extrair_topicos.py --metadados "D:\planilha.xlsx" --somente-metadados
```

Instale antes `08_instalar_extracao.bat` e configure a chave OpenAI. A extração usa o provedor original e pode ter custo. Sua saída intermediária fica em `dados/base_extracao.json`; índices de busca são preservados e ficam desatualizados apenas quando o texto da categoria muda. Não é necessário instalar essas dependências para explorar a base já importada. A interface desktop atual recebe bases JSON; o fluxo de extração continua disponível pelo terminal.

## Chaves e configuração

`embedding_credentials.py` mantém campos diretos `OPENAI_API_KEY_DIRETA` e `GEMINI_API_KEY_DIRETA`. Também aceita variáveis de ambiente, `.env.local` e o ponto de extensão `RESOLVEDOR_SEGURO`. A ordem é: resolvedor seguro, ambiente, arquivo local e campo direto. Ollama e modelos Sentence Transformers locais não usam chave.

## Verificação

`06_executar_testes.bat` executa testes offline com provedor simulado. `09_verificar_modelos_locais.bat` verifica dimensões e inferência real em português, além das oito categorias, busca, faixas, reordenação e exportação numa base sintética temporária. Não gera embeddings da base do usuário. `autorag_cli.py validar` verifica hashes, IDs, matrizes e artefatos dos experimentos salvos. Falhas da interface ficam em `logs/ultimo_erro.log`.

Se um processo for encerrado à força, pode restar `dados/.escrita.lock`: confirme que não há processamento ativo antes de removê-lo. A aplicação permite uma operação de escrita por vez. Fechar a janela durante processamento é bloqueado para permitir a conclusão do lote.
