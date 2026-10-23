# Variação dos embeddings por grupo de recuperação

Análise de 23/10/2026. EmbeddingGemma 256d. Cada grupo contém os seis recuperados de uma entrada. O vetor de entrada permanece separado.

## Resultados por categoria

| Base | Categoria | Consultas | Cosseno entre recuperados | Cosseno entrada–recuperados | CV mín–máx mediano (%) |
|---|---|---:|---:|---:|---:|
| base_completa | Palavras-chave | 282 | 0.6145 | 0.6019 | 21.48 |
| base_completa | Perguntas | 282 | 0.6487 | 0.6441 | 20.55 |
| base_completa | Respostas-síntese | 282 | 0.6692 | 0.6754 | 19.97 |
| base_completa | Resumos | 282 | 0.6625 | 0.6664 | 20.10 |
| itens_individuais | Palavras-chave | 113 | 0.5482 | 0.4956 | 23.29 |
| itens_individuais | Perguntas | 59 | 0.5791 | 0.5772 | 22.68 |

## Método e leitura

- Uma única transformação mín–máx por coordenada foi ajustada ao conjunto de 300 vetores do corpus, 1.128 entradas conjuntas e 172 entradas individuais. Os mesmos limites são usados nas duas bases e em todos os grupos. As fontes originais foram preservadas.
- DP principal populacional (ddof=0), descrevendo os seis recuperados observados. DP amostral (ddof=1) também é exportado. O CV principal é 100 × DP populacional / |média|. A versão clássica amostral com sinal também consta na tabela original.
- Média de módulo ≤ 1e-12 torna o CV indefinido. Colunas constantes na referência seriam transformadas em zero e sinalizadas. Não há recorte ou renormalização L2 após mín–máx.
- Mín–máx é normalização por amplitude, não padronização por escore z. DP e CV são calculados entre vetores na mesma coordenada, nunca entre as 256 coordenadas de um único vetor.
- A soma da variância intragrupo média com a variância das médias dos grupos reproduz a variância total por coordenada. A decomposição é descritiva: os grupos podem compartilhar recuperados e textos.
- CV em embeddings assinados é sensível a cancelamento e à média próxima de zero. O mín–máx torna os valores não negativos, mas o CV continua dependente da origem definida pelos extremos. CV baixo em uma coordenada não demonstra proximidade semântica do vetor completo.
- Cosseno original mede a geometria do modelo. Cosseno mín–máx responde à geometria transformada e tende a aumentar por os valores serem não negativos. As duas matrizes têm rótulos distintos.
- Matrizes de entradas: N × N, entre textos/vetores completos. Matrizes de coordenadas: 256 × 256, entre colunas das entradas ao longo das N consultas. Esta última não é uma matriz de correlação de Pearson.
- Ocorrências repetidas permanecem identificadas por id_consulta. O resumo adicional com peso igual por texto reduz o peso de textos repetidos. A base individual é uma seleção anterior por extremos espectrais e não representa todo o corpus.
- Não foram aplicados testes de significância, nem interpretados os 256 componentes como observações independentes. A transformação usa todas as entradas porque esta etapa é descritiva. Em modelagem futura, ajustar os extremos apenas no treino.

## Arquivos

- `index.html`: relatório navegável, com busca de entradas, comparação de DP/CV e coordenadas de menor e maior variação.
- `parametros_minmax.csv`: mínimos, máximos e amplitudes, nas 256 coordenadas.
- `resumo_categorias.csv` e `resumo_todos_grupos.csv`: tabelas resumidas.
- `coordenadas_entre_grupos.csv`: estatísticas das médias e decomposição de variância, por categoria e escala.
- Em cada subpasta: `coordenadas_intragrupos.csv`, `resumo_grupos.csv`, `pares_intragrupos.csv`, quatro matrizes de cosseno em CSV, `entradas_cossenos_extremos.csv` e arquivos NPZ completos.
- CSV em UTF-8 com BOM, separador ponto e vírgula e vírgula decimal. Índices de coordenada de 1 a 256. A precisão de exibição do HTML é reduzida; as tabelas mantêm 12 dígitos significativos e os NPZ mantêm float64.

## Validação

Erros máximos das identidades e conferências: `{"max_erro_cosseno_fontes": {"base_completa": 4.440892098500626e-16, "itens_individuais": 3.3306690738754696e-16}, "max_erro_decomposicao_variancia": 6.106226635438361e-16, "max_erro_rmse_decomposicao": 2.0816681711721685e-17, "max_erro_identidade_cosseno": 4.440892098500626e-16}`.

Fontes metodológicas: [NIST — coeficiente de variação](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/coefvari.htm) e [Sentence Transformers — similaridade](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html).