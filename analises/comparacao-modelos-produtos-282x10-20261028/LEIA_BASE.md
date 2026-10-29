# Base min–max e comparação preditiva — 28/10/2026

Esta análise usa os 2.820 pares dos 282 tópicos com as dez entradas e os mesmos rótulos temáticos Luna 6: 567 relacionados e 2.253 não relacionados. A fonte preservada é `../base_produtos_originais_282x10_20261028/base_produtos_originais.csv`.

## Colunas dos CSVs de dados

`base_minmax.csv`, `treino_minmax.csv` e `teste_minmax.csv` têm 258 colunas, na mesma ordem:

| Posição | Nome | Significado |
|---|---|---|
| 1 | `classificacao_luna6` | Alvo binário original: 1 relacionado ao tema da entrada; 0 não relacionado. Não foi transformado. |
| 2 | `distancia_cosseno` | Distância cosseno original, `1 − similaridade_cosseno`, transformada pelo min–max do treino. |
| 3–258 | `produto_dim_001` a `produto_dim_256` | Produtos entre coordenadas correspondentes da entrada e do tópico, transformados separadamente por coluna pelo min–max do treino. |

Primeiro foi calculado o produto com os embeddings originais: `produto_dim_j = vetor_entrada_j × vetor_topico_j`. Depois cada coluna de produtos foi transformada; não foram multiplicados vetores previamente normalizados por min–max.

Para uma coluna preditora original `x_j`, a transformação é:

`x_minmax_j = (x_j − minimo_treino_j) / (maximo_treino_j − minimo_treino_j)`.

Os parâmetros das 257 colunas foram estimados somente nas 2.250 linhas de treino externo. Os mesmos parâmetros foram aplicados às 570 linhas de teste, sem truncamento. Por isso, **valores de teste podem ficar abaixo de 0 ou acima de 1**: são 241 células de preditores fora de `[0,1]` nesta divisão. Isso preserva o teste fora do cálculo dos mínimos e máximos. Todas as colunas preditoras do treino atingem mínimo 0 e máximo 1.

## Divisão treino e teste

| Partição | Pares | Tópicos | Entradas | Relacionados | Não relacionados |
|---|---:|---:|---:|---:|---:|
| Treino | 2.250 | 225 | 10 | 450 | 1.800 |
| Teste | 570 | 57 | 10 | 117 | 453 |
| Total | 2.820 | 282 | 10 | 567 | 2.253 |

A divisão é aproximadamente 80%/20%, usando `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261028)`, com a primeira parte reservada para teste. **Os dez pares de cada tópico ficam juntos**; nenhum tópico aparece simultaneamente no treino e no teste.

Cada entrada tem 225 linhas de treino e 57 de teste. Isso atende à exigência de pelo menos uma linha de cada entrada no teste. O resultado mede a classificação de tópicos novos para as mesmas dez entradas; não mede generalização para entradas inéditas.

## Identificadores e ordem das linhas

Os três CSVs min–max contêm somente o alvo e os 257 preditores. Os identificadores estão em `divisao_base.csv`:

| Coluna | Uso |
|---|---|
| `indice_par` | Índice do par na base completa, começando em zero. A linha humana correspondente no CSV completo é `indice_par + 2`, contando o cabeçalho. |
| `par_id` | Identificador único do par entrada–tópico. |
| `entrada_id` | Identificador da entrada, de `nova_01` a `nova_10`. |
| `topico_id` | Identificador do tópico usado para manter grupos inteiros na divisão. |
| `y` | Mesmo rótulo de `classificacao_luna6`. |
| `particao` | `treino` ou `teste`. |
| `fold_validacao_interno` | Parte de validação cruzada interna, de 0 a 3, nas linhas de treino; vazio nas linhas de teste. |

`divisao_base.csv` e `base_minmax.csv` preservam a ordem completa original. **Filtrar `divisao_base.csv` por `particao=treino` ou `particao=teste`, preservando a ordem, produz os metadados na mesma ordem de `treino_minmax.csv` ou `teste_minmax.csv`.** Esses IDs, a partição e o número da parte interna são metadados, não preditores dos modelos.

`divisao.json` registra a distribuição, a semente e as assinaturas dos arquivos de origem. `parametros_minmax_treino.json` e `minmax_treino.joblib` guardam a transformação ajustada no treino externo.

## Seleção e ajuste dos três modelos

A seleção usa quatro partes internas de validação cruzada, também agrupadas por tópico e com semente 20261029. Em cada ajuste interno, o min–max, a seleção de variáveis e o modelo são refeitos somente com o treino daquela parte. **As bases CSV já escaladas pelo treino externo não foram usadas para ajustar a busca interna.** A busca recebe os preditores originais e estima o pré-processamento dentro de cada ajuste.

O critério principal é o menor log loss médio nas quatro partes. **O teste foi mantido fora de toda seleção e ajuste.** O limiar de classificação dos três modelos é fixo em 0,5. O relatório apresenta somente os três modelos finais congelados.

- **Logística:** ranking univariado pela maior entre AUC e `1 − AUC`, calculado no treino de cada ajuste; seleção entre 8, 16, 32, 64, 128 ou 257 efeitos principais; graus 1, 2 e 3; regularização L2 com `C` igual a 0,1, 1 ou 10. Duplas e triplas são combinações de variáveis distintas entre as seis primeiras do ranking, com seus termos componentes preservados. Não são incluídas potências da mesma variável. Os termos do desenho recebem um segundo min–max antes da regularização. O grau efetivamente selecionado fica registrado em `resultado_comparacao.json` e no relatório: grau 1 mantém somente efeitos principais; graus 2 e 3 mantêm também as interações correspondentes. Não são calculados p-valores.
- **Random forest:** 350 árvores; seleção de número mínimo por folha, quantidade de variáveis candidatas a cada divisão e profundidade máxima. O min–max faz parte da sequência de transformação e previsão.
- **Rede neural:** perceptron multicamadas com ativação ReLU e otimizador Adam; seleção entre uma camada de 32 neurônios e duas camadas de 32 e 16, além de três níveis de regularização. O min–max faz parte da sequência de transformação e previsão. O ajuste não usa o teste para interromper ou escolher o treinamento.

As configurações e os resultados de cada busca ficam em `cv_logistica.csv`, `cv_random_forest.csv` e `cv_rede_neural.csv`. `selecao_modelos.json` e `resultado_comparacao.json` registram os parâmetros selecionados, versões, caminhos e assinaturas dos modelos. A configuração escolhida minimiza log loss de validação; o maior valor de uma métrica isolada no teste não foi usado para refazer a escolha.

**Resultado da seleção logística:** entre 54 configurações, a menor perda média de validação foi **0,396774423959801**, com `k_main=257`, `degree=1` e `C=1`. Portanto, o modelo final manteve todos os 257 preditores principais. As alternativas com interações duplas e triplas foram testadas, mas não selecionadas pelo critério preditivo. Essa escolha foi concluída somente com a validação cruzada do treino, antes da integração e avaliação final do teste.

Na rede neural, a configuração final tem uma camada de 32 neurônios e `alpha=1`; o ajuste final executou **333 iterações, de um limite de 1.500**. Durante a busca, alguns outros candidatos atingiram o limite de 1.500 iterações. A menor perda média de validação da rede foi 0,3996298820653472. A floresta selecionada tem 350 árvores, `min_samples_leaf=3`, `max_features=0.3` e profundidade sem limite explícito; sua perda média de validação foi 0,4062164186771203. Esses parâmetros foram escolhidos no treino.

A auditoria independente aprovou a preparação, os modelos e os resultados finais. Ao refazer os quatro ajustes internos da configuração neural selecionada, a auditoria registrou 439, 310, 399 e 522 iterações, todos abaixo do limite de 1.500. Os registros estão em `auditoria_independente.json` e `auditoria_convergencia_rede_neural.json`.

## Como usar os modelos salvos

**Os modelos em `modelos/*.joblib` recebem os 257 preditores ORIGINAIS, antes do min–max, na ordem `distancia_cosseno`, `produto_dim_001`, ..., `produto_dim_256`.** Eles aplicam a transformação internamente. Não forneça a coluna do rótulo nem IDs.

**Não passe `base_minmax.csv`, `treino_minmax.csv` ou `teste_minmax.csv` diretamente aos modelos salvos: isso aplicaria o min–max duas vezes.** Esses CSVs servem para inspeção e para ajustes externos que assumam os valores já transformados. Para reproduzir os modelos salvos, use os 257 preditores da base original ou `X_original` em `dados_divididos.npz`.

`modelo_logistica.json` também recebe conceitualmente os preditores originais. Ele contém, nesta ordem, o min–max das colunas originais, os índices dos termos selecionados, o min–max do desenho, os coeficientes e o intercepto. A função `predict_proba_from_json` em `../modelo_logistica_preditiva_produtos.py` reproduz essa sequência.

`coeficientes_logistica.csv` contém coeficientes penalizados do desenho final. `coeficiente_sem_segundo_minmax` desfaz somente o segundo min–max, mantendo a escala min–max inicial das variáveis. `importancias_random_forest.csv` contém as importâncias por redução de impureza do modelo de floresta. `curva_treino_rede_neural.csv` registra a perda de treino por iteração.

## Matrizes em NPZ

`dados_divididos.npz` contém as matrizes na ordem completa de 2.820 pares:

- `X_original`: 257 preditores originais; `X_minmax`: os mesmos preditores transformados pelo treino externo; `y`: rótulos binários.
- `train_index`, `test_index`: índices das linhas de treino e teste nas matrizes completas.
- `topico_ids`, `entrada_ids`, `par_ids`: metadados alinhados às linhas; `feature_names`: nomes das 257 colunas preditoras.
- `fold_validacao_interno`: parte interna de validação para cada linha de treino; `−1` para as linhas de teste.
- `cv0_train`/`cv0_val` até `cv3_train`/`cv3_val`: índices relativos à matriz de treino externo, não à base completa. Use-os sobre `X_original[train_index]` e `y[train_index]`.
- `minmax_data_min`, `minmax_data_max`, `minmax_scale`, `minmax_min`: parâmetros do min–max externo. A forma equivalente é `X_minmax = X_original × minmax_scale + minmax_min`.

## Previsões, métricas e relatório

`predicoes_teste.csv` contém os 570 pares de teste, com `indice_par`, `par_id`, `entrada_id`, `topico_id`, `y` e as três probabilidades de relação temática: `prob_rede_neural`, `prob_logistica` e `prob_random_forest`.

`metricas_teste.csv` registra ROC-AUC, AP (precisão média), log loss, Brier, acurácia, acurácia balanceada, precisão, recall, F1 e as contagens de verdadeiros/falsos positivos/negativos. As métricas de classificação usam probabilidade maior ou igual a 0,5. Log loss e Brier menores são melhores; as demais métricas maiores são melhores.

`relatorio.html` apresenta a comparação, as curvas ROC e precisão–recall, matrizes de confusão, calibração e resultados por entrada. `metricas_teste_relatorio.csv` confere as métricas a partir das probabilidades congeladas; `metricas_por_entrada.csv` e `calibracao_faixas_teste.csv` detalham os grupos. O relatório não reajusta modelos. Uma entrada com somente uma classe não recebe ROC-AUC ou acurácia balanceada; AP e recall não são apresentados sem positivos.

Os rótulos representam relação temática avaliada pelo Luna, não julgamento humano de suficiência para responder. A distância cosseno e os produtos por dimensão possuem redundância algébrica para vetores unitários. A análise e os coeficientes após seleção são exploratórios.

Não foi produzida versão Excel.

## Matrizes por linha observada e valores preditivos

`matrizes_confusao_proporcoes_teste.png` apresenta as matrizes `[[VN, FP], [FN, VP]]` normalizadas **por linha da classificação observada Luna 6**. Cada célula mostra a contagem original dividida pelo total da sua linha e o percentual correspondente. O denominador é 453 na linha Luna 6 = 0 e 117 na linha Luna 6 = 1. Cada linha soma 100%; a cor representa essa proporção numa escala de 0 a 1. O limiar permanece em 0,5. O arquivo anterior `matrizes_confusao_teste.png`, com as contagens, foi preservado.

`valores_preditivos_prevalencias.csv` contém 33 registros: três modelos no conjunto global e três modelos para cada uma das dez entradas. As colunas `prevalencia`, `sensibilidade`, `especificidade`, `vpp_bayes`, `vpn_bayes`, `vpp_contagens` e `vpn_contagens` são proporções entre 0 e 1; o relatório as mostra em percentual. As contagens `tn`, `fp`, `fn`, `tp`, o tamanho e o identificador do grupo permitem reproduzir os cálculos.

A prevalência global de relação temática é `π = 117/570 = 20,5263%`. Cada entrada usa sua própria prevalência nos 57 pares de teste. Com `Se = VP/(VP+FN)` e `Sp = VN/(VN+FP)`, os cálculos são:

- `VPP = Se × π / [Se × π + (1 − Sp) × (1 − π)]`.
- `VPN = Sp × (1 − π) / [Sp × (1 − π) + (1 − Se) × π]`.

Para a prevalência do mesmo grupo, esses resultados coincidem, com tolerância de `1e-12`, com `VPP = VP/(VP+FP)` e `VPN = VN/(VN+FN)`. O relatório mostra também essas frações. Um VPP de 0% indica que houve previsões positivas, todas incorretas; VPP indefinido indica ausência de previsões positivas. Quando o denominador é zero, o CSV deixa o valor numérico vazio e registra `indefinido` em `vpp_status` ou `vpn_status`.

Esta atualização deriva das previsões finais congeladas. Preservou os modelos, os CSVs anteriores, as métricas originais e os gráficos ROC/PR e de calibração; atualizou somente a nova matriz, a seção VPP/VPN do relatório, o novo CSV e o manifesto de validação.
