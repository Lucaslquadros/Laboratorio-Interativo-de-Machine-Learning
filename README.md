# Laboratório Interativo de Machine Learning

Site interativo (Streamlit) para visualizar, na prática, como funcionam os
algoritmos de Machine Learning ensinados em aula — tanto o resultado visual
quanto o código Python que roda por trás de cada passo. Começou como um
laboratório só de Regressão Linear Simples e a meta agora é virar a
**consulta oficial de ML do semestre**: cobrir, um de cada vez, os 14
tópicos do syllabus da disciplina (ver "Roadmap" abaixo).

Antes de escolher o modelo, a aba **🧭 Passo 0** ensina a classificar o
problema (alvo rotulado ou não, contínuo ou categórico, binário ou
multiclasse) usando a taxonomia do professor, com um quiz interativo que
recomenda o Tipo de Tarefa certo -- ver seção "Passo 0" abaixo.

Hoje tem dois grandes "Tipos de Tarefa":

- **Regressão** (alvo contínuo): **Simples** (1 variável X), **Múltipla**
  (2+ variáveis X) e **Polinomial** (1 variável, curva de grau
  configurável), com um leaderboard que compara os três, mais uma aba de
  Diagnóstico dos Resíduos.
- **Classificação** (alvo categórico): **Regressão Logística**, com matriz
  de confusão, acurácia/precisão/revocação/F1 e curva ROC/AUC.

## O que tem dentro

- **Passo 0**: antes de treinar qualquer coisa, guia o aluno a classificar o
  problema (alvo rotulado ou não → contínuo ou categórico → binário ou
  multiclasse) com a taxonomia exata do professor e um quiz que recomenda o
  Tipo de Tarefa certo. Ver seção própria abaixo.
- **Teoria**: recapitulação de aprendizado supervisionado, regressão vs.
  classificação, a equação da reta (β₀, β₁), a extensão para regressão
  múltipla, R², MAE, MSE e como interpretar a correlação de Pearson — direto
  dos slides da aula.
- **Dados & Correlação**: visualização do dataset, estrutura, estatísticas
  descritivas e mapa de calor da matriz de correlação.
- **Treino / Teste**: divisão treino/teste com o `test_size` que você
  escolher (padrão 70/30, igual à aula).
- **Treinamento do Modelo**: o modelo do scikit-learn é treinado. No modo
  Simples, os coeficientes (β₀, β₁) são recalculados **na mão**, pela fórmula
  do slide, para provar que batem — a "caixa-preta" é aberta. No modo
  Múltipla, mostra o intercepto e uma tabela com um coeficiente por variável.
- **Avaliação**: R², MAE e MSE do modelo — com gráfico de dispersão + reta no
  modo Simples, ou gráfico Previsto vs. Real no modo Múltipla (não dá para
  desenhar uma reta em N dimensões).
- **Diagnóstico dos Resíduos**: Estudo de Adequação do Modelo -- gráfico de
  Resíduos vs. Previstos, teste de homocedasticidade (Goldfeld-Quandt),
  histograma dos resíduos e teste de normalidade (Shapiro-Wilk),
  independência dos resíduos (Ljung-Box + Durbin-Watson) e heatmap de
  colinearidade das variáveis X (modo Múltipla), todos com interpretação.
- **Previsão**: um input por variável X para você escolher novos valores e
  ver a previsão de y em tempo real.
- **Comparação**: leaderboard que treina automaticamente todos os modelos
  de regressão possíveis neste dataset (Simples, Múltipla, Polinomial) e
  ranqueia por R², sem afetar as outras abas. (Só existe no lado
  Regressão.)

No lado **Classificação**, as abas Treinamento/Avaliação/Previsão mudam de
conteúdo: coeficientes em log-odds + razão de chances, matriz de confusão
+ curva ROC/AUC, e classe prevista + probabilidade, respectivamente
(Diagnóstico dos Resíduos e Comparação não existem nesse lado, já que só
há 1 algoritmo de classificação por enquanto).

Em toda aba há um botão **"🔍 Ver o código Python"** que mostra o código-fonte
real (via `inspect.getsource`) da função que gerou aquele resultado — não é
um trecho decorativo, é exatamente o comando executado.

## Datasets incluídos

- `data/base_salarios.csv` — o mesmo usado no `exemplo1_regressaolinearsimples.py`
  (Anos de Experiência x Salário).
- `data/USA_Housing.csv` — o mesmo usado no `exemplo2_regressaolinearsimples.py`
  (várias colunas; escolha X pela correlação com Price).
- `data/distancia_consumo.csv` — os 10 carros do slide "Regressão Linear
  Simples" (Distância x Consumo). Bom para conferir se a conta bate com o
  professor: y = 0.5716 + 0.0663·X, R² = 0.8980, MSE = 0.2995.
- `data/saude_desenvolvimento.csv` — **Saúde & Desenvolvimento** (ODS 3, 7,
  8): 183 países, ano 2020, indicadores do World Bank Open Data. Expectativa
  de vida explicada por PIB per capita, gasto em saúde, acesso à
  eletricidade e emissões de CO2. Ótimo para Regressão Múltipla: R² sobe de
  ≈0.47 (melhor variável isolada) para ≈0.71 com as 4 variáveis.
- `data/mortalidade_infantil_desenvolvimento.csv` — **Mortalidade Infantil
  & Desenvolvimento** (ODS 3, 6, 8): 180 países, ano 2020, também do World
  Bank. Mortalidade de crianças menores de 5 anos explicada por PIB per
  capita, gasto em saúde, saneamento e água potável. Aqui saneamento sozinho
  já explica quase tudo (R²≈0.69) -- bom exemplo prático de colinearidade
  entre indicadores de desenvolvimento.
- `data/50_Startups.csv` — o mesmo usado no
  `exemplo1_regressaolinearmultipla.py` (50 startups: gastos em P&D,
  Administração e Marketing, e o Lucro resultante). Igual ao script de
  aula, a coluna de Estado do CSV original (variável categórica,
  codificada em dummies) foi removida da versão em `data/` para focar no
  que foi ensinado -- `R&D Spend` e `Marketing Spend` são as variáveis de
  maior correlação com `Profit` (R² ≈ 0.9431 com `test_size=0.3`,
  `random_state=0`).
- `data/diagnostico_cancer_mama.csv` — **Diagnóstico de Câncer de Mama**
  (ODS 3), único dataset do lado Classificação. Breast Cancer Wisconsin
  (Diagnostic) Data Set (UCI ML Repository / scikit-learn), 569 casos, 30
  medidas numéricas, alvo genuinamente categórico (Maligno/Benigno).
  Regressão Logística treina com acurácia estável entre 90% e 98%.
- `data/publicidade.csv` — o mesmo usado no `exemplo_teste_suposicao.py`
  (200 mercados: orçamento de publicidade em TV, Rádio e Jornal, e as
  Vendas resultantes). Bom para ver os testes de suposições da aba
  Diagnóstico em ação -- Jornal tem correlação fraca com Vendas (R² ≈
  0.8649 com as 3 variáveis, 0.8657 com o melhor subconjunto TV+Rádio,
  `test_size=0.3`, `random_state=0`).

Você também pode carregar qualquer outro `.csv` seu pelo painel lateral.

## Robustez (v2)

O núcleo de regressão vive em `core.py`, separado da interface, e é coberto
por testes automatizados:

- **Validação de dados** antes de treinar: linhas insuficientes, coluna sem
  variância, `test_size` que esvaziaria treino/teste, valores vazios (NaN)
  — tudo isso gera uma mensagem de erro clara na tela, não um traceback.
- **CSV tolerante**: aceita separador `,` ou `;` (comum em exportações do
  Excel em pt-BR), com erro amigável se o arquivo não puder ser lido.
- **Cache** (`st.cache_data`): trocar de aba não reprocessa o CSV do zero.
- **Testes** (`tests/test_core.py`, 43 casos): confere que o cálculo manual
  de β₀/β₁ bate com o scikit-learn, que os valores batem com o slide do
  professor, que todos os cenários de erro acima realmente disparam, e que
  a regressão múltipla (validação, treino, avaliação, previsão) funciona.

## Regressão Linear Múltipla (v3)

Além do modo Simples original, o painel lateral tem um seletor "Tipo de
Regressão":

- **Simples**: comportamento original, uma única variável X.
- **Múltipla**: escolha 2 ou mais variáveis X (`st.multiselect`); o modelo é
  treinado com `LinearRegression` sobre todas elas ao mesmo tempo.

As funções novas (`validar_dados_para_regressao_multipla`,
`treinar_modelo_regressao_multipla`, `prever_novo_valor_multiplo`) vivem em
`core.py`, ao lado das equivalentes do modo Simples, e reaproveitam a mesma
validação de linhas/`test_size` das duas.

## Datasets de ODS via API pública (v4)

`saude_desenvolvimento.csv` e `mortalidade_infantil_desenvolvimento.csv`
(ver seção "Datasets incluídos" acima) vêm da **API pública do World Bank
Open Data** (`https://api.worldbank.org/v2/`, sem autenticação), baixados e
montados pelo script `scripts/baixar_dados_ods.py`. Os CSVs ficam
versionados no repo (mesma abordagem dos outros datasets) para o app rodar
offline; rode o script de novo se quiser atualizar o ano de referência.

Um terceiro candidato -- CO2 per capita explicado por PIB, urbanização,
energia renovável, área de floresta e densidade populacional -- foi testado
e descartado: o R² variava demais (e às vezes ficava negativo) entre
diferentes `random_state` do split treino/teste, por causa de outliers de
países pequenos/petro-exportadores. Essa decisão está documentada em
`scripts/baixar_dados_ods.py` e em `ai-dlc/01-inception/INCEPTION.md`.

## Diagnóstico de Resíduos + Regressão Polinomial + Regressão Logística (v5)

Seguindo o syllabus da disciplina (Estudo de Adequação do Modelo, Regressão
Polinomial e Regressão Logística):

- **Diagnóstico dos Resíduos**: nova aba "🩺" usa `diagnosticar_residuos()`
  e `testar_normalidade_residuos()` (teste de Shapiro-Wilk, via `scipy`)
  para checar as suposições clássicas da regressão linear: resíduos
  aleatórios, centrados em 0 e aproximadamente normais. Disponível nos
  modos Simples, Múltipla e Polinomial.
- **Regressão Polinomial**: 3ª opção em "Tipo de Regressão", com um grau
  configurável (2 a 5). `treinar_modelo_regressao_polinomial()` usa um
  `Pipeline(PolynomialFeatures, LinearRegression)` -- ainda mínimos
  quadrados, só que sobre X elevado a potências, o que permite ajustar uma
  curva em vez de uma reta.
- **Leaderboard**: a aba Comparação agora treina Simples, Múltipla e
  Polinomial de uma vez e ranqueia por R², em vez de comparar só 2 modelos
  por vez.
- **Regressão Logística**: novo seletor "Tipo de Tarefa" (Regressão /
  Classificação) no topo da sidebar. No lado Classificação, o alvo (y) é
  categórico (2 classes) em vez de contínuo -- por isso as métricas mudam
  para acurácia, precisão, revocação, F1 e matriz de confusão, e a
  Avaliação ganha uma curva ROC com AUC. `treinar_modelo_regressao_logistica()`
  usa `sklearn.linear_model.LogisticRegression`; os coeficientes são
  interpretados em log-odds / razão de chances na aba Treinamento.

## Fluxo de ingestão de `aulas/` + Passo 0 (v7)

- **`aulas/`** é a caixa de entrada do Lucas: slides, datasets e scripts que
  o professor for passando ao longo do semestre, adicionados livremente
  (sem estrutura de subpastas obrigatória). A cada rodada de trabalho,
  arquivos novos são conferidos contra `data/`, `core.py`, `app.py` e o
  roadmap do syllabus -- o que já foi revisado (e o que foi feito com cada
  arquivo) fica registrado em `ai-dlc/aulas-log.md`, para não reanalisar
  tudo do zero toda vez.
- **Aba "🧭 Passo 0"**: primeira aba da interface (nos dois Tipos de
  Tarefa). Apresenta a taxonomia exata do professor (slide "Tipos de
  algoritmos em DM" de `Intro_AprendMaquina.pdf`: Supervisionado →
  Regressão/Classificação binária ou multiclasse; Não Supervisionado →
  Associação/Clusterização), unifica o vocabulário duplicado de X/y
  (atributos previsores = variáveis independentes; atributo-alvo = variável
  dependente) e destaca a pegadinha de que a Regressão Logística é
  classificação, apesar do nome. Termina com um quiz interativo
  (`classificar_tipo_problema()` em `core.py`) que recomenda qual Tipo de
  Tarefa usar, com base nas respostas do aluno.
- **Dataset novo**: `50_Startups.csv`, incorporado seguindo o script de
  aula (ver "Datasets incluídos" acima).

## Testes de suposições avançados + dataset Publicidade (v9)

Segunda rodada do fluxo de ingestão de `aulas/`: o script
`exemplo_teste_suposicao.py` (sobre as 6 suposições da regressão linear)
expôs 3 lacunas na aba Diagnóstico, que até então só cobria média dos
resíduos e normalidade:

- **Homocedasticidade**: teste de Goldfeld-Quandt (`statsmodels`),
  complementando o gráfico Resíduos vs. Previstos já existente.
- **Independência dos resíduos**: testes de Ljung-Box e Durbin-Watson
  lado a lado -- podem discordar entre si (cada um capta um tipo de
  autocorrelação), então os dois são mostrados juntos, com uma nota
  explicando a possível divergência.
- **Ausência de colinearidade**: heatmap de correlação entre as variáveis
  X escolhidas (modo Múltipla), duplicando o já existente em "Dados &
  Correlação" para reunir todas as suposições num só lugar.
- **Dataset novo**: `publicidade.csv` (ver "Datasets incluídos" acima).

`testar_homocedasticidade_residuos()` e `testar_independencia_residuos()`
vivem em `core.py`, ao lado de `testar_normalidade_residuos()`.

## Roadmap (v6 — mudança de visão)

O objetivo mudou de "laboratório de regressão" para **consulta oficial de
ML** do semestre: cobrir, um algoritmo por vez, os 14 tópicos do syllabus da
disciplina antes de cogitar algo fora do programa. Detalhes e decisões em
`ai-dlc/01-inception/INCEPTION.md`.

| Tópico | Status |
|--------|--------|
| 1. Introdução ao Aprendizado de Máquina | ✅ |
| 2. Regressão Linear Simples | ✅ |
| 3. Regressão Linear Múltipla | ✅ |
| 4. Estudo de Adequação do Modelo (resíduos) | ✅ |
| 5. Regressão Polinomial | ✅ |
| 6. Regressão Logística | ✅ |
| 7. Árvores de Decisão | ⬜ |
| 8. K-NN | ⬜ |
| 9. Random Forest | ⬜ |
| 10. Support Vector Machine (SVM) | ⬜ |
| 11. Compromisso Viés/Variância | ⬜ |
| 12. Seleção e Validação de Modelos (validação cruzada) | ⬜ |
| 13. Métricas de avaliação | 🟡 parcial (classificação já cobre acurácia/precisão/F1/matriz de confusão) |
| 14. Análise de Agrupamentos (Clustering) | ⬜ |

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

O navegador abre automaticamente em `http://localhost:8501`.

## Como rodar os testes

```bash
pytest
```

## Metodologia de desenvolvimento (AI-DLC)

Este projeto segue uma versão enxuta do **AI-DLC** (AI-Driven Development
Life Cycle, da AWS), adaptada para desenvolvimento solo. Veja a pasta
`ai-dlc/` — `CLAUDE.md` explica o fluxo, `01-inception/INCEPTION.md` tem o
objetivo e requisitos deste projeto, `02-construction/BOLTS.md` tem o
backlog de bolts (inclusive os que geraram a robustez da v2) e
`03-operations/OPERATIONS.md` tem instruções de uso e o changelog.

## Estrutura

```
app.py                      # interface (Streamlit) -- só UI
core.py                     # núcleo de regressão e classificação -- testável sem Streamlit
requirements.txt
data/
  base_salarios.csv
  USA_Housing.csv
  distancia_consumo.csv
  saude_desenvolvimento.csv
  mortalidade_infantil_desenvolvimento.csv
  diagnostico_cancer_mama.csv
  50_Startups.csv
  publicidade.csv
scripts/
  baixar_dados_ods.py            # baixa os 2 datasets de ODS do World Bank
  baixar_dados_classificacao.py  # monta o dataset de câncer de mama (scikit-learn)
tests/
  test_core.py               # pytest, 57 casos
aulas/                       # caixa de entrada: slides, datasets e scripts de aula
ai-dlc/                      # AI-DLC enxuto deste projeto (ver acima)
  CLAUDE.md
  README.md
  aulas-log.md                    # rastreio do que já foi revisado em aulas/
  01-inception/INCEPTION.md
  02-construction/BOLTS.md
  03-operations/OPERATIONS.md
README.md
```
