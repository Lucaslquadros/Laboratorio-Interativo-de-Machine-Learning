# Construction — Backlog de Bolts — Laboratório Interativo de Machine Learning

> Um **bolt** é uma unidade pequena de trabalho (minutos a poucas horas),
> com um objetivo único e verificável. Trabalhe **um bolt por vez**. Ao
> terminar um bolt, pare no checkpoint antes de seguir para o próximo.
>
> Status: `todo` → `em andamento` → `checkpoint` → `feito`

> **Sobre os bolts 1-29 abaixo:** cobrem v1 a v5 (Regressão Simples até
> Regressão Logística). A partir da v6, a visão do projeto se ampliou para
> cobrir os 14 tópicos do syllabus da disciplina (roadmap completo em
> `ai-dlc/01-inception/INCEPTION.md`). Os próximos algoritmos (Árvores de
> Decisão, KNN, Random Forest, SVM, Clustering, etc.) ainda **não** têm
> bolts quebrados aqui de propósito — cada um deve passar por um Inception
> próprio primeiro (perguntas de validação, decisões de escopo/arquitetura)
> antes de virar backlog de bolts, seguindo o mesmo fluxo já usado para
> Múltipla/Polinomial/Logística.

## Bolt 1 — Extrair o núcleo de regressão para `core.py`

- **Objetivo:** separar toda a lógica de regressão (carregar dados,
  correlação, split, treino, cálculo manual, avaliação, previsão) da
  interface Streamlit, para que vire código testável sem precisar subir o
  app inteiro.
- **Entregável:** `core.py` com as funções puras; `app.py` importa de lá.
- **Checkpoint:** `python -c "import core"` não executa nada de UI; app
  continua funcionando igual (mesmos números de antes).
- **Status:** feito

## Bolt 2 — Validação de dados antes de treinar

- **Objetivo:** checar, antes do `train_test_split`/`fit`, se há linhas
  suficientes, se X e y têm variância, e se não sobram linhas vazias
  (NaN) — e falhar com mensagem clara em vez de traceback.
- **Entregável:** `validar_dados_para_regressao()` em `core.py`, usada por
  `app.py` para mostrar `st.error()` amigável.
- **Checkpoint:** subir um CSV de 1 linha, ou com uma coluna constante,
  mostra mensagem de erro clara — o app não quebra.
- **Status:** feito

## Bolt 3 — Carregamento de CSV mais tolerante + cache

- **Objetivo:** aceitar CSV com `;` como separador (comum em export do
  Excel/BR), tratar erro de parsing com mensagem clara, e cachear a leitura
  (`st.cache_data`) para não reprocessar 10 mil linhas a cada clique.
- **Entregável:** `carregar_dataset()` atualizado em `core.py` +
  `@st.cache_data` no wrapper usado pelo `app.py`.
- **Checkpoint:** CSV com `;` carrega normalmente; medir que trocar de aba
  não reprocessa o CSV de 10k linhas do zero.
- **Status:** feito

## Bolt 4 — Suite de testes automatizados

- **Objetivo:** cobrir com `pytest` o essencial: β0/β1 manual bate com
  scikit-learn, métricas calculadas corretamente, e os casos de erro do
  Bolt 2 realmente disparam.
- **Entregável:** `tests/test_core.py`.
- **Checkpoint:** `pytest` roda e passa (`X passed, 0 failed`).
- **Status:** feito

## Bolt 5 — Núcleo de regressão múltipla em `core.py`

- **Objetivo:** adicionar `validar_dados_para_regressao_multipla`,
  `treinar_modelo_regressao_multipla` e `prever_novo_valor_multiplo`,
  reaproveitando a validação comum já usada pela regressão simples.
- **Entregável:** as três funções em `core.py`.
- **Checkpoint:** `python -c "import core"` sem erro; treino manual com 2-3
  colunas do `USA_Housing.csv` produz coeficientes plausíveis.
- **Status:** feito

## Bolt 6 — Testes automatizados do núcleo múltiplo

- **Objetivo:** cobrir com `pytest` a validação (erro com só 1 coluna X,
  linhas insuficientes) e o treino/avaliação/previsão da regressão múltipla.
- **Entregável:** casos novos em `tests/test_core.py`.
- **Checkpoint:** `pytest` passa, casos novos contam na suíte.
- **Status:** feito

## Bolt 7 — Sidebar: seletor Simples/Múltipla

- **Objetivo:** adicionar `st.sidebar.radio` de tipo de regressão e
  `st.multiselect` de X para o modo Múltipla, com branch no bloco de
  CÁLCULOS que treina o modelo certo e guarda um formato comum (`modelo`,
  `intercepto`, `coeficientes`) para as abas seguintes.
- **Entregável:** `app.py` atualizado.
- **Checkpoint:** `streamlit run app.py` sem traceback nos dois modos.
- **Status:** feito

## Bolt 8 — Aba "Treinamento do Modelo" com suporte a múltipla

- **Objetivo:** no modo Múltipla, mostrar tabela de coeficientes e equação
  dinâmica em vez de β0/β1 únicos; sem cálculo "na mão" nesse modo.
- **Entregável:** `app.py` — aba `aba_treino` com branch por modo.
- **Checkpoint:** aba renderiza nos dois modos, com 2 e com 5 colunas X.
- **Status:** feito

## Bolt 9 — Aba "Avaliação" com gráfico adaptado

- **Objetivo:** no modo Múltipla, trocar a reta de regressão por um gráfico
  Previsto vs. Real (scatter + diagonal de referência).
- **Entregável:** `app.py` — aba `aba_avaliacao` com branch por modo.
- **Checkpoint:** aba renderiza nos dois modos sem erro.
- **Status:** feito

## Bolt 10 — Aba "Previsão" com múltiplos inputs

- **Objetivo:** no modo Múltipla, um input por coluna X selecionada, usando
  `prever_novo_valor_multiplo`.
- **Entregável:** `app.py` — aba `aba_previsao` com branch por modo.
- **Checkpoint:** previsão funciona nos dois modos, valor plausível.
- **Status:** feito

## Bolt 11 — Nova aba "🆚 Comparação"

- **Objetivo:** treinar o modelo alternativo ao modo ativo (só para
  comparação) e mostrar tabela + gráfico de barras com R²/MAE/MSE dos dois.
- **Entregável:** nova aba em `app.py`.
- **Checkpoint:** aba funciona nos 3 datasets embutidos, sem traceback
  mesmo quando o modelo alternativo não é viável.
- **Status:** feito

## Bolt 12 — Teoria + Operations (fechamento)

- **Objetivo:** explicar Regressão Múltipla na aba Teoria; atualizar
  `README.md` e `OPERATIONS.md` (changelog v3, novos critérios de saúde);
  marcar bolts 5-12 como `feito` aqui.
- **Entregável:** `app.py` (aba Teoria), `README.md`, `OPERATIONS.md`,
  `BOLTS.md`.
- **Checkpoint:** `pytest` passa; app roda sem traceback nas 7 abas, nos
  dois modos, nos 3 datasets embutidos + 1 upload.
- **Status:** feito

## Bolt 13 — Script de coleta dos datasets de ODS

- **Objetivo:** buscar indicadores do World Bank Open Data (API pública,
  sem autenticação) e montar dois CSVs limpos (sem NaN) ligados aos ODS,
  bons para Regressão Múltipla.
- **Entregável:** `scripts/baixar_dados_ods.py` +
  `data/saude_desenvolvimento.csv` + `data/mortalidade_infantil_desenvolvimento.csv`.
- **Checkpoint:** rodar o script gera os dois CSVs sem NaN; R² da múltipla
  testado manualmente em vários `random_state` para garantir estabilidade
  (um terceiro candidato, com CO2 per capita, foi descartado nesse
  checkpoint por dar R² instável/negativo).
- **Status:** feito

## Bolt 14 — Integrar os datasets de ODS no app

- **Objetivo:** adicionar as duas novas entradas em `DATASETS` (`app.py`),
  com `y_padrao` e descrição explicando a fonte e os ODS envolvidos.
- **Entregável:** `app.py` atualizado.
- **Checkpoint:** os dois datasets aparecem no seletor da sidebar e
  carregam sem erro nos modos Simples e Múltipla.
- **Status:** feito

## Bolt 15 — Testes e documentação dos datasets de ODS

- **Objetivo:** testar os dois datasets no navegador (Simples, Múltipla,
  Previsão, Comparação) e documentar a decisão (INCEPTION.md, README.md,
  OPERATIONS.md).
- **Entregável:** documentação atualizada.
- **Checkpoint:** `pytest` passa; os dois datasets funcionam nas 7 abas
  sem traceback.
- **Status:** feito

## Bolt 16 — Núcleo de diagnóstico de resíduos em `core.py`

- **Objetivo:** `diagnosticar_residuos()` (real - previsto, média,
  desvio-padrão) e `testar_normalidade_residuos()` (Shapiro-Wilk, via
  `scipy.stats`) -- base do "Estudo de Adequação do Modelo" do syllabus.
- **Entregável:** as duas funções em `core.py`; `scipy` adicionado ao
  `requirements.txt`.
- **Checkpoint:** testado manualmente via console -- resíduos de um ajuste
  perfeito são 0; teste de normalidade roda numa amostra normal simulada e
  falha com mensagem clara com < 3 pontos.
- **Status:** feito

## Bolt 17 — Nova aba "Diagnóstico dos Resíduos"

- **Objetivo:** aba visível nos modos Simples e Múltipla (a Polinomial, da
  próxima fase, também vai usar) com gráfico Resíduos vs. Previstos,
  histograma dos resíduos, métricas e resultado interpretado do teste de
  Shapiro-Wilk.
- **Entregável:** nova aba em `app.py`, entre Avaliação e Previsão.
- **Checkpoint:** testado no navegador nos dois modos, incluindo o caso
  limite de só 3 resíduos no conjunto de teste (dataset Distância x
  Consumo) -- não quebra.
- **Status:** feito

## Bolt 18 — Testes automatizados do diagnóstico de resíduos

- **Objetivo:** cobrir com `pytest` o cálculo dos resíduos, o teste de
  normalidade (amostra normal simulada) e o encaixe direto na saída de
  `avaliar_modelo`.
- **Entregável:** 5 casos novos em `tests/test_core.py`.
- **Checkpoint:** `pytest` passa (33 casos no total).
- **Status:** feito

## Bolt 19 — Núcleo de regressão polinomial em `core.py`

- **Objetivo:** `treinar_modelo_regressao_polinomial(X, y, grau)` via
  `Pipeline(PolynomialFeatures, LinearRegression)` -- reaproveita
  `validar_dados_para_regressao` (1 variável, igual à Simples) e
  `avaliar_modelo`/`prever_novo_valor` (já genéricos, funcionam com
  `Pipeline` sem mudança nenhuma).
- **Entregável:** a função em `core.py`.
- **Checkpoint:** testado manualmente -- grau 1 bate exatamente com a
  Regressão Simples; R² melhora ou mantém com graus 2-5 em datasets
  maiores; `prever_novo_valor` funciona sem alteração.
- **Status:** feito

## Bolt 20 — Sidebar: 3ª opção "Polinomial" + grau configurável

- **Objetivo:** "Tipo de Regressão" ganha `MODO_POLINOMIAL`, reaproveitando
  o `selectbox` de 1 variável X da Simples; novo slider "Grau do polinômio"
  (2 a 5) quando esse modo está ativo.
- **Entregável:** `app.py` (sidebar + bloco de CÁLCULOS com 3 branches).
- **Checkpoint:** `streamlit run app.py` sem traceback alternando entre os
  3 modos.
- **Status:** feito

## Bolt 21 — Abas Treinamento/Avaliação/Previsão com suporte a Polinomial

- **Objetivo:** Treinamento mostra tabela de coeficientes por potência de X
  e a equação; Avaliação desenha a curva ajustada (via `np.linspace` sobre
  todo o intervalo de X, não só os pontos de teste); Previsão reaproveita a
  UI da Simples sem mudança (o `Pipeline` já faz a transformação
  polinomial internamente).
- **Entregável:** `app.py` -- 3 abas com branch por modo.
- **Checkpoint:** testado no navegador em graus 2 e 4, dataset com 5000
  linhas (USA_Housing) -- curva, tabela e previsão corretas.
- **Status:** feito

## Bolt 22 — Aba "Comparação" vira leaderboard

- **Objetivo:** em vez de comparar só o modo ativo com 1 alternativo,
  treina Simples (melhor X por correlação), Múltipla (todas as X
  numéricas) e Polinomial (grau 2, mesma X da Simples) sobre o mesmo split
  e ranqueia por R², com tratamento de erro independente por modelo (um
  modelo não viável não derruba os outros).
- **Entregável:** `app.py` -- aba `aba_comparacao` reescrita.
- **Checkpoint:** testado com dataset de 1 única coluna X candidata
  (Múltipla falha com mensagem clara, Simples e Polinomial funcionam) e com
  dataset de 5+ colunas (os 3 modelos aparecem ranqueados).
- **Status:** feito

## Bolt 23 — Testes automatizados da Regressão Polinomial

- **Objetivo:** cobrir com `pytest` que grau 1 bate com a Simples, que o
  número de coeficientes bate com o grau, e que o pipeline funciona com
  `avaliar_modelo`/`prever_novo_valor` sem alteração.
- **Entregável:** 3 casos novos em `tests/test_core.py`.
- **Checkpoint:** `pytest` passa (36 casos no total).
- **Status:** feito

## Bolt 24 — Dataset de classificação (Diagnóstico de Câncer de Mama)

- **Objetivo:** montar um dataset com alvo categórico **genuíno** (não
  binarizado por nós) para a Regressão Logística -- Breast Cancer
  Wisconsin (UCI ML Repository / scikit-learn), tema ODS 3.
- **Entregável:** `scripts/baixar_dados_classificacao.py` +
  `data/diagnostico_cancer_mama.csv` (569 casos, 30 variáveis, sem NaN).
- **Checkpoint:** testado antes de fixar -- acurácia da
  `LogisticRegression` estável entre 90% e 98% em vários `random_state`.
- **Status:** feito

## Bolt 25 — Núcleo de classificação em `core.py`

- **Objetivo:** `validar_dados_para_classificacao`,
  `treinar_modelo_regressao_logistica`, `avaliar_modelo_classificacao`
  (acurácia, precisão, revocação, F1, matriz de confusão) e
  `prever_classe_novo_valor`.
- **Entregável:** as quatro funções em `core.py`.
- **Checkpoint:** testado manualmente via console com o dataset do
  Bolt 24 -- acurácia ~90%, matriz de confusão coerente, erro tratado
  para coluna-alvo com número errado de categorias.
- **Status:** feito

## Bolt 26 — "Tipo de Tarefa" (Regressão / Classificação) em `app.py`

- **Objetivo:** novo controle no topo da sidebar que troca a lista de
  datasets, esconde "Tipo de Regressão" no lado Classificação, e adapta o
  conjunto de abas (Classificação não tem Diagnóstico de Resíduos nem
  Comparação -- só existe 1 algoritmo de classificação por enquanto).
- **Entregável:** `app.py` -- `tarefa`, `DATASETS_CLASSIFICACAO`, seleção
  condicional de `col_y`/`cols_x`, `st.tabs()` condicional.
- **Checkpoint:** testado no navegador -- alternar entre os dois Tipos de
  Tarefa sem traceback; lado Regressão continua idêntico ao de antes.
- **Status:** feito

## Bolt 27 — Abas Treinamento/Avaliação/Previsão para Classificação

- **Objetivo:** Treinamento mostra coeficientes em log-odds + razão de
  chances e a equação sigmoide; Avaliação mostra matriz de confusão
  (heatmap), acurácia/precisão/revocação/F1 e curva ROC com AUC; Previsão
  mostra a classe prevista + probabilidade da classe positiva.
- **Entregável:** `app.py` -- 3 abas com branch por `tarefa`.
- **Checkpoint:** testado no navegador com o dataset de câncer de mama --
  acurácia ~92%, matriz de confusão e curva ROC corretas, previsão
  plausível.
- **Status:** feito

## Bolt 28 — Testes automatizados da Regressão Logística

- **Objetivo:** cobrir com `pytest` a validação (alvo com número errado
  de categorias), o treino, a avaliação (métricas coerentes com a matriz
  de confusão) e a previsão de classe.
- **Entregável:** 7 casos novos em `tests/test_core.py`.
- **Checkpoint:** `pytest` passa (43 casos no total).
- **Status:** feito

## Bolt 29 — Teoria + documentação final da v5

- **Objetivo:** explicar Regressão Polinomial, Diagnóstico de Resíduos e
  Classificação/Regressão Logística na aba Teoria (pendências das Fases A
  e B); fechar `INCEPTION.md`, `OPERATIONS.md`, `README.md` para a v5.
- **Entregável:** `app.py` (aba Teoria), `README.md`, `OPERATIONS.md`,
  `INCEPTION.md`, `BOLTS.md`.
- **Checkpoint:** `pytest` passa; app roda sem traceback nos dois Tipos
  de Tarefa, testado manualmente no navegador.
- **Status:** feito

---

> **v7 (Fluxo de ingestão de `aulas/` + Passo 0):** decisões de Inception em
> `ai-dlc/01-inception/INCEPTION.md`. Diferente dos itens 7-14 do roadmap
> (que ainda esperam um Inception próprio), esta rodada já tem escopo
> fechado e validado com o Lucas — os bolts abaixo já podem ser executados.

## Bolt 30 — Manifesto de rastreio `ai-dlc/aulas-log.md`

- **Objetivo:** registrar quais arquivos de `aulas/` já foram revisados e o
  que foi feito com cada um, para rodadas futuras não reanalisarem tudo do
  zero quando o Lucas adicionar arquivos novos.
- **Entregável:** `ai-dlc/aulas-log.md` com tabela (arquivo, tipo, status,
  ação, data).
- **Checkpoint:** arquivo criado, cobre os 10 arquivos hoje presentes em
  `aulas/`.
- **Status:** feito

## Bolt 31 — Dataset `50_Startups.csv` integrado ao app

- **Objetivo:** copiar `aulas/50_Startups.csv` para `data/`, adicionar
  entrada em `DATASETS` (`app.py`) com `y_padrao="Profit"`, seguindo o
  script de aula (`exemplo1_regressaolinearmultipla.py`): X sugerido =
  `R&D Spend` + `Marketing Spend` (as colunas dummy de `State` ficam no CSV
  mas não são destacadas/sugeridas, por decisão de Inception v7).
- **Entregável:** `data/50_Startups.csv` + `app.py` atualizado.
- **Checkpoint:** dataset aparece no seletor da sidebar, carrega sem erro
  nos modos Simples e Múltipla.
- **Status:** feito

## Bolt 32 — Teste automatizado do dataset `50_Startups.csv`

- **Objetivo:** cobrir com `pytest` que a Múltipla com X=[`R&D Spend`,
  `Marketing Spend`], y=`Profit`, `test_size=0.3`, `random_state=0`
  reproduz R²≈0.9431, batendo com o script de aula.
- **Entregável:** caso novo em `tests/test_core.py`.
- **Checkpoint:** `pytest` passa.
- **Status:** feito

## Bolt 33 — Núcleo do Passo 0 em `core.py`

- **Objetivo:** função pura que recebe as respostas do quiz de classificação
  do problema (alvo rotulado? contínuo ou categórico? binário ou
  multiclasse?) e devolve a recomendação (qual família/Tipo de Tarefa usar)
  — testável sem subir o Streamlit, seguindo o padrão do resto de `core.py`.
- **Entregável:** função nova em `core.py` (ex.: `classificar_tipo_problema`).
- **Checkpoint:** testado manualmente via console com as combinações
  possíveis (rotulado+contínuo, rotulado+categórico binário,
  rotulado+categórico multiclasse, não rotulado).
- **Status:** feito

## Bolt 34 — Nova aba "🧭 Passo 0" em `app.py`

- **Objetivo:** primeira aba da lista (antes de Teoria), com a taxonomia do
  professor (`Intro_AprendMaquina.pdf`: Supervisionado/Não-supervisionado →
  Regressão/Classificação/Associação/Clusterização), o vocabulário X/y
  unificado (atributos previsores = variáveis independentes; atributo-alvo
  = variável dependente), a pegadinha da Regressão Logística ("é
  classificação, apesar do nome") como checagem de entendimento, e o quiz
  interativo usando a função do Bolt 33 para recomendar o Tipo de Tarefa.
- **Entregável:** `app.py` — nova aba.
- **Checkpoint:** testado com `streamlit.testing.v1.AppTest` — quiz responde
  corretamente às 4 combinações do Bolt 33; app roda sem traceback nos dois
  Tipos de Tarefa (9 abas em Regressão, 7 em Classificação); dataset novo
  seleciona `R&D Spend` + `Marketing Spend` por padrão no modo Múltipla.
- **Status:** feito

## Bolt 35 — Documentação de fechamento da v7

- **Objetivo:** atualizar `README.md` (nova aba Passo 0, novo dataset),
  `OPERATIONS.md` (changelog v7) e marcar bolts 30-35 como `feito` aqui.
- **Entregável:** `README.md`, `OPERATIONS.md`, `BOLTS.md`.
- **Checkpoint:** `pytest` passa; app roda sem traceback.
- **Status:** feito

---

> **v8 (Leaderboard: Múltipla usa o melhor subconjunto de variáveis):**
> decisão de Inception em `ai-dlc/01-inception/INCEPTION.md` -- o Lucas
> encontrou, via captura de tela do app rodando, que a linha "Múltipla" do
> leaderboard usava todas as colunas numéricas candidatas (comportamento do
> Bolt 22) e por isso ficava com R² pior do que a combinação de 2 variáveis
> já sugerida por padrão na aba Múltipla interativa. Escolhida a opção A
> (busca exaustiva pelo melhor subconjunto).

## Bolt 36 — Leaderboard: Múltipla busca o melhor subconjunto de variáveis

- **Objetivo:** trocar "Múltipla usa todas as colunas candidatas" por
  "Múltipla testa todas as combinações de 2..N colunas e usa a de maior R²
  de teste", para o leaderboard não mostrar uma Múltipla pior do que uma
  combinação menor e melhor (caso do dataset Startups: 3 variáveis dava
  R²=0.9355, 2 variáveis dá R²=0.9431).
- **Entregável:** `selecionar_melhor_subconjunto_multipla()` em `core.py`;
  `app.py` (bloco do leaderboard) passa a chamá-la em vez de treinar direto
  com `opcoes_x_ordenadas` inteiro, e a coluna "Detalhes" passa a listar os
  nomes das variáveis vencedoras (ex.: "R&D Spend + Marketing Spend") em
  vez de só a contagem ("3 variáveis").
- **Checkpoint:** `pytest` cobre que, no dataset Startups, a busca descarta
  `Administration` e reproduz R²≈0.9431 (batendo com a referência da aula);
  testado no navegador nos datasets Startups (3 candidatas), USA_Housing
  (4 candidatas) e confirmando que a aba Comparação carrega sem lentidão
  perceptível (busca exaustiva é 2^N, trivial para N≤5, que é o máximo
  hoje entre os datasets do app).
- **Status:** feito

## Bolt 37 — Documentação de fechamento da v8

- **Objetivo:** atualizar `OPERATIONS.md` (changelog v8) e marcar os bolts
  36-37 como `feito` aqui.
- **Entregável:** `OPERATIONS.md`, `BOLTS.md`.
- **Checkpoint:** `pytest` passa; leaderboard testado no navegador.
- **Status:** feito

---

## Registro de bolts concluídos

- **2026-08-18** — Bolt 1: lógica extraída para `core.py`, `app.py` só cuida
  de UI.
- **2026-08-18** — Bolt 2: validação de dados com mensagens de erro
  amigáveis antes de treinar.
- **2026-08-18** — Bolt 3: CSV com `;`, encoding e cache de leitura.
- **2026-08-18** — Bolt 4: `tests/test_core.py` com pytest, cobrindo os
  casos de sucesso e de erro.
- **2026-08-17** — Bolt 5: núcleo de regressão múltipla em `core.py`
  (validação, treino, previsão).
- **2026-08-17** — Bolt 6: suíte `pytest` ampliada para 28 casos, cobrindo
  regressão múltipla.
- **2026-08-17** — Bolt 7: seletor Simples/Múltipla no painel lateral.
- **2026-08-17** — Bolt 8: aba Treinamento com tabela de coeficientes e
  equação dinâmica no modo Múltipla.
- **2026-08-17** — Bolt 9: aba Avaliação com gráfico Previsto vs. Real no
  modo Múltipla.
- **2026-08-17** — Bolt 10: aba Previsão com um input por variável no modo
  Múltipla.
- **2026-08-17** — Bolt 11: nova aba "🆚 Comparação" (Simples vs. Múltipla).
- **2026-08-17** — Bolt 12: Teoria, README.md e OPERATIONS.md atualizados
  para a v3 (Regressão Linear Múltipla).
- **2026-08-17** — Bolt 13: `scripts/baixar_dados_ods.py` + 2 novos CSVs
  de ODS via API do World Bank.
- **2026-08-17** — Bolt 14: datasets de ODS integrados em `DATASETS`
  (`app.py`).
- **2026-08-17** — Bolt 15: testes no navegador + documentação da v4.
- **2026-08-17** — Bolt 16: `diagnosticar_residuos` e
  `testar_normalidade_residuos` em `core.py`.
- **2026-08-17** — Bolt 17: nova aba "Diagnóstico dos Resíduos" em `app.py`.
- **2026-08-17** — Bolt 18: testes automatizados do diagnóstico de
  resíduos (33 casos no total).
- **2026-08-18** — Bolt 19: `treinar_modelo_regressao_polinomial` em
  `core.py` (Pipeline PolynomialFeatures + LinearRegression).
- **2026-08-18** — Bolt 20: 3ª opção "Polinomial" na sidebar + grau
  configurável.
- **2026-08-18** — Bolt 21: abas Treinamento/Avaliação/Previsão com
  suporte a Polinomial.
- **2026-08-18** — Bolt 22: aba "Comparação" reescrita como leaderboard
  de todos os modelos de regressão.
- **2026-08-18** — Bolt 23: testes automatizados da Regressão Polinomial
  (36 casos no total).
- **2026-08-18** — Bolt 24: dataset de classificação (Breast Cancer
  Wisconsin, `diagnostico_cancer_mama.csv`).
- **2026-08-18** — Bolt 25: núcleo de classificação (validação, treino,
  avaliação, previsão) em `core.py`.
- **2026-08-18** — Bolt 26: "Tipo de Tarefa" (Regressão/Classificação) na
  sidebar de `app.py`.
- **2026-08-18** — Bolt 27: abas Treinamento/Avaliação/Previsão adaptadas
  para Classificação (log-odds, matriz de confusão, curva ROC).
- **2026-08-18** — Bolt 28: testes automatizados da Regressão Logística
  (43 casos no total).
- **2026-08-18** — Bolt 29: Teoria (Polinomial, Resíduos, Classificação) +
  documentação final da v5.
- **2026-08-19** — Bolt 30: `ai-dlc/aulas-log.md` criado, cobrindo os 10
  arquivos iniciais de `aulas/`.
- **2026-08-19** — Bolt 31: `data/50_Startups.csv` (sem dummies de Estado) +
  entrada nova em `DATASETS` (`app.py`).
- **2026-08-19** — Bolt 32: teste automatizado confirma R²≈0.9431 no
  dataset novo, batendo com o script de aula (48 casos no total).
- **2026-08-19** — Bolt 33: `classificar_tipo_problema()` em `core.py`,
  espelhando a taxonomia de `Intro_AprendMaquina.pdf`.
- **2026-08-19** — Bolt 34: nova aba "🧭 Passo 0" em `app.py` (primeira aba,
  nos dois Tipos de Tarefa), com taxonomia, vocabulário unificado, pegadinha
  da Regressão Logística e quiz interativo.
- **2026-08-19** — Bolt 35: `README.md` e `OPERATIONS.md` atualizados para
  a v7.
- **2026-08-19** — Bolt 36: `selecionar_melhor_subconjunto_multipla()` em
  `core.py`; leaderboard passa a mostrar a combinação de variáveis com
  maior R² em vez de sempre usar todas (50 testes no total).
- **2026-08-19** — Bolt 37: `OPERATIONS.md` atualizado para a v8.
