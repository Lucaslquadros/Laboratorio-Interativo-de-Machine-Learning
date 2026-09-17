# Inception — Laboratório Interativo de Machine Learning

> Preenchido de forma retroativa: o app já existia (v1) quando o AI-DLC foi
> adotado. A partir daqui, toda evolução do projeto segue o fluxo normal
> (Inception → Construction → Operations) antes de qualquer novo bolt.

## Visão do projeto (atualizada — v6)

O projeto nasceu enxuto: um app para visualizar Regressão Linear Simples.
Ele já cresceu bastante (v2-v5: robustez, Múltipla, datasets de ODS,
Polinomial, Diagnóstico de Resíduos, Regressão Logística) e a visão muda de
escopo a partir daqui — deixa de ser "o laboratório de regressão" e passa a
ser a **consulta oficial de algoritmos de Machine Learning** do Lucas: um
único app que cobre, de forma interativa e com o código real à mostra, os
algoritmos vistos ao longo da disciplina — e, depois, possivelmente além
dela.

**Prioridade combinada com o Lucas:** primeiro fechar os 14 tópicos do
syllabus da disciplina (lista completa abaixo); só depois disso avaliar
expandir para algoritmos fora do programa da matéria.

### Roadmap do syllabus (14 tópicos)

| # | Tópico | Status |
|---|--------|--------|
| 1 | Introdução ao Aprendizado de Máquina (conceitos, supervisionado x não-supervisionado) | ✅ aba Teoria |
| 2 | Regressão Linear Simples | ✅ v1 |
| 3 | Regressão Linear Múltipla | ✅ v3 |
| 4 | Estudo de Adequação do Modelo (resíduos, diagnóstico) | ✅ v5 (Fase A) |
| 5 | Regressão Polinomial | ✅ v5 (manual) + v12/Bolt 50-52 (grau automático via `GridSearchCV`, opção extra ao lado do manual) |
| 6 | Regressão Logística | ✅ v5 (Fase C) |
| 6b | **Regularização (Ridge / Lasso / ElasticNet)** — fora dos 14 originais, apresentada na Aula7; priorizada à frente dos itens 7-14 (ver regra acima) | ✅ v13 — toggle na Múltipla, escopo restrito a Múltipla+Regularização (sem Polinomial/Logística, não apresentado em aula) |
| 7 | Árvores de Decisão (estrutura, critérios de divisão, poda) | ⬜ pendente |
| 8 | K-NN (vizinhos mais próximos) | ⬜ pendente |
| 9 | Random Forest (ensemble) | ⬜ pendente |
| 10 | Support Vector Machine — SVM (kernel trick, linear/não-linear) | ⬜ pendente |
| 11 | Compromisso entre Viés e Variância (over/underfitting) | ⬜ pendente |
| 12 | Seleção e Validação de Modelos (validação cruzada) | ⬜ pendente |
| 13 | Métricas de avaliação (acurácia, precisão, F1, cobertura, matriz de confusão) | 🟡 parcialmente coberto (métricas de classificação já existem na Regressão Logística) — falta consolidar como conteúdo próprio |
| 14 | Análise de Agrupamentos / Clustering (k-médias, DBSCAN) | ⬜ pendente — primeiro tópico não-supervisionado do app |

Itens 7-10 e 14 são novos algoritmos (cada um vira seu próprio ciclo
Inception → Construction → Operations, como Múltipla/Polinomial/Logística
antes deles). Itens 11-13 são mais conceituais/transversais — tendem a virar
conteúdo na aba Teoria e, quando fizer sentido, ajustes na aba Comparação
(ex.: validação cruzada) em vez de "modelos" novos por si só.

**Regra de priorização (decidida na v12, substitui a prioridade "ordem do
syllabus" da v6 sempre que houver conflito):** conteúdo que o professor **já
apresentou em aula** (mesmo que fora dos 14 tópicos originais) tem prioridade
sobre tópicos do syllabus que **ainda não foram apresentados**. Motivo: o
app é uma consulta de estudo para o semestre em andamento — não faz sentido
adiantar Árvores de Decisão (item 7, não apresentado ainda) enquanto há
material de aula real (Regularização, item 6b) esperando revisão. Na
prática: **Regularização (Ridge/Lasso/ElasticNet)** — apresentada na Aula7,
não fazia parte da lista original de 14 — entra como **item 6b**, com
prioridade sobre os itens 7-14. Ver "Requisitos funcionais (v12)" abaixo.

### Decisões validadas com o Lucas (rodada da mudança de visão)

- **Nome do projeto:** renomeado de "Laboratório de Regressão Linear
  Simples" para **"Laboratório Interativo de Machine Learning"** — em
  `README.md`, `app.py` (docstring, `page_title`, `st.title`) e neste
  documento. O nome da pasta do projeto não muda.
- **Escopo:** os 14 tópicos do syllabus da disciplina, nesta ordem, antes de
  cogitar qualquer algoritmo fora do programa da matéria. Substitui a
  decisão anterior (v3) que excluía Árvore de Decisão, Random Forest, KNN e
  Ridge/Lasso "deste ciclo" — eles voltam ao escopo, só que em ciclos
  futuros, um de cada vez.
- **Arquitetura:** `app.py` (~1300 linhas) e a divisão em módulos por
  algoritmo/tarefa ficam como intenção registrada (ver
  `03-operations/OPERATIONS.md`, "Próximos passos"), não como decisão
  travada agora — a forma concreta de dividir o código deve ser avaliada no
  Inception do próximo algoritmo novo (Árvore de Decisão, item 7), quando
  ficar mais claro se o padrão atual (branches por modo dentro das mesmas
  abas) ainda aguenta crescer.

## Objetivo original (v1, mantido por contexto histórico)

Site interativo (Streamlit) que deixa visível, na prática, como a
Regressão Linear Simples ensinada em aula funciona — visualmente (gráficos,
métricas) e "por trás dos panos" (o código real de cada passo). Uso pessoal
de estudo, mas precisa aguentar uso real ao longo do semestre: rodar sem
travar com dados diferentes, sem monitoramento humano constante.

## Contexto

- Disciplina: Machine Learning (Prof. Dr. Daniel Trevisan Bravo).
- Material-base: slides "Introdução ao Aprendizado de Máquina" e "Regressão
  Linear Simples"; scripts `exemplo1_regressaolinearsimples.py` e
  `exemplo2_regressaolinearsimples.py`; datasets `base_salarios.csv` e
  `USA_Housing.csv` fornecidos em aula.
- Prazo: uso contínuo ao longo do semestre (sem entrega única).

## Requisitos funcionais (v1 — já entregues)

- [x] Escolher dataset (2 da aula + exemplo do slide + upload de CSV próprio)
- [x] Escolher X e y interativamente, com sugestão pela correlação
- [x] Mostrar matriz/mapa de calor de correlação com interpretação da força
- [x] Split treino/teste configurável (padrão 70/30)
- [x] Treinar `LinearRegression` e mostrar β0/β1, e o mesmo cálculo na mão
- [x] Avaliar com R², MAE, MSE + gráfico de dispersão com reta
- [x] Previsão interativa para novo valor de X
- [x] Expander "ver o código" com o código-fonte real de cada passo

## Requisitos não-funcionais (robustez, qualidade) — v2, motivo desta rodada

- [ ] Trata entradas inválidas/inesperadas sem quebrar (CSV com NaN, poucas
      linhas, coluna sem variância, separador ; em vez de ,)
- [ ] Tem teste automatizado (pytest) cobrindo o núcleo do cálculo
      (β0/β1 manual == scikit-learn, métricas, casos de erro)
- [ ] Lógica de regressão separada da interface (importável e testável sem
      subir o Streamlit)
- [ ] Cache de leitura de CSV (evita reprocessar 10 mil linhas a cada clique)
- [ ] Roda de novo do zero sem passos manuais escondidos (`pip install -r
      requirements.txt && streamlit run app.py`)

## Requisitos funcionais (v3 — Regressão Linear Múltipla, motivo desta rodada)

- [ ] Selecionar o "Tipo de Regressão" (Simples ou Múltipla) num painel lateral
- [ ] No modo Múltipla, escolher 2 ou mais variáveis X (em vez de uma só)
- [ ] Treinar `LinearRegression` com múltiplas colunas X e mostrar intercepto +
      tabela de coeficientes (um por variável) e a equação resultante
- [ ] Avaliação (R², MAE, MSE) funciona igual nos dois modos; gráfico se adapta
      (reta 2D no modo Simples, "Previsto vs Real" no modo Múltiplo, já que não
      dá para desenhar uma reta em N dimensões)
- [ ] Previsão interativa com um input por variável X no modo Múltiplo
- [ ] Nova aba de Comparação: treina o modelo "alternativo" ao modo ativo só
      para comparação lado a lado (R², MAE, MSE), sem afetar as outras abas
- [ ] Lógica nova (validação, treino, previsão) em `core.py`, testável sem
      subir o Streamlit, seguindo o mesmo padrão das funções já existentes

## Decisões validadas com o Lucas (perguntas de validação respondidas)

- **Modelo a adicionar:** só Regressão Linear Múltipla por enquanto. Árvore de
  decisão, Random Forest, KNN e Ridge/Lasso ficam de fora deste ciclo.
- **UI:** seletor de modo (Simples/Múltipla) + aba nova de comparação lado a
  lado — não é só trocar o modelo usado nas abas existentes.
- **Escopo de arquivos:** só `laboratorio_regressao_linear_1\laboratorio_regressao_linear`
  (a cópia mais avançada, com `core.py` testável). A cópia antiga
  `laboratorio_regressao_linear` (v1, sem `core.py` em uso) não é tocada.
- **Processo:** fluxo AI-DLC completo — Inception (este documento) →
  Construction em bolts pequenos com checkpoint (`BOLTS.md`) → Operations
  (changelog em `OPERATIONS.md`) ao final.

## Requisitos funcionais (v4 — Datasets de ODS via API pública, motivo desta rodada)

- [ ] Pelo menos 2 novos datasets padrão (embutidos, não upload) ligados aos
      Objetivos de Desenvolvimento Sustentável (ODS) da ONU
- [ ] Dados vindos de uma API pública de verdade (não inventados/sintéticos),
      com fonte documentada
- [ ] Bons para Regressão Múltipla: várias colunas numéricas candidatas a X,
      sem NaN, com R² razoável e **estável** entre diferentes `random_state`
      (testado antes de fixar no repo -- um candidato com CO2 per capita foi
      descartado por dar R² instável/negativo por causa de outliers)
- [ ] Processo de coleta reproduzível (script versionado, não só o CSV final)

## Decisões validadas com o Lucas (perguntas de validação respondidas)

- **Fonte de dados:** API pública do World Bank Open Data (sem
  autenticação, dados oficiais de desenvolvimento por país).
- **Datasets escolhidos:** "Saúde & Desenvolvimento" (Expectativa de vida
  explicada por PIB, gasto em saúde, acesso à eletricidade, CO2 -- ODS 3, 7,
  8) e "Mortalidade Infantil & Desenvolvimento" (Mortalidade < 5 anos
  explicada por PIB, gasto em saúde, saneamento, água potável -- ODS 3, 6,
  8). Um terceiro candidato ("Clima & Desenvolvimento", CO2 per capita) foi
  testado e descartado por instabilidade do R² entre splits.
- **Formato:** CSV estático salvo em `data/`, baixado uma vez via script
  (não busca ao vivo na API a cada execução do app) -- mesma abordagem dos
  datasets já existentes, para o app funcionar offline e com resultados
  reproduzíveis.

## Requisitos funcionais (v5 — Diagnóstico de Resíduos + Regressão Polinomial + Regressão Logística)

Motivo: próximos 3 tópicos do syllabus da disciplina (Estudo de Adequação
do Modelo, Regressão Polinomial, Regressão Logística). Plano completo em
`ai-dlc/02-construction/BOLTS.md` (Fases A/B/C).

- [x] **Fase A — Diagnóstico de Resíduos:** nova aba com resíduos vs.
      previstos, histograma e teste de Shapiro-Wilk, disponível nos modos
      de regressão (Simples/Múltipla, e futuramente Polinomial).
- [x] **Fase B — Regressão Polinomial:** 3ª opção em "Tipo de Regressão",
      grau configurável; aba Comparação vira leaderboard com todos os
      modelos de regressão.
- [x] **Fase C — Regressão Logística:** novo Tipo de Tarefa (Regressão /
      Classificação) com dataset próprio de alvo binário genuíno (tema
      ODS 3), métricas de classificação (matriz de confusão, acurácia,
      precisão, revocação, F1, ROC/AUC).

## Riscos / pontos de incerteza

- Dataset enviado pelo aluno pode ter formato inesperado (separador,
  encoding, colunas totalmente vazias) — precisa de mensagem clara, não
  stack trace.
- Datasets pequenos (poucas linhas) podem deixar `test_size` sem linhas de
  teste ou treino — precisa de validação antes do `train_test_split`.
- Coluna X ou y sem variância (todos os valores iguais) quebra o cálculo
  manual de β1 (divisão por zero).
- Modo Múltipla exige pelo menos 2 colunas numéricas candidatas a X além de
  y — datasets pequenos (ex.: upload com só 2 colunas numéricas) podem não
  ter isso; precisa de mensagem clara, não um `multiselect` vazio quebrando
  o treino.
- A aba de Comparação treina um segundo modelo "nos bastidores" — se esse
  modelo alternativo não for viável (ex.: só existe 1 coluna X candidata no
  dataset, então não dá para treinar a Múltipla de comparação), a aba deve
  avisar em vez de travar.

## Critérios de aceite

- Suite de testes (`pytest`) passa sem falhas.
- Upload de um CSV com uma coluna vazia, um CSV com poucas linhas, e um CSV
  com colunas constantes não derrubam o app — aparece mensagem de erro
  amigável em vez de traceback.
- App continua batendo com os valores de referência dos scripts da aula
  (R²≈0.8943 salário, R²≈0.4247 casas, y=0.5716+0.0663X no exemplo do slide)
  quando o modo Simples está ativo.
- Modo Múltipla treina sem erro nos 3 datasets embutidos (quando têm 2+
  colunas numéricas candidatas) e num CSV de upload com 3+ colunas numéricas.
- Aba de Comparação mostra R²/MAE/MSE dos dois modelos lado a lado nos 3
  datasets embutidos.

- Indicadores do World Bank para o mesmo ano/país costumam ser todos
  correlacionados entre si (um "fator" comum de nível de desenvolvimento) --
  então a Múltipla nem sempre ganha muito da Simples (ex: no dataset de
  Mortalidade Infantil, Saneamento sozinho já explica quase tudo). Isso foi
  documentado na descrição do dataset em vez de escondido/maquiado.

## Requisitos funcionais (v7 — Fluxo de ingestão de `aulas/` + Passo 0 de classificação do problema)

Motivo: o Lucas apontou uma lacuna real — o app deixa escolher o modelo (via
seletor "Tipo de Regressão"/"Tipo de Tarefa") sem nunca ensinar **como
classificar o problema** antes disso. Ele criou a pasta `aulas/` (fora do
controle do app até agora) para centralizar slides, datasets e códigos que o
professor for passando, com a ideia de que isso vire um **fluxo contínuo**:
ele adiciona arquivos em `aulas/` ao longo do semestre, e a cada rodada eu
confiro alinhamento entre o que está lá e o que o app cobre.

### O que já existe em `aulas/` (primeira leitura, 2026-08-19)

| Arquivo | O que é | Ação |
|---|---|---|
| `Intro_AprendMaquina.pdf` | Slides "Introdução ao Aprendizado de Máquina" | Fonte da taxonomia do Passo 0 (ver abaixo) |
| `RegressaoLinearSimples_ML.pdf` | Slides já refletidos na aba Teoria | Nenhuma ação nova |
| `RegressaoLinearMultipla_ML.pdf` | Slides de Regressão Múltipla | Lido; sem conceito novo além do já implementado (ver riscos) |
| `exemplo1_regressaolinearsimples.py`, `exemplo2_regressaolinearsimples.py` | Scripts já referenciados no README/`core.py` | Nenhuma ação nova |
| `exemplo1_regressaolinearmultipla.py` (50 Startups), `exemplo2_regressaolinearmultipla.py` (USA Housing) | Scripts novos de Regressão Múltipla | Confirmam que `treinar_modelo_regressao_multipla` já bate metodologicamente (mesmo `LinearRegression`, split 70/30 -- já é o default do app) |
| `base_salarios.csv`, `USA_Housing.csv` | Idênticos (`diff` bit-a-bit) aos já presentes em `data/` | Nenhuma ação |
| `50_Startups.csv` | **Dataset novo**, não está em `data/` ainda | Adicionar (ver abaixo) |

### Taxonomia extraída de `Intro_AprendMaquina.pdf` (para o Passo 0)

O professor (slide 17, "Tipos de algoritmos em DM") classifica assim,
usando estes termos exatos:

```
Aprendizado de Máquina (Data Mining)
├── Supervisionado (dados + rótulos/target conhecidos)
│   ├── Regressão → atributo-alvo CONTÍNUO
│   │   (exemplos citados pelo professor: Árvores de Decisão, Regressão
│   │    Linear, Naive Bayes, SVM, KNN -- quando usados p/ valor contínuo)
│   └── Classificação → atributo-alvo SEMPRE categórico
│       ├── Binária (2 classes, 1/0)
│       └── Multiclasse (3+ classes)
│       (Regressão Logística É classificação, apesar do nome -- o professor
│        destaca isso explicitamente como pegadinha conceitual)
├── Não supervisionado ("aprendizado descritivo", sem rótulo)
│   ├── Associação (Algoritmo Regra de Associação)
│   └── Clusterização/Agrupamento (K-Means, Hierárquicos, Grafos)
└── Aprendizado por Reforço (recompensas/punições)
```

Vocabulário duplicado usado pelo professor em aulas diferentes (o Passo 0
deve deixar isso explícito, para o aluno reconhecer os dois jeitos):
**X** = "atributos previsores" (Intro) = "variáveis independentes/preditoras"
(Múltipla). **y** = "atributo-alvo/target" (Intro) = "variável dependente/
resposta" (Múltipla).

O PDF **não tem** um fluxograma de "como escolher o algoritmo" -- é só a
taxonomia em tópicos acima. O Passo 0 vai ser uma **síntese didática nossa**
em cima dessa taxonomia (não uma reprodução de slide existente) -- vale
deixar isso claro na UI, para não parecer que é conteúdo literal do
professor.

- [ ] Nova seção/aba "Passo 0" que guia o aluno a classificar o problema
      **antes** de liberar a escolha do modelo: (1) o alvo é conhecido/
      rotulado? (supervisionado vs. não-supervisionado -- não-supervisionado
      fica só mencionado, ainda não implementado); (2) o alvo é contínuo ou
      categórico? (regressão vs. classificação); (3) se categórico, binário
      ou multiclasse?; termina indicando qual "Tipo de Tarefa" do app usar.
- [ ] Incluir a pegadinha da Regressão Logística ("é classificação, apesar
      do nome") como pergunta de checagem de entendimento.
- [ ] Unificar o vocabulário X/y (atributos previsores = variáveis
      independentes; atributo-alvo = variável dependente) em um único lugar
      visível.
- [ ] Adicionar `50_Startups.csv` a `data/` e ao seletor de datasets,
      seguindo o padrão dos scripts de aula (X = `R&D Spend` + `Marketing
      Spend`, y = `Profit`, R² de referência ≈0.9431 com `test_size=0.3,
      random_state=0`).
- [ ] Registrar um processo repetível de ingestão: quando o Lucas adicionar
      arquivo(s) novos em `aulas/`, eu confiro contra `data/`, `core.py`,
      `app.py` (aba Teoria) e o roadmap do syllabus, e reporto o que muda --
      sem exigir que ele descreva manualmente o que colocou lá.

### Riscos / pontos de incerteza (v7)

- `50_Startups.csv` já vem com a coluna `State` pré-codificada em 3 colunas
  dummy (`California`, `Florida`, `New York`) -- **conceito de variável
  dummy/one-hot encoding não aparece em nenhum dos dois PDFs de aula**
  (confirmado por leitura integral). O app não precisa implementar
  encoding (o CSV já vem pronto, e são colunas numéricas 0/1 como
  qualquer outra), mas expõe o aluno a um conceito que a aula não cobre
  explicitamente -- decidir se isso merece uma nota textual ou fica de fora
  por ora.
- Extração dos PDFs foi via texto (`pdftotext`), não via renderização de
  imagem -- diagramas e a fórmula matricial da Múltipla não foram vistos
  diretamente, só inferidos pelo texto ao redor. Se a notação exata de algum
  slide importar depois, pode ser preciso abrir o PDF manualmente.
- Sem um manifesto de "o que já foi revisado" em `aulas/`, cada rodada de
  checagem corre o risco de re-analisar tudo do zero (caro) ou perder
  arquivos novos silenciosos (arriscado). Precisa de um mecanismo leve de
  rastreio -- ver pergunta de validação abaixo.
- `aulas/` não deve virar fonte de verdade para os datasets em produção do
  app (isso continua sendo `data/`) -- é só a caixa de entrada; preciso
  manter claro que copiar/adaptar para `data/` é um passo explícito, não
  automático.

### Critérios de aceite (v7)

- `pytest` continua passando; novos casos cobrem `50_Startups.csv` se ele
  ganhar tratamento especial em `core.py` (não deve precisar, já que são
  colunas numéricas comuns).
- Passo 0 renderiza sem erro, usa a taxonomia/vocabulário acima, e termina
  indicando explicitamente qual seletor da sidebar usar.
- `50_Startups.csv` aparece no seletor de datasets e reproduz o R²≈0.9431
  do script de aula (mesma X, mesmo `test_size`/`random_state`).
- Existe um lugar único e documentado (a decidir nas perguntas abaixo) que
  registra quais arquivos de `aulas/` já foram revisados e o que foi feito
  com cada um.

### Decisões validadas com o Lucas (v7)

- **Passo 0:** aba nova e dedicada (não uma seção dentro da Teoria) — fica
  em destaque, antes das outras abas.
- **`50_Startups.csv`:** segue o script de aula (`exemplo1_regressaolinearmultipla.py`)
  1:1 — X = `R&D Spend` + `Marketing Spend`, y = `Profit`. As colunas dummy
  de `State` (`California`/`Florida`/`New York`) ficam no CSV mas não são
  expostas no multiselect por enquanto, para não introduzir um conceito
  (variável dummy) que a aula ainda não cobriu.
- **Rastreio de `aulas/`:** arquivo dedicado `ai-dlc/aulas-log.md`, com
  tabela simples (arquivo, tipo, status, ação tomada, data) — não reaproveita
  o changelog do `OPERATIONS.md`.

## Requisitos funcionais (v8 — Leaderboard: "Múltipla" ignora a melhor combinação de variáveis)

Motivo: o Lucas reportou, olhando a aba Comparação (dataset "🚀 Lucro de
Startups (aula)"), que o Leaderboard parecia inconsistente com o resto do
app -- a aba Múltipla interativa sugere por padrão 2 variáveis (`R&D Spend`
+ `Marketing Spend`, excluindo `Administration`, que tem baixa correlação
com `Profit`), mas o Leaderboard mostra "Múltipla: 3 variáveis".

**Confirmado no navegador (`localhost:8501`, dataset Startups,
`test_size=0.3`, `random_state=0`):**

| Onde | Variáveis usadas | R² |
|---|---|---|
| Aba Múltipla (padrão interativo, documentado desde a v7) | `R&D Spend` + `Marketing Spend` | **0.9431** |
| Aba Comparação → linha "Múltipla" do leaderboard | `R&D Spend` + `Administration` + `Marketing Spend` (todas) | 0.9355 |

Não é um erro de cálculo -- é comportamento **intencional desde o Bolt 22**
(`BOLTS.md`): a linha "Múltipla" do leaderboard sempre usa **todas** as
colunas numéricas candidatas (`opcoes_x_ordenadas` inteiro, em
`app.py:1343-1344`), sem nenhuma seleção de variáveis. Isso funcionava
enquanto os datasets tinham poucas colunas parecidas entre si (ex.: ODS),
mas no dataset Startups a variável `Administration` tem correlação fraca
com `Profit` e **piora** o R² de teste ao ser incluída -- então o
leaderboard, cujo texto promete ranquear "todos os modelos... por R²",
acaba mostrando uma Múltipla pior do que a "oficial" da aba interativa (e
pior do que o R²≈0.9431 documentado como referência da aula/Bolt 32). Quem
olha só o leaderboard pode concluir, errado, que a Múltipla "não vale a
pena" nesse dataset.

- [ ] Decidir (pergunta de validação abaixo) como a linha "Múltipla" do
      leaderboard deve escolher **quantas** e **quais** variáveis usar.
- [ ] Aplicar a decisão em `app.py` (bloco do leaderboard, ~linha 1339-1360).
- [ ] Testar no navegador nos datasets Startups (o caso que expôs o
      problema), USA_Housing e pelo menos 1 dataset de ODS (várias colunas
      correlacionadas entre si -- não pode regredir o R² desses).
- [ ] Cobrir com `pytest` o novo critério de escolha, se envolver lógica
      nova em `core.py`.

### Opções levantadas (perguntadas ao Lucas via `AskUserQuestion`)

- **A -- Melhor subconjunto (busca exaustiva):** testar todas as
  combinações de 2..N variáveis e usar a de maior R² de teste; "Detalhes"
  passa a listar os nomes escolhidos (ex.: "R&D Spend + Marketing Spend").
  Resolve o problema de raiz e responde de vez a pergunta "como eles
  escolhem as variáveis" -- passa a ser "a combinação com melhor R²",
  igual ao critério que a Simples já usa (maior correlação) e que o
  próprio dataset documenta como referência. Viável computacionalmente
  porque nenhum dataset do app hoje passa de ~5 colunas X candidatas
  (2^5 combinações, instantâneo). Ressalva: escolhe com base no R² do
  *conjunto de teste* -- mesma simplificação pedagógica que a Simples já
  comete (a correlação usada para escolher X é calculada no dataset
  inteiro, não só no treino), então não introduz um problema novo, só
  estende um existente.
- **B -- Mesmo padrão da aba interativa:** leaderboard usa
  `opcoes_x_ordenadas[:2]` (top-2 por correlação com y), igual ao valor
  default do `multiselect` da aba Múltipla. Simples de implementar e
  garante consistência entre abas, mas fixa "2" como mágico -- não se
  adapta a um dataset onde 3+ variáveis genuinamente ajudassem.
- **C -- Só ajustar a comunicação:** manter "todas as variáveis", mas
  deixar explícito no rótulo/legenda que não é necessariamente a melhor
  combinação (ex.: "Múltipla (todas as 3 variáveis)" + nota de rodapé).
  Menor esforço, não resolve a confusão de fundo -- só documenta a
  limitação.

**Recomendação:** opção A -- é a que de fato responde à pergunta original
do Lucas e deixa o leaderboard coerente com o resto do app.

**Decisão validada com o Lucas:** opção A (melhor subconjunto, busca
exaustiva). Implementada no Bolt 36 (`BOLTS.md`).

## Requisitos funcionais (v9 — Ingestão de `aulas/`: teste de suposições + dataset Publicidade)

Motivo: o Lucas adicionou dois arquivos novos em `aulas/`:
`exemplo_teste_suposicao.py` e `publicidade.csv`. Seguindo o processo
registrado na v7, a checagem de alinhamento encontrou lacunas reais no
tópico #4 do syllabus (Estudo de Adequação do Modelo), que hoje só está
parcialmente coberto pela aba **🩺 Diagnóstico dos Resíduos**.

### Leitura dos arquivos novos

| Arquivo | O que é | Ação |
|---|---|---|
| `publicidade.csv` | Dataset clássico "Advertising" (TV/Rádio/Jornal → Vendas, 200 linhas, 4 colunas, sem NaN) | **Dataset novo**, não está em `data/` ainda — integrar |
| `exemplo_teste_suposicao.py` | Script novo sobre as 6 suposições da regressão linear (linearidade, média dos resíduos, homocedasticidade, normalidade, independência, colinearidade), com testes estatísticos formais para cada uma | Confirma o que já existe (média dos resíduos, normalidade via Shapiro-Wilk) e expõe **3 lacunas** (ver abaixo) |

### Gap analysis: suposições cobertas x faltantes na aba Diagnóstico

| Suposição | Hoje no app | O que o script novo ensina a mais |
|---|---|---|
| Linearidade | Indireta (matriz de correlação em "Dados & Correlação") | Nada de novo — mesma ideia |
| Média dos resíduos = 0 | ✅ `diagnosticar_residuos` | Nada de novo |
| Homocedasticidade | Só visual (gráfico Resíduos vs. Previstos) | **Teste de Goldfeld-Quandt** (formal, com p-valor) |
| Normalidade | ✅ Shapiro-Wilk | Nada de novo |
| Independência dos resíduos | ❌ Não existe | **Teste de Ljung-Box e teste de Durbin-Watson** — no próprio script de aula os dois divergem no dataset de publicidade (Ljung-Box rejeita H0, Durbin-Watson não indica autocorrelação relevante), o que é pedagogicamente rico (mostra que testes diferentes captam coisas diferentes) |
| Ausência de colinearidade | Indireta (matriz de correlação geral em "Dados & Correlação", não citada como suposição) | Nada de novo tecnicamente, mas o script trata como suposição própria — decisão: duplicar o heatmap (só das colunas X escolhidas) dentro da aba Diagnóstico |

Nova dependência: `statsmodels` (usada pelo script para Goldfeld-Quandt,
Ljung-Box e Durbin-Watson; já estava instalada no ambiente, mas ausente do
`requirements.txt`).

- [ ] Adicionar `statsmodels` a `requirements.txt`.
- [ ] Integrar `publicidade.csv` a `data/` e ao seletor de datasets
      (`DATASETS` em `app.py`), seguindo o padrão já usado para
      `50_Startups.csv` (v7).
- [ ] `core.py`: `testar_homocedasticidade_residuos()` (Goldfeld-Quandt) e
      `testar_independencia_residuos()` (Ljung-Box + Durbin-Watson).
- [ ] Aba **🩺 Diagnóstico dos Resíduos**: nova seção de Homocedasticidade
      (teste formal, além do gráfico já existente), nova seção de
      Independência dos Resíduos (os dois testes, com nota explicando a
      divergência possível) e heatmap de colinearidade das variáveis X
      escolhidas.
- [ ] Atualizar `ai-dlc/aulas-log.md` com os 2 arquivos novos.

### Decisões validadas com o Lucas (v9)

- **Independência dos resíduos:** incluir os dois testes (Ljung-Box e
  Durbin-Watson), com nota textual explicando que podem divergir — em vez
  de escolher só um.
- **Colinearidade:** duplicar o heatmap de correlação (das variáveis X
  escolhidas) dentro da aba Diagnóstico, além do já existente em "Dados &
  Correlação" — não é redundância "ruim" porque a aba Diagnóstico deve
  reunir todas as suposições em um só lugar.

### Riscos / pontos de incerteza (v9)

- Goldfeld-Quandt (`statsmodels`) devolve `NaN` para F/p-valor com menos de
  ~6 resíduos (confirmado empiricamente) — `core.py` precisa validar um
  mínimo de linhas e lançar `DadosInvalidosError` amigável antes disso,
  igual ao padrão já usado no Shapiro-Wilk (mínimo de 3).
- O script de aula escalona X com `StandardScaler` antes de treinar;
  confirmado matematicamente e empiricamente que R² não muda com essa
  transformação afim (mesma reta em outra parametrização) — o app **não**
  precisa adotar escalonamento para bater com a referência da aula.
- `sms.het_goldfeldquandt` não ordena as observações por padrão (usa a
  ordem original do conjunto de teste) — mesma forma "crua" usada no
  script de aula; documentar isso no texto da UI para não parecer mais
  rigoroso do que realmente é.

### Critérios de aceite (v9)

- `pytest` continua passando; novos casos cobrem `publicidade.csv` (R² de
  referência com as 3 variáveis e com o melhor subconjunto) e as duas
  funções novas de `core.py`.
- R² de referência do dataset `publicidade.csv`, com `test_size=0.3,
  random_state=0`: **0.8649** com as 3 variáveis (TV, Rádio, Jornal);
  **0.8657** com o melhor subconjunto (TV + Rádio).
- Aba Diagnóstico renderiza sem erro nos datasets existentes e no novo
  `publicidade.csv`, nos modos Simples, Múltipla e Polinomial.
- `ai-dlc/aulas-log.md` reflete os 2 arquivos novos como revisados/integrados.

## Requisitos funcionais (v10 — Ingestão de `aulas/correcoes/`: RMSE + 3 datasets novos + organização de `data/`)

Motivo: o Lucas adicionou `aulas/correcoes/` com 3 pastas (baixadas do Google
Drive, nomes com timestamp) contendo scripts de correção do professor, cada
um ao lado de um dataset usado no exercício. Ele também notou que datasets
ficaram espalhados entre `aulas/` e `aulas/correcoes/` quando já existe
`data/` para isso, e pediu para consolidar tudo lá.

### Leitura dos arquivos novos

| Arquivo | O que é | Ação |
|---|---|---|
| `correcao_regsimples.py` + `finance_market.csv` (500 linhas) | Regressão Simples: X=`Indice_S&P500`, y=`ETF_Preco`, `test_size=0.3, random_state=42` | **Dataset novo** — integrar. Calcula **RMSE**, métrica que o app não tinha |
| `regressao_multipla.py` + `regressao_simples.py` + `agro_tech.csv` (450 linhas) | Simples (X=`precipitacao_anual`) e Múltipla (+`fertilizante_kg_ha`), y=`toneladas_por_hectare`, `test_size=0.3, random_state=0` | **Dataset novo** — integrar. Metodologia idêntica ao já implementado (heatmap, R²/MAE/MSE, reta) |
| `correcao_teste_suposicao.py` + `base_plano_saude_preparada.csv` (2772 linhas) | Múltipla com as 6 suposições da regressão (mesmas da v9), mas aqui o **modelo falha**: heterocedasticidade e resíduos não-normais confirmados pelo próprio script do professor | **Dataset novo** — integrar como exemplo intencional de "mau ajuste" na aba Diagnóstico. Sem técnica nova (mesmos testes da v9) |

### Gap de código confirmado

- `core.py::avaliar_modelo()` retorna só R², MAE, MSE — `correcao_regsimples.py`
  trata **RMSE** como métrica padrão de avaliação. Gap real, mesmo padrão das
  lacunas achadas em v9.

### Valores de referência (reproduzidos e conferidos)

- `finance_market.csv`: R²≈0.9987, MAE≈0.44, MSE≈0.30, RMSE≈0.55
  (`test_size=0.3, random_state=42` — diferente do padrão do app, que é
  `random_state=0`; documentar os dois).
- `agro_tech.csv`: Simples R²≈0.4847 (MAE≈2.3037, MSE≈7.3525) → Múltipla
  R²≈0.8705 (MAE≈1.1035, MSE≈1.8481), batendo exatamente com os números do
  script do professor. Top-2 por correlação com y já é
  `precipitacao_anual` + `fertilizante_kg_ha` — o app escolhe isso
  automaticamente sem configuração extra.
- `base_plano_saude_preparada.csv`: correlações fracas (máx. |r|=0.335,
  `Sudoeste`); R²≈0.18 usando todas as colunas — confirma que é mesmo um
  caso de mau ajuste, não um bug de leitura.

### Decisões validadas com o Lucas (v10, via `AskUserQuestion`)

- **RMSE:** adicionar como métrica padrão em `core.py` (`avaliar_modelo`),
  aba Avaliação e leaderboard da Comparação, coberta por `pytest`.
- **Organização de `data/`:** os 3 datasets novos são **movidos** (não
  copiados) de `aulas/correcoes/.../programas/` para `data/` — os scripts
  `.py` de correção continuam em `aulas/correcoes/` como referência, sem o
  CSV ao lado. Mesma lógica aplicada aos duplicados que já existiam em
  `aulas/` (nível raiz): `base_salarios.csv`, `USA_Housing.csv` e
  `publicidade.csv` são bit-idênticos aos de `data/` (confirmado por
  `diff`) — removidos de `aulas/` por serem puramente redundantes.
  `aulas/50_Startups.csv` (com as colunas dummy de `State`, nunca usadas)
  também é removido — `data/50_Startups.csv` (sem as dummies) já é a
  versão oficial desde a v7. Tudo sob controle de versão (`git`), então
  reversível se necessário.
- **Colunas dummy de `base_plano_saude_preparada.csv`:** ao contrário do
  precedente do `50_Startups.csv` (v7, dummies escondidas), aqui a decisão
  foi **incluir todas as colunas** (`Noroeste`/`Sudeste`/`Sudoeste`
  inclusive) como candidatas a X, replicando o que o script do professor
  faz sem ressalva — o app já seleciona por correlação automaticamente, sem
  precisar de tratamento especial no código.

### Riscos / pontos de incerteza (v10)

- `finance_market.csv` usa `random_state=42` como referência do professor,
  diferente do padrão de todos os outros datasets (`random_state=0`) — a
  descrição do dataset documenta os dois valores para não confundir quando
  o usuário mantiver o padrão do app.
- Nenhuma técnica estatística nova em `correcao_teste_suposicao.py` além da
  v9 — não requer bolt de `core.py` novo, só integração do dataset.

### Critérios de aceite (v10)

- `pytest` continua passando; novos casos cobrem os 3 datasets (R²/MAE/MSE
  de referência) e o RMSE (bate com `np.sqrt(MSE)` em casos já existentes).
- Os 3 datasets aparecem no seletor de datasets e treinam sem erro nos 3
  modos de regressão.
- RMSE aparece na aba Avaliação e no leaderboard da Comparação.
- `data/` passa a ser a única cópia de cada dataset — nenhum CSV duplicado
  sobra em `aulas/` ou `aulas/correcoes/`.
- `ai-dlc/aulas-log.md` reflete os arquivos novos revisados/integrados.

## Requisitos funcionais (v11 — Aba de Revisão para a prova, Inception abreviada)

Motivo: o Lucas tem prova prática amanhã (2026-08-27) e pediu uma página
única reunindo todo o conhecimento do semestre, com **todo o código Python
à vista** (a prova é prática). Pediu para usar o AI-DLC e decidir onde
encaixar no app. Dado o prazo, esta rodada usa uma Inception **abreviada**
(decisões tomadas diretamente, sem rodada de `AskUserQuestion`) -- validação
acontece no checkpoint do bolt único, testando no navegador.

- **Onde encaixar:** nova aba **"🎓 Revisão da Prova"**, primeira da lista
  (antes de Passo 0) nos dois Tipos de Tarefa -- vira a aba padrão ao abrir
  o app, ideal para consulta rápida. Estática (não depende do dataset
  selecionado na sidebar), para nunca quebrar independente da escolha atual.
- **Conteúdo, em duas camadas:**
  1. **Scripts prontos (estilo "cola de prova")** -- para cada algoritmo já
     coberto (Simples, Múltipla, Polinomial, Diagnóstico de Resíduos,
     Logística), um bloco de código único, sequencial, no mesmo estilo dos
     scripts do professor (`aulas/`, `aulas/correcoes/`) -- `pd.read_csv` →
     X/y → `train_test_split` → `.fit()` → métricas → predição -- pronto
     para adaptar rápido numa prova prática, sem precisar entender a
     modularização do app.
  2. **Implementação real do app** -- as mesmas funções de `core.py` via
     `inspect.getsource` (já usado em toda a UI), para conferir a versão
     "de produção" caso o script de cola não seja suficiente.
  3. **Tabela de valores de referência** (gabarito) dos datasets de aula --
     útil para conferir se o resultado da prova bate com o esperado.
- **Escopo:** só os tópicos já implementados no app (Simples, Múltipla,
  Polinomial, Diagnóstico de Resíduos/6 suposições, Regressão Logística,
  Passo 0/taxonomia) -- tópicos 7-14 do syllabus (Árvores, KNN, Random
  Forest, SVM, Clustering) não têm código no app ainda, então ficam fora.

### Critérios de aceite (v11)

- `pytest` continua passando.
- App abre sem traceback com a aba nova como padrão, nos dois Tipos de
  Tarefa, testado com `streamlit.testing.v1.AppTest`.
- Todo o código mostrado é executável (copiado de `core.py` real via
  `inspect.getsource`, ou testado manualmente no caso dos scripts de cola).

## Requisitos funcionais (v12 — Ingestão de Aula5/`correcao_AC1`, Aula6 e Aula7 + reprioridade do backlog)

Motivo: o Lucas adicionou pastas de aula novas (`Aula6`, `Aula7`) direto na
raiz de `ML/` (fora do `Estudo_1`) e pediu uma varredura completa de todas as
pastas de aula contra o que já está ingerido. A varredura achou 3 focos sem
revisão (`Aula5/correcao_AC1`, `Aula6`, `Aula7`) e a entrega própria da AC1
(`ativ1`, fora do fluxo de aula). Os arquivos foram copiados para `aulas/`
(`aulas/correcoes/` para scripts de correção do professor, `aulas/ac1_lucas/`
para a entrega própria). O Lucas também decidiu a regra de priorização
registrada acima: conteúdo já apresentado em aula (mesmo fora do syllabus
original) passa à frente de tópicos do syllabus ainda não apresentados.

### Leitura dos arquivos novos

| Arquivo | O que é | Ação |
|---|---|---|
| `correcoes/gabarito_prova_T1_exercicio1.py` (`Ecommerce Customers.csv`, ausente) | **Rótulo corrigido (feedback do Lucas, 2026-09-16):** não é correção da AC1 -- é o **gabarito da prova prática T1** (sala T1, canal V5), a primeira avaliação formal do semestre. As mesmas 6 suposições da regressão (linearidade, média dos resíduos, homocedasticidade, normalidade, independência, colinearidade), com `StandardScaler` aplicado a X | Revisado — **nenhuma técnica nova** (mesmos testes da v9); dataset ausente, não dá para reproduzir os números |
| `correcoes/gabarito_prova_T1_exercicio2.py` (`weatherHistory.csv`, ausente) | Mesmo roteiro do exercicio1, mesma prova T1, dataset diferente | Revisado — mesma conclusão |
| `exemplo_regr_polinomial.py` (`brazil_covid19.csv`, presente) | Regressão polinomial em série temporal (casos de COVID) — usa `Pipeline(PolynomialFeatures, LinearRegression)` **dentro de um `GridSearchCV`** (`cv=5`, `scoring='neg_mean_squared_error'`, testando grau 1 a 6) para **escolher o grau automaticamente** | **Gap real** (ver abaixo) |
| `correcoes/correcao_polinomial_p1.py` (`Ice Cream.csv`, ausente), `correcoes/correcao_polinomial_p2.py` (dataset não identificado, ausente) | Mesma técnica (`Pipeline` + `GridSearchCV` sobre `poly__degree: [1..6]`), uma versão simples e uma multivariada | Revisado — confirma que o `GridSearchCV` é o padrão do professor, não uma escolha isolada de um script só |
| `exemplo_regularizacao.py` (`diabetes.csv`, presente) | Regressão Múltipla com `StandardScaler` + comparação `LinearRegression` vs. **Ridge**, **Lasso** e **ElasticNet**, cada um com `GridSearchCV` (`cv=10`, `scoring='neg_mean_squared_error'`) para achar o melhor `alpha` (e `l1_ratio` no ElasticNet); também usa `cross_val_score` para a Múltipla sem regularização | **Gap real, algoritmo novo** (ver abaixo) |
| `correcoes/correcao_Hitters.py` (`Hitters.csv`, ausente) | Mesmo roteiro (Ridge/Lasso/ElasticNet via `GridSearchCV`), dataset de beisebol com variáveis dummy (`League`/`Division`/`NewLeague`) | Revisado — confirma metodologia, sem técnica adicional além do `exemplo_regularizacao.py` |
| `diabetes.csv`, `diabetes_nao_escalonado.csv` | Datasets do exemplo de Regularização — versão escalonada (a usada no treino) e não-escalonada (usada só para escalonar novos dados na predição, com uma fórmula de padronização manual — nota do próprio script diz que é específica desse dataset) | **Dataset novo** — integrar junto com o bolt de Regularização |
| `Regularizacao_20252.pdf`, `RegressaoPolinomial_20252.pdf` | Slides das duas aulas | Fonte teórica para as abas Teoria/Passo 0 quando os bolts forem implementados |
| `ac1_lucas/AC1_Codigos_LucasQuadros_*.py` + datasets | Entrega própria do Lucas na AC1 (Pacientes COVID, Plano de Saúde) — **não é material do professor** | Revisado, sem ação automática — ver pergunta de validação abaixo |

### Gap 1: Regressão Polinomial — grau escolhido manualmente vs. `GridSearchCV`

Hoje (`app.py`, modo Polinomial) o aluno escolhe o grau (2 a 5) num slider.
Nos 3 scripts de aula que usam Polinomial (`exemplo_regr_polinomial.py`,
`correcao_polinomial_p1.py`, `correcao_polinomial_p2.py`), o grau **nunca**
é escolhido manualmente — sempre via `GridSearchCV` sobre um `Pipeline`,
testando grau 1-6 e escolhendo o de menor MSE (validação cruzada). É a
mesma lógica que o app já usa no leaderboard de Múltipla (Bolt 36, busca
exaustiva) — só que para grau, não para conjunto de variáveis.

### Gap 2: Regularização — algoritmo/técnicas totalmente ausentes de `core.py`

Nenhuma das seguintes peças existe hoje no app: `Ridge`, `Lasso`,
`ElasticNet`, `StandardScaler`, `GridSearchCV`, `cross_val_score`,
`neg_mean_squared_error` como critério de seleção. É o primeiro caso do app
onde escalonar X **importa de verdade** para o resultado (ao contrário da
OLS pura, documentado como não-afetada em v9) — regularização penaliza o
tamanho do coeficiente, então a escala de cada variável muda o resultado.

### Riscos / pontos de incerteza (v12)

- **Datasets ausentes:** `Ecommerce Customers.csv`, `weatherHistory.csv`,
  `Ice Cream.csv`, o dataset do `correcao_polinomial_p2.py` e `Hitters.csv`
  não existem em nenhuma pasta de `ML/` — só o código foi copiado. Esses 5
  scripts servem só como **referência de metodologia**, não são
  reproduzíveis nem viram dataset novo no app.
- **Discrepância treino x teste nos resíduos:** `gabarito_prova_T1_exercicio1.py`
  calcula os resíduos das 6 suposições sobre o **conjunto de treino**
  (`y_train - y_pred_train`); o app (`app.py:1550`,
  `diagnosticar_residuos(y_teste, predicoes_modelo)`) usa o **conjunto de
  teste**. Ambos são defensáveis (a v9 já documentou a lógica de usar teste),
  mas vale registrar a diferença para não achar que é um erro se aparecer de
  novo em outra correção.
- **Escalonamento "não-tradicional" do `diabetes.csv`:** o próprio
  `exemplo_regularizacao.py` avisa que esse dataset foi escalonado de um
  jeito diferente do `StandardScaler` padrão do scikit-learn (divisão extra
  por `sqrt(N_train)`) — só para esse dataset; outros exercícios usam
  `StandardScaler` "tradicional". Se o app vier a reproduzir a predição de
  novos dados do script, precisa da fórmula específica, não do
  `StandardScaler` genérico.
- **`ac1_lucas/`:** são datasets e código **do Lucas**, não do professor —
  `base_plano_saude_preparada.csv` aqui é **diferente** (`diff` confirma) da
  versão já integrada em `data/` (que veio do professor, via Aula5). Não
  deve substituir a versão oficial sem decisão explícita.

### Critérios de aceite (v12 — só desta rodada de Inception, sem Construction ainda)

- `aulas-log.md` reflete todos os arquivos novos (feito, ver tabela abaixo).
- Roadmap atualizado com a reprioridade combinada (feito, ver tabela do
  topo) e o item 6b (Regularização).
- Nenhum código de `core.py`/`app.py` foi alterado nesta rodada — Gap 1 e
  Gap 2 viram bolts só depois que o Lucas responder as perguntas abaixo,
  seguindo a regra do AI-DLC de não pular para Construction sem confirmação.

### Decisões validadas com o Lucas (v12, via `AskUserQuestion`)

1. **Gap 1 (Polinomial):** opção extra, não substituição — slider manual
   continua existindo, com um radio novo ("Manual" / "Automático") para
   escolher. Implementado nos Bolts 50-52 (ver `BOLTS.md`); checkpoint
   validado (`pytest` 65/65 + `AppTest` sem exceção nos dois modos).
2. **Gap 2 (Regularização):** a Inception própria do item 6b começa **depois**
   de fechar o Gap 1 (feito agora) — próximo passo do projeto, ainda não
   iniciado.
3. **`ac1_lucas/`:** fica só como registro em `aulas/` — não vira conteúdo do
   app, sem bolt de integração.
4. **Datasets ausentes:** o Lucas pediu para eu checar se os arquivos
   (`Ecommerce Customers.csv`, `weatherHistory.csv`, `Ice Cream.csv`, dataset
   do `correcao_polinomial_p2.py`, `Hitters.csv`) estão em alguma pasta de
   aula que a varredura inicial não tenha pego. Busca ampla refeita
   (nome exato + palavras-chave + conteúdo de todos os `.zip` de `ML/`):
   **confirmado, não existem em nenhum lugar do diretório `ML/`** (nem
   soltos, nem dentro de `entrega_pacientes_covid.zip`,
   `entrega_plano_saude.zip` ou `drive-download-...zip`). Os 5 scripts que
   os usam continuam como referência de metodologia apenas — se o Lucas
   encontrar os arquivos em outro lugar (Drive, Moodle), é só adicionar a
   `aulas/` que a próxima varredura pega.

## Requisitos funcionais (v13 — Regularização: Ridge/Lasso/ElasticNet, item 6b do roadmap)

Motivo: fechado o Gap 1 (Polinomial) na v12, o Lucas pediu para começar a
Inception do item **6b** do roadmap — Regularização — a próxima peça
priorizada à frente dos tópicos 7-14 (ainda não apresentados em aula).

### Visão

Regularização não é um algoritmo do zero -- é a Regressão Múltipla (que o
app já tem) com um termo de penalidade sobre o tamanho dos coeficientes.
Os dois scripts de aula lidos na v12 (`exemplo_regularizacao.py`,
`diabetes.csv`; `correcao_Hitters.py`, `Hitters.csv` ausente) seguem o mesmo
roteiro:

1. Treinar `LinearRegression` "normal" com todas as variáveis (X já vem
   escalonado com `StandardScaler` no `diabetes.csv` de aula) e olhar os
   coeficientes -- alguns ficam grandes por causa de multicolinearidade
   (ex.: `s1`/`s2` correlacionados em 0.896 no diabetes).
2. Para **Ridge**, **Lasso** e **ElasticNet**: usar `GridSearchCV` (`cv=10`,
   `scoring='neg_mean_squared_error'`) para achar o melhor `alpha` (e
   `l1_ratio` no ElasticNet) -- mesmo padrão de "hiperparâmetro escolhido
   por validação cruzada" que o Gap 1 do Polinomial acabou de implementar em
   `escolher_grau_polinomial_cv()`.
3. Comparar os coeficientes dos 3 modelos num gráfico único -- Ridge encolhe
   todos em direção a zero sem zerar; Lasso zera alguns (seleção de
   variável implícita); ElasticNet fica entre os dois (`l1_ratio` controla
   a mistura).
4. (Só no `exemplo_regularizacao.py`) Prever valores novos com o melhor
   modelo (Lasso, `alpha=0.06`), escalonando os dados novos com uma fórmula
   manual específica desse dataset (ver risco abaixo).

### Decisão de escopo (validada com o Lucas antes das perguntas abaixo)

Regularização não é um algoritmo à parte -- é um modificador (penalidade
sobre o tamanho dos coeficientes) que em tese poderia ser acoplado a
qualquer modelo linear: Múltipla, Polinomial (comum na prática, para conter
overfitting de grau alto) e até Logística (o `sklearn.linear_model.
LogisticRegression` já aplica L2 por padrão, nunca exposto ao aluno). Mas
**nenhum dos 2 scripts de aula lidos** (`exemplo_regularizacao.py` com
`diabetes.csv`, `correcao_Hitters.py` com `Hitters.csv`) combina
regularização com expansão polinomial ou com classificação -- os dois
aplicam Ridge/Lasso/ElasticNet só sobre X multivariado "cru" (Múltipla).
**Decisão:** esta rodada cobre só **Múltipla + Regularização**, para não
extrapolar além do que foi apresentado em aula (mesma regra de priorização
da v12). Polinomial+Regularização e penalidade na Logística ficam anotados
como possível expansão futura, sem virar requisito agora.

### Requisitos funcionais (rascunho, sujeito às perguntas de validação)

- [ ] `core.py`: funções para treinar Ridge/Lasso/ElasticNet com `alpha`
      escolhido por `GridSearchCV` (`cv`, `neg_mean_squared_error`),
      reaproveitando o padrão já criado em `escolher_grau_polinomial_cv()`.
- [ ] `StandardScaler` entra no pipeline -- primeira vez que o app escalona
      X antes de treinar (documentado desde a v9 que **não** era necessário
      para OLS puro; aqui **é**, porque a penalidade depende da escala).
- [ ] UI: encaixar Regularização em algum lugar do seletor "Tipo de
      Regressão" (ver pergunta 1 abaixo).
- [ ] Mostrar comparação de coeficientes (com/sem regularização) -- gráfico
      ou tabela, replicando a ideia do `comparing_models` do script de aula.
- [ ] Dataset `diabetes.csv` integrado a `data/` (10 variáveis numéricas já
      escalonadas, 442 linhas, alvo `diabetes_measure` contínuo).

### Riscos / pontos de incerteza (v13)

- **Convenção de avaliação diferente do resto do app:** todas as abas hoje
  usam `train_test_split` + métricas no conjunto de **teste**. O script de
  Regularização nunca faz esse split -- ele usa `cross_val_score`/
  `GridSearchCV` (CV) sobre o dataset **inteiro**. Adotar um dos dois jeitos
  (ou os dois) é decisão de design, não só estética -- ver pergunta 2.
- **Escalonamento não-tradicional do `diabetes.csv` na Previsão:** o próprio
  script avisa que esse dataset foi escalonado de um jeito específico (não
  o `StandardScaler` genérico) para poder desfazer a normalização em dados
  novos -- fórmula manual com `sqrt(N_train)`. Se o app quiser oferecer
  "prever novo valor" nesse dataset, precisa dessa fórmula específica, não
  do `StandardScaler` padrão do scikit-learn (que não é diretamente
  invertível para o `diabetes.csv` da forma como ele já vem pré-escalonado).
- **`Hitters.csv` ausente:** `correcao_Hitters.py` usa variáveis dummy
  (`League`/`Division`/`NewLeague`) além de Ridge/Lasso/ElasticNet -- só
  serve como confirmação de metodologia, sem dataset para integrar.
- **Múltiplos hiperparâmetros no ElasticNet:** `alpha` e `l1_ratio` juntos --
  o `GridSearchCV` já lida bem (grade 2D), mas a exibição na UI precisa
  deixar claro que são 2 valores escolhidos, não 1.

### Opções levantadas (para `AskUserQuestion`)

**1. Onde entra na UI** (já restrito a Múltipla + Regularização, ver
"Decisão de escopo" acima):
- **A -- Novo modo em "Tipo de Regressão"**: "Regularização (Ridge/Lasso/
  ElasticNet)" ao lado de Simples/Múltipla/Polinomial, com sua própria
  seleção de variáveis X (igual à Múltipla). Consistente com o padrão já
  usado para Polinomial/Logística, mas duplica a lógica de seleção de X que
  a Múltipla já tem.
- **B -- Toggle dentro do modo Múltipla existente**: aparece quando 2+
  variáveis X estão escolhidas ("Regularizar? Nenhuma/Ridge/Lasso/
  ElasticNet"). Reaproveita a seleção de X já existente e deixa explícito
  que é a mesma Múltipla, só com penalidade -- mas mistura dois conceitos
  na mesma aba/leaderboard.

**2. Convenção de avaliação:**
- **A -- Padrão do app**: `train_test_split` (como todo o resto),
  `GridSearchCV`/`cross_val_score` rodando só dentro do treino, métricas
  finais (R²/MAE/MSE/RMSE) no teste -- mais consistente com as outras abas,
  mas diverge do script do professor.
- **B -- Padrão do script de aula**: sem split explícito, `GridSearchCV`
  direto no dataset inteiro, métrica reportada é o `neg_mean_squared_error`
  médio de CV -- bate exatamente com os números que aparecem no script
  (ex.: -2986,37 para o Lasso no diabetes), mas foge do padrão do resto do
  app.

**3. Quais modelos expor:**
- **A -- Os 3 juntos, comparados** (Ridge + Lasso + ElasticNet + "sem
  regularização"), igual ao gráfico do script -- mais fiel à aula, mais
  trabalho de UI.
- **B -- Um de cada vez**, escolhido num seletor -- mais simples, menos
  imediato para comparar.

**4. Dataset `diabetes.csv`:**
- **A -- Integrar agora**, com "Previsão" desabilitada/com aviso (por causa
  do escalonamento não-padrão) até decidir a fórmula.
- **B -- Integrar agora com a fórmula específica implementada**, replicando
  a Previsão também.
- **C -- Adiar a integração do dataset**, focar só no algoritmo primeiro
  (usar outro dataset já existente, ex. `50_Startups` ou `saude_desenvolvimento`,
  para os quais X multivariado já está pronto e escalonar é direto).

### Decisões validadas com o Lucas (v13, parcial)

- **Pergunta 1 (UI):** toggle dentro do modo Múltipla existente (opção B) --
  não vira um "Tipo de Regressão" à parte.
- **Pergunta 2 (avaliação):** padrão do script de aula -- CV puro
  (`GridSearchCV`/`cross_val_score`, `cv=10`, sem `train_test_split`
  explícito). Validado com o Lucas o porquê: `cross_val_score`/
  `GridSearchCV` já fazem split internamente (k-fold, 10 rodadas treino/
  teste em vez de uma só) -- não é ausência de validação, é validação
  cruzada em vez de split único. A ressalva registrada: escolher o `alpha`
  (e `l1_ratio`) pelo mesmo `neg_mean_squared_error` de CV que depois anuncia
  o "vencedor" entre Ridge/Lasso/ElasticNet tem um viés de seleção sutil
  (uma CV aninhada eliminaria isso, mas foge do que o script faz) -- decisão
  consciente de seguir fiel ao script mesmo com essa limitação conhecida.

### Decisões validadas com o Lucas (v13, fechamento)

- **Pergunta 3 (modelos expostos):** os 3 juntos, comparados (Sem
  regularização + Ridge + Lasso + ElasticNet), igual ao gráfico do script.
- **Pergunta 4 (dataset):** não integrar `diabetes.csv` agora (escalonamento
  não-padrão complica demais a Previsão). Em vez disso, usar
  `mortalidade_infantil_desenvolvimento.csv` (já em `data/`) -- tem
  colinearidade real entre X (`Saneamento_pct` x `Agua_Potavel_pct`,
  r≈0.906, quase idêntico ao par `s1`/`s2` do diabetes, r≈0.896), já citado
  desde a v7 do README como bom exemplo de colinearidade. Reproduz a mesma
  lição pedagógica sem a complicação do escalonamento.
- **Decisão de arquitetura (não perguntada, mas registrada):** o toggle de
  Regularização é **puramente aditivo** -- só adiciona uma seção nova na
  aba Treinamento (tabela + gráfico comparativo), computada via CV no
  dataset completo. As abas Avaliação, Previsão, Diagnóstico dos Resíduos e
  Comparação **não mudam** -- continuam usando a Múltipla "padrão" (sem
  regularização) com o split treino/teste de sempre. Isso evita reescrever
  o pipeline compartilhado do app (que assume um único split usado por
  todas as abas) só para acomodar uma convenção de avaliação diferente
  (CV puro) usada só neste recurso -- e fica explícito na UI (texto da
  seção) para não confundir o Lucas.

### Critérios de aceite (v13)

- `pytest` continua passando (68 casos, 3 novos para `treinar_regularizacao_cv`).
- Toggle "4b. Regularizar?" aparece só no modo Múltipla; "Comparar Ridge /
  Lasso / ElasticNet" mostra tabela (score CV + hiperparâmetros) e gráfico
  de coeficientes sem gerar exceção (`streamlit.testing.v1.AppTest`).
- No dataset "👶 Mortalidade Infantil & Desenvolvimento" com X =
  `Saneamento_pct` + `Agua_Potavel_pct` + `PIB_per_capita`: Ridge/ElasticNet
  reduzem a diferença entre os coeficientes das duas variáveis colineares
  em relação à OLS sem regularização (redistribuem peso em vez de uma
  dominar a outra) -- confirmado manualmente.
- Abas Avaliação/Previsão/Diagnóstico/Comparação continuam idênticas ao
  comportamento anterior à v13 (nenhuma mudança de código nelas).

### Revisão pós-checkpoint (mesma rodada v13): feedback do Lucas ao testar

Depois do checkpoint acima, o Lucas testou o app e trouxe 3 observações que
expuseram um problema real de arquitetura, não só de UI:

1. Regularização só aparecia na aba Treinamento.
2. Não tinha controle manual de `alpha`/`l1_ratio` (só automático).
3. **O mais importante:** a tabela comparava os 4 modelos com CV pura no
   dataset inteiro, enquanto a Múltipla "oficial" (Avaliação) usa R²/MAE/
   MSE/RMSE no conjunto de teste -- métricas diferentes, partições
   diferentes (a CV da Regularização incluía linhas que na Múltipla estavam
   reservadas como teste). **Não dava pra responder "estou ganhando ao
   regularizar?" de forma confiável.**

**Decisão revista:** abandonar a convenção "CV pura" (decidida antes) em
favor de **GridSearchCV rodando só dentro de `X_treinamento`** (mesmo
padrão já usado em `escolher_grau_polinomial_cv`), com os 4 modelos
avaliados no mesmo `X_teste`/`y_teste` da Múltipla oficial, usando
`avaliar_modelo()` (R²/MAE/MSE/RMSE) -- diretamente comparável. Essa troca é
também metodologicamente melhor: resolve o viés de seleção que a v13
original já tinha identificado como limitação conhecida do método do
professor. Controle manual dos parâmetros ficou fora desta rodada (decisão
do Lucas). Exibição: tabela de hiperparâmetros + gráfico de coeficientes na
aba Treinamento; tabela de métricas (destacando o melhor R²) na aba
Avaliação, ao lado dos números da Múltipla oficial; e Ridge/Lasso/ElasticNet
ganharam linhas no leaderboard da Comparação (com **todas** as colunas
candidatas, não o melhor subconjunto -- a própria penalidade já seleciona
variável).

**Confirmado no navegador** (dataset "Mortalidade Infantil", X =
`Saneamento_pct`+`Agua_Potavel_pct`+`PIB_per_capita`, `test_size=0.3`,
`random_state=0`): "Sem regularização" bate exatamente com a Múltipla
oficial (R²=0.6973 nos dois lugares) e Ridge supera os dois (R²=0.7010) --
um ganho real e visível, exatamente o que o Lucas queria enxergar.

## Requisitos funcionais (v14 — Aba "Revisão da Prova" cresce para "Revisão da Prova Parcial")

Motivo: correção de rótulo importante (ver acima) revelou que
`gabarito_prova_T1_exercicio1/2.py` são o gabarito da **prova T1** (a
primeira avaliação formal do semestre), não uma correção de atividade. O
Lucas vai ter agora a **Prova Parcial** -- cumulativa desde a T1, com peso
maior, cobrindo tudo até aqui **mais** o que ainda for apresentado em aula
antes da prova. Pediu uma página de estudo, no mesmo espírito da aba
"🎓 Revisão da Prova" (criada na v11 para a T1), com foco em deixar claro o
entendimento do pipeline (é prova prática).

### Decisões validadas com o Lucas (via `AskUserQuestion`)

- **Extender vs. nova aba:** a aba "🎓 Revisão da Prova" já era desenhada
  como página cumulativa ("cobre só os tópicos já implementados") -- só
  faltava Polinomial automático e Regularização (adicionados depois da T1).
  Decisão: **fazer crescer a mesma aba** em vez de duplicar conteúdo numa
  aba nova -- uma só fonte de verdade, que continua servindo pra qualquer
  prova futura também.
- **Visão comparativa dos pipelines:** o Lucas quis uma seção nova,
  específica, comparando os 5 algoritmos lado a lado (o que muda/repete em
  cada etapa) -- além das seções por algoritmo que já existiam.

### Implementado

- Título/intro atualizados: "🎓 Revisão da Prova Parcial", explicando que é
  cumulativa (T1 + Polinomial automático + Regularização).
- **6️⃣ Regressão Polinomial -- grau automático (GridSearchCV):** script de
  cola + `ver_codigo(escolher_grau_polinomial_cv)`.
- **7️⃣ Regularização -- Ridge/Lasso/ElasticNet:** script de cola (escalona
  X, `GridSearchCV` pros 3 modelos) + `ver_codigo(treinar_regularizacao_cv)`.
- **🔀 Comparação lado a lado dos pipelines:** tabela nova
  (`TABELA_COMPARATIVA_PIPELINES`) cruzando 6 etapas (nº de variáveis,
  escalonamento, hiperparâmetro via CV, classe do scikit-learn, métrica de
  avaliação, suposições, pegadinha comum) x 5 algoritmos.
- Checklist geral do pipeline: passo 4b (escalonar X, só Regularização) e
  nota sobre `GridSearchCV` dentro do treino no passo 5.
- Tabela de gabarito (`REFERENCIA_DATASETS`): 2 linhas novas --
  `comissao.csv` (Simples x Polinomial grau 2) e Mortalidade Infantil com
  Regularização (R² 0.6961 → 0.7010 com Ridge).

### Critérios de aceite (v14)

- `pytest` continua passando (70 casos, sem mudança de lógica em `core.py`).
- Aba renderiza sem erro (`streamlit.testing.v1.AppTest`), com as 4 novas
  seções presentes (6️⃣, 7️⃣, comparação, gabarito atualizado).

## Perguntas de validação

- Resolvidas nesta rodada (via perguntas equivalentes de Inception antes da
  Construction): qual modelo adicionar (Múltipla), como expor na UI (seletor
  + aba de comparação), em qual cópia do projeto trabalhar (só a `_1`), e se
  seguir o fluxo AI-DLC completo (sim). Ver seção "Decisões validadas" acima.
- **Resolvidas na rodada de mudança de visão (v6):** renomear o projeto
  (sim, para "Laboratório Interativo de Machine Learning"), escopo dos
  próximos algoritmos (os 14 tópicos do syllabus, nesta ordem — ver
  "Roadmap do syllabus" acima) e se a modularização de `app.py` deveria ser
  decidida agora (não — só registrada como intenção, decisão adiada para o
  Inception do próximo algoritmo). Ver seção "Visão do projeto" acima.
- **Resolvidas na v7:** Passo 0 virou aba dedicada; `50_Startups.csv` fica
  restrito a `R&D Spend`/`Marketing Spend` (dummies de `State` ficam no CSV
  mas fora do multiselect); manifesto de `aulas/` registrado em
  `ai-dlc/aulas-log.md`.
- **Resolvida na v8:** opção A (melhor subconjunto por busca exaustiva) foi
  a escolhida -- ver "Decisão validada" acima. Nenhuma pergunta em aberto
  no momento.
- **Resolvidas na v9:** independência dos resíduos usa os dois testes
  (Ljung-Box + Durbin-Watson, com nota de divergência); colinearidade
  duplica o heatmap dentro da aba Diagnóstico. Ver "Decisões validadas
  (v9)" acima. Nenhuma pergunta em aberto no momento.
