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
| 5 | Regressão Polinomial | ✅ v5 (Fase B) |
| 6 | Regressão Logística | ✅ v5 (Fase C) |
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
