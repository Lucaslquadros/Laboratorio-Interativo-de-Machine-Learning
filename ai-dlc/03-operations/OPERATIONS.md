# Operations — Laboratório Interativo de Machine Learning

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Como rodar os testes

```bash
pytest
```

## Como atualizar os datasets de ODS (opcional)

Os CSVs de `saude_desenvolvimento.csv` e
`mortalidade_infantil_desenvolvimento.csv` são snapshots estáticos (ano
2020) do World Bank Open Data. Para buscar um ano mais recente ou refazer
o download (precisa de internet, sem chave de API):

```bash
python scripts/baixar_dados_ods.py
```

## Como atualizar o dataset de classificação (opcional)

`diagnostico_cancer_mama.csv` vem do Breast Cancer Wisconsin (Diagnostic)
Data Set, bundlado no scikit-learn (não precisa de internet):

```bash
python scripts/baixar_dados_classificacao.py
```

## Como manter o app alinhado com o que é ensinado em aula

A pasta `aulas/` (fora do controle "oficial" do app) é a caixa de entrada do
Lucas: sempre que ele adiciona um slide, dataset ou script novo ali, a
próxima rodada de trabalho confere alinhamento contra `data/`, `core.py`,
`app.py` (aba Teoria/Passo 0) e o roadmap do syllabus, registrando o
resultado em `ai-dlc/aulas-log.md` (o que já foi revisado e o que foi feito
com cada arquivo). Isso evita reanalisar tudo do zero a cada rodada.

## Como verificar que está saudável

- [x] `pytest` passa sem falhas (70 casos em `tests/test_core.py`)
- [x] Dataset "💰 Comissão x Quantidade Vendida (aula)" (`comissao.csv`)
      aparece no seletor; modo Polinomial automático converge para grau 2
      com R²≈1.0
- [x] Aba "🎓 Revisão da Prova Parcial" renderiza sem erro com as seções
      6️⃣ (Polinomial automático), 7️⃣ (Regularização) e 🔀 (comparação de
      pipelines) — testado com `streamlit.testing.v1.AppTest`
- [x] Modo Polinomial: opção "Automático (GridSearchCV)" escolhe o grau por
      validação cruzada (igual ao método da Aula6) e mostra a tabela de
      scores por grau; opção "Manual" continua funcionando como antes
      (testado com `streamlit.testing.v1.AppTest`, ambos os modos)
- [x] Modo Múltipla: toggle "Regularizar?" -> "Comparar Ridge / Lasso /
      ElasticNet" mostra, na Treinamento, hiperparâmetros + gráfico de
      coeficientes, e na Avaliação, tabela de R²/MAE/MSE/RMSE dos 4 modelos
      (mesmo split treino/teste da Múltipla oficial) sem erro; no dataset
      "👶 Mortalidade Infantil & Desenvolvimento"
      (`Saneamento_pct`+`Agua_Potavel_pct`+`PIB_per_capita`), "Sem
      regularização" bate exatamente com a Múltipla oficial (R²=0.6973) e
      Ridge supera os dois (R²=0.7010) -- confirmado manualmente
- [x] Leaderboard da Comparação inclui Ridge/Lasso/ElasticNet **independente**
      do toggle da sidebar (mesmo padrão de Simples/Múltipla/Polinomial),
      usando todas as colunas candidatas a X
- [x] `avaliar_modelo()` devolve RMSE além de R²/MAE/MSE; aparece na aba
      Avaliação (4ª métrica) e no leaderboard da Comparação
- [x] Datasets "📈 Mercado Financeiro (correção)" (`finance_market.csv`,
      R²≈0.9987), "🌾 Produtividade Agrícola (correção)" (`agro_tech.csv`,
      Simples R²≈0.4847 → Múltipla R²≈0.8705) e "🏥 Gastos com Plano de
      Saúde (correção)" (`base_plano_saude_preparada.csv`, R²≈0.18 --
      exemplo intencional de mau ajuste) aparecem no seletor e treinam sem
      erro nos 3 modos de regressão (testado com `streamlit.testing.v1.AppTest`)
- [x] `data/` é a única cópia de cada dataset -- nenhum CSV duplicado resta
      em `aulas/` ou `aulas/correcoes/`
- [x] Aba Diagnóstico dos Resíduos: teste de Goldfeld-Quandt (homocedasticidade),
      testes de Ljung-Box + Durbin-Watson (independência, com nota de
      divergência) e heatmap de colinearidade (variáveis X escolhidas)
      renderizam sem erro nos modos Simples, Múltipla e Polinomial --
      testado no navegador com o dataset "📢 Publicidade x Vendas"
- [x] Dataset "📢 Publicidade x Vendas (aula)" (`publicidade.csv`) aparece
      no seletor; modo Múltipla com TV+Rádio+Jornal reproduz R²≈0.8649, e o
      leaderboard/melhor subconjunto descarta Jornal e reproduz R²≈0.8657
- [x] Aba Comparação → leaderboard: a linha "Múltipla" testa todas as
      combinações de variáveis candidatas e usa a de maior R² (não mais
      "todas as colunas"); no dataset Startups escolhe `R&D Spend` +
      `Marketing Spend` (R²≈0.9431), descartando `Administration`, em vez
      de usar as 3 e cair para R²≈0.9355
- [x] App abre sem traceback nas 9 abas do lado Regressão (Passo 0, Teoria,
      Dados & Correlação, Treino/Teste, Treinamento, Avaliação, Diagnóstico
      dos Resíduos, Previsão, Comparação) e nas 7 abas do lado Classificação
      (Passo 0 + Teoria + as mesmas, sem Diagnóstico de Resíduos nem
      Comparação)
- [x] Aba "🧭 Passo 0" responde corretamente ao quiz (testado com
      `streamlit.testing.v1.AppTest`): alvo não rotulado → recomenda
      Não Supervisionado; alvo contínuo → recomenda Regressão; alvo
      categórico binário/multiclasse → recomenda Classificação
- [x] Dataset "🚀 Lucro de Startups (aula)" (`50_Startups.csv`, sem as
      colunas dummy de Estado) aparece no seletor, e o modo Múltipla sugere
      por padrão `R&D Spend` + `Marketing Spend` (maior correlação com
      `Profit`), reproduzindo R²≈0.9431 do script de aula
- [x] Com o dataset "Salário x Experiência", R² fica ≈ 0.8943 (mesmo valor
      do `exemplo1_regressaolinearsimples.py`)
- [x] Com o dataset "Preço de Casas", escolhendo `Avg. Area Income` como X,
      R² fica ≈ 0.4247 (mesmo valor do `exemplo2_regressaolinearsimples.py`)
- [ ] Com o exemplo do slide (Distância x Consumo), a equação fica
      y = 0.5716 + 0.0663·X, R² ≈ 0.898 -- *só bate exatamente treinando com
      TODAS as 10 linhas (é o que `tests/test_core.py` faz); no app, com
      `test_size` padrão (0.3) e só 10 linhas no dataset, o split
      treino/teste deixa poucas linhas de cada lado e os coeficientes saem
      um pouco diferentes -- comportamento esperado, não é bug.*
- [x] Modo Múltipla treina sem erro no dataset "Preço de Casas" (2+ colunas
      numéricas) e mostra R² maior que o melhor modelo Simples equivalente
      (ex.: R²≈0.625 com 2 variáveis vs. R²≈0.4247 com 1)
- [x] Aba Comparação mostra os dois modelos (ativo + alternativo) lado a
      lado, nos dois sentidos (Simples→Múltipla e Múltipla→Simples)
- [x] Upload de um CSV com poucas linhas, colunas constantes, ou separador
      `;` não derruba o app — aparece mensagem de erro clara
- [x] Dataset "Saúde & Desenvolvimento" carrega e treina sem erro nos dois
      modos (R² ≈ 0.71 múltipla vs. 0.47 simples, com `random_state=0`)
- [x] Dataset "Mortalidade Infantil & Desenvolvimento" carrega e treina
      sem erro nos dois modos (R² ≈ 0.70 múltipla vs. 0.69 simples --
      ganho pequeno de propósito, ver nota de colinearidade na descrição
      do dataset)
- [x] Aba Diagnóstico dos Resíduos funciona nos modos Simples, Múltipla e
      Polinomial, inclusive no caso limite de só 3 resíduos no teste
      (dataset Distância x Consumo)
- [x] Modo Polinomial treina sem erro em graus 2-5, com curva ajustada
      visível na aba Avaliação e Previsão funcionando sem alteração no
      código (o `Pipeline` já trata a transformação internamente)
- [x] Aba Comparação (leaderboard) ranqueia corretamente os modelos
      disponíveis e trata modelos não-viáveis (ex.: Múltipla com só 1
      coluna candidata) sem quebrar a aba inteira
- [x] "Tipo de Tarefa" = Classificação troca a lista de datasets, esconde
      "Tipo de Regressão" e reduz as abas para 6 (sem Diagnóstico de
      Resíduos nem Comparação); dataset "Diagnóstico de Câncer de Mama"
      treina com acurácia ~90-92%, matriz de confusão e curva ROC/AUC
      corretas, previsão de classe + probabilidade funcionando
- [x] Alternar entre Regressão e Classificação e voltar não quebra nada
      (testado manualmente no navegador)

## Changelog

- **2026-09-16** — v14 (Rótulo corrigido + "Revisão da Prova" cresce para
  "Revisão da Prova Parcial", Bolts 60-61 do AI-DLC): o Lucas apontou que
  `exercicio1/2_correcao_AC1.py` (Aula5) não são correção da AC1 dele (que é
  sobre preparação de dados: imputação, outliers, dummies, escalonamento) --
  são o **gabarito da prova prática T1** (sala T1, canal V5), a primeira
  avaliação formal do semestre. Arquivos renomeados para
  `gabarito_prova_T1_exercicio1/2.py`; rótulo corrigido em `aulas-log.md` e
  `INCEPTION.md`. A próxima avaliação é a **Prova Parcial** -- cumulativa
  desde a T1, com peso maior -- então a aba "🎓 Revisão da Prova" (criada na
  v11 para a T1) foi **atualizada** (não duplicada) com os 2 tópicos que
  faltavam: Regressão Polinomial com grau automático (`GridSearchCV`) e
  Regularização (Ridge/Lasso/ElasticNet). Ganhou também uma seção nova
  comparando os 5 pipelines lado a lado (nº de variáveis, escalonamento,
  hiperparâmetro via CV, classe do scikit-learn, métricas, suposições,
  pegadinha comum) -- pedido do Lucas por ser prova prática. `pytest`
  continua em 70 casos (nenhuma mudança de lógica em `core.py`).
- **2026-09-16** — v13.2 (Dataset `comissao.xlsx` integrado, Bolt 59 do
  AI-DLC): o Lucas perguntou se havia bases de aula fora do app -- 4
  encontradas (`comissao.xlsx`, `brazil_covid19.csv`, `diabetes*.csv`,
  datasets da AC1), das quais só `comissao.xlsx` era candidata de baixo
  esforço para integrar agora. Convertida para `data/comissao.csv` (50
  linhas, `quantidade`→`comissao`, dataset construído como parábola exata):
  grau 2 dá R²≈1.0, e o modo Simples (grau 1) reproduz o ponto pedagógico do
  script -- intercepto negativo, previsão de comissão negativa para
  quantidades baixas. Novo dataset "💰 Comissão x Quantidade Vendida (aula)"
  no seletor. Suíte `pytest` ampliada para 70 casos.
- **2026-09-16** — v13.1 (Revisão pós-checkpoint da Regularização, Bolts
  56-58 do AI-DLC): o Lucas testou a v13 e trouxe 3 observações -- só
  aparecia na Treinamento, sem controle manual de parâmetros, e (o
  problema de fundo) a comparação por CV pura no dataset inteiro não dava
  pra confrontar de forma justa com a Múltipla oficial (métricas
  diferentes, partições diferentes, incluindo linhas de teste na CV).
  **Convenção revista:** `GridSearchCV` passa a rodar só dentro de
  `X_treinamento` (mesmo padrão do Gap 1/Polinomial), os 4 modelos avaliados
  no mesmo `X_teste` com R²/MAE/MSE/RMSE -- diretamente comparável, e
  também resolve o viés de seleção que já tinha sido identificado como
  limitação do método CV-puro do professor. Controle manual dos parâmetros
  ficou fora desta rodada. UI: tabela de métricas + destaque do melhor R²
  também na aba Avaliação; linhas Ridge/Lasso/ElasticNet no leaderboard da
  Comparação, **independente do toggle** (mesmo padrão de Simples/Múltipla/
  Polinomial). Confirmado no dataset "Mortalidade Infantil &
  Desenvolvimento": "Sem regularização" bate exatamente com a Múltipla
  oficial (R²=0.6973) e Ridge supera os dois (R²=0.7010) -- ganho real e
  visível. Suíte `pytest` ampliada para 69 casos.
- **2026-09-16** — v13 (Regularização: Ridge/Lasso/ElasticNet, item 6b do
  roadmap, Bolts 53-55 do AI-DLC): Inception própria pedida pelo Lucas após
  fechar o Gap 1. Decisões: (1) regularização é um modificador, não um
  algoritmo à parte -- em tese aplicável a Múltipla, Polinomial e até
  Logística, mas **só** os scripts de Múltipla (`exemplo_regularizacao.py`,
  `correcao_Hitters.py`) foram apresentados em aula, então o escopo desta
  rodada ficou restrito a **Múltipla + Regularização** (mesma regra de
  priorização da v12); (2) toggle **dentro** do modo Múltipla existente, não
  um "Tipo de Regressão" à parte; (3) convenção de avaliação **igual ao
  script** -- sem `train_test_split`, cada modelo (Ridge/Lasso/ElasticNet)
  avaliado por `GridSearchCV`/`cross_val_score` (`cv=10`,
  `neg_mean_squared_error`) no dataset inteiro (decisão consciente, discutida
  o porquê o k-fold já faz o papel de treino/teste repetido, com a ressalva
  de que comparar o "vencedor" pelo mesmo score usado pra escolher o
  hiperparâmetro tem um viés de seleção sutil que uma CV aninhada evitaria);
  (4) os 3 modelos mostrados juntos, comparados (tabela + gráfico de
  coeficientes), igual ao script; (5) `diabetes.csv` (dataset original do
  script) **não** foi integrado -- seu escalonamento não-padrão complicaria
  demais a aba Previsão -- em vez disso, usa-se
  `mortalidade_infantil_desenvolvimento.csv` (já em `data/`), que tem
  colinearidade real entre X (`Saneamento_pct`/`Agua_Potavel_pct`, r≈0.906,
  quase idêntica ao par `s1`/`s2` do diabetes) e reproduz a mesma lição.
  Implementação puramente aditiva: `treinar_regularizacao_cv()` novo em
  `core.py`; as abas Avaliação/Previsão/Diagnóstico/Comparação não mudaram.
  Suíte `pytest` ampliada para 68 casos.
- **2026-09-16** — v12 (Varredura completa de `Aula3`-`Aula7`/`ativ1` +
  reprioridade do backlog + Gap 1 do Polinomial, Bolts 50-52 do AI-DLC): o
  Lucas pediu uma varredura de todas as pastas de aula do diretório `ML/`
  contra o que já estava em `aulas-log.md`. Achados: `Aula3`/`Aula4` 100%
  cobertos; `Aula5/correcao_AC1` (2 scripts), `Aula6` (Regressão Polinomial)
  e `Aula7` (Regularização) nunca revisados; `ativ1` (entrega própria da AC1)
  fora do fluxo. Todos os arquivos foram copiados para `aulas/` e revisados.
  **Nova regra de priorização:** conteúdo já apresentado em aula (mesmo fora
  dos 14 tópicos originais do syllabus) passa à frente de tópicos ainda não
  apresentados -- por isso **Regularização (Ridge/Lasso/ElasticNet)** vira
  item **6b** do roadmap, à frente de Árvores de Decisão (item 7) e
  seguintes; sua própria Inception começa numa rodada futura. Fechado nesta
  rodada: **Gap 1 -- Regressão Polinomial**, achado ao ler
  `exemplo_regr_polinomial.py`/`correcao_polinomial_p1/p2.py` (Aula6): o app
  deixava escolher o grau só manualmente (slider), enquanto o método real de
  aula sempre escolhe o grau via `GridSearchCV` (Pipeline
  `PolynomialFeatures`+`LinearRegression`, `cv=5`,
  `neg_mean_squared_error`, graus 1-6). Implementado como **opção extra**
  (decisão do Lucas), não substituição: `escolher_grau_polinomial_cv()` nova
  em `core.py`, radio "Manual"/"Automático" na sidebar do modo Polinomial, e
  uma caixa na aba Treinamento com o grau escolhido + tabela de scores por
  grau. `ac1_lucas/` (entrega própria) fica só como registro em `aulas/`,
  sem virar conteúdo do app. 5 scripts de correção (`exercicio1/2_correcao_AC1`,
  `correcao_polinomial_p1/p2`, `correcao_Hitters`) referenciam datasets que
  **não existem em lugar nenhum do diretório `ML/`** (confirmado por busca
  ampla, inclusive dentro dos `.zip`) -- ficam só como referência de
  metodologia. Suíte `pytest` ampliada para 65 casos.
- **2026-08-26** — v11 (Aba "🎓 Revisão da Prova", Bolt 49 do AI-DLC,
  Inception abreviada por causa de prova em 2026-08-27): nova aba estática,
  primeira da lista nos dois Tipos de Tarefa (vira a aba padrão ao abrir o
  app). Reúne, para os tópicos já implementados (Simples, Múltipla,
  Polinomial, Diagnóstico de Resíduos, Regressão Logística): um checklist
  geral do pipeline, o recap do Passo 0, um "script de cola" por algoritmo
  (estilo direto dos exercícios do professor, pronto para adaptar numa
  prova prática) e o código real de `core.py` via `inspect.getsource` para
  quem quiser conferir a implementação "de produção". Fecha com uma tabela
  de valores de referência (gabarito) de todos os datasets de aula.
- **2026-08-26** — v10 (Ingestão de `aulas/correcoes/`: RMSE + 3 datasets
  novos + organização de `data/`, Bolts 44-48 do AI-DLC): o Lucas adicionou
  `aulas/correcoes/` com 3 pastas de correção do professor (script +
  dataset cada). `correcao_regsimples.py` expôs uma métrica que o app não
  tinha -- **RMSE** foi adicionada a `avaliar_modelo()` (`core.py`) e
  passou a aparecer na aba Avaliação e no leaderboard da Comparação. Os 3
  datasets novos foram integrados: "📈 Mercado Financeiro (correção)"
  (`finance_market.csv`, R²≈0.9987), "🌾 Produtividade Agrícola (correção)"
  (`agro_tech.csv`, Simples R²≈0.4847 → Múltipla R²≈0.8705, batendo com os
  scripts de aula) e "🏥 Gastos com Plano de Saúde (correção)"
  (`base_plano_saude_preparada.csv`) -- este último é o primeiro dataset do
  app onde o próprio script do professor conclui que o modelo **não**
  satisfaz as suposições da regressão linear (heterocedasticidade e
  resíduos não-normais), então foi mantido como exemplo intencional de mau
  ajuste na aba Diagnóstico, com todas as colunas (incluindo as dummies de
  região) como candidatas a X, replicando o script sem ressalva. Além
  disso, `data/` virou a única fonte de verdade para datasets: os 3 CSVs
  novos foram movidos (não copiados) de `aulas/correcoes/` para `data/`, e
  os duplicados que já existiam em `aulas/` (raiz) --
  `base_salarios.csv`/`USA_Housing.csv`/`publicidade.csv` (bit-idênticos) e
  `50_Startups.csv` (superado pela versão sem dummies desde a v7) -- foram
  removidos por serem redundantes (tudo sob `git`, reversível). Suíte
  `pytest` ampliada para 62 casos.
- **2026-08-20** — v9 (Ingestão de `aulas/`: teste de suposições + dataset
  Publicidade, Bolts 38-43 do AI-DLC): o Lucas adicionou
  `exemplo_teste_suposicao.py` e `publicidade.csv` em `aulas/`; a checagem
  de alinhamento (processo da v7) encontrou 3 lacunas reais na aba
  "🩺 Diagnóstico dos Resíduos" (tópico #4 do syllabus), que antes só cobria
  média dos resíduos e normalidade (Shapiro-Wilk). Novo dataset "📢
  Publicidade x Vendas (aula)" (`publicidade.csv`, 200 mercados, TV/Rádio/
  Jornal → Vendas) integrado a `data/` e ao seletor. `core.py` ganhou
  `testar_homocedasticidade_residuos()` (teste de Goldfeld-Quandt) e
  `testar_independencia_residuos()` (Ljung-Box + Durbin-Watson) --
  `statsmodels` adicionado a `requirements.txt`. A aba Diagnóstico ganhou 3
  seções novas: teste formal de homocedasticidade (complementa o gráfico já
  existente), independência dos resíduos com os dois testes lado a lado
  (decisão de Inception: mostrar os dois, com nota explicando que podem
  discordar -- confirmado na prática com o dataset de Publicidade em modo
  Simples) e heatmap de colinearidade das variáveis X escolhidas (decisão
  de Inception: duplicar o heatmap já existente em "Dados & Correlação",
  para reunir todas as suposições num só lugar). Suíte `pytest` ampliada
  para 57 casos. `ai-dlc/aulas-log.md` atualizado com os 2 arquivos novos.
- **2026-08-19** — v8 (Leaderboard: Múltipla busca o melhor subconjunto de
  variáveis, Bolts 36-37 do AI-DLC): corrigido bug de design reportado pelo
  Lucas (screenshot ao vivo do app) -- a linha "Múltipla" da aba Comparação
  usava sempre todas as colunas numéricas candidatas, então no dataset
  Startups incluía `Administration` (baixa correlação com `Profit`) e caía
  para R²≈0.9355, pior do que a combinação de 2 variáveis (`R&D Spend` +
  `Marketing Spend`, R²≈0.9431) já sugerida por padrão na aba Múltipla
  interativa. Nova `selecionar_melhor_subconjunto_multipla()` em `core.py`
  testa todas as combinações de 2..N colunas e usa a de maior R² de teste;
  "Detalhes" no leaderboard agora lista os nomes das variáveis vencedoras.
  Decisão de Inception (opção "melhor subconjunto" entre 3 alternativas)
  registrada em `01-inception/INCEPTION.md`. Suíte `pytest` ampliada para
  50 casos.
- **2026-08-19** — v7 (Fluxo de ingestão de `aulas/` + Passo 0, Bolts 30-35
  do AI-DLC): nova pasta `aulas/` vira a caixa de entrada oficial de
  material de aula (slides, datasets, scripts), com rastreio em
  `ai-dlc/aulas-log.md`. Nova aba "🧭 Passo 0" (primeira da lista, nos dois
  Tipos de Tarefa) ensina a taxonomia do professor (Supervisionado ->
  Regressão/Classificação binária-multiclasse; Não Supervisionado ->
  Associação/Clusterização; extraída de `Intro_AprendMaquina.pdf`), unifica
  o vocabulário X/y (atributos previsores = variáveis independentes;
  atributo-alvo = variável dependente), destaca a pegadinha da Regressão
  Logística ("é classificação, apesar do nome") e termina com um quiz
  interativo (`classificar_tipo_problema()` em `core.py`) que recomenda o
  Tipo de Tarefa certo. Novo dataset "🚀 Lucro de Startups (aula)"
  (`50_Startups.csv`, sem as colunas dummy de Estado, por decisão de
  Inception) reproduz o R²≈0.9431 do script `exemplo1_regressaolinearmultipla.py`.
  Suíte `pytest` ampliada para 48 casos.
- **2026-08-18** — v6 (mudança de visão, documentação apenas — sem novo
  algoritmo neste ciclo): projeto renomeado de "Laboratório de Regressão
  Linear Simples" para "Laboratório Interativo de Machine Learning"
  (`README.md`, `app.py`, `ai-dlc/`). Visão ampliada: em vez de fechar no
  syllabus de regressão, o app passa a ter como meta cobrir os 14 tópicos
  do syllabus completo da disciplina (Árvores de Decisão, KNN, Random
  Forest, SVM, Viés/Variância, Validação de Modelos, Métricas, Clustering),
  um de cada vez, com Inception próprio para cada novo algoritmo. Roadmap
  detalhado em `01-inception/INCEPTION.md` e `README.md`. Decisão anterior
  que excluía Árvore de Decisão/Random Forest/KNN/Ridge-Lasso "deste ciclo"
  foi revogada. Nenhuma mudança de código funcional nesta rodada além dos
  títulos/branding.
- **2026-08-18** — v5 (Diagnóstico de Resíduos + Regressão Polinomial +
  Regressão Logística, Fases A/B/C do AI-DLC, completa). Fase A (Bolts
  16-18): `diagnosticar_residuos` e `testar_normalidade_residuos` em
  `core.py`; nova aba "🩺 Diagnóstico dos Resíduos". `scipy` adicionado ao
  `requirements.txt`. Fase B (Bolts 19-23):
  `treinar_modelo_regressao_polinomial` em `core.py`; 3ª opção
  "Polinomial" no seletor de Tipo de Regressão com grau configurável (2-5);
  abas Treinamento/Avaliação/Previsão adaptadas (tabela de coeficientes por
  potência, curva ajustada); aba "🆚 Comparação" reescrita como leaderboard
  que treina Simples, Múltipla e Polinomial e ranqueia por R². Fase C
  (Bolts 24-29): novo dataset `diagnostico_cancer_mama.csv` (Breast Cancer
  Wisconsin, UCI/scikit-learn, ODS 3) via
  `scripts/baixar_dados_classificacao.py`; `validar_dados_para_classificacao`,
  `treinar_modelo_regressao_logistica`, `avaliar_modelo_classificacao`,
  `prever_classe_novo_valor` em `core.py`; novo seletor "Tipo de Tarefa"
  (Regressão/Classificação) na sidebar, que troca dataset, esconde
  controles irrelevantes e adapta as abas (log-odds na Treinamento, matriz
  de confusão + ROC/AUC na Avaliação, classe + probabilidade na Previsão).
  Teoria ganhou seções sobre Regressão Polinomial, Diagnóstico de Resíduos
  e Classificação/Regressão Logística (pendências das Fases A/B). Suíte
  `pytest` com 43 casos.
- **2026-08-17** — v4 (Datasets de ODS via API pública, Bolts 13-15 do
  AI-DLC): `scripts/baixar_dados_ods.py` baixa indicadores do World Bank
  Open Data e monta 2 novos datasets embutidos --
  `saude_desenvolvimento.csv` (Expectativa de vida, ODS 3/7/8, 183 países)
  e `mortalidade_infantil_desenvolvimento.csv` (Mortalidade < 5 anos, ODS
  3/6/8, 180 países) -- ambos sem NaN e testados para R² estável entre
  splits antes de fixar no repo.
- **2026-08-17** — v3 (Regressão Linear Múltipla, Bolts 5-12 do AI-DLC):
  seletor "Tipo de Regressão" (Simples/Múltipla) no painel lateral;
  `validar_dados_para_regressao_multipla`, `treinar_modelo_regressao_multipla`
  e `prever_novo_valor_multiplo` em `core.py`; abas Treinamento, Avaliação e
  Previsão adaptadas para os dois modos (tabela de coeficientes, gráfico
  Previsto vs. Real, um input por variável); nova aba "🆚 Comparação" que
  treina o modelo alternativo só para comparar R²/MAE/MSE; suíte `pytest`
  ampliada para 28 casos.
- **2026-08-18** — v2 (robustez / Bolts 1-4 do AI-DLC): lógica extraída
  para `core.py`; validação de dados com mensagens amigáveis (linhas
  insuficientes, coluna constante, test_size inválido, NaN); CSV tolerante
  a `,`/`;` com cache via `st.cache_data`; suíte `pytest` com 21 casos.
- **2026-08-17** — v1: primeira versão do laboratório interativo (6 abas,
  3 datasets + upload, expanders "ver o código").

## Próximos passos / dívidas conhecidas

- Sem testes de interface (só do núcleo `core.py`) — se crescer, considerar
  `streamlit.testing` para testar a UI também.
- Sem CI configurado (rodar `pytest` continua manual).
- Datasets embutidos (`data/*.csv`) ficam junto do código -- ok para uso
  solo, mas se o projeto crescer vale considerar não versionar CSVs grandes.
- Modo Múltipla não tem cálculo "na mão" (só o scikit-learn) -- a fórmula
  fechada matricial ficou fora de escopo desta rodada; se fizer sentido
  pedagogicamente, dá para adicionar depois em `core.py`.
- `app.py` cresceu bastante (de ~250 para ~1300 linhas) com as Fases A/B/C
  -- ainda organizado em seções claras, mas se crescer mais vale considerar
  dividir em módulos (ex.: um arquivo por "PASSO" ou por tarefa
  Regressão/Classificação). **Com o roadmap de v6** (mais 8 tópicos do
  syllabus a caminho -- Árvores de Decisão, KNN, Random Forest, SVM,
  Clustering, entre outros), essa divisão deixa de ser só "se crescer mais"
  e vira algo a decidir explicitamente no Inception do próximo algoritmo
  novo (ver `01-inception/INCEPTION.md`), antes de continuar empilhando
  branches por modo no mesmo `app.py`.
- Classificação só tem 1 algoritmo (Regressão Logística) e 1 dataset --
  sem leaderboard nem diagnóstico próprios ainda; ficaria natural adicionar
  mais adiante, seguindo o mesmo padrão do lado Regressão.
