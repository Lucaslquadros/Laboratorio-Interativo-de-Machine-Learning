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

- [x] `pytest` passa sem falhas (50 casos em `tests/test_core.py`)
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
