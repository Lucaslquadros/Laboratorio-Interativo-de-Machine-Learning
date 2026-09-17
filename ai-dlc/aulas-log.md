# Log de ingestão de `aulas/`

> Registro de quais arquivos da pasta `aulas/` (fora do controle de versão
> "oficial" do app, é a caixa de entrada do Lucas) já foram revisados e o
> que foi feito com cada um. Objetivo: quando um arquivo novo aparecer em
> `aulas/`, dá pra rodar `diff` mental contra esta tabela em vez de
> reanalisar tudo do zero. Atualizado a cada rodada de Inception que mexer
> em `aulas/`.

| Arquivo | Tipo | Status | Ação tomada | Data |
|---|---|---|---|---|
| `Intro_AprendMaquina.pdf` | Slide | Revisado | Fonte da taxonomia do Passo 0 (INCEPTION.md v7) | 2026-08-19 |
| `RegressaoLinearSimples_ML.pdf` | Slide | Revisado | Já refletido na aba Teoria (v1); nenhuma ação nova | 2026-08-19 |
| `RegressaoLinearMultipla_ML.pdf` | Slide | Revisado | Lido na íntegra; sem conceito novo além do já implementado em `core.py`/`app.py` (ver riscos v7 no INCEPTION.md — nenhuma menção a dummy encoding, multicolinearidade, backward elimination, adjusted R² ou pressupostos) | 2026-08-19 |
| `exemplo1_regressaolinearsimples.py` | Código | Revisado | Já referenciado no README/`core.py` desde v1; nenhuma ação nova | 2026-08-19 |
| `exemplo2_regressaolinearsimples.py` | Código | Revisado | Já referenciado no README/`core.py` desde v1; nenhuma ação nova | 2026-08-19 |
| `exemplo1_regressaolinearmultipla.py` | Código | Revisado | Confirma metodologia de `treinar_modelo_regressao_multipla` (mesmo `LinearRegression`, split 70/30); usado como referência de R² para `50_Startups.csv` | 2026-08-19 |
| `exemplo2_regressaolinearmultipla.py` | Código | Revisado | Confirma metodologia sobre `USA_Housing.csv`; nenhuma ação nova (dataset e função já existem) | 2026-08-19 |
| `base_salarios.csv` | Dataset | Revisado | Idêntico (`diff` bit-a-bit) a `data/base_salarios.csv`; nenhuma ação | 2026-08-19 |
| `USA_Housing.csv` | Dataset | Revisado | Idêntico (`diff` bit-a-bit) a `data/USA_Housing.csv`; nenhuma ação | 2026-08-19 |
| `50_Startups.csv` | Dataset | Integrado | Copiado para `data/` sem as colunas dummy de Estado; entrada nova em `DATASETS` (`app.py`), R²≈0.9431 confirmado por teste (Bolts 31-32) | 2026-08-19 |
| `publicidade.csv` | Dataset | Integrado | Copiado para `data/`; entrada nova em `DATASETS` (`app.py`); R²≈0.8649 (3 vars) / 0.8657 (melhor subconjunto TV+Rádio) confirmado por teste (Bolts 38-39) | 2026-08-20 |
| `exemplo_teste_suposicao.py` | Código | Integrado | Expôs 3 lacunas na aba Diagnóstico (homocedasticidade formal, independência dos resíduos, colinearidade) — implementadas em `core.py`/`app.py` (Bolts 40-42); linearidade e média dos resíduos já estavam cobertas | 2026-08-20 |
| `correcao_regsimples.py` | Código (correção) | Integrado | Expôs lacuna de métrica: RMSE, adicionada em `avaliar_modelo()` (`core.py`) e propagada para `app.py`/leaderboard (Bolt 45) | 2026-08-26 |
| `finance_market.csv` | Dataset (correção) | Integrado | Movido de `aulas/correcoes/` para `data/`; entrada nova em `DATASETS` (`app.py`); R²≈0.9987 confirmado por teste (Bolts 44, 46-47) | 2026-08-26 |
| `regressao_multipla.py`, `regressao_simples.py` | Código (correção) | Revisado | Confirmam metodologia já implementada (heatmap, R²/MAE/MSE, reta); usados como referência de R² para `agro_tech.csv` | 2026-08-26 |
| `agro_tech.csv` | Dataset (correção) | Integrado | Movido de `aulas/correcoes/` para `data/`; entrada nova em `DATASETS` (`app.py`); Simples R²≈0.4847 → Múltipla R²≈0.8705 confirmado por teste (Bolts 44, 46-47) | 2026-08-26 |
| `correcao_teste_suposicao.py` | Código (correção) | Revisado | Mesmas 6 suposições já implementadas na v9 (nenhuma técnica nova); único script cuja conclusão é que o modelo **não** satisfaz as suposições | 2026-08-26 |
| `base_plano_saude_preparada.csv` | Dataset (correção) | Integrado | Movido de `aulas/correcoes/` para `data/`; entrada nova em `DATASETS` (`app.py`) como exemplo intencional de mau ajuste (R²≈0.18); Bolts 44, 46-47 | 2026-08-26 |
| `base_salarios.csv`, `USA_Housing.csv`, `publicidade.csv` (cópias em `aulas/`) | Dataset | Removido | Bit-idênticos aos já presentes em `data/` — removidos de `aulas/` para eliminar duplicidade (Bolt 44); `data/` passa a ser a única fonte de verdade | 2026-08-26 |
| `50_Startups.csv` (cópia em `aulas/`, com dummies de Estado) | Dataset | Removido | Superado pela versão em `data/` (sem dummies) desde a v7 — removido de `aulas/` (Bolt 44) | 2026-08-26 |
| `correcoes/gabarito_prova_T1_exercicio1.py` (Aula5) | Código (**gabarito de prova**, não correção de atividade) | Revisado | **Rótulo corrigido (2026-09-16, feedback do Lucas):** é o gabarito da prova prática T1 (sala T1, canal V5) — a primeira avaliação formal do semestre, não a AC1 (que é de preparação de dados, ver `ac1_lucas/`). Mesmas 6 suposições da v9, sem técnica nova; dataset `Ecommerce Customers.csv` ausente, não reproduzível | 2026-09-16 |
| `correcoes/gabarito_prova_T1_exercicio2.py` (Aula5) | Código (**gabarito de prova**) | Revisado | Mesmo roteiro do exercicio1, mesma prova T1; dataset `weatherHistory.csv` ausente | 2026-09-16 |
| `exemplo_regr_polinomial.py` (Aula6) | Código | Integrado (Gap 1 aberto) | Expõe gap: grau do polinômio escolhido via `GridSearchCV` (CV, grau 1-6), não manualmente como no app hoje — Inception v12, aguardando decisão (Bolt a definir) | 2026-09-16 |
| `correcoes/correcao_polinomial_p1.py`, `correcoes/correcao_polinomial_p2.py` (Aula6) | Código (correção) | Revisado | Confirmam o padrão `GridSearchCV` para grau; datasets (`Ice Cream.csv` e outro) ausentes | 2026-09-16 |
| `exemplo_polinomial.py` (Aula6) | Código | Revisado | Confirma o padrão `GridSearchCV`/`Pipeline` para grau (já registrado no Gap 1) | 2026-09-16 |
| `comissao.xlsx` (Aula6) | Dataset | Integrado | Convertido para `data/comissao.csv` (mesmo padrão CSV dos outros datasets); entrada nova em `DATASETS` (`app.py`); GridSearchCV confirma grau=2, R²≈1.0 (dataset construído como parábola exata); removido de `aulas/` após integrar | 2026-09-16 |
| `brazil_covid19.csv` (Aula6) | Dataset | Revisado | Dataset de série temporal do `exemplo_regr_polinomial.py`; não integrado a `data/` ainda (aguarda decisão do Gap 1) | 2026-09-16 |
| `RegressaoPolinomial_20252.pdf` (Aula6) | Slide | Revisado | Fonte teórica para quando o Gap 1 virar bolt | 2026-09-16 |
| `exemplo_regularizacao.py` (Aula7) | Código | Integrado (Gap 2 aberto — item 6b do roadmap) | Ridge/Lasso/ElasticNet via `GridSearchCV` + `StandardScaler`, algoritmo totalmente novo para o app — Inception v12, aguardando decisão | 2026-09-16 |
| `correcoes/correcao_Hitters.py` (Aula7) | Código (correção) | Revisado | Confirma metodologia de Regularização; dataset `Hitters.csv` ausente | 2026-09-16 |
| `diabetes.csv`, `diabetes_nao_escalonado.csv` (Aula7) | Dataset | Revisado | Datasets do exemplo de Regularização; não integrados a `data/` ainda (aguarda decisão do Gap 2) | 2026-09-16 |
| `Regularizacao_20252.pdf` (Aula7) | Slide | Revisado | Fonte teórica para quando o Gap 2 virar bolt | 2026-09-16 |
| `ac1_lucas/AC1_Codigos_LucasQuadros_PacientesCovid.py`, `AC1_Codigos_LucasQuadros_PlanoSaude.py` + datasets (`ativ1`) | Código/Dataset (entrega própria) | Revisado | Entrega da AC1 do próprio Lucas, não é material do professor — aguardando decisão se vira conteúdo do app (pergunta de validação v12) | 2026-09-16 |
