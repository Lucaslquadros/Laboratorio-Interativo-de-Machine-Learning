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
