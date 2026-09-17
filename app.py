# -*- coding: utf-8 -*-
"""
Laboratório Interativo de Machine Learning
====================================================

Site interativo (Streamlit) construído a partir do material da disciplina de
Machine Learning (Prof. Dr. Daniel Trevisan Bravo). Nasceu como um laboratório
só de Regressão Linear Simples e foi crescendo junto com o syllabus da
disciplina -- a meta agora é cobrir, ao longo do semestre, todos os 14 tópicos
do programa (ver roadmap em `ai-dlc/01-inception/INCEPTION.md`), virando a
referência de consulta para os algoritmos de ML vistos em aula.

A ideia: cada aba corresponde a um "PASSO" igual aos scripts da aula, e em
cada passo existe um botão "Ver o código Python" que mostra o código-fonte
REAL que está sendo executado (via `inspect.getsource`) -- ou seja, o que
você vê na tela é literalmente o comando que gerou o resultado. Nada de
código decorativo.

Para rodar:
    pip install -r requirements.txt
    streamlit run app.py
"""

import inspect
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

import core
from core import (
    DadosInvalidosError,
    avaliar_modelo,
    avaliar_modelo_classificacao,
    calcular_coeficientes_na_mao,
    carregar_dataset,
    classificar_tipo_problema,
    diagnosticar_residuos,
    dividir_treino_teste,
    escolher_grau_polinomial_cv,
    gerar_matriz_correlacao,
    interpretar_correlacao,
    prever_classe_novo_valor,
    prever_novo_valor,
    prever_novo_valor_multiplo,
    selecionar_melhor_subconjunto_multipla,
    testar_homocedasticidade_residuos,
    testar_independencia_residuos,
    testar_normalidade_residuos,
    treinar_modelo_regressao_logistica,
    treinar_modelo_regressao_multipla,
    treinar_modelo_regressao_polinomial,
    treinar_modelo_regressao_simples,
    treinar_regularizacao_cv,
    validar_dados_para_classificacao,
    validar_dados_para_regressao,
    validar_dados_para_regressao_multipla,
)

MODO_SIMPLES = "Simples (1 variável)"
MODO_MULTIPLA = "Múltipla (2+ variáveis)"
MODO_POLINOMIAL = "Polinomial (1 variável, curva)"

TAREFA_REGRESSAO = "Regressão (alvo contínuo)"
TAREFA_CLASSIFICACAO = "Classificação (alvo categórico)"

# ----------------------------------------------------------------------------
# Configuração geral da página
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Laboratório Interativo de Machine Learning",
    page_icon="📈",
    layout="wide",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ============================================================================
# Carregamento com cache (Bolt 3) -- evita reprocessar o CSV a cada clique
# ============================================================================

@st.cache_data(show_spinner="Carregando dataset...")
def carregar_dataset_do_disco(caminho_arquivo):
    """Wrapper cacheado de core.carregar_dataset() para os datasets prontos
    (arquivo no disco) -- a chave de cache é o caminho do arquivo."""
    return core.carregar_dataset(caminho_arquivo)


@st.cache_data(show_spinner="Carregando o seu CSV...")
def carregar_dataset_upload(nome_arquivo, conteudo_bytes):
    """Wrapper cacheado de core.carregar_dataset() para upload do usuário --
    a chave de cache é (nome do arquivo, bytes), então reenviar o mesmo
    arquivo não reprocessa do zero."""
    return core.carregar_dataset(io.BytesIO(conteudo_bytes))


# ============================================================================
# Utilitários de interface
# ============================================================================

def ver_codigo(funcao, titulo="Ver o código Python deste passo", expanded=False):
    with st.expander(f"🔍 {titulo}", expanded=expanded):
        st.code(inspect.getsource(funcao), language="python")


# ============================================================================
# Aba "🎓 Revisão da Prova" -- página única de consulta, estática (não
# depende do dataset escolhido na sidebar). Duas camadas de código:
#   1) "cola de prova" -- scripts flat, no estilo direto do professor
#      (aulas/, aulas/correcoes/): pd.read_csv -> X/y -> split -> fit ->
#      métricas -> previsão, prontos para adaptar rápido numa prova prática.
#   2) implementação real do app -- as mesmas funções de core.py via
#      inspect.getsource, para conferir a versão "de produção".
# ============================================================================

SCRIPT_COLA_SIMPLES = '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Carregar os dados
df = pd.read_csv("dataset.csv")

# 2. Ver a matriz de correlação -- escolher X com maior |correlação| com y
print(df.corr(numeric_only=True))

# 3. Definir X (1 variável) e y
X = df[["coluna_x"]]
y = df["coluna_y"]

# 4. Dividir treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 5. Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

b0 = modelo.intercept_
b1 = modelo.coef_[0]
print(f"Equação: y = {b0:.4f} + {b1:.4f} * X")

# 5b. Cálculo "na mão" de b0/b1 (bate com o modelo acima)
x_arr = X_train["coluna_x"].values
y_arr = y_train.values
media_x, media_y = x_arr.mean(), y_arr.mean()
b1_manual = np.sum((x_arr - media_x) * (y_arr - media_y)) / np.sum((x_arr - media_x) ** 2)
b0_manual = media_y - b1_manual * media_x

# 6. Prever no conjunto de teste e avaliar
y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"R2={r2:.4f}  MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}")

# 7. Prever um novo valor
novo = pd.DataFrame({"coluna_x": [valor_novo]})
print("Previsão:", modelo.predict(novo)[0])
'''

SCRIPT_COLA_MULTIPLA = '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_csv("dataset.csv")
print(df.corr(numeric_only=True))

# X com 2+ colunas (as de maior correlação com y); y é sempre 1 coluna
features = ["coluna_x1", "coluna_x2"]
X = df[features]
y = df["coluna_y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

intercepto = modelo.intercept_
coeficientes = modelo.coef_  # 1 por coluna de X, na mesma ordem de `features`
print(f"Intercepto: {intercepto:.4f}")
print(f"Coeficientes: {dict(zip(features, coeficientes))}")

y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"R2={r2:.4f}  MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}")

novo = pd.DataFrame({"coluna_x1": [valor1], "coluna_x2": [valor2]})
print("Previsão:", modelo.predict(novo)[0])
'''

SCRIPT_COLA_POLINOMIAL = '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("dataset.csv")
X = df[["coluna_x"]]  # Polinomial usa só 1 variável X, igual à Simples
y = df["coluna_y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

grau = 2  # grau 1 == Regressão Simples; graus maiores = curvas mais flexíveis
modelo = Pipeline([
    ("polinomio", PolynomialFeatures(degree=grau, include_bias=False)),
    ("regressao_linear", LinearRegression()),
])
modelo.fit(X_train, y_train)

intercepto = modelo.named_steps["regressao_linear"].intercept_
coeficientes = modelo.named_steps["regressao_linear"].coef_  # X^1, X^2, ..., X^grau
print(f"Intercepto: {intercepto:.4f}  Coeficientes: {coeficientes}")

y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R2={r2:.4f}  MAE={mae:.4f}  MSE={mse:.4f}")

# `modelo.predict()` já aplica a transformação polinomial internamente --
# previsão funciona igual à Simples, sem precisar expandir X na mão.
novo = pd.DataFrame({"coluna_x": [valor_novo]})
print("Previsão:", modelo.predict(novo)[0])
'''

SCRIPT_COLA_DIAGNOSTICO = '''import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from statsmodels.stats import diagnostic as diag

# resíduo = valor real - valor previsto (no conjunto de TESTE)
residuos = y_test.values - y_pred

# 1. LINEARIDADE -- inspecionar visualmente (scatter X vs y, ou pairplot)
sns.scatterplot(x=X_test.iloc[:, 0], y=y_test); plt.show()

# 2. MÉDIA DOS RESÍDUOS -- deve ser ~0 (sem viés sistemático)
print("Média dos resíduos:", np.mean(residuos))

# 3. HOMOCEDASTICIDADE -- variância dos resíduos deve ser constante
sns.scatterplot(x=y_pred, y=residuos); plt.axhline(0, color="red"); plt.show()
# Teste de Goldfeld-Quandt: H0 = homocedástico. p > 0.05 -> não rejeita H0.
estatistica_gq, p_valor_gq, _ = sms.het_goldfeldquandt(residuos, X_test)
print(f"Goldfeld-Quandt: estat={estatistica_gq:.4f}  p-valor={p_valor_gq:.4f}")

# 4. NORMALIDADE DOS RESÍDUOS -- Shapiro-Wilk. H0 = normal. p > 0.05 -> não rejeita H0.
from scipy import stats
estatistica_sw, p_valor_sw = stats.shapiro(residuos)
print(f"Shapiro-Wilk: estat={estatistica_sw:.4f}  p-valor={p_valor_sw:.4f}")
sns.histplot(residuos, kde=True); plt.show()

# 5. INDEPENDÊNCIA DOS RESÍDUOS (ausência de autocorrelação)
# Ljung-Box: H0 = sem autocorrelação em nenhum lag. p > 0.05 em todos -> não rejeita H0.
lb = diag.acorr_ljungbox(residuos, lags=min(40, len(residuos) - 1), return_df=True)
print("Menor p-valor (Ljung-Box):", lb["lb_pvalue"].min())
# Durbin-Watson: ~2 = sem autocorrelação; <1.5 positiva; >2.5 negativa.
dw = sms.durbin_watson(residuos)
print(f"Durbin-Watson: {dw:.4f}")

# 6. AUSÊNCIA DE COLINEARIDADE (só importa na Múltipla) -- heatmap entre as X
sns.heatmap(X_train.corr(), annot=True, cmap="coolwarm"); plt.show()
'''

SCRIPT_COLA_POLINOMIAL_AUTOMATICO = '''import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("dataset.csv")
X = df[["coluna_x"]]  # Polinomial usa só 1 variável X, igual à Simples
y = df["coluna_y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Em vez de escolher o grau na mão, testa 1..6 por validação cruzada (cv=5)
# e escolhe o de menor MSE médio -- método usado nos scripts de aula (Aula6).
pipeline = Pipeline([
    ("polinomio", PolynomialFeatures(include_bias=False)),
    ("regressao_linear", LinearRegression()),
])
grid = GridSearchCV(
    pipeline, {"polinomio__degree": [1, 2, 3, 4, 5, 6]},
    scoring="neg_mean_squared_error", cv=5,
)
grid.fit(X_train, y_train)  # o CV roda só dentro do treino

print("Melhor grau:", grid.best_params_["polinomio__degree"])
modelo = grid.best_estimator_  # já treinado com o melhor grau, pronto p/ usar

y_pred = modelo.predict(X_test)
print(f"R2={r2_score(y_test, y_pred):.4f}")

# modelo.predict() já aplica a expansão polinomial internamente
novo = pd.DataFrame({"coluna_x": [valor_novo]})
print("Previsão:", modelo.predict(novo)[0])
'''

SCRIPT_COLA_REGULARIZACAO = '''import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("dataset.csv")
# Use TODAS as variáveis candidatas -- a própria penalidade seleciona/encolhe
# as menos úteis; não precisa de busca de melhor subconjunto como na Múltipla.
features = ["coluna_x1", "coluna_x2", "coluna_x3"]
X = df[features]
y = df["coluna_y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# CUIDADO: Ridge/Lasso/ElasticNet são sensíveis à ESCALA de cada variável
# (a penalidade incide sobre o tamanho do coeficiente) -- ao contrário da
# OLS pura, aqui escalonar é obrigatório. Ajuste o scaler só no treino!
scaler = StandardScaler().fit(X_train)
X_train_esc = scaler.transform(X_train)
X_test_esc = scaler.transform(X_test)

alphas = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}

ridge = GridSearchCV(Ridge(), alphas, scoring="neg_mean_squared_error", cv=10)
ridge.fit(X_train_esc, y_train)

lasso = GridSearchCV(Lasso(), alphas, scoring="neg_mean_squared_error", cv=10)
lasso.fit(X_train_esc, y_train)

elastic = GridSearchCV(
    ElasticNet(), {**alphas, "l1_ratio": [0.1, 0.5, 0.9, 1]},
    scoring="neg_mean_squared_error", cv=10,
)
elastic.fit(X_train_esc, y_train)

for nome, busca in [("Ridge", ridge), ("Lasso", lasso), ("ElasticNet", elastic)]:
    modelo = busca.best_estimator_
    y_pred = modelo.predict(X_test_esc)
    print(f"{nome} (melhores params: {busca.best_params_})")
    print(f"  R2={r2_score(y_test, y_pred):.4f}  coeficientes={dict(zip(features, modelo.coef_.round(4)))}")

# Ridge encolhe todos os coeficientes em direção a zero, sem zerar nenhum.
# Lasso PODE zerar coeficientes por completo -- é uma seleção de variável
# implícita. ElasticNet fica entre os dois, conforme o l1_ratio escolhido.
'''

SCRIPT_COLA_LOGISTICA = '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
)

df = pd.read_csv("dataset.csv")
# y precisa ter EXATAMENTE 2 categorias (classificação binária)
X = df[["coluna_x1", "coluna_x2"]]
y = df["coluna_y_categorica"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# coeficientes ficam em escala de log-odds (log da razão de chances)
print("Intercepto:", modelo.intercept_[0])
print("Coeficientes (log-odds):", modelo.coef_[0])

y_pred = modelo.predict(X_test)
classe_positiva = modelo.classes_[-1]

acuracia = accuracy_score(y_test, y_pred)
precisao = precision_score(y_test, y_pred, pos_label=classe_positiva)
revocacao = recall_score(y_test, y_pred, pos_label=classe_positiva)
f1 = f1_score(y_test, y_pred, pos_label=classe_positiva)
matriz = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
print(f"Acurácia={acuracia:.4f}  Precisão={precisao:.4f}  Revocação={revocacao:.4f}  F1={f1:.4f}")
print("Matriz de confusão:\\n", matriz)

# Curva ROC / AUC (usa probabilidade, não a classe prevista)
probabilidades = modelo.predict_proba(X_test)[:, -1]
y_test_binario = (y_test == classe_positiva).astype(int)
taxa_fp, taxa_vp, _ = roc_curve(y_test_binario, probabilidades)
auc = roc_auc_score(y_test_binario, probabilidades)
print("AUC:", auc)

# Previsão de um novo caso
novo = pd.DataFrame({"coluna_x1": [valor1], "coluna_x2": [valor2]})
print("Classe prevista:", modelo.predict(novo)[0])
print("Probabilidade da classe positiva:", modelo.predict_proba(novo)[0][-1])
'''

REFERENCIA_DATASETS = pd.DataFrame(
    [
        {"Dataset": "base_salarios.csv", "Config": "y=Salary, Simples", "Referência": "R² ≈ 0.8943"},
        {"Dataset": "USA_Housing.csv", "Config": "X=Avg. Area Income, Simples", "Referência": "R² ≈ 0.4247"},
        {"Dataset": "distancia_consumo.csv", "Config": "Simples, todas as 10 linhas", "Referência": "y=0.5716+0.0663X, R²=0.8980, MSE=0.2995"},
        {"Dataset": "50_Startups.csv", "Config": "X=R&D Spend+Marketing Spend, test_size=0.3, rs=0", "Referência": "R² ≈ 0.9431"},
        {"Dataset": "publicidade.csv", "Config": "X=TV+Rádio+Jornal (e melhor subconjunto TV+Rádio)", "Referência": "R² ≈ 0.8649 / 0.8657"},
        {"Dataset": "agro_tech.csv", "Config": "Simples (precipitação) / Múltipla (+fertilizante)", "Referência": "R² ≈ 0.4847 / 0.8705"},
        {"Dataset": "finance_market.csv", "Config": "Simples, test_size=0.3, rs=42", "Referência": "R² ≈ 0.9987, RMSE ≈ 0.55"},
        {"Dataset": "saude_desenvolvimento.csv", "Config": "Múltipla (4 vars) vs. melhor Simples", "Referência": "R² ≈ 0.71 vs. 0.47"},
        {"Dataset": "mortalidade_infantil_desenvolvimento.csv", "Config": "Múltipla vs. melhor Simples", "Referência": "R² ≈ 0.70 vs. 0.69"},
        {"Dataset": "base_plano_saude_preparada.csv", "Config": "Múltipla, todas as colunas", "Referência": "R² ≈ 0.18 -- mau ajuste (suposições violadas)"},
        {"Dataset": "diagnostico_cancer_mama.csv", "Config": "Regressão Logística", "Referência": "Acurácia entre 90% e 98%"},
        {"Dataset": "comissao.csv", "Config": "Simples (grau 1) vs. Polinomial grau 2", "Referência": "Simples prevê comissão negativa (não faz sentido); grau 2 dá R² ≈ 1.0"},
        {"Dataset": "mortalidade_infantil_desenvolvimento.csv", "Config": "Múltipla (4 vars) vs. Ridge/Lasso/ElasticNet", "Referência": "R² 0.6961 (sem regularização) → 0.7010 (Ridge); Lasso zera Gasto_Saude_pct_PIB"},
    ]
)

TABELA_COMPARATIVA_PIPELINES = pd.DataFrame(
    [
        {"Etapa": "Nº de variáveis X", "Simples": "1", "Múltipla": "2+", "Polinomial": "1 (expandida em potências)", "Logística": "1+", "Regularização": "2+ (usa todas, sem busca de subconjunto)"},
        {"Etapa": "Escalonamento (StandardScaler)", "Simples": "não precisa", "Múltipla": "não precisa", "Polinomial": "não precisa", "Logística": "não precisa", "Regularização": "**obrigatório** -- penalidade depende da escala"},
        {"Etapa": "Hiperparâmetro via GridSearchCV", "Simples": "não tem", "Múltipla": "não tem", "Polinomial": "grau (só no modo automático)", "Logística": "não tem (neste app)", "Regularização": "alpha (Ridge/Lasso) + l1_ratio (ElasticNet)"},
        {"Etapa": "Classe do scikit-learn", "Simples": "LinearRegression", "Múltipla": "LinearRegression", "Polinomial": "Pipeline(PolynomialFeatures, LinearRegression)", "Logística": "LogisticRegression", "Regularização": "Ridge / Lasso / ElasticNet"},
        {"Etapa": "Métrica de avaliação (teste)", "Simples": "R², MAE, MSE, RMSE", "Múltipla": "R², MAE, MSE, RMSE", "Polinomial": "R², MAE, MSE, RMSE", "Logística": "Acurácia, Precisão, Revocação, F1, AUC", "Regularização": "R², MAE, MSE, RMSE"},
        {"Etapa": "Suposições da regressão (6)", "Simples": "checar", "Múltipla": "checar", "Polinomial": "checar", "Logística": "não se aplica (é classificação)", "Regularização": "mesmas da Múltipla (é Múltipla + penalidade)"},
        {"Etapa": "Pegadinha comum", "Simples": "confundir correlação com causalidade", "Múltipla": "esquecer que `coef_` agora é um vetor, não um escalar", "Polinomial": "achar que grau alto é sempre melhor (overfitting)", "Logística": "achar que é regressão -- é classificação!", "Regularização": "usar `alpha` sem escalonar X antes"},
    ]
)


def renderizar_revisao_da_prova():
    st.header("🎓 Revisão da Prova Parcial -- Pipeline completo em Python")
    st.info(
        "Página estática de consulta (não depende do dataset escolhido na "
        "barra lateral) -- **cumulativa**: cresce a cada novo tópico coberto "
        "pelo app, então sempre reflete tudo o que já foi ensinado até aqui. "
        "A prova T1 (a primeira do semestre) cobriu até Regressão Logística; "
        "a Parcial soma **Regressão Polinomial com grau automático** e "
        "**Regularização (Ridge/Lasso/ElasticNet)**, os 2 tópicos mais "
        "recentes. Cada bloco tem: (1) um **script pronto**, no mesmo estilo "
        "direto dos exercícios do professor, para adaptar rápido numa prova "
        "prática, e (2) a **implementação real do app** (`core.py`), caso "
        "precise conferir detalhes."
    )

    with st.expander("✅ Checklist geral do pipeline (qualquer algoritmo)", expanded=True):
        st.markdown(
            """
1. **Carregar** o CSV (`pd.read_csv`) e olhar `df.head()`/`df.info()`.
2. **Correlação** (`df.corr(numeric_only=True)`) -- escolher X pela maior
   correlação em módulo com y (ou usar todas, na Múltipla).
3. **Definir X e y**: `X = df[[...]]` (sempre uma lista, mesmo com 1
   coluna), `y = df["..."]`.
4. **Dividir treino/teste**: `train_test_split(X, y, test_size=0.3,
   random_state=0)` -- 70/30 é o padrão usado em aula.
4b. **(Só Regularização) Escalonar X**: `StandardScaler().fit(X_train)` e
   aplicar em treino e teste -- Ridge/Lasso/ElasticNet são sensíveis à
   escala; os outros algoritmos não precisam disso.
5. **Treinar**: `modelo.fit(X_train, y_train)`. Se tiver hiperparâmetro pra
   escolher (grau do Polinomial, `alpha` da Regularização), envolva o
   `.fit()` num `GridSearchCV` rodando **só dentro do treino**.
6. **Avaliar no conjunto de TESTE** (nunca no treino):
   - Regressão: R², MAE, MSE, RMSE.
   - Classificação: acurácia, precisão, revocação, F1, matriz de confusão,
     ROC/AUC.
7. **(Regressão) Checar as suposições**, se pedido: linearidade, média dos
   resíduos ≈ 0, homocedasticidade (Goldfeld-Quandt), normalidade
   (Shapiro-Wilk), independência (Ljung-Box/Durbin-Watson), colinearidade.
8. **Prever um novo valor**: monte um `pd.DataFrame` com as mesmas colunas
   de X e chame `modelo.predict(...)` (escalone o valor novo também, se o
   modelo for Regularização).
            """
        )

    with st.expander("🧭 Passo 0 -- Como classificar o problema antes de escolher o modelo", expanded=True):
        st.markdown(
            """
```
Aprendizado de Máquina
├── Supervisionado (tem atributo-alvo/y conhecido)
│   ├── Regressão → y CONTÍNUO (Regressão Linear Simples/Múltipla/Polinomial)
│   └── Classificação → y CATEGÓRICO
│       ├── Binária (2 classes) -- ex: Regressão Logística
│       └── Multiclasse (3+ classes)
└── Não supervisionado (sem y) → Associação / Clusterização (fora do escopo do app)
```
**Pegadinha clássica do professor:** Regressão Logística faz
**Classificação**, apesar do nome ter "Regressão".

**Vocabulário duplicado (mesma coisa, nomes diferentes):** X = "atributos
previsores" = "variáveis independentes/preditoras". y = "atributo-alvo/
target" = "variável dependente/resposta".
            """
        )
        ver_codigo(classificar_tipo_problema, "Ver o código: classificar_tipo_problema()")

    st.subheader("1️⃣ Regressão Linear Simples")
    st.code(SCRIPT_COLA_SIMPLES, language="python")
    ver_codigo(carregar_dataset, "core.py: carregar_dataset()")
    ver_codigo(treinar_modelo_regressao_simples, "core.py: treinar_modelo_regressao_simples()")
    ver_codigo(calcular_coeficientes_na_mao, "core.py: calcular_coeficientes_na_mao() -- cálculo manual")
    ver_codigo(avaliar_modelo, "core.py: avaliar_modelo() -- R²/MAE/MSE/RMSE")
    ver_codigo(prever_novo_valor, "core.py: prever_novo_valor()")

    st.subheader("2️⃣ Regressão Linear Múltipla")
    st.code(SCRIPT_COLA_MULTIPLA, language="python")
    ver_codigo(treinar_modelo_regressao_multipla, "core.py: treinar_modelo_regressao_multipla()")
    ver_codigo(prever_novo_valor_multiplo, "core.py: prever_novo_valor_multiplo()")
    ver_codigo(selecionar_melhor_subconjunto_multipla, "core.py: selecionar_melhor_subconjunto_multipla() -- busca do melhor R²")

    st.subheader("3️⃣ Regressão Polinomial")
    st.code(SCRIPT_COLA_POLINOMIAL, language="python")
    ver_codigo(treinar_modelo_regressao_polinomial, "core.py: treinar_modelo_regressao_polinomial()")

    st.subheader("4️⃣ Diagnóstico de Resíduos -- as 6 suposições da Regressão Linear")
    st.code(SCRIPT_COLA_DIAGNOSTICO, language="python")
    with st.expander("📋 Regra de decisão de cada teste (H0, limiar)", expanded=True):
        st.markdown(
            """
| Suposição | Teste | H0 (hipótese nula) | Regra |
|---|---|---|---|
| Média dos resíduos | -- | -- | deve ser ≈ 0 |
| Homocedasticidade | Goldfeld-Quandt | variância constante | p > 0.05 → não rejeita H0 (ok) |
| Normalidade | Shapiro-Wilk | resíduos normais | p > 0.05 → não rejeita H0 (ok) |
| Independência | Ljung-Box | sem autocorrelação | p > 0.05 em todos os lags → não rejeita H0 (ok) |
| Independência | Durbin-Watson | -- | entre 1.5 e 2.5 → sem autocorrelação relevante |
| Colinearidade | heatmap de correlação entre X | -- | |correlação| alta entre X's → colinearidade |
            """
        )
    ver_codigo(diagnosticar_residuos, "core.py: diagnosticar_residuos()")
    ver_codigo(testar_normalidade_residuos, "core.py: testar_normalidade_residuos() -- Shapiro-Wilk")
    ver_codigo(testar_homocedasticidade_residuos, "core.py: testar_homocedasticidade_residuos() -- Goldfeld-Quandt")
    ver_codigo(testar_independencia_residuos, "core.py: testar_independencia_residuos() -- Ljung-Box + Durbin-Watson")

    st.subheader("5️⃣ Regressão Logística (Classificação binária)")
    st.code(SCRIPT_COLA_LOGISTICA, language="python")
    ver_codigo(validar_dados_para_classificacao, "core.py: validar_dados_para_classificacao()")
    ver_codigo(treinar_modelo_regressao_logistica, "core.py: treinar_modelo_regressao_logistica()")
    ver_codigo(avaliar_modelo_classificacao, "core.py: avaliar_modelo_classificacao()")
    ver_codigo(prever_classe_novo_valor, "core.py: prever_classe_novo_valor()")

    st.subheader("6️⃣ Regressão Polinomial -- grau automático (GridSearchCV)")
    st.caption(
        "Novo desde a v12 (Aula6): em vez de escolher o grau no slider, "
        "deixa o `GridSearchCV` testar os graus 1-6 por validação cruzada "
        "(dentro do treino) e escolher o de menor erro médio -- método usado "
        "nos scripts de aula, não uma invenção do app."
    )
    st.code(SCRIPT_COLA_POLINOMIAL_AUTOMATICO, language="python")
    ver_codigo(escolher_grau_polinomial_cv, "core.py: escolher_grau_polinomial_cv()")

    st.subheader("7️⃣ Regularização -- Ridge / Lasso / ElasticNet")
    st.caption(
        "Novo desde a v13 (Aula7): é a Múltipla com uma penalidade sobre o "
        "tamanho dos coeficientes -- ajuda quando há colinearidade real "
        "entre as variáveis X (ex.: `Saneamento_pct` e `Agua_Potavel_pct` no "
        "dataset de Mortalidade Infantil, correlacionados em 0.906). Sem "
        "colinearidade, os 3 convergem pra quase-OLS (não é bug, é o "
        "`GridSearchCV` dizendo que menos regularização já é o melhor)."
    )
    st.code(SCRIPT_COLA_REGULARIZACAO, language="python")
    ver_codigo(treinar_regularizacao_cv, "core.py: treinar_regularizacao_cv()")

    st.subheader("🔀 Comparação lado a lado dos pipelines")
    st.caption(
        "O que muda e o que se repete entre os algoritmos -- útil pra não "
        "confundir na hora da prova qual passo é específico de qual modelo."
    )
    st.dataframe(TABELA_COMPARATIVA_PIPELINES, use_container_width=True, hide_index=True)

    st.subheader("📎 Gabarito -- valores de referência dos datasets de aula")
    st.caption(
        "Se você usar um destes datasets na prova (ou um parecido), confira "
        "se o resultado bate aproximadamente com a referência abaixo."
    )
    st.dataframe(REFERENCIA_DATASETS, use_container_width=True, hide_index=True)


DATASETS = {
    "💼 Salário x Anos de Experiência (aula)": {
        "arquivo": "base_salarios.csv",
        "y_padrao": "Salary",
        "descricao": (
            "Dataset usado no `exemplo1_regressaolinearsimples.py`. Relaciona anos "
            "de experiência profissional com o salário anual."
        ),
    },
    "🏠 Preço de Casas nos EUA (aula)": {
        "arquivo": "USA_Housing.csv",
        "y_padrao": "Price",
        "descricao": (
            "Dataset usado no `exemplo2_regressaolinearsimples.py`. Tem várias "
            "colunas numéricas -- aqui é sua vez de escolher, pela matriz de "
            "correlação, qual variável independente usar."
        ),
    },
    "🚗 Distância x Consumo (exemplo do slide)": {
        "arquivo": "distancia_consumo.csv",
        "y_padrao": "Consumo",
        "descricao": (
            "Os mesmos 10 carros do slide 'Regressão Linear Simples'. Bom para "
            "conferir se a conta bate exatamente com o professor: "
            "y = 0.5716 + 0.0663·X, R² = 0.8980, MSE = 0.2995."
        ),
    },
    "🚀 Lucro de Startups (aula)": {
        "arquivo": "50_Startups.csv",
        "y_padrao": "Profit",
        "descricao": (
            "Dataset usado no `exemplo1_regressaolinearmultipla.py`: 50 "
            "startups com gastos em P&D, Administração e Marketing, e o "
            "lucro (Profit) resultante. Igual ao script de aula, a coluna "
            "de Estado (variável categórica, codificada em dummies no CSV "
            "original) foi removida para focar no que foi ensinado -- "
            "aqui as variáveis com maior correlação com o lucro são "
            "'R&D Spend' e 'Marketing Spend' (R² ≈ 0.9431 com "
            "test_size=0.3, random_state=0, batendo com a aula)."
        ),
    },
    "❤️ Saúde & Desenvolvimento (ODS 3, 7, 8)": {
        "arquivo": "saude_desenvolvimento.csv",
        "y_padrao": "Expectativa_Vida",
        "descricao": (
            "Indicadores do World Bank Open Data (183 países, 2020) ligados aos "
            "Objetivos de Desenvolvimento Sustentável da ONU: Saúde e "
            "Bem-Estar (ODS 3), Energia Limpa e Acessível (ODS 7) e Trabalho "
            "Decente e Crescimento Econômico (ODS 8). PIB per capita, gasto em "
            "saúde, acesso à eletricidade e emissões de CO2 explicam boa parte "
            "da expectativa de vida entre países -- ótimo para Regressão "
            "Múltipla (R² ≈ 0.69 com as 4 variáveis, bem acima de qualquer "
            "variável isolada)."
        ),
    },
    "👶 Mortalidade Infantil & Desenvolvimento (ODS 3, 6, 8)": {
        "arquivo": "mortalidade_infantil_desenvolvimento.csv",
        "y_padrao": "Mortalidade_Infantil",
        "descricao": (
            "Indicadores do World Bank Open Data (180 países, 2020) ligados aos "
            "ODS de Saúde e Bem-Estar (ODS 3, meta 3.2 -- mortalidade de "
            "crianças menores de 5 anos), Água Potável e Saneamento (ODS 6) e "
            "Trabalho Decente e Crescimento Econômico (ODS 8). PIB per capita, "
            "gasto em saúde, acesso a saneamento e a água potável explicam bem "
            "a mortalidade infantil entre países (R² ≈ 0.70, estável entre "
            "splits). Saneamento sozinho já explica quase tudo aqui -- bom "
            "exemplo de colinearidade (Saneamento e Água Potável andam "
            "juntos), então a Múltipla ganha pouco da Simples neste caso "
            "-- ao contrário do dataset de Saúde acima, onde a Múltipla "
            "ganha bem mais. Compare os dois na aba **Comparação**."
        ),
    },
    "📢 Publicidade x Vendas (aula)": {
        "arquivo": "publicidade.csv",
        "y_padrao": "Vendas",
        "descricao": (
            "Dataset usado no `exemplo_teste_suposicao.py`: vendas de um "
            "produto em 200 mercados, explicadas pelo orçamento de "
            "publicidade em TV, Rádio e Jornal. Bom para ver os **testes "
            "de suposições** da aba Diagnóstico em ação -- 'Jornal' tem "
            "correlação fraca com Vendas (R² ≈ 0.8649 com as 3 variáveis, "
            "0.8657 com o melhor subconjunto TV+Rádio, test_size=0.3, "
            "random_state=0)."
        ),
    },
    "📈 Mercado Financeiro (correção)": {
        "arquivo": "finance_market.csv",
        "y_padrao": "ETF_Preco",
        "descricao": (
            "Dataset usado em `correcao_regsimples.py`: preço de um ETF "
            "explicado pelo Índice S&P500 (500 observações, sem NaN). Ajuste "
            "quase perfeito -- R² ≈ 0.9987, MAE ≈ 0.44, MSE ≈ 0.30, "
            "RMSE ≈ 0.55 com `test_size=0.3, random_state=42` (o script do "
            "professor usa essa semente; com o `random_state=0` padrão do "
            "app o resultado é parecido, mas não idêntico)."
        ),
    },
    "🌾 Produtividade Agrícola (correção)": {
        "arquivo": "agro_tech.csv",
        "y_padrao": "toneladas_por_hectare",
        "descricao": (
            "Dataset usado em `regressao_simples.py`/`regressao_multipla.py`: "
            "produtividade agrícola (ton/ha) explicada por clima e insumos "
            "(450 observações, sem NaN). Bom para comparar Simples vs. "
            "Múltipla -- Simples com `precipitacao_anual` dá R² ≈ 0.4847, "
            "Múltipla com `precipitacao_anual` + `fertilizante_kg_ha` "
            "(as 2 variáveis de maior correlação, escolhidas automaticamente) "
            "sobe para R² ≈ 0.8705, batendo com o script de aula "
            "(`test_size=0.3, random_state=0`)."
        ),
    },
    "🏥 Gastos com Plano de Saúde (correção)": {
        "arquivo": "base_plano_saude_preparada.csv",
        "y_padrao": "gastos_plano",
        "descricao": (
            "Dataset usado em `correcao_teste_suposicao.py`: gastos com "
            "plano de saúde explicados por idade, IMC, filhos, gênero, "
            "hábito de fumar e região (já codificada em colunas dummy -- "
            "2772 observações, sem NaN). **Exemplo intencional de mau "
            "ajuste**: as correlações com o alvo são fracas (|r| ≤ 0.34) e "
            "o próprio script do professor conclui que o modelo viola as "
            "suposições de homocedasticidade e normalidade dos resíduos -- "
            "bom para testar a aba **Diagnóstico dos Resíduos** num caso "
            "que realmente falha, ao contrário dos outros datasets."
        ),
    },
    "💰 Comissão x Quantidade Vendida (aula)": {
        "arquivo": "comissao.csv",
        "y_padrao": "comissao",
        "descricao": (
            "Dataset usado no `exemplo_polinomial.py` (Aula6): 50 observações, "
            "`quantidade` vendida x `comissao` recebida -- feito sob medida "
            "para mostrar por que uma curva bate melhor que uma reta. No modo "
            "**Simples** (grau 1), a equação vira "
            "`comissão = -626 + 178·quantidade`, prevendo comissão **negativa** "
            "pra quantidades baixas (não faz sentido). No modo **Polinomial** "
            "com grau 2 -- manual ou via 'Automático (GridSearchCV)' -- o "
            "ajuste fica essencialmente perfeito (R² ≈ 1.0, RMSE ≈ 0), porque "
            "os dados foram construídos como uma parábola exata. Ótimo para "
            "comparar Simples x Polinomial no leaderboard e ver a diferença "
            "na prática."
        ),
    },
}

DATASETS_CLASSIFICACAO = {
    "🩺 Diagnóstico de Câncer de Mama (ODS 3)": {
        "arquivo": "diagnostico_cancer_mama.csv",
        "y_padrao": "Diagnostico",
        "descricao": (
            "Breast Cancer Wisconsin (Diagnostic) Data Set -- UCI ML Repository / "
            "scikit-learn. 569 casos, 30 medidas numéricas extraídas de exames de "
            "imagem, alvo genuinamente categórico: diagnóstico **Maligno** ou "
            "**Benigno** de um tumor. Tema ODS 3 (Saúde e Bem-Estar). Bom para "
            "Regressão Logística: acurácia estável entre 90% e 98% dependendo "
            "das variáveis e do split escolhidos."
        ),
    },
}


# ============================================================================
# BARRA LATERAL -- controla o estado global do laboratório
# ============================================================================

st.sidebar.title("⚙️ Painel de controle")

tarefa = st.sidebar.radio(
    "0. Tipo de Tarefa",
    [TAREFA_REGRESSAO, TAREFA_CLASSIFICACAO],
    help=(
        "Regressão: o alvo (y) é um valor contínuo (preço, salário, "
        "expectativa de vida...). Classificação: o alvo é uma categoria "
        "(ex: diagnóstico maligno/benigno)."
    ),
)
st.sidebar.markdown("---")

datasets_disponiveis = DATASETS if tarefa == TAREFA_REGRESSAO else DATASETS_CLASSIFICACAO

nome_dataset = st.sidebar.selectbox(
    "1. Escolha o dataset", list(datasets_disponiveis.keys()) + ["📁 Carregar meu próprio CSV"]
)

try:
    if nome_dataset == "📁 Carregar meu próprio CSV":
        arquivo_upload = st.sidebar.file_uploader("Envie um arquivo .csv", type=["csv"])
        if arquivo_upload is None:
            st.sidebar.info("Envie um CSV para continuar, ou escolha um dataset pronto acima.")
            st.stop()
        df = carregar_dataset_upload(arquivo_upload.name, arquivo_upload.getvalue())
        descricao_dataset = "Dataset carregado por você."
        y_sugerido = df.columns[-1]
    else:
        info = datasets_disponiveis[nome_dataset]
        caminho = os.path.join(DATA_DIR, info["arquivo"])
        df = carregar_dataset_do_disco(caminho)
        descricao_dataset = info["descricao"]
        y_sugerido = info["y_padrao"]
except DadosInvalidosError as erro:
    st.sidebar.error(str(erro))
    st.stop()

st.sidebar.caption(descricao_dataset)

colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
matriz_corr_completa = gerar_matriz_correlacao(df)

if tarefa == TAREFA_CLASSIFICACAO:
    colunas_binarias = [c for c in df.columns if df[c].nunique() == 2]
    if not colunas_binarias:
        st.sidebar.error(
            "Este dataset não tem nenhuma coluna com exatamente 2 categorias -- "
            "não dá para escolher um alvo binário para a Regressão Logística."
        )
        st.stop()
    col_y = st.sidebar.selectbox(
        "2. Variável DEPENDENTE (y, alvo categórico)",
        colunas_binarias,
        index=colunas_binarias.index(y_sugerido) if y_sugerido in colunas_binarias else 0,
    )
    opcoes_x = [c for c in colunas_numericas if c != col_y]
    if not opcoes_x:
        st.sidebar.error("Este dataset não tem colunas numéricas para usar como variáveis X.")
        st.stop()
else:
    if len(colunas_numericas) < 2:
        st.sidebar.error("Este dataset precisa de pelo menos 2 colunas numéricas.")
        st.stop()
    col_y = st.sidebar.selectbox(
        "2. Variável DEPENDENTE (y, alvo)",
        colunas_numericas,
        index=colunas_numericas.index(y_sugerido) if y_sugerido in colunas_numericas else 0,
    )
    opcoes_x = [c for c in colunas_numericas if c != col_y]

if col_y in matriz_corr_completa:
    opcoes_x_ordenadas = sorted(
        opcoes_x, key=lambda c: abs(matriz_corr_completa[col_y][c]), reverse=True
    )
else:
    opcoes_x_ordenadas = opcoes_x

if tarefa == TAREFA_CLASSIFICACAO:
    modo_regressao = None
    cols_x = st.sidebar.multiselect(
        "3. Variáveis INDEPENDENTES (X, preditoras)",
        opcoes_x_ordenadas,
        default=opcoes_x_ordenadas[:4] if len(opcoes_x_ordenadas) >= 4 else opcoes_x_ordenadas[:1],
        help="Escolha 1 ou mais colunas numéricas para prever a classe.",
    )
    if not cols_x:
        st.sidebar.info("Selecione pelo menos 1 variável X para treinar a Regressão Logística.")
        st.stop()
    col_x = cols_x[0]  # coluna de referência p/ trechos que ainda mostram só 1 variável
else:
    modo_regressao = st.sidebar.radio(
        "3. Tipo de Regressão",
        [MODO_SIMPLES, MODO_MULTIPLA, MODO_POLINOMIAL],
        help=(
            "Simples: uma única variável X explica y (uma reta). Múltipla: duas "
            "ou mais variáveis X combinadas tentam explicar y melhor. "
            "Polinomial: uma única variável X, mas ajustando uma curva em vez "
            "de uma reta."
        ),
    )

    if modo_regressao == MODO_MULTIPLA:
        if len(opcoes_x_ordenadas) < 2:
            st.sidebar.error(
                "Este dataset só tem 1 coluna numérica candidata a X -- não dá "
                "para fazer Regressão Múltipla. Escolha outro dataset ou o modo Simples."
            )
            st.stop()
        cols_x = st.sidebar.multiselect(
            "4. Variáveis INDEPENDENTES (X, preditoras)",
            opcoes_x_ordenadas,
            default=opcoes_x_ordenadas[:2],
            help="Escolha 2 ou mais colunas. Já vêm ordenadas pela força da correlação com y.",
        )
        if len(cols_x) < 2:
            st.sidebar.info("Selecione pelo menos 2 variáveis X para treinar a Regressão Múltipla.")
            st.stop()
        col_x = cols_x[0]  # coluna de referência p/ trechos que ainda mostram só 1 variável

        modo_regularizacao = st.sidebar.radio(
            "4b. Regularizar?",
            ["Nenhuma (Múltipla padrão)", "Comparar Ridge / Lasso / ElasticNet"],
            help="Reproduz o método dos scripts de aula (Aula7): escalona X com "
            "StandardScaler e compara a Múltipla sem regularização com Ridge, Lasso "
            "e ElasticNet, cada um com o hiperparâmetro escolhido por validação "
            "cruzada (GridSearchCV, sem separar treino/teste -- a comparação usa o "
            "neg_mean_squared_error médio da CV, igual ao script). Não afeta as "
            "abas Avaliação/Previsão/Comparação, que continuam usando a Múltipla "
            "'padrão' com o split treino/teste de sempre.",
        )
    else:
        col_x = st.sidebar.selectbox(
            "4. Variável INDEPENDENTE (X, preditora)",
            opcoes_x_ordenadas,
            help="Dica: na aula, escolhe-se a variável com maior correlação (em módulo) com o alvo.",
        )
        cols_x = [col_x]

    if modo_regressao == MODO_POLINOMIAL:
        modo_grau_polinomial = st.sidebar.radio(
            "4b. Como escolher o grau?",
            ["Manual (slider)", "Automático (GridSearchCV, igual à aula)"],
            help="'Automático' reproduz o método usado nos scripts de aula (Aula6): "
            "testa cada grau candidato via validação cruzada (GridSearchCV) e escolhe "
            "o de menor MSE médio, em vez de você escolher o grau na mão.",
        )
        if modo_grau_polinomial == "Manual (slider)":
            grau_polinomial = st.sidebar.slider(
                "4c. Grau do polinômio", min_value=2, max_value=5, value=2, step=1,
                help="Grau 1 seria igual à Regressão Simples. Graus mais altos ajustam curvas "
                "mais flexíveis, mas arriscam overfitting com poucos dados.",
            )
        else:
            grau_polinomial = None  # escolhido depois do split, via GridSearchCV (precisa de X_treinamento/y_treinamento)

st.sidebar.markdown("---")

test_size = st.sidebar.slider(
    "5. Tamanho do conjunto de TESTE", min_value=0.1, max_value=0.5, value=0.3, step=0.05,
    help="Na aula: 70% treino / 30% teste (valor padrão = 0.3).",
)

random_state = st.sidebar.number_input(
    "6. random_state (semente aleatória)", min_value=0, max_value=9999, value=0, step=1,
    help="Mesma semente = mesma divisão treino/teste sempre que rodar de novo.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Este painel controla TODAS as abas ao lado. Mude o dataset, as variáveis "
    "ou o tamanho do teste e veja o laboratório inteiro recalcular na hora."
)


# ============================================================================
# VALIDAÇÃO (Bolt 2) -- falha com mensagem clara em vez de traceback
# ============================================================================

try:
    if tarefa == TAREFA_CLASSIFICACAO:
        df_limpo, linhas_removidas = validar_dados_para_classificacao(df, cols_x, col_y, test_size)
    elif modo_regressao == MODO_MULTIPLA:
        df_limpo, linhas_removidas = validar_dados_para_regressao_multipla(df, cols_x, col_y, test_size)
    else:
        df_limpo, linhas_removidas = validar_dados_para_regressao(df, col_x, col_y, test_size)
except DadosInvalidosError as erro:
    st.error(f"⚠️ Não foi possível treinar o modelo: {erro}")
    st.stop()

if linhas_removidas > 0:
    st.sidebar.warning(
        f"{linhas_removidas} linha(s) com valor vazio em {', '.join(cols_x)!r} ou '{col_y}' "
        "foram ignoradas."
    )


# ============================================================================
# CÁLCULOS (compartilhados por todas as abas)
#
# No lado Regressão, o resultado é guardado num formato comum --
# `intercepto` (float) + `coeficientes` (pd.Series indexada pelas colunas
# de X) -- para que as abas abaixo não precisem saber qual modo está
# ativo. `beta0_manual`/`beta1_manual` só existem no modo Simples (cálculo
# na mão só faz sentido fechado para 1 variável). No lado Classificação,
# as métricas são outras (acurácia, precisão, revocação, F1, matriz de
# confusão) por isso ficam em variáveis próprias.
# ============================================================================

X = df_limpo[cols_x]
y = df_limpo[col_y]

X_treinamento, X_teste, y_treinamento, y_teste = dividir_treino_teste(
    X, y, test_size=test_size, random_state=int(random_state)
)

try:
    if tarefa == TAREFA_CLASSIFICACAO:
        modelo, intercepto, coeficientes = treinar_modelo_regressao_logistica(X_treinamento, y_treinamento)
    elif modo_regressao == MODO_MULTIPLA:
        modelo, intercepto, coeficientes = treinar_modelo_regressao_multipla(X_treinamento, y_treinamento)
        beta0_manual, beta1_manual = None, None
    elif modo_regressao == MODO_POLINOMIAL:
        tabela_cv_graus = None
        if modo_grau_polinomial == "Manual (slider)":
            modelo, intercepto, coefs_array = treinar_modelo_regressao_polinomial(
                X_treinamento, y_treinamento, grau_polinomial
            )
        else:
            grau_polinomial, modelo, tabela_cv_graus = escolher_grau_polinomial_cv(
                X_treinamento, y_treinamento
            )
            intercepto = modelo.named_steps["regressao_linear"].intercept_
            coefs_array = modelo.named_steps["regressao_linear"].coef_
        coeficientes = pd.Series(coefs_array, index=[f"{col_x}^{p}" for p in range(1, grau_polinomial + 1)])
        beta0_manual, beta1_manual = None, None
    else:
        modelo, intercepto, coeficiente = treinar_modelo_regressao_simples(X_treinamento, y_treinamento)
        coeficientes = pd.Series([coeficiente], index=[col_x])
        beta0_manual, beta1_manual = calcular_coeficientes_na_mao(X_treinamento[col_x], y_treinamento)
except DadosInvalidosError as erro:
    st.error(
        f"⚠️ Não foi possível treinar o modelo com esta divisão treino/teste: {erro} "
        "Tente outro random_state ou tamanho de teste."
    )
    st.stop()

if tarefa == TAREFA_CLASSIFICACAO:
    predicoes_modelo, acuracia, precisao, revocacao, f1, matriz_confusao, classe_positiva = (
        avaliar_modelo_classificacao(modelo, X_teste, y_teste)
    )
else:
    predicoes_modelo, r2, mae, mse, rmse = avaliar_modelo(modelo, X_teste, y_teste)

# Regularização (Ridge/Lasso/ElasticNet) -- calculado uma única vez aqui
# (não em cada aba) porque as abas Treinamento e Avaliação mostram os
# mesmos resultados sob ângulos diferentes (coeficientes x métricas).
resultados_regularizacao = None
erro_regularizacao = None
if (
    tarefa == TAREFA_REGRESSAO
    and modo_regressao == MODO_MULTIPLA
    and modo_regularizacao == "Comparar Ridge / Lasso / ElasticNet"
):
    try:
        resultados_regularizacao = treinar_regularizacao_cv(X_treinamento, y_treinamento, X_teste, y_teste)
    except DadosInvalidosError as erro:
        erro_regularizacao = str(erro)


# ============================================================================
# CORPO PRINCIPAL
# ============================================================================

st.title("📈 Laboratório Interativo de Machine Learning")
st.markdown(
    "Consulta prática dos algoritmos de ML da disciplina "
    "(Prof. Dr. Daniel Trevisan Bravo) -- começou por Regressão Linear e vai "
    "crescendo ao longo do semestre. "
    "Mexa nos controles à esquerda e observe cada etapa do algoritmo acontecer "
    "em tempo real -- com o código Python de cada passo à mostra."
)

if tarefa == TAREFA_CLASSIFICACAO:
    aba_revisao, aba_passo0, aba_teoria, aba_dados, aba_split, aba_treino, aba_avaliacao, aba_previsao = st.tabs(
        [
            "🎓 Revisão da Prova",
            "🧭 Passo 0",
            "📚 Teoria",
            "📊 Dados & Correlação",
            "✂️ Treino / Teste",
            "🧮 Treinamento do Modelo",
            "📐 Avaliação",
            "🔮 Previsão",
        ]
    )
    aba_diagnostico = aba_comparacao = None  # não existem no lado Classificação
else:
    aba_revisao, aba_passo0, aba_teoria, aba_dados, aba_split, aba_treino, aba_avaliacao, aba_diagnostico, aba_previsao, aba_comparacao = st.tabs(
        [
            "🎓 Revisão da Prova",
            "🧭 Passo 0",
            "📚 Teoria",
            "📊 Dados & Correlação",
            "✂️ Treino / Teste",
            "🧮 Treinamento do Modelo",
            "📐 Avaliação",
            "🩺 Diagnóstico dos Resíduos",
            "🔮 Previsão",
            "🆚 Comparação",
        ]
    )

# ------------------------------------------------------- Revisão da Prova --
with aba_revisao:
    renderizar_revisao_da_prova()

# --------------------------------------------------------------- Passo 0 ---
with aba_passo0:
    st.header("🧭 Passo 0 -- Antes de escolher o modelo, classifique o problema")
    st.markdown(
        """
Este laboratório deixa você escolher o modelo direto num seletor -- mas na
prática **essa escolha depende de entender o problema primeiro**. Antes de
sair testando modelos, responda a si mesmo:

1. **Eu tenho um atributo-alvo conhecido (rótulo) para cada exemplo?**
2. Se sim, **esse alvo é um valor contínuo ou uma categoria?**
3. Se for categoria, **são 2 categorias ou 3+?**

É exatamente essa sequência de perguntas que define qual "família" de
algoritmo usar -- e é isso que o quiz abaixo simula.
        """
    )

    st.subheader("A taxonomia do professor")
    st.markdown(
        """
Segundo os slides de **"Introdução ao Aprendizado de Máquina"**
(Prof. Dr. Daniel Trevisan Bravo), os algoritmos de Data Mining/ML se
dividem assim:

```
Aprendizado de Máquina (Data Mining)
├── Supervisionado (dados + rótulos/target conhecidos)
│   ├── Regressão → atributo-alvo CONTÍNUO
│   │   (ex.: Árvores de Decisão, Regressão Linear, Naive Bayes, SVM, KNN
│   │    -- quando usados p/ valor contínuo)
│   └── Classificação → atributo-alvo SEMPRE categórico
│       ├── Binária (2 classes, 1/0)
│       └── Multiclasse (3+ classes)
│       (Regressão Logística é CLASSIFICAÇÃO, apesar do nome!)
├── Não supervisionado ("aprendizado descritivo", sem rótulo)
│   ├── Associação (Algoritmo Regra de Associação)
│   └── Clusterização/Agrupamento (K-Means, Hierárquicos, Grafos)
└── Aprendizado por Reforço (recompensas/punições)
```

Este laboratório hoje cobre só o ramo **Supervisionado** (Regressão e
Classificação) -- Clusterização é o item 14 do roadmap
(`ai-dlc/01-inception/INCEPTION.md`), ainda não implementado.
        """
    )

    st.subheader("Vocabulário: dois nomes para a mesma coisa")
    st.markdown(
        """
O professor usa nomes diferentes para X e y em aulas diferentes -- são
sinônimos, vale reconhecer os dois:

| Símbolo | Nomes que você vai ver |
|---|---|
| **X** | atributos previsores · variáveis independentes · variáveis preditoras |
| **y** | atributo-alvo (target) · variável dependente · variável resposta |
        """
    )

    st.warning(
        "⚠️ **Pegadinha clássica**: apesar do nome, a **Regressão Logística "
        "é um algoritmo de Classificação**, não de Regressão -- o alvo dela "
        "é categórico (ex.: maligno/benigno), só o cálculo interno usa uma "
        "função logística em cima de uma combinação linear."
    )

    st.subheader("🔎 Quiz: classifique o seu problema")
    resposta_rotulado = st.radio(
        "1. O seu dataset tem um atributo-alvo (y) conhecido para cada exemplo?",
        ["Sim, tenho um alvo conhecido", "Não, só tenho os dados, sem alvo"],
        key="passo0_rotulado",
    )
    rotulado = resposta_rotulado.startswith("Sim")

    tipo_alvo_resposta = None
    num_classes_resposta = None
    if rotulado:
        tipo_alvo_resposta = st.radio(
            "2. O atributo-alvo (y) é um valor contínuo ou uma categoria?",
            ["Contínuo (ex.: preço, salário, expectativa de vida)", "Categórico (ex.: sim/não, maligno/benigno)"],
            key="passo0_tipo_alvo",
        )
        if tipo_alvo_resposta.startswith("Categórico"):
            num_classes_resposta = st.radio(
                "3. Quantas categorias o atributo-alvo pode assumir?",
                [2, 3, 4, 5],
                key="passo0_num_classes",
                help="2 = binária (ex.: fraude/não-fraude). 3 ou mais = multiclasse.",
            )

    tipo_alvo_arg = "categórico" if (tipo_alvo_resposta and tipo_alvo_resposta.startswith("Categórico")) else "contínuo"
    resultado_quiz = classificar_tipo_problema(
        rotulado=rotulado, tipo_alvo=tipo_alvo_arg, num_classes=num_classes_resposta
    )

    if resultado_quiz["recomendacao_app"] is not None:
        st.success(
            f"**Categoria:** {resultado_quiz['categoria']}"
            + (f" ({resultado_quiz['subtipo']})" if resultado_quiz["subtipo"] else "")
            + f"\n\n{resultado_quiz['explicacao']}"
            + f"\n\n👉 Vá para **'0. Tipo de Tarefa'** na barra lateral e "
              f"escolha **'{resultado_quiz['recomendacao_app']}'**."
        )
    else:
        st.info(f"**Categoria:** {resultado_quiz['categoria']}\n\n{resultado_quiz['explicacao']}")

    ver_codigo(classificar_tipo_problema, "Ver o código Python deste passo")

# ---------------------------------------------------------------- Teoria ---
with aba_teoria:
    st.header("O que é Aprendizado de Máquina supervisionado?")
    st.markdown(
        """
É um campo da inteligência artificial em que os sistemas aprendem padrões a
partir de dados, em vez de receber instruções explícitas para cada tarefa.

No **aprendizado supervisionado**, os dados são divididos em:

- **X** (atributos previsores / independentes): o que se usa para prever.
- **y** (atributo-alvo / dependente): o que se quer prever.

Dentro do aprendizado supervisionado existem dois grandes problemas:

- **Regressão**: o alvo é um valor contínuo (preço de um imóvel, salário,
  consumo de combustível...). É o caso deste laboratório.
- **Classificação**: o alvo é uma categoria (fraude ou não, tumor benigno ou
  maligno...).
        """
    )

    st.header("Regressão Linear Simples")
    st.markdown(
        r"""
A regressão linear simples modela a relação entre **uma** variável
independente $X$ e uma variável dependente $y$ através de uma reta:

$$\hat{y} = \beta_0 + \beta_1 \cdot X$$

- $\beta_0$ (**intercepto**): onde a reta cruza o eixo y.
- $\beta_1$ (**coeficiente angular**): o quanto $y$ varia para cada unidade
  a mais em $X$.

Os coeficientes que minimizam o erro entre a reta e os pontos reais são
calculados por:

$$\beta_1 = \dfrac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
\qquad\qquad
\beta_0 = \bar{y} - \beta_1 \bar{x}$$

É exatamente essa conta que o `LinearRegression().fit()` do scikit-learn
faz por baixo dos panos -- você vai conferir isso na aba **Treinamento do
Modelo**.
        """
    )

    st.header("Regressão Linear Múltipla")
    st.markdown(
        r"""
Quando **duas ou mais** variáveis independentes ($X_1, X_2, \dots, X_n$)
ajudam a explicar $y$, a regressão vira múltipla -- em vez de uma reta em 2D,
o modelo ajusta um plano (ou hiperplano, com mais variáveis):

$$\hat{y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n$$

Cada $\beta_i$ mede o quanto $y$ varia para cada unidade a mais em $X_i$,
**mantendo as outras variáveis fixas**. Os coeficientes que minimizam o erro
não têm mais uma fórmula fechada tão simples quanto no caso de 1 variável --
o scikit-learn resolve via álgebra matricial (a equação normal
$\beta = (X^TX)^{-1}X^Ty$), mas o princípio é o mesmo: mínimos quadrados.

Vale a pena usar Múltipla em vez de Simples quando várias variáveis têm
correlação relevante com o alvo -- o R² tende a subir, já que o modelo tem
mais informação para trabalhar (compare os dois na aba **Comparação**).
        """
    )

    st.header("Regressão Polinomial")
    st.markdown(
        r"""
Quando a relação entre $X$ e $y$ não é bem descrita por uma reta, dá para
ajustar uma **curva** em vez disso -- sem trocar de algoritmo, só
adicionando potências de $X$ como se fossem novas variáveis:

$$\hat{y} = \beta_0 + \beta_1 X + \beta_2 X^2 + \dots + \beta_g X^g$$

onde $g$ é o **grau** do polinômio. É por isso que ainda é "regressão
**linear**" -- a equação é linear nos coeficientes $\beta_i$, só que $X$
aparece elevado a potências. Na prática, o `PolynomialFeatures` do
scikit-learn gera as colunas $X, X^2, \dots, X^g$, e depois o mesmo
`LinearRegression().fit()` de sempre ajusta os coeficientes.

Cuidado com **overfitting**: graus muito altos ajustam até o ruído dos
dados de treino, e a curva fica bonita no treino mas erra feio no teste --
compare o R² de treino e teste em diferentes graus para perceber isso.
        """
    )

    st.header("Como avaliar se o modelo é bom?")
    st.markdown(
        """
- **R² (R-quadrado)**: % da variação de y que o modelo explica. Varia de 0
  (não explica nada) a 1 (explica tudo). Quanto mais perto de 1, melhor.
- **MAE (erro médio absoluto)**: a média de "quanto o modelo erra", na
  mesma unidade de y.
- **MSE (erro médio quadrático)**: parecido com o MAE, mas eleva os erros
  ao quadrado antes de tirar a média -- por isso penaliza mais forte os
  erros grandes. Quanto mais perto de 0, melhor.
        """
    )

    st.header("Estudo de Adequação do Modelo (Análise de Resíduos)")
    st.markdown(
        r"""
R², MAE e MSE dizem **o quanto** o modelo erra, mas não dizem se o modelo
é *adequado* -- para isso, olha-se para os **resíduos**
($\text{resíduo} = y_{real} - \hat{y}$). A regressão linear por mínimos
quadrados assume que os resíduos são:

- **Aleatórios**: sem padrão quando plotados contra os valores previstos
  (se formam uma curva ou um funil, o modelo está deixando algo de fora).
- **Centrados em 0**: a média dos resíduos deve ser ~0.
- **Homocedásticos**: variância aproximadamente constante (não deveriam
  "abrir" ou "fechar" conforme os valores previstos crescem).
- **Aproximadamente normais**: útil para os testes estatísticos formais do
  modelo (intervalos de confiança, testes de hipótese sobre os $\beta_i$).

O **teste de Shapiro-Wilk** checa formalmente a suposição de normalidade:
$H_0$ = os resíduos vêm de uma distribuição normal. Se o p-valor for maior
que 0.05, não há evidência para rejeitar $H_0$. Tudo isso está na aba
**🩺 Diagnóstico dos Resíduos**.
        """
    )

    st.header("Correlação de Pearson: como interpretar")
    st.markdown(
        """
| Valor absoluto | Força da correlação |
|---|---|
| ≥ 0.9 | muito forte |
| 0.7 – 0.9 | forte |
| 0.5 – 0.7 | moderada |
| 0.3 – 0.5 | fraca |
| < 0.3 | muito fraca / desprezível |

A escolha da variável independente $X$, quando há mais de uma disponível,
costuma recair sobre aquela com **maior correlação (em módulo)** com o
alvo $y$ -- é isso que a aba **Dados & Correlação** te ajuda a visualizar.
        """
    )

    st.header("Classificação e Regressão Logística")
    st.markdown(
        r"""
Até aqui, $y$ era sempre um valor **contínuo** (regressão). Quando $y$ é
uma **categoria** -- diagnóstico maligno/benigno, cliente vai cancelar ou
não, e-mail é spam ou não -- o problema vira **classificação**, e R²/MAE/
MSE deixam de fazer sentido (não existe "erro médio" entre categorias).

A **Regressão Logística** é o ponto de partida clássico para classificação
binária (2 classes). Em vez de prever $y$ diretamente, ela prevê a
**probabilidade** de $y$ pertencer à classe positiva, passando a
combinação linear de $X$ por uma função sigmoide:

$$P(y = 1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_n X_n)}}$$

O resultado é sempre um número entre 0 e 1 (uma probabilidade) -- daí o
modelo escolher a classe prevista comparando essa probabilidade com um
limiar (0.5 por padrão). Os coeficientes $\beta_i$ são interpretados em
**log-odds** (log da razão de chances): positivo aumenta a chance da
classe positiva, negativo diminui, mantendo as outras variáveis fixas.

Como o alvo agora é categórico, as métricas de avaliação também mudam:

- **Acurácia**: % de previsões corretas no total.
- **Precisão**: das vezes que o modelo previu a classe positiva, quantas
  vezes acertou.
- **Revocação (recall)**: dos casos que realmente eram da classe positiva,
  quantos o modelo conseguiu identificar.
- **F1**: média harmônica entre precisão e revocação -- útil quando as
  classes são desbalanceadas.
- **Matriz de confusão**: tabela com acertos e erros de cada classe lado a
  lado (verdadeiros/falsos positivos e negativos).

No painel lateral, mude o **"Tipo de Tarefa"** para *Classificação* para
explorar tudo isso na prática.
        """
    )

# ------------------------------------------------------- Dados & Correlação
with aba_dados:
    st.header("PASSO 0 e 1 — Dados e correlação")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Dados brutos")
        st.dataframe(df.head(15), use_container_width=True)
        st.caption(f"{df.shape[0]} linhas × {df.shape[1]} colunas.")
        ver_codigo(carregar_dataset, "Ver o código: carregar o dataset")

    with col_b:
        st.subheader("Estrutura do DataFrame")
        buffer_info = []
        for coluna in df.columns:
            buffer_info.append(
                {"coluna": coluna, "tipo": str(df[coluna].dtype), "nulos": int(df[coluna].isna().sum())}
            )
        st.dataframe(pd.DataFrame(buffer_info), use_container_width=True, hide_index=True)
        st.subheader("Estatísticas descritivas")
        st.dataframe(df[colunas_numericas].describe().round(2), use_container_width=True)

    st.markdown("---")
    st.subheader("Matriz de correlação (Pearson)")

    matriz_correlacao = gerar_matriz_correlacao(df)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        matriz_correlacao, annot=True, fmt=".2f", cmap="coolwarm", square=True,
        linecolor="white", linewidths=0.5, ax=ax,
    )
    st.pyplot(fig, use_container_width=False)
    ver_codigo(gerar_matriz_correlacao, "Ver o código: matriz de correlação")

    if tarefa == TAREFA_CLASSIFICACAO:
        st.subheader(f"Média de cada variável X por classe de {col_y}")
        st.caption(
            "Correlação de Pearson não se aplica diretamente a um alvo categórico "
            "-- em vez disso, compare a média de cada variável X entre as classes. "
            "Diferenças grandes indicam variáveis mais úteis para separar as classes."
        )
        tabela_media_por_classe = df_limpo.groupby(col_y)[cols_x].mean().T
        st.dataframe(
            tabela_media_por_classe.style.format("{:.4f}"),
            use_container_width=True,
        )
    elif modo_regressao == MODO_MULTIPLA:
        tabela_corr_x = pd.DataFrame(
            {
                "variável X": cols_x,
                "correlação com y": [matriz_corr_completa[col_y][c] for c in cols_x],
            }
        )
        tabela_corr_x["força"] = tabela_corr_x["correlação com y"].apply(interpretar_correlacao)
        st.dataframe(
            tabela_corr_x.style.format({"correlação com y": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Estas são as {len(cols_x)} variáveis independentes escolhidas para a "
            f"Regressão Múltipla, cada uma com sua correlação (em módulo) com **{col_y}**."
        )
    else:
        corr_xy = matriz_corr_completa[col_y][col_x]
        st.info(
            f"Correlação entre **{col_x}** e **{col_y}**: **{corr_xy:.4f}** "
            f"→ correlação **{interpretar_correlacao(corr_xy)}**. "
            f"Por isso {col_x!r} foi escolhida como variável independente (X) e "
            f"{col_y!r} como variável dependente / alvo (y)."
        )

        if len(opcoes_x) > 1:
            st.caption(
                "Este dataset tem mais de uma variável numérica candidata a X. "
                "Troque a variável independente no painel à esquerda e veja como "
                "a correlação (e, mais adiante, o R²) muda."
            )

# ------------------------------------------------------------ Treino/Teste
with aba_split:
    st.header("PASSO 2 — Dividir em Treinamento e Teste")
    st.markdown(
        f"""
- O modelo é **treinado** usando **{int((1 - test_size) * 100)}%** dos dados.
- O modelo é **avaliado** com os **{int(test_size * 100)}%** restantes --
  dados que ele nunca viu durante o treinamento.
        """
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total de linhas", df.shape[0])
    col_b.metric("Linhas de treinamento", X_treinamento.shape[0])
    col_c.metric("Linhas de teste", X_teste.shape[0])

    fig_split, ax_split = plt.subplots(figsize=(6, 1.2))
    ax_split.barh([0], [X_treinamento.shape[0]], color="#4C72B0", label="Treinamento")
    ax_split.barh([0], [X_teste.shape[0]], left=[X_treinamento.shape[0]], color="#DD8452", label="Teste")
    ax_split.set_yticks([])
    ax_split.set_xlabel("Número de linhas")
    ax_split.legend(loc="upper center", bbox_to_anchor=(0.5, 1.6), ncol=2, frameon=False)
    st.pyplot(fig_split, use_container_width=False)

    ver_codigo(dividir_treino_teste, "Ver o código: train_test_split")

    st.subheader("Como ficaram os conjuntos")
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("X_treinamento (primeiras linhas)")
        st.dataframe(X_treinamento.head(), use_container_width=True)
    with col_b:
        st.caption("X_teste (primeiras linhas)")
        st.dataframe(X_teste.head(), use_container_width=True)

# --------------------------------------------------------------- Treino ---
with aba_treino:
    st.header("PASSO 3 — Criação e treinamento do modelo")

    if tarefa == TAREFA_CLASSIFICACAO:
        st.markdown(
            "`LogisticRegression().fit(X_treinamento, y_treinamento)` calcula o "
            "intercepto e **um coeficiente por variável X**, em escala de "
            "log-odds, que maximizam a probabilidade dos dados observados "
            "-- diferente da regressão linear (mínimos quadrados), aqui o "
            "método é **máxima verossimilhança**."
        )

        st.subheader("Calculado pelo scikit-learn")
        st.metric("Intercepto (log-odds)", f"{intercepto:.4f}")
        tabela_coeficientes = coeficientes.rename("coeficiente (log-odds)").reset_index()
        tabela_coeficientes.columns = ["variável X", "coeficiente (log-odds)"]
        tabela_coeficientes["razão de chances (e^coef)"] = np.exp(coeficientes.values)
        st.dataframe(
            tabela_coeficientes.style.format(
                {"coeficiente (log-odds)": "{:.4f}", "razão de chances (e^coef)": "{:.4f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Classe positiva (a que os coeficientes 'empurram' para cima): "
            f"**{classe_positiva}**. Razão de chances > 1 aumenta a chance dessa "
            "classe a cada unidade a mais na variável; < 1 diminui."
        )

        termos = " + ".join(f"{coef:.4f} \\cdot {nome}" for nome, coef in coeficientes.items())
        st.markdown(
            f"### Equação do modelo\n"
            f"$$P(y = \\text{{{classe_positiva}}}) = "
            f"\\dfrac{{1}}{{1 + e^{{-({intercepto:.4f} + {termos})}}}}$$"
        )

        ver_codigo(treinar_modelo_regressao_logistica, "Ver o código: treinar com scikit-learn")
    elif modo_regressao == MODO_MULTIPLA:
        st.markdown(
            "`LinearRegression().fit(X_treinamento, y_treinamento)` calcula o "
            "intercepto (β₀) e **um coeficiente por variável X** que minimizam "
            "o erro entre o modelo e os pontos de treinamento (Método dos "
            "Mínimos Quadrados, resolvido via álgebra matricial -- a equação "
            "normal $\\beta = (X^TX)^{-1}X^Ty$ -- em vez da fórmula fechada de "
            "1 variável do modo Simples)."
        )

        st.subheader("Calculado pelo scikit-learn")
        st.metric("Intercepto (β₀)", f"{intercepto:.4f}")
        tabela_coeficientes = coeficientes.rename("coeficiente").reset_index()
        tabela_coeficientes.columns = ["variável X", "coeficiente"]
        st.dataframe(
            tabela_coeficientes.style.format({"coeficiente": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        termos = " + ".join(f"{coef:.4f} \\cdot {nome}" for nome, coef in coeficientes.items())
        st.markdown(
            f"### Equação do modelo\n"
            f"$$\\hat{{y}} = {intercepto:.4f} + {termos}$$"
        )

        ver_codigo(treinar_modelo_regressao_multipla, "Ver o código: treinar com scikit-learn")

        if modo_regularizacao == "Comparar Ridge / Lasso / ElasticNet":
            st.markdown("---")
            st.subheader("🪢 Regularização: coeficientes (Ridge / Lasso / ElasticNet)")
            st.markdown(
                "Reproduzindo o método dos scripts de aula (Aula7), mas no "
                "**mesmo split treino/teste** do painel lateral (para comparar "
                "de forma justa com a Múltipla acima -- ver métricas na aba "
                "Avaliação): X é escalonado com `StandardScaler` (ajustado só "
                "no treino) e `alpha` (Ridge/Lasso) / `alpha`+`l1_ratio` "
                "(ElasticNet) são escolhidos por `GridSearchCV` (validação "
                "cruzada **dentro do treino**, 10 dobras)."
            )
            if erro_regularizacao is not None:
                st.warning(f"⚠️ Não foi possível comparar os modelos regularizados: {erro_regularizacao}")
            else:
                tabela_hiperparametros = pd.DataFrame(
                    {
                        "modelo": nome,
                        "hiperparâmetro(s)": ", ".join(
                            f"{chave}={valor}" for chave, valor in info["hiperparametros"].items()
                        )
                        or "--",
                    }
                    for nome, info in resultados_regularizacao.items()
                )
                st.dataframe(tabela_hiperparametros, use_container_width=True, hide_index=True)

                fig_regularizacao, ax_regularizacao = plt.subplots(figsize=(8, 4.5))
                for nome, info in resultados_regularizacao.items():
                    ax_regularizacao.plot(
                        info["coeficientes"].index, info["coeficientes"].values, marker="o", label=nome
                    )
                ax_regularizacao.axhline(0, color="red", linestyle="solid", linewidth=1)
                ax_regularizacao.set_ylabel("coeficiente (X escalonado)")
                ax_regularizacao.tick_params(axis="x", rotation=30)
                ax_regularizacao.legend()
                ax_regularizacao.set_title("Coeficientes: sem regularização x Ridge x Lasso x ElasticNet")
                st.pyplot(fig_regularizacao, use_container_width=False)

                st.caption(
                    "Ridge encolhe todos os coeficientes em direção a zero sem "
                    "zerar nenhum; Lasso pode zerar completamente uma variável "
                    "(seleção de variável implícita); ElasticNet fica entre os "
                    "dois, conforme `l1_ratio`. Métricas de desempenho (R²/MAE/"
                    "MSE/RMSE) dos 4 modelos, lado a lado com a Múltipla oficial: "
                    "aba **Avaliação**."
                )

                ver_codigo(treinar_regularizacao_cv, "Ver o código: treinar com scikit-learn")
    elif modo_regressao == MODO_POLINOMIAL:
        st.markdown(
            f"`PolynomialFeatures(degree={grau_polinomial})` expande **{col_x}** em "
            f"potências (X¹, X²{', ..., X' + str(grau_polinomial) if grau_polinomial > 2 else ''}) "
            "e depois `LinearRegression().fit()` ajusta os coeficientes -- "
            "continua sendo mínimos quadrados, só que sobre as colunas "
            "expandidas em vez de X sozinho."
        )

        if tabela_cv_graus is not None:
            st.info(
                f"🔍 **Grau escolhido automaticamente: {grau_polinomial}** -- "
                "`GridSearchCV` testou os graus abaixo por validação cruzada "
                "(5 dobras, `neg_mean_squared_error`) e escolheu o de menor erro "
                "médio, igual ao método usado nos scripts de aula (Aula6)."
            )
            tabela_cv_exibicao = tabela_cv_graus.rename("neg_MSE médio (CV)").reset_index()
            tabela_cv_exibicao.columns = ["grau", "neg_MSE médio (CV)"]
            st.dataframe(
                tabela_cv_exibicao.style.format({"neg_MSE médio (CV)": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
            ver_codigo(escolher_grau_polinomial_cv, "Ver o código: escolher o grau via GridSearchCV")

        st.subheader("Calculado pelo scikit-learn")
        st.metric("Intercepto (β₀)", f"{intercepto:.4f}")
        tabela_coeficientes = coeficientes.rename("coeficiente").reset_index()
        tabela_coeficientes.columns = ["potência de X", "coeficiente"]
        st.dataframe(
            tabela_coeficientes.style.format({"coeficiente": "{:.6f}"}),
            use_container_width=True,
            hide_index=True,
        )

        termos = " + ".join(
            f"{coef:.6f} \\cdot X^{{{p}}}" for p, coef in zip(range(1, grau_polinomial + 1), coeficientes)
        )
        st.markdown(
            f"### Equação do modelo (grau {grau_polinomial})\n"
            f"$$\\hat{{y}} = {intercepto:.4f} + {termos}$$"
        )

        ver_codigo(treinar_modelo_regressao_polinomial, "Ver o código: treinar com scikit-learn")
    else:
        st.markdown(
            "`LinearRegression().fit(X_treinamento, y_treinamento)` calcula o "
            "intercepto (β₀) e o coeficiente (β₁) que minimizam o erro entre a "
            "reta e os pontos de treinamento (Método dos Mínimos Quadrados)."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Calculado pelo scikit-learn")
            st.metric("Intercepto (β₀)", f"{intercepto:.4f}")
            st.metric("Coeficiente (β₁)", f"{coeficiente:.4f}")
        with col_b:
            st.subheader("Calculado na mão (fórmula do slide)")
            st.metric("Intercepto (β₀)", f"{beta0_manual:.4f}")
            st.metric("Coeficiente (β₁)", f"{beta1_manual:.4f}")

        st.success(
            "Os dois lados batem: o `LinearRegression()` não é uma caixa-preta, "
            "ele resolve exatamente a fórmula dos mínimos quadrados do slide."
        )

        st.markdown(
            f"### Equação do modelo\n"
            f"$$\\hat{{y}} = {intercepto:.4f} + {coeficiente:.4f} \\cdot X$$"
        )

        ver_codigo(treinar_modelo_regressao_simples, "Ver o código: treinar com scikit-learn")
        ver_codigo(calcular_coeficientes_na_mao, "Ver o código: calcular β₀ e β₁ na mão")

# ------------------------------------------------------------ Avaliação ---
with aba_avaliacao:
    st.header("PASSO 4 — Avaliação do modelo")

    if tarefa == TAREFA_CLASSIFICACAO:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Acurácia", f"{acuracia:.4f}")
        col_b.metric(f"Precisão ({classe_positiva})", f"{precisao:.4f}")
        col_c.metric(f"Revocação ({classe_positiva})", f"{revocacao:.4f}")
        col_d.metric(f"F1 ({classe_positiva})", f"{f1:.4f}")

        st.info(
            f"O modelo acertou **{acuracia * 100:.2f}%** das previsões no conjunto "
            f"de teste, usando {len(cols_x)} variável(is) para prever **{col_y}**. "
            f"Classe positiva (usada em precisão/revocação/F1): **{classe_positiva}**."
        )

        ver_codigo(avaliar_modelo_classificacao, "Ver o código: acurácia, precisão, revocação, F1 e matriz de confusão")

        st.markdown("---")
        st.subheader("Matriz de confusão")
        st.caption(
            "Linhas = classe real, colunas = classe prevista. A diagonal principal "
            "são os acertos; fora dela, os erros do modelo."
        )

        fig_matriz, ax_matriz = plt.subplots(figsize=(5, 4.5))
        sns.heatmap(
            matriz_confusao, annot=True, fmt="d", cmap="Blues", square=True,
            xticklabels=modelo.classes_, yticklabels=modelo.classes_, ax=ax_matriz,
            cbar=False,
        )
        ax_matriz.set_xlabel("Previsto")
        ax_matriz.set_ylabel("Real")
        st.pyplot(fig_matriz, use_container_width=False)

        st.markdown("---")
        st.subheader("Curva ROC")
        st.caption(
            "Mostra o trade-off entre taxa de verdadeiros positivos e falsos "
            "positivos conforme o limiar de decisão muda. Quanto mais a curva "
            "sobe em direção ao canto superior esquerdo (e mais a área sob a "
            "curva -- AUC -- se aproxima de 1), melhor o classificador."
        )

        probabilidades_teste = modelo.predict_proba(X_teste)[:, -1]
        y_teste_binario = (y_teste == classe_positiva).astype(int)
        taxa_fp, taxa_vp, _ = roc_curve(y_teste_binario, probabilidades_teste)
        auc = roc_auc_score(y_teste_binario, probabilidades_teste)

        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(taxa_fp, taxa_vp, color="steelblue", linewidth=2, label=f"Regressão Logística (AUC = {auc:.4f})")
        ax_roc.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle="--", label="Classificador aleatório")
        ax_roc.set_xlabel("Taxa de Falsos Positivos")
        ax_roc.set_ylabel("Taxa de Verdadeiros Positivos")
        ax_roc.set_title("Curva ROC")
        ax_roc.legend()
        st.pyplot(fig_roc, use_container_width=False)
    else:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("R² (coeficiente de determinação)", f"{r2:.4f}")
        col_b.metric("MAE (erro médio absoluto)", f"{mae:,.2f}")
        col_c.metric("MSE (erro médio quadrático)", f"{mse:,.2f}")
        col_d.metric("RMSE (raiz do erro quadrático médio)", f"{rmse:,.2f}")

        if modo_regressao == MODO_MULTIPLA:
            origem_x = f"das {len(cols_x)} variáveis selecionadas ({', '.join(cols_x)})"
        else:
            origem_x = f"**{col_x}**"
        st.info(
            f"O modelo conseguiu explicar **{r2 * 100:.2f}%** da variação de "
            f"**{col_y}** a partir de {origem_x}, no conjunto de teste."
        )

        ver_codigo(avaliar_modelo, "Ver o código: R², MAE, MSE e RMSE")

        st.markdown("---")

        if modo_regressao == MODO_MULTIPLA:
            st.subheader("Previsto vs. Real (conjunto de teste)")
            st.caption(
                "Com 2 ou mais variáveis X não dá mais para desenhar uma reta 2D -- "
                "em vez disso, cada ponto compara o valor real de "
                f"**{col_y}** com o valor que o modelo previu. Quanto mais perto "
                "da linha diagonal, melhor o ajuste."
            )

            fig_disp, ax_disp = plt.subplots(figsize=(6, 6))
            ax_disp.scatter(y_teste, predicoes_modelo, color="steelblue", alpha=0.6, label="Previsões (teste)")
            limite_min = min(float(np.min(y_teste)), float(np.min(predicoes_modelo)))
            limite_max = max(float(np.max(y_teste)), float(np.max(predicoes_modelo)))
            ax_disp.plot(
                [limite_min, limite_max], [limite_min, limite_max],
                color="red", linewidth=2, linestyle="--", label="Previsão perfeita (y = ŷ)",
            )
            ax_disp.set_xlabel(f"{col_y} real")
            ax_disp.set_ylabel(f"{col_y} previsto")
            ax_disp.set_title("Previsto vs. Real")
            ax_disp.legend()
            st.pyplot(fig_disp, use_container_width=False)

            if modo_regularizacao == "Comparar Ridge / Lasso / ElasticNet":
                st.markdown("---")
                st.subheader("🪢 Regularização: métricas lado a lado com a Múltipla")
                st.caption(
                    "Mesmo split treino/teste da Múltipla acima -- a linha "
                    "'Sem regularização' deve bater com R²/MAE/MSE/RMSE mostrados "
                    "no topo desta aba (escalonar X não muda essas métricas numa "
                    "OLS pura; só Ridge/Lasso/ElasticNet, que penalizam o tamanho "
                    "do coeficiente, são sensíveis à escala)."
                )
                if erro_regularizacao is not None:
                    st.warning(f"⚠️ Não foi possível comparar os modelos regularizados: {erro_regularizacao}")
                else:
                    tabela_metricas_regularizacao = pd.DataFrame(
                        {
                            "modelo": nome,
                            "R²": info["r2"],
                            "MAE": info["mae"],
                            "MSE": info["mse"],
                            "RMSE": info["rmse"],
                        }
                        for nome, info in resultados_regularizacao.items()
                    )
                    melhor_regularizacao = tabela_metricas_regularizacao.loc[
                        tabela_metricas_regularizacao["R²"].idxmax(), "modelo"
                    ]
                    st.dataframe(
                        tabela_metricas_regularizacao.style.format(
                            {"R²": "{:.4f}", "MAE": "{:,.2f}", "MSE": "{:,.2f}", "RMSE": "{:,.2f}"}
                        ).apply(
                            lambda linha: [
                                "font-weight: bold" if linha["modelo"] == melhor_regularizacao else ""
                                for _ in linha
                            ],
                            axis=1,
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    if melhor_regularizacao == "Sem regularização":
                        st.info(
                            "Neste dataset e split, nenhuma das 3 regularizações "
                            "melhorou o R² de teste em relação à Múltipla sem "
                            "regularização -- regularizar nem sempre ajuda, "
                            "principalmente com poucas variáveis/pouca colinearidade."
                        )
                    else:
                        st.success(
                            f"Neste dataset e split, **{melhor_regularizacao}** teve o "
                            f"maior R² de teste, superando a Múltipla sem regularização."
                        )
        elif modo_regressao == MODO_POLINOMIAL:
            st.subheader(f"Dispersão dos dados + curva ajustada (grau {grau_polinomial})")
            st.caption(
                "A curva é calculada num intervalo suave de valores de X (não só "
                "nos pontos de teste), para mostrar a forma real da curva ajustada."
            )

            fig_disp, ax_disp = plt.subplots(figsize=(8, 5))
            ax_disp.scatter(X, y, color="steelblue", alpha=0.5, label="Todos os dados")
            x_min, x_max = float(X[col_x].min()), float(X[col_x].max())
            x_curva = np.linspace(x_min, x_max, 200).reshape(-1, 1)
            y_curva = modelo.predict(pd.DataFrame(x_curva, columns=[col_x]))
            ax_disp.plot(
                x_curva, y_curva, color="red", linewidth=2,
                label=f"Curva ajustada (grau {grau_polinomial})",
            )
            ax_disp.set_xlabel(col_x)
            ax_disp.set_ylabel(col_y)
            ax_disp.set_title(f"{col_x} x {col_y}")
            ax_disp.legend()
            st.pyplot(fig_disp, use_container_width=False)
        else:
            st.subheader("Dispersão dos dados + reta de regressão")

            fig_disp, ax_disp = plt.subplots(figsize=(8, 5))
            ax_disp.scatter(X, y, color="steelblue", alpha=0.5, label="Todos os dados")
            ordem = np.argsort(X_teste[col_x].values)
            ax_disp.plot(
                X_teste[col_x].values[ordem], predicoes_modelo[ordem],
                color="red", linewidth=2, label="Reta de regressão (previsão no teste)",
            )
            ax_disp.set_xlabel(col_x)
            ax_disp.set_ylabel(col_y)
            ax_disp.set_title(f"{col_x} x {col_y}")
            ax_disp.legend()
            st.pyplot(fig_disp, use_container_width=False)

# ------------------------------------------------- Diagnóstico dos Resíduos
if tarefa == TAREFA_REGRESSAO:
    with aba_diagnostico:
        st.header("Estudo de Adequação do Modelo — Análise de Resíduos")
        st.markdown(
            "Resíduo = valor real − valor previsto. As suposições clássicas da "
            "regressão linear (mínimos quadrados) esperam que os resíduos sejam "
            "**aleatórios** (sem padrão), **centrados em 0** e com **variância "
            "constante** (homocedasticidade) -- é isso que os gráficos e o teste "
            "abaixo ajudam a checar, usando o conjunto de **teste**."
        )

        diagnostico = diagnosticar_residuos(y_teste, predicoes_modelo)
        residuos = diagnostico["residuos"]

        col_a, col_b = st.columns(2)
        col_a.metric("Resíduo médio", f"{diagnostico['media']:,.4f}")
        col_b.metric("Desvio-padrão dos resíduos", f"{diagnostico['desvio_padrao']:,.4f}")

        ver_codigo(diagnosticar_residuos, "Ver o código: calcular os resíduos")

        st.markdown("---")
        st.subheader("Resíduos vs. Previstos")
        st.caption(
            "Se o modelo é adequado, os pontos devem ficar espalhados "
            "aleatoriamente ao redor da linha 0, sem formar funil, curva ou "
            "outro padrão visível."
        )
        fig_resid, ax_resid = plt.subplots(figsize=(8, 4.5))
        ax_resid.scatter(predicoes_modelo, residuos, color="steelblue", alpha=0.6)
        ax_resid.axhline(0, color="red", linewidth=2, linestyle="--")
        ax_resid.set_xlabel(f"{col_y} previsto")
        ax_resid.set_ylabel("Resíduo (real − previsto)")
        st.pyplot(fig_resid, use_container_width=False)

        st.markdown("**Teste estatístico (Goldfeld-Quandt)**")
        try:
            estatistica_gq, p_valor_gq = testar_homocedasticidade_residuos(residuos, X_teste)
            if p_valor_gq > 0.05:
                st.success(
                    f"p-valor = {p_valor_gq:.4f} (> 0.05) -- não há evidência para "
                    "rejeitar a hipótese de que a variância dos resíduos é "
                    "constante (homocedasticidade)."
                )
            else:
                st.warning(
                    f"p-valor = {p_valor_gq:.4f} (≤ 0.05) -- há evidência de "
                    "**heterocedasticidade** (variância dos resíduos não é "
                    "constante), o que pode afetar a confiabilidade dos "
                    "testes estatísticos do modelo."
                )
            st.caption(f"Estatística F: {estatistica_gq:.4f}")
        except DadosInvalidosError as erro:
            st.info(str(erro))

        ver_codigo(testar_homocedasticidade_residuos, "Ver o código: teste de Goldfeld-Quandt")

        st.subheader("Histograma dos resíduos")
        st.caption("Se o modelo é adequado, a forma deve lembrar um sino (distribuição normal).")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
        n_bins = min(20, max(5, len(residuos) // 2))
        ax_hist.hist(residuos, bins=n_bins, color="steelblue", edgecolor="white")
        ax_hist.axvline(0, color="red", linewidth=2, linestyle="--")
        ax_hist.set_xlabel("Resíduo")
        ax_hist.set_ylabel("Frequência")
        st.pyplot(fig_hist, use_container_width=False)

        st.markdown("---")
        st.subheader("Teste de normalidade (Shapiro-Wilk)")
        try:
            estatistica, p_valor = testar_normalidade_residuos(residuos)
            if p_valor > 0.05:
                st.success(
                    f"p-valor = {p_valor:.4f} (> 0.05) -- não há evidência para "
                    "rejeitar a hipótese de que os resíduos seguem uma "
                    "distribuição normal. Boa notícia para a adequação do "
                    "modelo linear."
                )
            else:
                st.warning(
                    f"p-valor = {p_valor:.4f} (≤ 0.05) -- há evidência de que os "
                    "resíduos NÃO seguem uma distribuição normal. Pode ser sinal "
                    "de que um modelo linear não é o mais adequado para estes "
                    "dados (compare com outros modelos na aba Comparação)."
                )
            st.caption(f"Estatística de teste: {estatistica:.4f}")
        except DadosInvalidosError as erro:
            st.info(str(erro))

        ver_codigo(testar_normalidade_residuos, "Ver o código: teste de Shapiro-Wilk")

        st.markdown("---")
        st.subheader("Independência dos resíduos (autocorrelação)")
        st.caption(
            "Os resíduos não deveriam ter padrão entre si -- se o valor de um "
            "resíduo ajuda a prever o próximo, o modelo está deixando "
            "informação na mesa. Dois testes diferentes, que podem discordar "
            "entre si (cada um capta um tipo de padrão)."
        )
        try:
            p_valor_lb, estatistica_dw = testar_independencia_residuos(residuos)
            col_lb, col_dw = st.columns(2)
            with col_lb:
                st.markdown("**Ljung-Box**")
                if p_valor_lb > 0.05:
                    st.success(f"p-valor (mínimo entre os lags) = {p_valor_lb:.4f} (> 0.05) -- não rejeita H0 (sem autocorrelação).")
                else:
                    st.warning(f"p-valor (mínimo entre os lags) = {p_valor_lb:.4f} (≤ 0.05) -- rejeita H0 (indício de autocorrelação em algum lag).")
            with col_dw:
                st.markdown("**Durbin-Watson**")
                st.metric("Estatística", f"{estatistica_dw:.4f}")
                if estatistica_dw < 1.5:
                    st.warning("< 1.5 -- indício de autocorrelação positiva.")
                elif estatistica_dw > 2.5:
                    st.warning("> 2.5 -- indício de autocorrelação negativa.")
                else:
                    st.success("Entre 1.5 e 2.5 -- sem evidência significativa de autocorrelação.")
            st.info(
                "Os dois testes podem discordar (Ljung-Box olha vários lags de "
                "uma vez, Durbin-Watson foca só na autocorrelação entre "
                "resíduos vizinhos) -- quando isso acontece, vale interpretar "
                "com cautela em vez de confiar cegamente em um só teste."
            )
        except DadosInvalidosError as erro:
            st.info(str(erro))

        ver_codigo(testar_independencia_residuos, "Ver o código: testes de Ljung-Box e Durbin-Watson")

        st.markdown("---")
        st.subheader("Ausência de colinearidade entre as variáveis X")
        if len(cols_x) < 2:
            st.caption(
                "Este modo usa 1 única variável X -- colinearidade só é um "
                "risco quando há 2 ou mais variáveis independentes (modo "
                "Múltipla)."
            )
        else:
            st.caption(
                "Em Regressão Múltipla, as variáveis X idealmente não deveriam "
                "estar fortemente correlacionadas entre si -- caso contrário, "
                "fica difícil separar o efeito de cada uma sobre o alvo. "
                "Correlações próximas de 1 ou -1 fora da diagonal são o sinal "
                "de alerta (veja também a aba **📊 Dados & Correlação**)."
            )
            fig_colin, ax_colin = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                X_teste.corr(), annot=True, cmap="RdYlGn", square=True, ax=ax_colin, vmin=-1, vmax=1
            )
            st.pyplot(fig_colin, use_container_width=False)

# ------------------------------------------------------------- Previsão ---
with aba_previsao:
    st.header("PASSO 5 — Previsão para um novo valor")

    if tarefa == TAREFA_CLASSIFICACAO:
        st.markdown(
            f"Escolha um valor para cada uma das {len(cols_x)} variáveis X e "
            f"veja qual classe de **{col_y}** o modelo prevê."
        )

        valores_x = {}
        for c in cols_x:
            minimo_c = float(X[c].min())
            maximo_c = float(X[c].max())
            valor_padrao_c = float(X[c].median())
            valores_x[c] = st.slider(
                f"Valor de {c}", min_value=minimo_c, max_value=maximo_c, value=valor_padrao_c,
                key=f"previsao_{c}",
            )

        classe_prevista, probabilidade = prever_classe_novo_valor(modelo, valores_x)

        col_a, col_b = st.columns(2)
        col_a.metric(f"Classe prevista de {col_y}", str(classe_prevista))
        col_b.metric(f"Probabilidade ({classe_positiva})", f"{probabilidade:.2%}")

        ver_codigo(prever_classe_novo_valor, "Ver o código: prever a classe de um novo valor")

        st.subheader("Valores usados na previsão")
        st.dataframe(
            pd.DataFrame([valores_x]).style.format("{:.4f}"),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"O modelo prevê a classe '{classe_prevista}', com "
            f"{probabilidade:.2%} de probabilidade de ser '{classe_positiva}'."
        )
    elif modo_regressao == MODO_MULTIPLA:
        st.markdown(
            f"Escolha um valor para cada uma das {len(cols_x)} variáveis X e "
            f"veja o que o modelo prevê para **{col_y}**."
        )

        valores_x = {}
        for c in cols_x:
            minimo_c = float(X[c].min())
            maximo_c = float(X[c].max())
            valor_padrao_c = float(X[c].median())
            valores_x[c] = st.slider(
                f"Valor de {c}", min_value=minimo_c, max_value=maximo_c, value=valor_padrao_c,
                key=f"previsao_{c}",
            )

        predicao = prever_novo_valor_multiplo(modelo, valores_x)

        st.metric(f"Previsão de {col_y}", f"{predicao:,.2f}")

        ver_codigo(prever_novo_valor_multiplo, "Ver o código: prever um novo valor")

        st.subheader("Valores usados na previsão")
        st.dataframe(
            pd.DataFrame([valores_x]).style.format("{:.4f}"),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"O valor previsto de {col_y} para os valores acima é {predicao:,.2f}."
        )
    else:
        st.markdown(
            f"Escolha um valor de **{col_x}** e veja o que o modelo prevê para **{col_y}**."
        )

        minimo = float(X[col_x].min())
        maximo = float(X[col_x].max())
        valor_padrao = float(X[col_x].median())

        valor_x = st.slider(
            f"Valor de {col_x}", min_value=minimo, max_value=maximo, value=valor_padrao,
        )

        predicao = prever_novo_valor(modelo, col_x, valor_x)

        st.metric(f"Previsão de {col_y}", f"{predicao:,.2f}")

        ver_codigo(prever_novo_valor, "Ver o código: prever um novo valor")

        rotulo_curva = f"Curva ajustada (grau {grau_polinomial})" if modo_regressao == MODO_POLINOMIAL else "Reta de regressão"

        fig_pred, ax_pred = plt.subplots(figsize=(8, 5))
        ax_pred.scatter(X, y, color="steelblue", alpha=0.4, label="Dados observados")
        x_linha = np.linspace(minimo, maximo, 100).reshape(-1, 1)
        y_linha = modelo.predict(pd.DataFrame(x_linha, columns=[col_x]))
        ax_pred.plot(x_linha, y_linha, color="red", linewidth=2, label=rotulo_curva)
        ax_pred.scatter([valor_x], [predicao], color="black", s=120, zorder=5, marker="*", label="Sua previsão")
        ax_pred.set_xlabel(col_x)
        ax_pred.set_ylabel(col_y)
        ax_pred.legend()
        st.pyplot(fig_pred, use_container_width=False)

        st.caption(
            f"print(f\"O valor previsto de {col_y} para {col_x} = {valor_x:.2f} "
            f"é {{predicao:.2f}}\")  →  O valor previsto de {col_y} é {predicao:,.2f}"
        )

# ----------------------------------------------------------- Comparação ---
if tarefa == TAREFA_REGRESSAO:
    with aba_comparacao:
        st.header("🆚 Leaderboard — todos os modelos de regressão")
        st.markdown(
            "Treina automaticamente **todos os modelos de regressão** possíveis "
            "neste dataset (Simples, Múltipla, Polinomial, Ridge, Lasso e "
            "ElasticNet), usando a mesma divisão treino/teste do painel lateral, "
            "e ranqueia por R² -- sem afetar as outras abas, que continuam "
            "mostrando só o modo ativo escolhido no painel lateral. Na Múltipla, "
            "testa **todas as combinações possíveis** de variáveis candidatas e "
            "usa a de maior R² -- por isso o número de variáveis em \"Detalhes\" "
            "pode ser menor que o total de colunas disponíveis. Já Ridge/Lasso/"
            "ElasticNet usam **todas** as colunas candidatas de uma vez -- a "
            "própria penalidade já reduz/zera as variáveis menos úteis, sem "
            "precisar de uma busca de subconjunto."
        )

        linhas_leaderboard = []
        erros_leaderboard = []

        # Simples -- variável X de maior correlação (em módulo) com y
        try:
            col_x_simples = opcoes_x_ordenadas[0]
            df_s, _ = validar_dados_para_regressao(df, col_x_simples, col_y, test_size)
            X_s, y_s = df_s[[col_x_simples]], df_s[col_y]
            X_s_tr, X_s_te, y_s_tr, y_s_te = dividir_treino_teste(
                X_s, y_s, test_size=test_size, random_state=int(random_state)
            )
            modelo_s, _, _ = treinar_modelo_regressao_simples(X_s_tr, y_s_tr)
            _, r2_s, mae_s, mse_s, rmse_s = avaliar_modelo(modelo_s, X_s_te, y_s_te)
            linhas_leaderboard.append(
                {"Modelo": "Simples", "Detalhes": col_x_simples, "R²": r2_s, "MAE": mae_s, "MSE": mse_s, "RMSE": rmse_s}
            )
        except DadosInvalidosError as erro:
            erros_leaderboard.append(f"Simples: {erro}")

        # Múltipla -- melhor subconjunto entre as colunas numéricas candidatas a X
        # (busca exaustiva por maior R² de teste, não só "todas as colunas";
        # ver ai-dlc/01-inception/INCEPTION.md, rodada v8)
        try:
            if len(opcoes_x_ordenadas) < 2:
                raise DadosInvalidosError("este dataset só tem 1 coluna numérica candidata a X.")
            cols_m, modelo_m, r2_m, mae_m, mse_m, rmse_m = selecionar_melhor_subconjunto_multipla(
                df, opcoes_x_ordenadas, col_y, test_size, random_state=int(random_state)
            )
            linhas_leaderboard.append(
                {
                    "Modelo": "Múltipla",
                    "Detalhes": " + ".join(cols_m),
                    "R²": r2_m,
                    "MAE": mae_m,
                    "MSE": mse_m,
                    "RMSE": rmse_m,
                }
            )
        except DadosInvalidosError as erro:
            erros_leaderboard.append(f"Múltipla: {erro}")

        # Polinomial -- mesma variável X da Simples, grau padrão (2)
        try:
            col_x_poli = opcoes_x_ordenadas[0]
            grau_leaderboard = 2
            df_p, _ = validar_dados_para_regressao(df, col_x_poli, col_y, test_size)
            X_p, y_p = df_p[[col_x_poli]], df_p[col_y]
            X_p_tr, X_p_te, y_p_tr, y_p_te = dividir_treino_teste(
                X_p, y_p, test_size=test_size, random_state=int(random_state)
            )
            modelo_p, _, _ = treinar_modelo_regressao_polinomial(X_p_tr, y_p_tr, grau_leaderboard)
            _, r2_p, mae_p, mse_p, rmse_p = avaliar_modelo(modelo_p, X_p_te, y_p_te)
            linhas_leaderboard.append(
                {
                    "Modelo": f"Polinomial (grau {grau_leaderboard})",
                    "Detalhes": col_x_poli,
                    "R²": r2_p,
                    "MAE": mae_p,
                    "MSE": mse_p,
                    "RMSE": rmse_p,
                }
            )
        except DadosInvalidosError as erro:
            erros_leaderboard.append(f"Polinomial: {erro}")

        # Regularização -- Ridge/Lasso/ElasticNet com TODAS as colunas
        # candidatas a X (ao contrário da Múltipla acima, a penalidade já
        # faz a seleção de variável sozinha -- não precisa de busca de
        # melhor subconjunto); alpha/l1_ratio escolhidos por GridSearchCV
        # dentro do treino, avaliados no mesmo teste (Inception v13).
        try:
            if len(opcoes_x_ordenadas) < 2:
                raise DadosInvalidosError("este dataset só tem 1 coluna numérica candidata a X.")
            df_r, _ = validar_dados_para_regressao_multipla(df, opcoes_x_ordenadas, col_y, test_size)
            X_r, y_r = df_r[opcoes_x_ordenadas], df_r[col_y]
            X_r_tr, X_r_te, y_r_tr, y_r_te = dividir_treino_teste(
                X_r, y_r, test_size=test_size, random_state=int(random_state)
            )
            resultados_reg_leaderboard = treinar_regularizacao_cv(X_r_tr, y_r_tr, X_r_te, y_r_te)
            for nome_modelo in ("Ridge", "Lasso", "ElasticNet"):
                info_modelo = resultados_reg_leaderboard[nome_modelo]
                hiperparametros_fmt = ", ".join(
                    f"{chave}={valor}" for chave, valor in info_modelo["hiperparametros"].items()
                )
                linhas_leaderboard.append(
                    {
                        "Modelo": nome_modelo,
                        "Detalhes": f"{len(opcoes_x_ordenadas)} variáveis ({hiperparametros_fmt})",
                        "R²": info_modelo["r2"],
                        "MAE": info_modelo["mae"],
                        "MSE": info_modelo["mse"],
                        "RMSE": info_modelo["rmse"],
                    }
                )
        except DadosInvalidosError as erro:
            erros_leaderboard.append(f"Regularização (Ridge/Lasso/ElasticNet): {erro}")

        if linhas_leaderboard:
            tabela_leaderboard = (
                pd.DataFrame(linhas_leaderboard).sort_values("R²", ascending=False).reset_index(drop=True)
            )
            tabela_leaderboard.insert(0, "Posição", range(1, len(tabela_leaderboard) + 1))
            st.dataframe(
                tabela_leaderboard.style.format(
                    {"R²": "{:.4f}", "MAE": "{:,.2f}", "MSE": "{:,.2f}", "RMSE": "{:,.2f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

            fig_leader, ax_leader = plt.subplots(figsize=(6, 1.2 + 0.6 * len(tabela_leaderboard)))
            ordem_grafico = tabela_leaderboard.sort_values("R²")
            cores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
            ax_leader.barh(ordem_grafico["Modelo"], ordem_grafico["R²"], color=cores[: len(ordem_grafico)])
            ax_leader.set_xlabel("R² (quanto maior, melhor)")
            r2_min = min(0.0, float(tabela_leaderboard["R²"].min()))
            r2_max = max(1.0, float(tabela_leaderboard["R²"].max()))
            ax_leader.set_xlim(r2_min - 0.05, r2_max + 0.05)
            st.pyplot(fig_leader, use_container_width=False)

            melhor = tabela_leaderboard.iloc[0]
            st.success(
                f"Neste dataset e divisão treino/teste, **{melhor['Modelo']}** "
                f"({melhor['Detalhes']}) teve o maior R² ({melhor['R²']:.4f})."
            )
        else:
            st.warning("Não foi possível treinar nenhum modelo de regressão para comparação.")

        for msg in erros_leaderboard:
            st.info(f"Não foi possível treinar o modelo {msg}")
