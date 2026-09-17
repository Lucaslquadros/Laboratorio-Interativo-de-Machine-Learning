# -*- coding: utf-8 -*-
"""
core.py — núcleo de Regressão Linear Simples, sem nenhuma dependência de UI.

Extraído do app.py (Bolt 1 do AI-DLC) para que a lógica seja testável de
forma isolada, com `pytest`, sem precisar subir o Streamlit. `app.py`
importa estas funções e cuida só da parte visual.

Convenções: os nomes das funções e variáveis seguem o mesmo estilo dos
scripts da aula (`exemplo1_regressaolinearsimples.py` e
`exemplo2_regressaolinearsimples.py`), para que o código continue
reconhecível para quem aprendeu com eles.
"""

import io
import itertools

import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
from scipy import stats
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class DadosInvalidosError(ValueError):
    """Erro com mensagem amigável, pensada para ser mostrada direto ao
    usuário via st.error() — sem stack trace."""


# ============================================================================
# PASSO 0 — Carregar dados
# ============================================================================

def carregar_dataset(caminho_ou_buffer):
    """Carrega um CSV de forma tolerante a vírgula ou ponto-e-vírgula
    (comum em exportações do Excel em pt-BR), e traduz erros de parsing
    em uma mensagem clara em vez de deixar a exceção crua subir para a UI.
    """
    def _tentar_ler(sep):
        # se for um buffer de upload, precisa voltar ao início a cada tentativa
        if hasattr(caminho_ou_buffer, "seek"):
            caminho_ou_buffer.seek(0)
        return pd.read_csv(caminho_ou_buffer, sep=sep)

    ultimo_erro = None
    for separador in (",", ";"):
        try:
            df = _tentar_ler(separador)
        except Exception as erro:  # noqa: BLE001 - queremos capturar qualquer parsing error
            ultimo_erro = erro
            continue

        # heurística: se só achou 1 coluna, o separador provavelmente está errado
        if df.shape[1] > 1:
            return df

    raise DadosInvalidosError(
        "Não consegui ler este CSV. Confira se o arquivo está separado por "
        "vírgula ou ponto-e-vírgula e se tem cabeçalho na primeira linha. "
        f"Detalhe técnico: {ultimo_erro}"
    )


# ============================================================================
# PASSO 1 — Correlação
# ============================================================================

def gerar_matriz_correlacao(df):
    """Tabela de correlação de Pearson entre as variáveis numéricas.

    - 0.9 ou mais (em módulo): correlação muito forte
    - 0.7 a 0.9: correlação forte
    - 0.5 a 0.7: correlação moderada
    - 0.3 a 0.5: correlação fraca
    - abaixo de 0.3: correlação muito fraca / desprezível
    """
    return df.corr(numeric_only=True)


def interpretar_correlacao(valor):
    v = abs(valor)
    if v >= 0.9:
        return "muito forte"
    elif v >= 0.7:
        return "forte"
    elif v >= 0.5:
        return "moderada"
    elif v >= 0.3:
        return "fraca"
    else:
        return "muito fraca / desprezível"


# ============================================================================
# Validação (Bolt 2) — roda ANTES de dividir/treinar
# ============================================================================

def _validar_linhas_e_test_size(df_limpo, len_df_original, test_size, linhas_minimas, descricao_colunas):
    """Checagens comuns entre a validação simples e a múltipla: linhas
    suficientes após remover NaN, e test_size que não zere treino/teste.
    Levanta DadosInvalidosError com mensagem amigável se não der."""
    linhas_removidas = len_df_original - len(df_limpo)

    if len(df_limpo) < linhas_minimas:
        raise DadosInvalidosError(
            f"Este dataset tem só {len(df_limpo)} linha(s) válida(s) para "
            f"{descricao_colunas} (removendo linhas vazias). São "
            f"necessárias pelo menos {linhas_minimas} para dividir em "
            "treino e teste de forma confiável."
        )

    n_teste = round(len(df_limpo) * test_size)
    n_treino = len(df_limpo) - n_teste
    if n_teste < 1 or n_treino < 1:
        raise DadosInvalidosError(
            f"Com {len(df_limpo)} linhas e test_size={test_size}, o conjunto "
            f"de treino ou de teste ficaria vazio. Ajuste o tamanho do "
            "teste no painel lateral."
        )

    return linhas_removidas


def validar_dados_para_regressao(df, col_x, col_y, test_size, linhas_minimas=6):
    """Confere se dá para treinar uma regressão simples com estes dados.
    Levanta DadosInvalidosError com mensagem amigável se não der.

    Retorna o DataFrame já limpo (sem linhas com NaN em X ou y) -- quem
    chamar deve usar o retorno, não o df original, para treinar.
    """
    if col_x not in df.columns:
        raise DadosInvalidosError(f"A coluna '{col_x}' não existe neste dataset.")
    if col_y not in df.columns:
        raise DadosInvalidosError(f"A coluna '{col_y}' não existe neste dataset.")
    if col_x == col_y:
        raise DadosInvalidosError("A variável independente (X) e a dependente (y) não podem ser a mesma coluna.")

    df_limpo = df[[col_x, col_y]].dropna()
    linhas_removidas = _validar_linhas_e_test_size(
        df_limpo, len(df), test_size, linhas_minimas, f"'{col_x}' x '{col_y}'"
    )

    if df_limpo[col_x].nunique() < 2:
        raise DadosInvalidosError(
            f"A coluna '{col_x}' tem sempre o mesmo valor -- não é possível "
            "calcular uma reta de regressão sem variação em X."
        )
    if df_limpo[col_y].nunique() < 2:
        raise DadosInvalidosError(
            f"A coluna '{col_y}' tem sempre o mesmo valor -- não há "
            "variação em y para o modelo explicar."
        )

    return df_limpo, linhas_removidas


def validar_dados_para_regressao_multipla(df, cols_x, col_y, test_size, linhas_minimas=6):
    """Confere se dá para treinar uma regressão MÚLTIPLA (2+ variáveis X)
    com estes dados. Levanta DadosInvalidosError com mensagem amigável se
    não der. Retorna o DataFrame já limpo (sem NaN em nenhuma das colunas
    envolvidas), igual a `validar_dados_para_regressao`."""
    if len(cols_x) < 2:
        raise DadosInvalidosError(
            "A regressão múltipla precisa de pelo menos 2 variáveis "
            "independentes (X). Selecione mais colunas no painel lateral."
        )
    for col_x in cols_x:
        if col_x not in df.columns:
            raise DadosInvalidosError(f"A coluna '{col_x}' não existe neste dataset.")
    if col_y not in df.columns:
        raise DadosInvalidosError(f"A coluna '{col_y}' não existe neste dataset.")
    if col_y in cols_x:
        raise DadosInvalidosError("A variável dependente (y) não pode estar também entre as variáveis independentes (X).")

    df_limpo = df[list(cols_x) + [col_y]].dropna()
    linhas_removidas = _validar_linhas_e_test_size(
        df_limpo, len(df), test_size, linhas_minimas, f"{cols_x} x '{col_y}'"
    )

    for col_x in cols_x:
        if df_limpo[col_x].nunique() < 2:
            raise DadosInvalidosError(
                f"A coluna '{col_x}' tem sempre o mesmo valor -- não é possível "
                "usá-la como variável independente sem variação."
            )
    if df_limpo[col_y].nunique() < 2:
        raise DadosInvalidosError(
            f"A coluna '{col_y}' tem sempre o mesmo valor -- não há "
            "variação em y para o modelo explicar."
        )

    return df_limpo, linhas_removidas


# ============================================================================
# PASSO 2 — Treino / Teste
# ============================================================================

def dividir_treino_teste(X, y, test_size, random_state):
    """Divide os dados em conjunto de TREINAMENTO e de TESTE."""
    X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_treinamento, X_teste, y_treinamento, y_teste


# ============================================================================
# PASSO 3 — Treinar
# ============================================================================

def treinar_modelo_regressao_simples(X_treinamento, y_treinamento):
    """Cria e treina o modelo de Regressão Linear Simples (mínimos
    quadrados) e devolve o intercepto (beta0) e o coeficiente (beta1)."""
    modelo_regressao_simples = LinearRegression()
    modelo_regressao_simples.fit(X_treinamento, y_treinamento)
    intercepto = modelo_regressao_simples.intercept_
    coeficiente = modelo_regressao_simples.coef_[0]
    return modelo_regressao_simples, intercepto, coeficiente


def treinar_modelo_regressao_multipla(X_treinamento, y_treinamento):
    """Cria e treina o modelo de Regressão Linear Múltipla (mínimos
    quadrados, resolvido via equação normal pelo scikit-learn) e devolve o
    intercepto (beta0) e os coeficientes (um por coluna de X, como
    pd.Series indexada pelo nome da coluna -- facilita montar tabela na UI)."""
    modelo_regressao_multipla = LinearRegression()
    modelo_regressao_multipla.fit(X_treinamento, y_treinamento)
    intercepto = modelo_regressao_multipla.intercept_
    coeficientes = pd.Series(modelo_regressao_multipla.coef_, index=X_treinamento.columns)
    return modelo_regressao_multipla, intercepto, coeficientes


def treinar_modelo_regressao_polinomial(X_treinamento, y_treinamento, grau):
    """Cria e treina um modelo de Regressão Polinomial: expande a única
    coluna de X em potências (X, X², ..., X^grau) via `PolynomialFeatures`
    e ajusta uma `LinearRegression` sobre essas colunas expandidas -- o
    modelo continua "linear nos coeficientes" (mínimos quadrados), só que
    a curva resultante em função de X não é mais uma reta.

    Devolve o pipeline treinado (usável direto em `.predict()`, igual aos
    outros modelos), o intercepto e os coeficientes (um por potência de X,
    na ordem X¹, X², ..., X^grau)."""
    modelo_regressao_polinomial = Pipeline(
        [
            ("polinomio", PolynomialFeatures(degree=grau, include_bias=False)),
            ("regressao_linear", LinearRegression()),
        ]
    )
    modelo_regressao_polinomial.fit(X_treinamento, y_treinamento)
    intercepto = modelo_regressao_polinomial.named_steps["regressao_linear"].intercept_
    coeficientes = modelo_regressao_polinomial.named_steps["regressao_linear"].coef_
    return modelo_regressao_polinomial, intercepto, coeficientes


def escolher_grau_polinomial_cv(X_treinamento, y_treinamento, graus=range(1, 7), cv=5):
    """Escolhe o grau do polinômio automaticamente por validação cruzada,
    igual ao método usado nos scripts de aula (`exemplo_regr_polinomial.py`,
    `correcao_polinomial_p1.py`/`p2.py`, Aula6): um `GridSearchCV` testa cada
    grau candidato dentro do mesmo `Pipeline(PolynomialFeatures,
    LinearRegression)` do modo manual, pontuado por MSE negativo médio em
    `cv` dobras, e escolhe o grau de maior `neg_mean_squared_error` (ou seja,
    menor erro).

    Devolve o grau escolhido, o pipeline já treinado com esse grau (nos
    dados passados) e uma `pd.Series` (grau -> neg_MSE médio) para mostrar
    na UI como a aula mostra o resultado do `GridSearchCV`."""
    n_linhas = len(X_treinamento)
    if n_linhas < cv:
        raise DadosInvalidosError(
            f"Poucos dados para validação cruzada com cv={cv}: são "
            f"necessárias pelo menos {cv} linhas de treino, há {n_linhas}."
        )

    pipeline = Pipeline(
        [
            ("polinomio", PolynomialFeatures(include_bias=False)),
            ("regressao_linear", LinearRegression()),
        ]
    )
    grid = GridSearchCV(
        pipeline,
        {"polinomio__degree": list(graus)},
        scoring="neg_mean_squared_error",
        cv=cv,
    )
    grid.fit(X_treinamento, y_treinamento)

    grau_escolhido = grid.best_params_["polinomio__degree"]
    tabela_scores = pd.Series(
        grid.cv_results_["mean_test_score"],
        index=list(graus),
        name="neg_mean_squared_error_medio",
    )
    return grau_escolhido, grid.best_estimator_, tabela_scores


_ALPHAS_PADRAO = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100]
_L1_RATIOS_PADRAO = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]


def treinar_regularizacao_cv(
    X_treinamento, y_treinamento, X_teste, y_teste, cv=10, alphas=_ALPHAS_PADRAO, l1_ratios=_L1_RATIOS_PADRAO
):
    """Compara Regressão Múltipla sem regularização com Ridge, Lasso e
    ElasticNet, no mesmo split treino/teste usado pelo resto do app (ao
    contrário do script de aula -- `exemplo_regularizacao.py`,
    `correcao_Hitters.py`, Aula7 -- que nunca separa treino/teste; decisão
    revista após o Lucas notar que isso impedia comparar de forma justa com
    a Múltipla "oficial" do app, que já usa split):

    - X é escalonado com `StandardScaler`, ajustado só no TREINO (evita
      vazamento de dados do teste) -- ao contrário da OLS pura, a penalidade
      de Ridge/Lasso/ElasticNet depende da escala de cada variável.
    - O `alpha` (Ridge/Lasso) e `alpha`+`l1_ratio` (ElasticNet) são
      escolhidos por `GridSearchCV` (`cv` dobras, `neg_mean_squared_error`)
      rodando **só dentro do treino** -- mesmo padrão já usado em
      `escolher_grau_polinomial_cv()` para o grau do polinômio.
    - Os 4 modelos são então avaliados no mesmo `X_teste`/`y_teste`, com as
      mesmas métricas (R², MAE, MSE, RMSE) já usadas em `avaliar_modelo()`
      -- comparável diretamente com a Múltipla "oficial" mostrada na aba
      Avaliação (o modelo "Sem regularização" aqui deve bater com ela, já
      que escalonar X não muda R²/MAE/MSE/RMSE de uma OLS pura).

    Devolve um dicionário {nome_do_modelo: {"modelo", "coeficientes"
    (pd.Series indexada pelas colunas de X, nos dados ESCALONADOS), "r2",
    "mae", "mse", "rmse", "hiperparametros" (dict, vazio para "Sem
    regularização")}}, na ordem Sem regularização -> Ridge -> Lasso ->
    ElasticNet -- pronto para montar tabela/gráfico comparativo na UI."""
    n_linhas_treino = len(X_treinamento)
    if n_linhas_treino < cv:
        raise DadosInvalidosError(
            f"Poucos dados de treino para validação cruzada com cv={cv}: são "
            f"necessárias pelo menos {cv} linhas, há {n_linhas_treino}."
        )

    colunas = X_treinamento.columns
    escalonador = StandardScaler().fit(X_treinamento)
    X_treinamento_escalonado = escalonador.transform(X_treinamento)
    X_teste_escalonado = escalonador.transform(X_teste)

    def _avaliar(modelo, hiperparametros):
        _, r2, mae, mse, rmse = avaliar_modelo(modelo, X_teste_escalonado, y_teste)
        return {
            "modelo": modelo,
            "coeficientes": pd.Series(modelo.coef_, index=colunas),
            "r2": r2,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "hiperparametros": hiperparametros,
        }

    resultados = {}

    sem_regularizacao = LinearRegression().fit(X_treinamento_escalonado, y_treinamento)
    resultados["Sem regularização"] = _avaliar(sem_regularizacao, {})

    grid_ridge = GridSearchCV(
        Ridge(), {"alpha": list(alphas)}, scoring="neg_mean_squared_error", cv=cv
    )
    grid_ridge.fit(X_treinamento_escalonado, y_treinamento)
    resultados["Ridge"] = _avaliar(grid_ridge.best_estimator_, grid_ridge.best_params_)

    grid_lasso = GridSearchCV(
        Lasso(), {"alpha": list(alphas)}, scoring="neg_mean_squared_error", cv=cv
    )
    grid_lasso.fit(X_treinamento_escalonado, y_treinamento)
    resultados["Lasso"] = _avaliar(grid_lasso.best_estimator_, grid_lasso.best_params_)

    grid_elastic = GridSearchCV(
        ElasticNet(),
        {"alpha": list(alphas), "l1_ratio": list(l1_ratios)},
        scoring="neg_mean_squared_error",
        cv=cv,
    )
    grid_elastic.fit(X_treinamento_escalonado, y_treinamento)
    resultados["ElasticNet"] = _avaliar(grid_elastic.best_estimator_, grid_elastic.best_params_)

    return resultados


def calcular_coeficientes_na_mao(x, y):
    """Cálculo MANUAL de beta0 e beta1, pela fórmula do slide:

        beta1 = soma((xi - media_x) * (yi - media_y)) / soma((xi - media_x)^2)
        beta0 = media_y - beta1 * media_x
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    media_x = x.mean()
    media_y = y.mean()
    denominador = np.sum((x - media_x) ** 2)
    if denominador == 0:
        raise DadosInvalidosError("X não tem variação suficiente para calcular beta1 (divisão por zero).")
    beta1 = np.sum((x - media_x) * (y - media_y)) / denominador
    beta0 = media_y - beta1 * media_x
    return beta0, beta1


# ============================================================================
# PASSO 4 — Avaliar
# ============================================================================

def avaliar_modelo(modelo, X_teste, y_teste):
    """R-quadrado, MAE (erro médio absoluto), MSE e RMSE (erro médio quadrático e sua raiz)."""
    predicoes_modelo = modelo.predict(X_teste)
    r2 = r2_score(y_teste, predicoes_modelo)
    mae = mean_absolute_error(y_teste, predicoes_modelo)
    mse = mean_squared_error(y_teste, predicoes_modelo)
    rmse = np.sqrt(mse)
    return predicoes_modelo, r2, mae, mse, rmse


def selecionar_melhor_subconjunto_multipla(df, cols_x_candidatas, col_y, test_size, random_state, tamanho_minimo=2):
    """Testa TODAS as combinações de `tamanho_minimo`..N das colunas
    candidatas e devolve a combinação com maior R² no conjunto de teste --
    usada pelo leaderboard (aba Comparação) para achar a Múltipla que
    realmente tem o melhor R², em vez de sempre usar todas as colunas
    candidatas (que pode incluir uma variável de baixa correlação e piorar
    o resultado).

    Nenhum dataset do app tem mais de ~5 colunas X candidatas, então a
    busca exaustiva (no máximo 2^N combinações) é instantânea.

    Devolve (colunas_escolhidas, modelo, r2, mae, mse, rmse)."""
    if len(cols_x_candidatas) < tamanho_minimo:
        raise DadosInvalidosError(
            f"É preciso pelo menos {tamanho_minimo} colunas numéricas candidatas "
            "a X para buscar a melhor combinação de Regressão Múltipla."
        )

    melhor = None
    for tamanho in range(tamanho_minimo, len(cols_x_candidatas) + 1):
        for combo in itertools.combinations(cols_x_candidatas, tamanho):
            cols = list(combo)
            df_limpo, _ = validar_dados_para_regressao_multipla(df, cols, col_y, test_size)
            X, y = df_limpo[cols], df_limpo[col_y]
            X_treinamento, X_teste, y_treinamento, y_teste = dividir_treino_teste(
                X, y, test_size=test_size, random_state=random_state
            )
            modelo, _, _ = treinar_modelo_regressao_multipla(X_treinamento, y_treinamento)
            _, r2, mae, mse, rmse = avaliar_modelo(modelo, X_teste, y_teste)
            if melhor is None or r2 > melhor[2]:
                melhor = (cols, modelo, r2, mae, mse, rmse)

    return melhor


# ============================================================================
# Estudo de Adequação do Modelo -- Análise de Resíduos
# ============================================================================

def diagnosticar_residuos(y_real, y_previsto):
    """Resíduo = valor real - valor previsto. Um bom modelo linear tem
    resíduos sem padrão (aleatórios), centrados em 0 e com variância
    constante -- é isso que as suposições da regressão linear assumem."""
    residuos = np.asarray(y_real, dtype=float) - np.asarray(y_previsto, dtype=float)
    return {
        "residuos": residuos,
        "media": float(residuos.mean()),
        "desvio_padrao": float(residuos.std(ddof=1)) if len(residuos) > 1 else 0.0,
    }


def testar_normalidade_residuos(residuos):
    """Teste de Shapiro-Wilk: H0 = os resíduos vêm de uma distribuição
    normal. p-valor > 0.05 -> não há evidência para rejeitar H0 (resíduos
    parecem normais). Precisa de pelo menos 3 resíduos para rodar."""
    residuos = np.asarray(residuos, dtype=float)
    if len(residuos) < 3:
        raise DadosInvalidosError(
            "São necessários pelo menos 3 resíduos para o teste de "
            "normalidade de Shapiro-Wilk -- aumente o conjunto de teste."
        )
    estatistica, p_valor = stats.shapiro(residuos)
    return float(estatistica), float(p_valor)


def testar_homocedasticidade_residuos(residuos, X_teste):
    """Teste de Goldfeld-Quandt: H0 = os resíduos têm variância constante
    (homocedasticidade). p-valor > 0.05 -> não há evidência para rejeitar
    H0 (variância parece constante). Segue a mesma metodologia de
    `aulas/exemplo_teste_suposicao.py`: não reordena as observações antes
    de dividir a amostra em duas partes -- usa a ordem em que já vêm no
    conjunto de teste."""
    residuos = np.asarray(residuos, dtype=float)
    if len(residuos) < 6:
        raise DadosInvalidosError(
            "São necessários pelo menos 6 resíduos para o teste de "
            "homocedasticidade de Goldfeld-Quandt -- aumente o conjunto de teste."
        )
    estatistica, p_valor, _ = sms.het_goldfeldquandt(residuos, np.asarray(X_teste))
    return float(estatistica), float(p_valor)


def testar_independencia_residuos(residuos):
    """Dois testes de independência (ausência de autocorrelação) dos
    resíduos, como em `aulas/exemplo_teste_suposicao.py`:

    - Ljung-Box: H0 = ausência de autocorrelação em qualquer um dos lags
      testados. p-valor > 0.05 em TODOS os lags -> não rejeita H0.
    - Durbin-Watson: estatística entre 0 e 4; próxima de 2 indica ausência
      de autocorrelação (< 1.5 sugere autocorrelação positiva, > 2.5
      sugere autocorrelação negativa).

    Os dois testes podem discordar entre si (Ljung-Box é sensível a vários
    lags ao mesmo tempo; Durbin-Watson foca só na autocorrelação de lag 1)
    -- por isso os dois são reportados juntos, em vez de só um."""
    residuos = np.asarray(residuos, dtype=float)
    if len(residuos) < 3:
        raise DadosInvalidosError(
            "São necessários pelo menos 3 resíduos para os testes de "
            "independência (Ljung-Box/Durbin-Watson) -- aumente o "
            "conjunto de teste."
        )
    lags = min(40, len(residuos) - 1)
    resultado_ljungbox = sms.acorr_ljungbox(residuos, lags=lags, return_df=True)
    p_valor_ljungbox = float(resultado_ljungbox["lb_pvalue"].min())
    estatistica_durbin_watson = float(sms.durbin_watson(residuos))
    return p_valor_ljungbox, estatistica_durbin_watson


# ============================================================================
# PASSO 5 — Prever novo valor
# ============================================================================

def prever_novo_valor(modelo, nome_coluna_x, valor_x):
    """Prevê y para um novo valor de X (dicionário -> DataFrame -> .predict())."""
    novo_dado = {nome_coluna_x: valor_x}
    df_novo = pd.DataFrame(data=novo_dado, index=[0])
    predicao = modelo.predict(df_novo)
    return predicao[0]


def prever_novo_valor_multiplo(modelo, valores_x):
    """Prevê y para um novo conjunto de valores de X (dicionário
    {coluna: valor} -> DataFrame de 1 linha -> .predict())."""
    df_novo = pd.DataFrame(data=valores_x, index=[0])
    predicao = modelo.predict(df_novo)
    return predicao[0]


# ============================================================================
# Regressão Logística -- Classificação binária
#
# Diferente dos modelos acima (alvo y contínuo), aqui o alvo é uma
# categoria com exatamente 2 classes -- por isso as métricas de avaliação
# também são outras (acurácia, precisão, revocação, F1, matriz de
# confusão, em vez de R²/MAE/MSE).
# ============================================================================

def validar_dados_para_classificacao(df, cols_x, col_y, test_size, linhas_minimas=10):
    """Confere se dá para treinar uma Regressão Logística (classificação
    binária) com estes dados. Levanta DadosInvalidosError com mensagem
    amigável se não der. Retorna o DataFrame já limpo (sem NaN nas colunas
    envolvidas)."""
    for col_x in cols_x:
        if col_x not in df.columns:
            raise DadosInvalidosError(f"A coluna '{col_x}' não existe neste dataset.")
    if col_y not in df.columns:
        raise DadosInvalidosError(f"A coluna '{col_y}' não existe neste dataset.")
    if col_y in cols_x:
        raise DadosInvalidosError(
            "A variável dependente (y) não pode estar também entre as variáveis independentes (X)."
        )

    df_limpo = df[list(cols_x) + [col_y]].dropna()
    linhas_removidas = _validar_linhas_e_test_size(
        df_limpo, len(df), test_size, linhas_minimas, f"{cols_x} x '{col_y}'"
    )

    classes = df_limpo[col_y].unique()
    if len(classes) != 2:
        raise DadosInvalidosError(
            f"A coluna '{col_y}' precisa ter exatamente 2 categorias para a "
            f"Regressão Logística -- encontrei {len(classes)}."
        )

    contagem_classes = df_limpo[col_y].value_counts()
    if contagem_classes.min() < 2:
        raise DadosInvalidosError(
            f"Uma das categorias de '{col_y}' tem menos de 2 exemplos -- não "
            "dá para dividir em treino e teste de forma confiável."
        )

    return df_limpo, linhas_removidas


def treinar_modelo_regressao_logistica(X_treinamento, y_treinamento):
    """Cria e treina o modelo de Regressão Logística e devolve o
    intercepto e os coeficientes (um por variável X, em escala de
    log-odds -- o quanto o log da razão de chances muda para cada unidade
    a mais em cada X, mantendo as outras fixas)."""
    modelo_regressao_logistica = LogisticRegression(max_iter=1000)
    modelo_regressao_logistica.fit(X_treinamento, y_treinamento)
    intercepto = modelo_regressao_logistica.intercept_[0]
    coeficientes = pd.Series(modelo_regressao_logistica.coef_[0], index=X_treinamento.columns)
    return modelo_regressao_logistica, intercepto, coeficientes


def avaliar_modelo_classificacao(modelo, X_teste, y_teste):
    """Acurácia, precisão, revocação, F1 e matriz de confusão -- as
    métricas clássicas para avaliar um classificador binário. A "classe
    positiva" usada em precisão/revocação/F1 é `modelo.classes_[-1]`
    (a última em ordem alfabética); devolvida junto para a UI deixar
    claro qual classe está sendo medida."""
    predicoes = modelo.predict(X_teste)
    classe_positiva = modelo.classes_[-1]
    acuracia = accuracy_score(y_teste, predicoes)
    precisao = precision_score(y_teste, predicoes, pos_label=classe_positiva)
    revocacao = recall_score(y_teste, predicoes, pos_label=classe_positiva)
    f1 = f1_score(y_teste, predicoes, pos_label=classe_positiva)
    matriz_confusao = confusion_matrix(y_teste, predicoes, labels=modelo.classes_)
    return predicoes, acuracia, precisao, revocacao, f1, matriz_confusao, classe_positiva


def prever_classe_novo_valor(modelo, valores_x):
    """Prevê a classe e a probabilidade da classe positiva
    (`modelo.classes_[-1]`) para um novo conjunto de valores de X."""
    df_novo = pd.DataFrame(data=valores_x, index=[0])
    classe_prevista = modelo.predict(df_novo)[0]
    probabilidade_classe_positiva = modelo.predict_proba(df_novo)[0][-1]
    return classe_prevista, probabilidade_classe_positiva


# ============================================================================
# Passo 0 -- Classificação do problema (antes de escolher o modelo)
#
# Espelha a taxonomia de `aulas/Intro_AprendMaquina.pdf` (slide 17, "Tipos
# de algoritmos em DM"): Supervisionado (Regressão / Classificação binária
# ou multiclasse) x Não Supervisionado (Associação / Clusterização). O
# laboratório hoje só implementa o ramo Supervisionado -- o Não Supervisionado
# aparece na resposta só como orientação (item 14 do roadmap do syllabus).
# ============================================================================

def classificar_tipo_problema(rotulado, tipo_alvo=None, num_classes=None):
    """Dadas as respostas do quiz do Passo 0, devolve a família do problema
    (segundo a taxonomia do professor) e qual "Tipo de Tarefa" da barra
    lateral usar.

    - `rotulado`: True se o dataset tem um atributo-alvo (y) conhecido.
    - `tipo_alvo`: "contínuo" ou "categórico" (só importa se `rotulado`).
    - `num_classes`: nº de categorias do alvo (só importa se `tipo_alvo`
      for "categórico"); 2 = binária, 3+ = multiclasse.
    """
    if not rotulado:
        return {
            "categoria": "Aprendizado Não Supervisionado",
            "subtipo": "Associação ou Clusterização/Agrupamento",
            "recomendacao_app": None,
            "explicacao": (
                "Sem atributo-alvo conhecido, o problema é de aprendizado "
                "não supervisionado (Associação ou Clusterização/"
                "Agrupamento) -- este laboratório ainda não implementa esses "
                "algoritmos (item 14 do roadmap do syllabus)."
            ),
        }

    if tipo_alvo == "categórico":
        subtipo = "Binária (2 classes)" if num_classes == 2 else "Multiclasse (3+ classes)"
        return {
            "categoria": "Aprendizado Supervisionado -- Classificação",
            "subtipo": subtipo,
            "recomendacao_app": "Classificação",
            "explicacao": (
                f"Atributo-alvo (y) é categórico -- {subtipo} -- então é um "
                "problema de Classificação. Use o Tipo de Tarefa "
                "'Classificação' na barra lateral. (Atenção: apesar do "
                "nome, a Regressão Logística é um algoritmo de "
                "Classificação, não de Regressão.)"
            ),
        }

    # tipo_alvo == "contínuo" (default do ramo Supervisionado)
    return {
        "categoria": "Aprendizado Supervisionado -- Regressão",
        "subtipo": None,
        "recomendacao_app": "Regressão",
        "explicacao": (
            "Atributo-alvo (y) é um valor contínuo, então é um problema de "
            "Regressão. Use o Tipo de Tarefa 'Regressão' na barra lateral."
        ),
    }
