# -*- coding: utf-8 -*-
"""
Testes do núcleo de Regressão Linear Simples (core.py).

Bolt 4 do AI-DLC: cobre o essencial --
  1) o cálculo manual de beta0/beta1 bate com o scikit-learn;
  2) as métricas de avaliação são coerentes;
  3) os casos de erro do Bolt 2 (validação) realmente disparam, com
     mensagem amigável, em vez de deixar a exceção "crua" do pandas/sklearn
     subir.

Rodar com:  pytest
"""

import io
import os

import numpy as np
import pandas as pd
import pytest

import core

DIR_DESTE_ARQUIVO = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR_DESTE_ARQUIVO, "..", "data")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def df_distancia_consumo():
    """O mesmo dataset de 10 carros do slide da aula -- serve de "gabarito"
    porque sabemos os valores exatos esperados (y = 0.5716 + 0.0663X)."""
    return core.carregar_dataset(os.path.join(DATA_DIR, "distancia_consumo.csv"))


@pytest.fixture
def df_mortalidade_infantil():
    """Dataset com colinearidade real entre X (Saneamento_pct x
    Agua_Potavel_pct, r~0.906) -- usado para testar Regularização porque
    reproduz a mesma situação do diabetes.csv do script de aula (s1/s2,
    r~0.896), sem precisar do escalonamento não-padrão daquele dataset."""
    return core.carregar_dataset(os.path.join(DATA_DIR, "mortalidade_infantil_desenvolvimento.csv"))


# ============================================================================
# 1) Cálculo manual == scikit-learn (a "caixa-preta" foi aberta com sucesso)
# ============================================================================

def test_coeficientes_na_mao_batem_com_sklearn(df_distancia_consumo):
    X = df_distancia_consumo[["Distancia"]]
    y = df_distancia_consumo["Consumo"]

    modelo, intercepto_sklearn, coef_sklearn = core.treinar_modelo_regressao_simples(X, y)
    beta0_manual, beta1_manual = core.calcular_coeficientes_na_mao(X["Distancia"], y)

    assert intercepto_sklearn == pytest.approx(beta0_manual, abs=1e-6)
    assert coef_sklearn == pytest.approx(beta1_manual, abs=1e-6)


def test_coeficientes_batem_com_o_slide_da_aula(df_distancia_consumo):
    """Confere contra os números exatos que aparecem no slide do professor:
    y = 0.5716 + 0.0663 X"""
    X = df_distancia_consumo[["Distancia"]]
    y = df_distancia_consumo["Consumo"]

    _, intercepto, coeficiente = core.treinar_modelo_regressao_simples(X, y)

    assert intercepto == pytest.approx(0.5716, abs=1e-3)
    assert coeficiente == pytest.approx(0.0663, abs=1e-3)


# ============================================================================
# 2) Métricas de avaliação
# ============================================================================

def test_avaliar_modelo_com_predicao_perfeita_da_r2_igual_a_1():
    # y = 2x, sem ruído -> ajuste perfeito
    X = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    y = pd.Series([2, 4, 6, 8, 10])

    modelo, _, _ = core.treinar_modelo_regressao_simples(X, y)
    _, r2, mae, mse, rmse = core.avaliar_modelo(modelo, X, y)

    assert r2 == pytest.approx(1.0, abs=1e-9)
    assert mae == pytest.approx(0.0, abs=1e-9)
    assert mse == pytest.approx(0.0, abs=1e-9)
    assert rmse == pytest.approx(0.0, abs=1e-9)


def test_avaliar_modelo_rmse_e_a_raiz_do_mse():
    X = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    y = pd.Series([2, 5, 6, 7, 11])

    modelo, _, _ = core.treinar_modelo_regressao_simples(X, y)
    _, _, _, mse, rmse = core.avaliar_modelo(modelo, X, y)

    assert rmse == pytest.approx(np.sqrt(mse))


def test_prever_novo_valor_usa_a_equacao_da_reta():
    X = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    y = pd.Series([2, 4, 6, 8, 10])
    modelo, intercepto, coeficiente = core.treinar_modelo_regressao_simples(X, y)

    predicao = core.prever_novo_valor(modelo, "x", 10)

    assert predicao == pytest.approx(intercepto + coeficiente * 10, abs=1e-9)


# ============================================================================
# 3) Interpretação da força da correlação (régua do slide)
# ============================================================================

@pytest.mark.parametrize(
    "valor,esperado",
    [
        (0.95, "muito forte"),
        (-0.95, "muito forte"),
        (0.8, "forte"),
        (0.6, "moderada"),
        (0.4, "fraca"),
        (0.1, "muito fraca / desprezível"),
    ],
)
def test_interpretar_correlacao(valor, esperado):
    assert core.interpretar_correlacao(valor) == esperado


# ============================================================================
# 4) Validação de dados (Bolt 2) -- os casos de erro devem falhar "bonito"
# ============================================================================

def test_validacao_ok_com_dataset_da_aula(df_distancia_consumo):
    df_limpo, linhas_removidas = core.validar_dados_para_regressao(
        df_distancia_consumo, "Distancia", "Consumo", test_size=0.3
    )
    assert linhas_removidas == 0
    assert len(df_limpo) == len(df_distancia_consumo)


def test_validacao_falha_com_poucas_linhas():
    df_pequeno = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_regressao(df_pequeno, "x", "y", test_size=0.3, linhas_minimas=6)


def test_validacao_falha_com_coluna_x_constante():
    df_constante = pd.DataFrame({"x": [5] * 10, "y": range(10)})
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_regressao(df_constante, "x", "y", test_size=0.3)


def test_validacao_falha_com_coluna_y_constante():
    df_constante = pd.DataFrame({"x": range(10), "y": [5] * 10})
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_regressao(df_constante, "x", "y", test_size=0.3)


def test_validacao_falha_com_colunas_iguais():
    df = pd.DataFrame({"x": range(10)})
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_regressao(df, "x", "x", test_size=0.3)


def test_validacao_falha_com_test_size_que_esvazia_o_teste():
    # 6 linhas, test_size muito pequeno -> arredondaria para 0 linhas de teste
    df = pd.DataFrame({"x": range(6), "y": range(6)})
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_regressao(df, "x", "y", test_size=0.05, linhas_minimas=6)


def test_validacao_remove_linhas_com_nan_e_avisa():
    df_com_nan = pd.DataFrame(
        {"x": [1, 2, 3, 4, None, 6, 7, 8], "y": [1, 2, 3, None, 5, 6, 7, 8]}
    )
    df_limpo, linhas_removidas = core.validar_dados_para_regressao(
        df_com_nan, "x", "y", test_size=0.3, linhas_minimas=3
    )
    assert linhas_removidas == 2
    assert df_limpo.isna().sum().sum() == 0


def test_validacao_falha_com_coluna_inexistente():
    df = pd.DataFrame({"x": range(10), "y": range(10)})
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_regressao(df, "coluna_que_nao_existe", "y", test_size=0.3)


# ============================================================================
# 5) Carregamento de CSV tolerante (Bolt 3)
# ============================================================================

def test_carregar_dataset_aceita_separador_ponto_e_virgula():
    csv_com_ponto_e_virgula = "x;y\n1;2\n3;4\n5;6\n"
    buffer = io.BytesIO(csv_com_ponto_e_virgula.encode("utf-8"))

    df = core.carregar_dataset(buffer)

    assert list(df.columns) == ["x", "y"]
    assert len(df) == 3


def test_carregar_dataset_aceita_separador_virgula():
    csv_com_virgula = "x,y\n1,2\n3,4\n5,6\n"
    buffer = io.BytesIO(csv_com_virgula.encode("utf-8"))

    df = core.carregar_dataset(buffer)

    assert list(df.columns) == ["x", "y"]
    assert len(df) == 3


def test_carregar_dataset_com_conteudo_invalido_da_erro_amigavel():
    buffer = io.BytesIO(b"")  # arquivo vazio
    with pytest.raises(core.DadosInvalidosError):
        core.carregar_dataset(buffer)


# ============================================================================
# 6) Regressão Linear Múltipla
# ============================================================================

@pytest.fixture
def df_usa_housing():
    """Dataset com várias colunas numéricas -- usado para testar regressão
    com 2+ variáveis independentes ao mesmo tempo."""
    return core.carregar_dataset(os.path.join(DATA_DIR, "USA_Housing.csv"))


@pytest.fixture
def df_diagnostico_cancer_mama():
    """Dataset com alvo categórico genuíno (Maligno/Benigno) -- usado para
    testar a Regressão Logística (classificação binária)."""
    return core.carregar_dataset(os.path.join(DATA_DIR, "diagnostico_cancer_mama.csv"))


def test_validacao_multipla_falha_com_menos_de_duas_colunas_x(df_usa_housing):
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_regressao_multipla(
            df_usa_housing, ["Avg. Area Income"], "Price", test_size=0.3
        )


def test_validacao_multipla_falha_com_coluna_y_entre_as_colunas_x(df_usa_housing):
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_regressao_multipla(
            df_usa_housing, ["Avg. Area Income", "Price"], "Price", test_size=0.3
        )


def test_validacao_multipla_falha_com_poucas_linhas():
    df_pequeno = pd.DataFrame({"x1": [1, 2, 3], "x2": [3, 2, 1], "y": [1, 2, 3]})
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_regressao_multipla(
            df_pequeno, ["x1", "x2"], "y", test_size=0.3, linhas_minimas=6
        )


def test_validacao_multipla_falha_com_coluna_x_constante():
    df = pd.DataFrame({"x1": range(10), "x2": [5] * 10, "y": range(10)})
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_regressao_multipla(df, ["x1", "x2"], "y", test_size=0.3)


def test_validacao_multipla_ok_remove_linhas_com_nan(df_usa_housing):
    cols_x = ["Avg. Area Income", "Avg. Area House Age"]
    df_limpo, linhas_removidas = core.validar_dados_para_regressao_multipla(
        df_usa_housing, cols_x, "Price", test_size=0.3
    )
    assert linhas_removidas == 0
    assert df_limpo.isna().sum().sum() == 0
    assert list(df_limpo.columns) == cols_x + ["Price"]


def test_treinar_modelo_multiplo_devolve_um_coeficiente_por_coluna(df_usa_housing):
    cols_x = ["Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms"]
    df_limpo, _ = core.validar_dados_para_regressao_multipla(df_usa_housing, cols_x, "Price", test_size=0.3)
    X, y = df_limpo[cols_x], df_limpo["Price"]
    X_treinamento, X_teste, y_treinamento, y_teste = core.dividir_treino_teste(
        X, y, test_size=0.3, random_state=0
    )

    modelo, intercepto, coeficientes = core.treinar_modelo_regressao_multipla(X_treinamento, y_treinamento)

    assert isinstance(coeficientes, pd.Series)
    assert list(coeficientes.index) == cols_x
    assert len(coeficientes) == len(cols_x)

    # avaliar_modelo (genérico) e prever_novo_valor_multiplo devem funcionar
    # normalmente com um modelo de múltiplas variáveis.
    _, r2, mae, mse, rmse = core.avaliar_modelo(modelo, X_teste, y_teste)
    assert 0.0 <= r2 <= 1.0
    assert mae >= 0.0
    assert mse >= 0.0

    predicao = core.prever_novo_valor_multiplo(modelo, {c: float(X[c].median()) for c in cols_x})
    esperado = intercepto + sum(coeficientes[c] * X[c].median() for c in cols_x)
    assert predicao == pytest.approx(esperado, rel=1e-6)


def test_regressao_multipla_bate_com_r2_do_script_de_aula_50_startups():
    """`aulas/exemplo1_regressaolinearmultipla.py` treina X=['R&D Spend',
    'Marketing Spend'] -> y='Profit' com test_size=0.3, random_state=0 e
    obtém R²=0.9431 -- confere que `data/50_Startups.csv` (mesmas colunas,
    sem as dummies de Estado) reproduz esse valor de referência."""
    df = core.carregar_dataset(os.path.join(DATA_DIR, "50_Startups.csv"))
    cols_x = ["R&D Spend", "Marketing Spend"]
    col_y = "Profit"

    df_limpo, _ = core.validar_dados_para_regressao_multipla(df, cols_x, col_y, test_size=0.3)
    X, y = df_limpo[cols_x], df_limpo[col_y]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)
    modelo, _, _ = core.treinar_modelo_regressao_multipla(X_tr, y_tr)
    _, r2, _, _, _ = core.avaliar_modelo(modelo, X_te, y_te)

    assert r2 == pytest.approx(0.9431, abs=1e-3)


def test_melhor_subconjunto_multipla_descarta_variavel_de_baixa_correlacao_50_startups():
    """No dataset 50 Startups, incluir 'Administration' (baixa correlação
    com Profit) piora o R² de teste -- a busca de melhor subconjunto deve
    descartá-la e escolher só ['R&D Spend', 'Marketing Spend'], batendo
    com a referência da aula (R²≈0.9431). Ver ai-dlc/01-inception/
    INCEPTION.md, rodada v8: antes deste bolt, o leaderboard usava sempre
    as 3 colunas e caía para R²≈0.9355."""
    df = core.carregar_dataset(os.path.join(DATA_DIR, "50_Startups.csv"))
    candidatas = ["R&D Spend", "Administration", "Marketing Spend"]
    col_y = "Profit"

    cols_escolhidas, _, r2, _, _, _ = core.selecionar_melhor_subconjunto_multipla(
        df, candidatas, col_y, test_size=0.3, random_state=0
    )

    assert set(cols_escolhidas) == {"R&D Spend", "Marketing Spend"}
    assert r2 == pytest.approx(0.9431, abs=1e-3)


def test_melhor_subconjunto_multipla_falha_com_menos_de_duas_colunas_candidatas(df_usa_housing):
    with pytest.raises(core.DadosInvalidosError):
        core.selecionar_melhor_subconjunto_multipla(
            df_usa_housing, ["Avg. Area Income"], "Price", test_size=0.3, random_state=0
        )


def test_regressao_multipla_explica_pelo_menos_tanto_quanto_a_melhor_simples(df_usa_housing):
    """Com mais informação disponível (mais colunas X), o R² da múltipla
    não deveria ficar pior que o da melhor variável simples isolada."""
    col_y = "Price"
    cols_x = ["Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms"]

    df_limpo, _ = core.validar_dados_para_regressao_multipla(df_usa_housing, cols_x, col_y, test_size=0.3)
    X, y = df_limpo[cols_x], df_limpo[col_y]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)
    modelo_multiplo, _, _ = core.treinar_modelo_regressao_multipla(X_tr, y_tr)
    _, r2_multiplo, _, _, _ = core.avaliar_modelo(modelo_multiplo, X_te, y_te)

    melhor_r2_simples = 0.0
    for col_x in cols_x:
        df_s, _ = core.validar_dados_para_regressao(df_usa_housing, col_x, col_y, test_size=0.3)
        X_s_tr, X_s_te, y_s_tr, y_s_te = core.dividir_treino_teste(
            df_s[[col_x]], df_s[col_y], test_size=0.3, random_state=0
        )
        modelo_simples, _, _ = core.treinar_modelo_regressao_simples(X_s_tr, y_s_tr)
        _, r2_simples, _, _, _ = core.avaliar_modelo(modelo_simples, X_s_te, y_s_te)
        melhor_r2_simples = max(melhor_r2_simples, r2_simples)

    assert r2_multiplo >= melhor_r2_simples - 1e-9


# ============================================================================
# 7) Estudo de Adequação do Modelo -- Análise de Resíduos
# ============================================================================

def test_diagnosticar_residuos_com_ajuste_perfeito():
    y_real = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_previsto = [1.0, 2.0, 3.0, 4.0, 5.0]

    diagnostico = core.diagnosticar_residuos(y_real, y_previsto)

    assert (diagnostico["residuos"] == 0).all()
    assert diagnostico["media"] == pytest.approx(0.0)
    assert diagnostico["desvio_padrao"] == pytest.approx(0.0)


def test_diagnosticar_residuos_calcula_real_menos_previsto():
    y_real = [10.0, 20.0, 30.0]
    y_previsto = [8.0, 22.0, 33.0]

    diagnostico = core.diagnosticar_residuos(y_real, y_previsto)

    assert diagnostico["residuos"] == pytest.approx([2.0, -2.0, -3.0])


def test_testar_normalidade_residuos_com_amostra_normal():
    rng = np.random.default_rng(42)
    residuos_normais = rng.normal(loc=0, scale=1, size=200)

    estatistica, p_valor = core.testar_normalidade_residuos(residuos_normais)

    assert 0.0 <= estatistica <= 1.0
    assert p_valor > 0.05  # não deve rejeitar normalidade de dados gerados como normais


def test_testar_normalidade_residuos_falha_com_poucos_pontos():
    with pytest.raises(core.DadosInvalidosError):
        core.testar_normalidade_residuos([1.0, 2.0])


def test_diagnostico_de_residuos_no_fluxo_completo(df_distancia_consumo):
    """Confere que diagnosticar_residuos encaixa direto na saída de
    avaliar_modelo, como app.py usa."""
    X = df_distancia_consumo[["Distancia"]]
    y = df_distancia_consumo["Consumo"]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)
    modelo, _, _ = core.treinar_modelo_regressao_simples(X_tr, y_tr)
    predicoes, _, _, _, _ = core.avaliar_modelo(modelo, X_te, y_te)

    diagnostico = core.diagnosticar_residuos(y_te, predicoes)

    assert len(diagnostico["residuos"]) == len(y_te)
    estatistica, p_valor = core.testar_normalidade_residuos(diagnostico["residuos"])
    assert 0.0 <= p_valor <= 1.0


# ============================================================================
# 8) Regressão Polinomial
# ============================================================================

def test_regressao_polinomial_grau_1_bate_com_regressao_simples(df_distancia_consumo):
    """Grau 1 do polinômio não passa de uma reta -- deve reproduzir
    exatamente os mesmos coeficientes da Regressão Linear Simples."""
    X = df_distancia_consumo[["Distancia"]]
    y = df_distancia_consumo["Consumo"]

    _, intercepto_simples, coeficiente_simples = core.treinar_modelo_regressao_simples(X, y)
    _, intercepto_poli, coeficientes_poli = core.treinar_modelo_regressao_polinomial(X, y, grau=1)

    assert intercepto_poli == pytest.approx(intercepto_simples, abs=1e-6)
    assert coeficientes_poli[0] == pytest.approx(coeficiente_simples, abs=1e-6)


def test_regressao_polinomial_devolve_um_coeficiente_por_grau(df_distancia_consumo):
    X = df_distancia_consumo[["Distancia"]]
    y = df_distancia_consumo["Consumo"]

    for grau in (2, 3):
        _, _, coeficientes = core.treinar_modelo_regressao_polinomial(X, y, grau=grau)
        assert len(coeficientes) == grau


def test_regressao_polinomial_funciona_com_avaliar_modelo_e_prever_novo_valor(df_distancia_consumo):
    """O pipeline (PolynomialFeatures + LinearRegression) precisa funcionar
    sem alterações nas funções genéricas já existentes."""
    X = df_distancia_consumo[["Distancia"]]
    y = df_distancia_consumo["Consumo"]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)

    modelo, _, _ = core.treinar_modelo_regressao_polinomial(X_tr, y_tr, grau=2)

    _, r2, mae, mse, rmse = core.avaliar_modelo(modelo, X_te, y_te)
    assert mae >= 0.0
    assert mse >= 0.0

    predicao = core.prever_novo_valor(modelo, "Distancia", 50.0)
    assert isinstance(predicao, (int, float, np.floating))


def test_escolher_grau_polinomial_cv_acerta_grau_conhecido():
    """Dados gerados por y = x² sem ruído: o GridSearchCV deve escolher
    grau 2 (ou próximo, com poucos dados o CV pode empatar), nunca grau 1
    (reta), que erraria feio uma parábola."""
    rng = np.random.RandomState(0)
    x = np.linspace(-10, 10, 60)
    y = x**2 + rng.normal(scale=0.1, size=len(x))
    X = pd.DataFrame({"x": x})
    y = pd.Series(y)

    grau, modelo, tabela_scores = core.escolher_grau_polinomial_cv(X, y, graus=range(1, 5), cv=5)

    assert grau in (2, 3, 4)
    assert grau != 1
    assert list(tabela_scores.index) == [1, 2, 3, 4]
    assert tabela_scores[grau] == tabela_scores.max()


def test_escolher_grau_polinomial_cv_pipeline_treinado_funciona_com_avaliar_modelo(df_distancia_consumo):
    """O pipeline devolvido já vem treinado (fit no X/y completos passados)
    e deve encaixar sem alteração nas funções genéricas já existentes."""
    X = df_distancia_consumo[["Distancia"]]
    y = df_distancia_consumo["Consumo"]

    grau, modelo, _ = core.escolher_grau_polinomial_cv(X, y, cv=3)

    _, r2, mae, mse, rmse = core.avaliar_modelo(modelo, X, y)
    assert mae >= 0.0
    assert isinstance(grau, (int, np.integer))


def test_escolher_grau_polinomial_cv_reproduz_exemplo_polinomial_comissao():
    """`exemplo_polinomial.py` (Aula6): dataset construído como uma parábola
    exata -- GridSearchCV deve escolher grau 2 e o ajuste deve ficar
    essencialmente perfeito (R² ≈ 1.0), reproduzindo o script de aula."""
    df = core.carregar_dataset(os.path.join(DATA_DIR, "comissao.csv"))
    X, y = df[["quantidade"]], df["comissao"]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)

    grau, modelo, _ = core.escolher_grau_polinomial_cv(X_tr, y_tr)
    _, r2, _, _, rmse = core.avaliar_modelo(modelo, X_te, y_te)

    assert grau == 2
    assert r2 == pytest.approx(1.0, abs=1e-6)
    assert rmse < 1e-6


def test_escolher_grau_polinomial_cv_poucas_linhas_levanta_erro():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 4.0, 9.0])

    with pytest.raises(core.DadosInvalidosError):
        core.escolher_grau_polinomial_cv(X, y, cv=5)


# ============================================================================
# 8b) Regularização -- Ridge / Lasso / ElasticNet (Inception v13)
# ============================================================================

def test_treinar_regularizacao_cv_devolve_os_4_modelos_na_ordem(df_mortalidade_infantil):
    X = df_mortalidade_infantil[["Saneamento_pct", "Agua_Potavel_pct", "PIB_per_capita"]]
    y = df_mortalidade_infantil["Mortalidade_Infantil"]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)

    resultados = core.treinar_regularizacao_cv(X_tr, y_tr, X_te, y_te, cv=5)

    assert list(resultados.keys()) == ["Sem regularização", "Ridge", "Lasso", "ElasticNet"]
    for nome, info in resultados.items():
        assert set(info.keys()) == {"modelo", "coeficientes", "r2", "mae", "mse", "rmse", "hiperparametros"}
        assert len(info["coeficientes"]) == 3
        assert info["mae"] >= 0.0
        assert info["mse"] >= 0.0
        assert info["rmse"] == pytest.approx(info["mse"] ** 0.5)

    assert resultados["Sem regularização"]["hiperparametros"] == {}
    assert "alpha" in resultados["Ridge"]["hiperparametros"]
    assert "alpha" in resultados["Lasso"]["hiperparametros"]
    assert {"alpha", "l1_ratio"} == set(resultados["ElasticNet"]["hiperparametros"].keys())


def test_treinar_regularizacao_cv_sem_regularizacao_bate_com_multipla_sem_escalonar(df_mortalidade_infantil):
    """Escalonar X não muda R²/MAE/MSE/RMSE de uma OLS pura (mesma
    conclusão já validada na v9 para o Diagnóstico dos Resíduos) -- então a
    linha "Sem regularização" (calculada com X escalonado) deve bater com a
    Múltipla "oficial" do app (calculada sem escalonar), no mesmo split."""
    X = df_mortalidade_infantil[["Saneamento_pct", "Agua_Potavel_pct", "PIB_per_capita"]]
    y = df_mortalidade_infantil["Mortalidade_Infantil"]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)

    modelo_oficial, _, _ = core.treinar_modelo_regressao_multipla(X_tr, y_tr)
    _, r2_oficial, mae_oficial, mse_oficial, rmse_oficial = core.avaliar_modelo(modelo_oficial, X_te, y_te)

    resultados = core.treinar_regularizacao_cv(X_tr, y_tr, X_te, y_te, cv=5)
    sem_regularizacao = resultados["Sem regularização"]

    assert sem_regularizacao["r2"] == pytest.approx(r2_oficial, abs=1e-8)
    assert sem_regularizacao["mae"] == pytest.approx(mae_oficial, abs=1e-6)
    assert sem_regularizacao["mse"] == pytest.approx(mse_oficial, abs=1e-6)
    assert sem_regularizacao["rmse"] == pytest.approx(rmse_oficial, abs=1e-6)


def test_treinar_regularizacao_cv_ridge_encolhe_coeficientes_da_ols(df_mortalidade_infantil):
    """Com colinearidade real entre X (Saneamento_pct x Agua_Potavel_pct),
    Ridge deve reduzir a magnitude média dos coeficientes em relação à OLS
    sem regularização -- é a lição central do script de aula."""
    X = df_mortalidade_infantil[["Saneamento_pct", "Agua_Potavel_pct", "PIB_per_capita"]]
    y = df_mortalidade_infantil["Mortalidade_Infantil"]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)

    resultados = core.treinar_regularizacao_cv(X_tr, y_tr, X_te, y_te, cv=5)

    norma_sem_regularizacao = np.abs(resultados["Sem regularização"]["coeficientes"]).sum()
    norma_ridge = np.abs(resultados["Ridge"]["coeficientes"]).sum()
    assert norma_ridge <= norma_sem_regularizacao


def test_treinar_regularizacao_cv_poucas_linhas_de_treino_levanta_erro():
    X_tr = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 1.0, 4.0]})
    y_tr = pd.Series([1.0, 4.0, 9.0])
    X_te = pd.DataFrame({"a": [1.5], "b": [2.5]})
    y_te = pd.Series([2.0])

    with pytest.raises(core.DadosInvalidosError):
        core.treinar_regularizacao_cv(X_tr, y_tr, X_te, y_te, cv=5)


# ============================================================================
# 9) Regressão Logística -- Classificação binária
# ============================================================================

def test_validacao_classificacao_ok_com_dataset_de_cancer(df_diagnostico_cancer_mama):
    cols_x = ["mean radius", "mean texture"]
    df_limpo, linhas_removidas = core.validar_dados_para_classificacao(
        df_diagnostico_cancer_mama, cols_x, "Diagnostico", test_size=0.3
    )
    assert linhas_removidas == 0
    assert set(df_limpo["Diagnostico"].unique()) == {"Maligno", "Benigno"}


def test_validacao_classificacao_falha_com_alvo_de_3_categorias():
    df = pd.DataFrame({"x": range(9), "y": ["A", "B", "C"] * 3})
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_classificacao(df, ["x"], "y", test_size=0.3)


def test_validacao_classificacao_falha_com_alvo_de_1_categoria():
    df = pd.DataFrame({"x": range(9), "y": ["A"] * 9})
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_classificacao(df, ["x"], "y", test_size=0.3)


def test_validacao_classificacao_falha_com_coluna_x_igual_a_y():
    df = pd.DataFrame({"x": ["A", "B"] * 5})
    with pytest.raises(core.DadosInvalidosError):
        core.validar_dados_para_classificacao(df, ["x"], "x", test_size=0.3)


def test_treinar_modelo_logistico_devolve_um_coeficiente_por_coluna(df_diagnostico_cancer_mama):
    cols_x = ["mean radius", "mean texture", "mean concavity"]
    df_limpo, _ = core.validar_dados_para_classificacao(
        df_diagnostico_cancer_mama, cols_x, "Diagnostico", test_size=0.3
    )
    X, y = df_limpo[cols_x], df_limpo["Diagnostico"]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)

    modelo, intercepto, coeficientes = core.treinar_modelo_regressao_logistica(X_tr, y_tr)

    assert isinstance(coeficientes, pd.Series)
    assert list(coeficientes.index) == cols_x
    assert isinstance(intercepto, float)


def test_avaliar_modelo_classificacao_devolve_metricas_coerentes(df_diagnostico_cancer_mama):
    cols_x = ["mean radius", "mean texture", "mean concavity", "mean symmetry"]
    df_limpo, _ = core.validar_dados_para_classificacao(
        df_diagnostico_cancer_mama, cols_x, "Diagnostico", test_size=0.3
    )
    X, y = df_limpo[cols_x], df_limpo["Diagnostico"]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)
    modelo, _, _ = core.treinar_modelo_regressao_logistica(X_tr, y_tr)

    predicoes, acuracia, precisao, revocacao, f1, matriz_confusao, classe_positiva = (
        core.avaliar_modelo_classificacao(modelo, X_te, y_te)
    )

    assert 0.0 <= acuracia <= 1.0
    assert 0.0 <= precisao <= 1.0
    assert 0.0 <= revocacao <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert classe_positiva == "Maligno"  # última em ordem alfabética (Benigno < Maligno)
    assert matriz_confusao.shape == (2, 2)
    assert matriz_confusao.sum() == len(y_te)
    # acurácia por definição deve bater com a soma da diagonal / total
    assert acuracia == pytest.approx(np.trace(matriz_confusao) / matriz_confusao.sum())


def test_prever_classe_novo_valor(df_diagnostico_cancer_mama):
    cols_x = ["mean radius", "mean texture"]
    df_limpo, _ = core.validar_dados_para_classificacao(
        df_diagnostico_cancer_mama, cols_x, "Diagnostico", test_size=0.3
    )
    X, y = df_limpo[cols_x], df_limpo["Diagnostico"]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)
    modelo, _, _ = core.treinar_modelo_regressao_logistica(X_tr, y_tr)

    classe_prevista, probabilidade = core.prever_classe_novo_valor(
        modelo, {c: float(X[c].median()) for c in cols_x}
    )

    assert classe_prevista in ("Maligno", "Benigno")
    assert 0.0 <= probabilidade <= 1.0


# ============================================================================
# 9) Passo 0 -- Classificação do problema
# ============================================================================

def test_classificar_problema_nao_rotulado_recomenda_nao_supervisionado():
    resultado = core.classificar_tipo_problema(rotulado=False)
    assert resultado["categoria"] == "Aprendizado Não Supervisionado"
    assert resultado["recomendacao_app"] is None


def test_classificar_problema_rotulado_continuo_recomenda_regressao():
    resultado = core.classificar_tipo_problema(rotulado=True, tipo_alvo="contínuo")
    assert resultado["categoria"] == "Aprendizado Supervisionado -- Regressão"
    assert resultado["recomendacao_app"] == "Regressão"


def test_classificar_problema_rotulado_categorico_binario_recomenda_classificacao():
    resultado = core.classificar_tipo_problema(rotulado=True, tipo_alvo="categórico", num_classes=2)
    assert resultado["categoria"] == "Aprendizado Supervisionado -- Classificação"
    assert resultado["subtipo"] == "Binária (2 classes)"
    assert resultado["recomendacao_app"] == "Classificação"


def test_classificar_problema_rotulado_categorico_multiclasse_recomenda_classificacao():
    resultado = core.classificar_tipo_problema(rotulado=True, tipo_alvo="categórico", num_classes=5)
    assert resultado["categoria"] == "Aprendizado Supervisionado -- Classificação"
    assert resultado["subtipo"] == "Multiclasse (3+ classes)"
    assert resultado["recomendacao_app"] == "Classificação"


# ============================================================================
# 10) Dataset `publicidade.csv` (ingestão de aulas/, v9)
# ============================================================================

def test_regressao_multipla_bate_com_r2_do_script_de_aula_publicidade():
    """`aulas/exemplo_teste_suposicao.py` treina X=['TV', 'Radio', 'Jornal']
    -> y='Vendas' com test_size=0.3, random_state=0. Confere que
    `data/publicidade.csv` reproduz o R² calculado como referência
    (0.8649) para essa mesma combinação de variáveis."""
    df = core.carregar_dataset(os.path.join(DATA_DIR, "publicidade.csv"))
    cols_x = ["TV", "Radio", "Jornal"]
    col_y = "Vendas"

    df_limpo, _ = core.validar_dados_para_regressao_multipla(df, cols_x, col_y, test_size=0.3)
    X, y = df_limpo[cols_x], df_limpo[col_y]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)
    modelo, _, _ = core.treinar_modelo_regressao_multipla(X_tr, y_tr)
    _, r2, _, _, _ = core.avaliar_modelo(modelo, X_te, y_te)

    assert r2 == pytest.approx(0.8649, abs=1e-3)


def test_melhor_subconjunto_multipla_descarta_jornal_no_dataset_publicidade():
    """'Jornal' tem correlação fraca com Vendas -- a busca de melhor
    subconjunto deve preferir só TV+Radio (R²≈0.8657), batendo com a
    conclusão do próprio script de aula ('Jornal' não tem forma
    específica de relação com Vendas)."""
    df = core.carregar_dataset(os.path.join(DATA_DIR, "publicidade.csv"))
    candidatas = ["TV", "Radio", "Jornal"]
    col_y = "Vendas"

    cols_escolhidas, _, r2, _, _, _ = core.selecionar_melhor_subconjunto_multipla(
        df, candidatas, col_y, test_size=0.3, random_state=0
    )

    assert set(cols_escolhidas) == {"TV", "Radio"}
    assert r2 == pytest.approx(0.8657, abs=1e-3)


# ============================================================================
# 11) Suposições avançadas: Homocedasticidade e Independência dos Resíduos
#     (v9 -- alinhamento com aulas/exemplo_teste_suposicao.py)
# ============================================================================

def test_testar_homocedasticidade_residuos_com_variancia_constante():
    rng = np.random.default_rng(42)
    residuos = rng.normal(loc=0, scale=1, size=100)
    X_teste = pd.DataFrame({"x": rng.normal(size=100)})

    estatistica, p_valor = core.testar_homocedasticidade_residuos(residuos, X_teste)

    assert not np.isnan(estatistica)
    assert 0.0 <= p_valor <= 1.0


def test_testar_homocedasticidade_residuos_falha_com_poucos_pontos():
    X_teste = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(core.DadosInvalidosError):
        core.testar_homocedasticidade_residuos([1.0, 2.0, 3.0], X_teste)


def test_testar_independencia_residuos_com_amostra_aleatoria():
    rng = np.random.default_rng(42)
    residuos = rng.normal(loc=0, scale=1, size=100)

    p_valor_ljungbox, estatistica_durbin_watson = core.testar_independencia_residuos(residuos)

    assert 0.0 <= p_valor_ljungbox <= 1.0
    # resíduos i.i.d. -> Durbin-Watson deve ficar perto de 2 (sem autocorrelação)
    assert 1.5 < estatistica_durbin_watson < 2.5


def test_testar_independencia_residuos_falha_com_poucos_pontos():
    with pytest.raises(core.DadosInvalidosError):
        core.testar_independencia_residuos([1.0, 2.0])


def test_diagnostico_de_suposicoes_no_fluxo_completo_publicidade():
    """Confere que os testes novos encaixam direto na saída de
    avaliar_modelo/diagnosticar_residuos, como app.py usa, no dataset que
    motivou esta rodada (publicidade.csv)."""
    df = core.carregar_dataset(os.path.join(DATA_DIR, "publicidade.csv"))
    cols_x = ["TV", "Radio", "Jornal"]
    col_y = "Vendas"
    df_limpo, _ = core.validar_dados_para_regressao_multipla(df, cols_x, col_y, test_size=0.3)
    X, y = df_limpo[cols_x], df_limpo[col_y]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)
    modelo, _, _ = core.treinar_modelo_regressao_multipla(X_tr, y_tr)
    predicoes, _, _, _, _ = core.avaliar_modelo(modelo, X_te, y_te)
    residuos = core.diagnosticar_residuos(y_te, predicoes)["residuos"]

    estatistica_gq, p_valor_gq = core.testar_homocedasticidade_residuos(residuos, X_te)
    assert 0.0 <= p_valor_gq <= 1.0

    p_valor_lb, estatistica_dw = core.testar_independencia_residuos(residuos)
    assert 0.0 <= p_valor_lb <= 1.0
    assert 0.0 <= estatistica_dw <= 4.0


# ============================================================================
# 12) Datasets de `aulas/correcoes/` (ingestão v10) + RMSE
# ============================================================================

def test_regressao_simples_bate_com_r2_da_correcao_finance_market():
    """`correcao_regsimples.py` treina X='Indice_S&P500' -> y='ETF_Preco'
    com test_size=0.3, random_state=42. Confere R²/MAE/MSE/RMSE de
    referência calculados a partir do mesmo dataset."""
    df = core.carregar_dataset(os.path.join(DATA_DIR, "finance_market.csv"))
    col_x, col_y = "Indice_S&P500", "ETF_Preco"

    df_limpo, _ = core.validar_dados_para_regressao(df, col_x, col_y, test_size=0.3)
    X, y = df_limpo[[col_x]], df_limpo[col_y]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=42)
    modelo, _, _ = core.treinar_modelo_regressao_simples(X_tr, y_tr)
    _, r2, mae, mse, rmse = core.avaliar_modelo(modelo, X_te, y_te)

    assert r2 == pytest.approx(0.9987, abs=1e-3)
    assert mae == pytest.approx(0.4448, abs=1e-2)
    assert mse == pytest.approx(0.2982, abs=1e-2)
    assert rmse == pytest.approx(np.sqrt(mse))


def test_regressao_simples_e_multipla_batem_com_a_correcao_agro_tech():
    """`regressao_simples.py`/`regressao_multipla.py` treinam com
    test_size=0.3, random_state=0: Simples (precipitacao_anual) dá
    R²≈0.4847, Múltipla (+fertilizante_kg_ha) sobe para R²≈0.8705 --
    os mesmos números impressos no script de correção."""
    df = core.carregar_dataset(os.path.join(DATA_DIR, "agro_tech.csv"))
    col_y = "toneladas_por_hectare"

    df_s, _ = core.validar_dados_para_regressao(df, "precipitacao_anual", col_y, test_size=0.3)
    X_s, y_s = df_s[["precipitacao_anual"]], df_s[col_y]
    X_s_tr, X_s_te, y_s_tr, y_s_te = core.dividir_treino_teste(X_s, y_s, test_size=0.3, random_state=0)
    modelo_s, _, _ = core.treinar_modelo_regressao_simples(X_s_tr, y_s_tr)
    _, r2_s, mae_s, mse_s, _ = core.avaliar_modelo(modelo_s, X_s_te, y_s_te)

    assert r2_s == pytest.approx(0.4847, abs=1e-3)
    assert mae_s == pytest.approx(2.3037, abs=1e-2)
    assert mse_s == pytest.approx(7.3525, abs=1e-2)

    cols_m = ["precipitacao_anual", "fertilizante_kg_ha"]
    df_m, _ = core.validar_dados_para_regressao_multipla(df, cols_m, col_y, test_size=0.3)
    X_m, y_m = df_m[cols_m], df_m[col_y]
    X_m_tr, X_m_te, y_m_tr, y_m_te = core.dividir_treino_teste(X_m, y_m, test_size=0.3, random_state=0)
    modelo_m, _, _ = core.treinar_modelo_regressao_multipla(X_m_tr, y_m_tr)
    _, r2_m, mae_m, mse_m, _ = core.avaliar_modelo(modelo_m, X_m_te, y_m_te)

    assert r2_m == pytest.approx(0.8705, abs=1e-3)
    assert mae_m == pytest.approx(1.1035, abs=1e-2)
    assert mse_m == pytest.approx(1.8481, abs=1e-2)


def test_melhor_subconjunto_multipla_escolhe_precipitacao_e_fertilizante_agro_tech():
    """As 2 variáveis de maior correlação com o alvo já são
    precipitacao_anual + fertilizante_kg_ha -- a busca de melhor
    subconjunto deve convergir para a mesma combinação do script de aula."""
    df = core.carregar_dataset(os.path.join(DATA_DIR, "agro_tech.csv"))
    candidatas = [
        "precipitacao_anual", "temp_media", "fertilizante_kg_ha",
        "horas_sol_dia", "ph_solo", "altitude_metros", "umidade_relativa",
    ]
    col_y = "toneladas_por_hectare"

    cols_escolhidas, _, r2, _, _, _ = core.selecionar_melhor_subconjunto_multipla(
        df, candidatas, col_y, test_size=0.3, random_state=0
    )

    assert {"precipitacao_anual", "fertilizante_kg_ha"} <= set(cols_escolhidas)
    assert r2 >= 0.8705 - 1e-3


def test_regressao_multipla_no_dataset_plano_saude_reproduz_mau_ajuste():
    """`correcao_teste_suposicao.py` treina com TODAS as colunas (incluindo
    as dummies de região) e conclui que o modelo não é bom -- confere que
    `data/base_plano_saude_preparada.csv` reproduz um R² baixo (≈0.18),
    consistente com a conclusão do script do professor."""
    df = core.carregar_dataset(os.path.join(DATA_DIR, "base_plano_saude_preparada.csv"))
    col_y = "gastos_plano"
    cols_x = [c for c in df.columns if c != col_y]

    df_limpo, _ = core.validar_dados_para_regressao_multipla(df, cols_x, col_y, test_size=0.3)
    X, y = df_limpo[cols_x], df_limpo[col_y]
    X_tr, X_te, y_tr, y_te = core.dividir_treino_teste(X, y, test_size=0.3, random_state=0)
    modelo, _, _ = core.treinar_modelo_regressao_multipla(X_tr, y_tr)
    _, r2, _, _, _ = core.avaliar_modelo(modelo, X_te, y_te)

    assert r2 == pytest.approx(0.1795, abs=1e-2)
