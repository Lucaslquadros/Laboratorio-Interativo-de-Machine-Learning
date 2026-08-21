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

def ver_codigo(funcao, titulo="Ver o código Python deste passo"):
    with st.expander(f"🔍 {titulo}"):
        st.code(inspect.getsource(funcao), language="python")


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
    else:
        col_x = st.sidebar.selectbox(
            "4. Variável INDEPENDENTE (X, preditora)",
            opcoes_x_ordenadas,
            help="Dica: na aula, escolhe-se a variável com maior correlação (em módulo) com o alvo.",
        )
        cols_x = [col_x]

    if modo_regressao == MODO_POLINOMIAL:
        grau_polinomial = st.sidebar.slider(
            "4b. Grau do polinômio", min_value=2, max_value=5, value=2, step=1,
            help="Grau 1 seria igual à Regressão Simples. Graus mais altos ajustam curvas "
            "mais flexíveis, mas arriscam overfitting com poucos dados.",
        )

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
        modelo, intercepto, coefs_array = treinar_modelo_regressao_polinomial(
            X_treinamento, y_treinamento, grau_polinomial
        )
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
    predicoes_modelo, r2, mae, mse = avaliar_modelo(modelo, X_teste, y_teste)


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
    aba_passo0, aba_teoria, aba_dados, aba_split, aba_treino, aba_avaliacao, aba_previsao = st.tabs(
        [
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
    aba_passo0, aba_teoria, aba_dados, aba_split, aba_treino, aba_avaliacao, aba_diagnostico, aba_previsao, aba_comparacao = st.tabs(
        [
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
    elif modo_regressao == MODO_POLINOMIAL:
        st.markdown(
            f"`PolynomialFeatures(degree={grau_polinomial})` expande **{col_x}** em "
            f"potências (X¹, X²{', ..., X' + str(grau_polinomial) if grau_polinomial > 2 else ''}) "
            "e depois `LinearRegression().fit()` ajusta os coeficientes -- "
            "continua sendo mínimos quadrados, só que sobre as colunas "
            "expandidas em vez de X sozinho."
        )

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
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("R² (coeficiente de determinação)", f"{r2:.4f}")
        col_b.metric("MAE (erro médio absoluto)", f"{mae:,.2f}")
        col_c.metric("MSE (erro médio quadrático)", f"{mse:,.2f}")

        if modo_regressao == MODO_MULTIPLA:
            origem_x = f"das {len(cols_x)} variáveis selecionadas ({', '.join(cols_x)})"
        else:
            origem_x = f"**{col_x}**"
        st.info(
            f"O modelo conseguiu explicar **{r2 * 100:.2f}%** da variação de "
            f"**{col_y}** a partir de {origem_x}, no conjunto de teste."
        )

        ver_codigo(avaliar_modelo, "Ver o código: R², MAE e MSE")

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
            "neste dataset (Simples, Múltipla e Polinomial), usando a mesma "
            "divisão treino/teste do painel lateral, e ranqueia por R² -- sem "
            "afetar as outras abas, que continuam mostrando só o modo ativo "
            "escolhido no painel lateral. Na Múltipla, testa **todas as "
            "combinações possíveis** de variáveis candidatas e usa a de maior "
            "R² -- por isso o número de variáveis em \"Detalhes\" pode ser "
            "menor que o total de colunas disponíveis."
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
            _, r2_s, mae_s, mse_s = avaliar_modelo(modelo_s, X_s_te, y_s_te)
            linhas_leaderboard.append(
                {"Modelo": "Simples", "Detalhes": col_x_simples, "R²": r2_s, "MAE": mae_s, "MSE": mse_s}
            )
        except DadosInvalidosError as erro:
            erros_leaderboard.append(f"Simples: {erro}")

        # Múltipla -- melhor subconjunto entre as colunas numéricas candidatas a X
        # (busca exaustiva por maior R² de teste, não só "todas as colunas";
        # ver ai-dlc/01-inception/INCEPTION.md, rodada v8)
        try:
            if len(opcoes_x_ordenadas) < 2:
                raise DadosInvalidosError("este dataset só tem 1 coluna numérica candidata a X.")
            cols_m, modelo_m, r2_m, mae_m, mse_m = selecionar_melhor_subconjunto_multipla(
                df, opcoes_x_ordenadas, col_y, test_size, random_state=int(random_state)
            )
            linhas_leaderboard.append(
                {
                    "Modelo": "Múltipla",
                    "Detalhes": " + ".join(cols_m),
                    "R²": r2_m,
                    "MAE": mae_m,
                    "MSE": mse_m,
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
            _, r2_p, mae_p, mse_p = avaliar_modelo(modelo_p, X_p_te, y_p_te)
            linhas_leaderboard.append(
                {
                    "Modelo": f"Polinomial (grau {grau_leaderboard})",
                    "Detalhes": col_x_poli,
                    "R²": r2_p,
                    "MAE": mae_p,
                    "MSE": mse_p,
                }
            )
        except DadosInvalidosError as erro:
            erros_leaderboard.append(f"Polinomial: {erro}")

        if linhas_leaderboard:
            tabela_leaderboard = (
                pd.DataFrame(linhas_leaderboard).sort_values("R²", ascending=False).reset_index(drop=True)
            )
            tabela_leaderboard.insert(0, "Posição", range(1, len(tabela_leaderboard) + 1))
            st.dataframe(
                tabela_leaderboard.style.format({"R²": "{:.4f}", "MAE": "{:,.2f}", "MSE": "{:,.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

            fig_leader, ax_leader = plt.subplots(figsize=(6, 1.2 + 0.6 * len(tabela_leaderboard)))
            ordem_grafico = tabela_leaderboard.sort_values("R²")
            cores = ["#4C72B0", "#DD8452", "#55A868"]
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
