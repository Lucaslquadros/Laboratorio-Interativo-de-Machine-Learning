# -*- coding: utf-8 -*-
"""
baixar_dados_classificacao.py — monta o dataset de classificação para a
Regressão Logística a partir do "Breast Cancer Wisconsin (Diagnostic)"
Data Set, publicado pelo UCI Machine Learning Repository e distribuído
junto com o scikit-learn (`sklearn.datasets.load_breast_cancer`).

Por que este dataset: alvo **genuinamente categórico** (diagnóstico
Maligno/Benigno de um tumor, a partir de medidas do exame -- não é uma
coluna contínua binarizada por nós), tema ODS 3 (Saúde e Bem-Estar), 569
casos, 30 variáveis numéricas, sem nenhum valor faltante. Testado antes de
fixar aqui: `LogisticRegression` treina com acurácia estável entre 94% e
98% em vários `random_state` diferentes (mesmo critério de estabilidade
usado para os datasets de ODS em `baixar_dados_ods.py`).

Rodar com:
    python scripts/baixar_dados_classificacao.py

Gera (sobrescreve):
    data/diagnostico_cancer_mama.csv
"""

from sklearn.datasets import load_breast_cancer


def montar_dataset():
    dados = load_breast_cancer(as_frame=True)
    df = dados.frame.copy()
    # target: 0 = malignant, 1 = benign (ver dados.target_names) -- convertido
    # para texto em português para ficar claro que é uma variável categórica,
    # não numérica contínua.
    df["Diagnostico"] = df["target"].map({0: "Maligno", 1: "Benigno"})
    df = df.drop(columns=["target"])
    return df


def main():
    df = montar_dataset()
    df.to_csv("data/diagnostico_cancer_mama.csv", index=False)
    print(f"data/diagnostico_cancer_mama.csv: {len(df)} casos, {df.shape[1] - 1} variáveis numéricas")
    print(df["Diagnostico"].value_counts())


if __name__ == "__main__":
    main()
