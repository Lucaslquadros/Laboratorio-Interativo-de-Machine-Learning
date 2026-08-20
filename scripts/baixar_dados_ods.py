# -*- coding: utf-8 -*-
"""
baixar_dados_ods.py — baixa e monta os datasets "Saúde & Desenvolvimento" e
"Mortalidade Infantil & Desenvolvimento" a partir da API pública do World
Bank (Open Data, sem autenticação), para uso como datasets padrão do
laboratório.

Fonte: https://api.worldbank.org/v2/ -- indicadores oficiais de
desenvolvimento por país, os mesmos usados para compor vários dos
Objetivos de Desenvolvimento Sustentável (ODS) da ONU. Ano de referência:
2020 (mais recente com boa cobertura simultânea dos indicadores escolhidos
-- checado manualmente antes de fixar aqui).

Um terceiro candidato -- "Clima & Desenvolvimento" (CO2 per capita
explicado por PIB, urbanização, energia renovável, área de floresta e
densidade populacional) -- foi testado e descartado: o R² da regressão
múltipla variava muito (e às vezes ficava negativo) dependendo do
random_state do split treino/teste, por causa de outliers fortes de países
pequenos/petro-exportadores (ex: Palau, Catar). Trocado por Mortalidade
Infantil, que testou consistentemente com R² entre 0.70 e 0.83 em vários
splits -- bem mais estável para fins didáticos.

Rodar com:
    python scripts/baixar_dados_ods.py

Gera (sobrescreve):
    data/saude_desenvolvimento.csv
    data/mortalidade_infantil_desenvolvimento.csv

Só países de verdade entram nos CSVs -- a API também devolve agregados
regionais (ex: "Arab World", "European Union"), que são descartados via
`região != "Aggregates"`. Linhas com qualquer indicador faltando também são
descartadas, para que os CSVs saiam prontos para treinar sem necessidade de
tratamento de NaN adicional.
"""

import json
import urllib.request

import pandas as pd

ANO_REFERENCIA = 2020

INDICADORES_SAUDE = {
    "Expectativa_Vida": "SP.DYN.LE00.IN",           # ODS 3 -- Saúde e Bem-Estar
    "PIB_per_capita": "NY.GDP.PCAP.CD",              # ODS 8 -- Trabalho Decente e Crescimento Econômico
    "Gasto_Saude_pct_PIB": "SH.XPD.CHEX.GD.ZS",      # ODS 3
    "Acesso_Eletricidade_pct": "EG.ELC.ACCS.ZS",     # ODS 7 -- Energia Limpa e Acessível
    "CO2_per_capita": "EN.GHG.CO2.PC.CE.AR5",        # ODS 13 -- Ação Contra a Mudança Global do Clima
}

INDICADORES_MORTALIDADE = {
    "Mortalidade_Infantil": "SH.DYN.MORT",           # ODS 3 -- meta 3.2 (mortalidade < 5 anos, por 1.000 nascidos vivos)
    "PIB_per_capita": "NY.GDP.PCAP.CD",              # ODS 8 -- Trabalho Decente e Crescimento Econômico
    "Gasto_Saude_pct_PIB": "SH.XPD.CHEX.GD.ZS",      # ODS 3
    "Saneamento_pct": "SH.STA.BASS.ZS",              # ODS 6 -- Água Potável e Saneamento
    "Agua_Potavel_pct": "SH.H2O.BASW.ZS",            # ODS 6
}


def buscar_indicador(codigo, ano=ANO_REFERENCIA):
    """Busca um indicador do World Bank para todos os países/agregados,
    num único ano. Devolve {iso3: valor}, só com valores não-nulos."""
    url = (
        f"https://api.worldbank.org/v2/country/all/indicator/{codigo}"
        f"?format=json&date={ano}&per_page=400"
    )
    with urllib.request.urlopen(url) as resposta:
        dados = json.load(resposta)
    linhas = dados[1] or []
    return {
        linha["countryiso3code"]: linha["value"]
        for linha in linhas
        if linha["value"] is not None
    }


def buscar_paises_reais():
    """Lista de países de verdade (exclui agregados regionais como
    'Arab World' ou 'European Union', que a API mistura com países)."""
    url = "https://api.worldbank.org/v2/country?format=json&per_page=400"
    with urllib.request.urlopen(url) as resposta:
        dados = json.load(resposta)
    return {
        pais["id"]: pais["name"]
        for pais in dados[1]
        if pais["region"]["value"] != "Aggregates"
    }


def montar_dataset(indicadores, nomes_paises):
    """Cruza vários indicadores num único DataFrame (1 linha por país),
    mantendo só países com TODOS os indicadores disponíveis."""
    valores_por_indicador = {
        nome_coluna: buscar_indicador(codigo) for nome_coluna, codigo in indicadores.items()
    }

    linhas = []
    for iso3, nome_pais in nomes_paises.items():
        if all(iso3 in valores_por_indicador[nome_coluna] for nome_coluna in indicadores):
            linha = {"Pais": nome_pais}
            for nome_coluna in indicadores:
                linha[nome_coluna] = valores_por_indicador[nome_coluna][iso3]
            linhas.append(linha)

    return pd.DataFrame(linhas).sort_values("Pais").reset_index(drop=True)


def main():
    nomes_paises = buscar_paises_reais()

    df_saude = montar_dataset(INDICADORES_SAUDE, nomes_paises)
    df_saude.to_csv("data/saude_desenvolvimento.csv", index=False)
    print(f"data/saude_desenvolvimento.csv: {len(df_saude)} países, ano {ANO_REFERENCIA}")

    df_mortalidade = montar_dataset(INDICADORES_MORTALIDADE, nomes_paises)
    df_mortalidade.to_csv("data/mortalidade_infantil_desenvolvimento.csv", index=False)
    print(
        f"data/mortalidade_infantil_desenvolvimento.csv: "
        f"{len(df_mortalidade)} países, ano {ANO_REFERENCIA}"
    )


if __name__ == "__main__":
    main()
