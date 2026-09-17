# -*- coding: utf-8 -*-
"""
Aluno: Lucas Quadros
Matrícula:
Atividade 1 - Revisão de preparação de dados
Base: base_pacientes_covid.csv
Data: 13/08/26
"""
#%% carregando as bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#%% ############# Carregamento dos dados #############
df_original = pd.read_csv("base_pacientes_covid.csv")
df_original.info()

#%%
df_original.head()

#%%
df_original.columns.tolist()
'''
data_teste              -> data do teste (dd/mm/aaaa)
sintoma_tosse            -> categórica (Sim/Nao)
sintoma_febre             -> categórica (Sim/Nao)
sintoma_dor_garganta      -> categórica (Sim/Nao)
sintoma_falta_ar          -> categórica (Sim/Nao)
sintoma_dor_cabeca        -> categórica (Sim/Nao)
resultado_teste           -> categórica (Positivo/Negativo)
idade_60_abaixo           -> categórica (Sim/Nao)
genero                    -> categórica (Masculino/Feminino)
indicacao_teste           -> categórica (3 categorias)

Toda a base é composta por variáveis CATEGÓRICAS. Não existe, originalmente,
nenhuma variável numérica/métrica. Como os itens de outlier e escalonamento
pedem tratamento de variáveis métricas, uma variável numérica é construída
a partir das categóricas antes dessas etapas:
- qtd_sintomas: contagem de sintomas relatados (0 a 5)

'data_teste' NÃO é transformada em variável numérica (ex.: dias desde o
início). Como não há identificação de paciente, cada linha é um teste
isolado, não a trajetória de uma mesma pessoa ao longo do tempo — então
"dias desde o início" não representaria idade/tempo de acompanhamento de
ninguém, só a data em que aquele teste em particular foi feito. A coluna
é mantida na base apenas como informação, sem entrar no tratamento de
outliers/escalonamento nem na criação de dummies (teria quase 50 datas
únicas, o que não faz sentido como categoria).
'''

#%% trabalhar em uma cópia
df = df_original.copy()

#%% ############# 1. TRATAMENTO DE DADOS AUSENTES #############

#%% Verificação de valores ausentes
df.isnull().sum()

#%% Verificação de possíveis "ausentes disfarçados" (strings vazias, "NA" etc.)
for col in df.columns:
    valores = df[col].astype(str).str.strip()
    disfarcados = valores.isin(['', 'nan', 'NA', 'None', 'null', '-', 'ignorado'])
    if disfarcados.sum() > 0:
        print(col, '->', disfarcados.sum(), 'valores ausentes disfarçados')
print("Verificação concluída.")
'''
A base não apresenta valores ausentes (nem explícitos, nem disfarçados em
formato de texto) em nenhuma das 10 colunas. Não há necessidade de imputação.
'''

#%% ############# CRIAÇÃO DAS VARIÁVEIS NUMÉRICAS AUXILIARES #############
# (necessárias para poder aplicar outlier/escalonamento nesta base)

#%% qtd_sintomas: soma dos 5 sintomas binários
colunas_sintomas = ['sintoma_tosse', 'sintoma_febre', 'sintoma_dor_garganta',
                     'sintoma_falta_ar', 'sintoma_dor_cabeca']

for col in colunas_sintomas:
    df[col] = df[col].map({'Sim': 1, 'Nao': 0})

df['qtd_sintomas'] = df[colunas_sintomas].sum(axis=1)

# desfazer a conversão numérica das colunas de sintoma original
# (serão recriadas como dummy na etapa 3, a partir da base original)
for col in colunas_sintomas:
    df[col] = df_original[col]

df[['qtd_sintomas']].describe()

#%% ############# 2. TRATAMENTO DE OUTLIERS #############

#%% Distribuição da variável numérica construída
plt.figure(figsize=(4, 4))
sns.boxplot(y=df['qtd_sintomas'])
plt.title("qtd_sintomas")
plt.show()

df['qtd_sintomas'].value_counts().sort_index()

#%% Função para identificar limites de outliers pelo método do IQR
def limites_iqr(serie, k=1.5):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - k * iqr
    limite_superior = q3 + k * iqr
    return limite_inferior, limite_superior

#%% qtd_sintomas: verificação
li_sint, ls_sint = limites_iqr(df['qtd_sintomas'])
n_out_sint = ((df['qtd_sintomas'] > ls_sint)).sum()
print(f"qtd_sintomas: limite superior {ls_sint:.2f} -> {n_out_sint} outliers "
      f"({n_out_sint/len(df)*100:.1f}% da base)")
'''
A maioria dos pacientes (95,9%) não relata nenhum sintoma, o que torna a
distribuição extremamente concentrada em zero (Q1 = Q3 = 0). Nessas condições
o critério do IQR classifica QUALQUER paciente com 1 ou mais sintomas como
"outlier", o que não faz sentido clínico: ter sintomas é um valor válido e
esperado da variável, limitada naturalmente entre 0 e 5. Por isso, NÃO se
aplica capping/remoção nesta variável — a decisão correta é reconhecer que
o IQR não é adequado para uma contagem discreta e fortemente concentrada em
zero, e manter os valores originais.
'''

#%% ############# 3. CRIAÇÃO DE VARIÁVEIS DUMMY #############

#%% Verificando as categorias antes de codificar
for col in colunas_sintomas + ['resultado_teste', 'idade_60_abaixo', 'genero', 'indicacao_teste']:
    print(col, '->', df[col].unique())

#%% Transformando variáveis categóricas (dummy, drop_first evita redundância)
colunas_categoricas = colunas_sintomas + ['resultado_teste', 'idade_60_abaixo',
                                           'genero', 'indicacao_teste']

df_dummy = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)

df_dummy.columns.tolist()

#%% ############# 4. ESCALONAMENTO DOS DADOS #############

colunas_metricas = ['qtd_sintomas']

scaler = StandardScaler()
df_padronizado = df_dummy.copy()
df_padronizado[colunas_metricas] = scaler.fit_transform(df_dummy[colunas_metricas])

#%% Conferindo estatísticas após a padronização (média ~0, desvio ~1)
df_padronizado[colunas_metricas].describe().round(4)

#%% ############# BASE FINAL PREPARADA #############
df_padronizado.info()
df_padronizado.head()

#%% Exportação da base preparada
df_padronizado.to_csv("base_pacientes_COVID_preparada.csv", index=False, encoding='utf-8-sig')
print("Arquivo 'base_pacientes_COVID_preparada.csv' salvo com sucesso.")
