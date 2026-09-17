# -*- coding: utf-8 -*-
"""
Aluno: Lucas Quadros
Matrícula:
Atividade 1 - Revisão de preparação de dados
Base: base_plano_saude.csv
Data: 13/08/26
"""
#%% carregando as bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#%% ############# Carregamento dos dados #############
df_original = pd.read_csv("base_plano_saude.csv", sep=';', encoding='utf-8-sig')
df_original.info()

#%%
df_original.head()

#%%
df_original.columns.tolist()
'''
idade            -> numérica (métrica)
genero           -> categórica (Masculino/Feminino)
imc               -> numérica (métrica) -> POSSUI DADOS AUSENTES
filhos            -> numérica (métrica, contagem)
fumante           -> categórica (Sim/Nao)
regiao            -> categórica (4 categorias)
gastos_plano      -> numérica (métrica) -> alvo/variável de custo, forte assimetria
'''

#%% trabalhar em uma cópia
df = df_original.copy()

#%% ############# 1. TRATAMENTO DE DADOS AUSENTES #############

#%% Verificação de valores ausentes
df.isnull().sum()
'''
Apenas a variável 'imc' possui dados ausentes (333 de 2772 registros, ~12%).
'''

#%% Analisando a distribuição do IMC antes de decidir o tratamento
df['imc'].describe()
print("Assimetria (skew) do IMC:", df['imc'].skew())

plt.figure(figsize=(6, 4))
sns.histplot(df['imc'], bins=20, kde=True)
plt.title("Distribuição do IMC (antes da imputação)")
plt.show()
'''
A assimetria é baixa (skew ~0.32) e média e mediana são próximas,
mas optamos pela MEDIANA por ser uma medida robusta a outliers,
que ainda serão tratados na próxima etapa.
'''

#%% Imputação pela mediana
mediana_imc = df['imc'].median()
df['imc'] = df['imc'].fillna(mediana_imc)

#%% Confirmando que não restaram valores ausentes
df.isnull().sum()

#%% ############# 2. TRATAMENTO DE OUTLIERS #############

#%% Boxplot das variáveis métricas (visão geral antes do tratamento)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sns.boxplot(y=df['idade'], ax=axes[0]).set_title("idade")
sns.boxplot(y=df['imc'], ax=axes[1]).set_title("imc")
sns.boxplot(y=df['filhos'], ax=axes[2]).set_title("filhos")
plt.tight_layout()
plt.show()

plt.figure(figsize=(4, 4))
sns.boxplot(y=df['gastos_plano'])
plt.title("gastos_plano (antes do tratamento)")
plt.show()

#%% Função para identificar limites de outliers pelo método do IQR
def limites_iqr(serie, k=1.5):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - k * iqr
    limite_superior = q3 + k * iqr
    return limite_inferior, limite_superior

#%% idade e filhos: verificação (não apresentam outliers pelo IQR)
for col in ['idade', 'filhos']:
    li, ls = limites_iqr(df[col])
    n_out = ((df[col] < li) | (df[col] > ls)).sum()
    print(f"{col}: limites [{li:.2f}, {ls:.2f}] -> {n_out} outliers")
'''
'idade' e 'filhos' não apresentam outliers pelo critério do IQR
(valores dentro do esperado: idade 18-64, filhos 0-5). Não precisam de tratamento.
'''

#%% imc: identificação e tratamento (poucos outliers, na cauda superior)
li_imc, ls_imc = limites_iqr(df['imc'])
n_out_imc = ((df['imc'] < li_imc) | (df['imc'] > ls_imc)).sum()
print(f"imc: limites [{li_imc:.2f}, {ls_imc:.2f}] -> {n_out_imc} outliers")

# Winsorização (capping): substitui os valores extremos pelo limite do IQR,
# preservando o registro em vez de excluí-lo
df['imc'] = np.where(df['imc'] > ls_imc, ls_imc, df['imc'])
df['imc'] = np.where(df['imc'] < li_imc, li_imc, df['imc'])

#%% gastos_plano: identificação
li_gp, ls_gp = limites_iqr(df['gastos_plano'])
n_out_gp = ((df['gastos_plano'] > ls_gp)).sum()
print(f"gastos_plano: limite superior {ls_gp:.2f} -> {n_out_gp} outliers "
      f"({n_out_gp/len(df)*100:.1f}% da base)")
'''
O critério clássico do IQR aponta ~15% dos registros como outliers em
'gastos_plano'. Isso não indica erro de digitação, e sim a forte assimetria
natural de custos de saúde (pacientes fumantes e/ou com IMC alto tendem a
gerar custos muito mais altos). Remover ~15% da base descartaria informação
relevante. Por isso optamos por WINSORIZAR (capping) apenas o limite
superior, preservando o efeito da variável sem deixar os valores extremos
dominarem etapas futuras (ex.: padronização).
'''
df['gastos_plano'] = np.where(df['gastos_plano'] > ls_gp, ls_gp, df['gastos_plano'])

#%% Boxplot após o tratamento dos outliers
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
sns.boxplot(y=df['imc'], ax=axes[0]).set_title("imc (depois)")
sns.boxplot(y=df['gastos_plano'], ax=axes[1]).set_title("gastos_plano (depois)")
plt.tight_layout()
plt.show()

#%% ############# 3. CRIAÇÃO DE VARIÁVEIS DUMMY #############

#%% Verificando as categorias antes de codificar
df['genero'].value_counts()
df['fumante'].value_counts()
df['regiao'].value_counts()

#%% Transformando variáveis categóricas (dummy, drop_first evita redundância)
df_dummy = pd.get_dummies(
    df,
    columns=['genero', 'fumante', 'regiao'],
    drop_first=True
)

df_dummy.columns.tolist()

#%% ############# 4. ESCALONAMENTO DOS DADOS #############

colunas_metricas = ['idade', 'imc', 'filhos', 'gastos_plano']

scaler = StandardScaler()
df_padronizado = df_dummy.copy()
df_padronizado[colunas_metricas] = scaler.fit_transform(df_dummy[colunas_metricas])

#%% Conferindo estatísticas após a padronização (média ~0, desvio ~1)
df_padronizado[colunas_metricas].describe().round(4)

#%% ############# BASE FINAL PREPARADA #############
df_padronizado.info()
df_padronizado.head()

#%% Exportação da base preparada
df_padronizado.to_csv("base_plano_saude_preparada.csv", sep=';', index=False, encoding='utf-8-sig')
print("Arquivo 'base_plano_saude_preparada.csv' salvo com sucesso.")
