import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. Leitura da base de dados (Multivariada)
try:
    df_energia = pd.read_csv('eficiencia_energetica.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'eficiencia_energetica.csv' não encontrado.")
    exit()

print("\n******* DADOS DO DATAFRAME *******")
print(df_energia.head())

# Estrutura do DataFrame
print("\n******* ESTRUTURA DO DATAFRAME *******")
print(df_energia.info())

df_energia = df_energia.astype({"carga_resfriamento": float})

# 2. Análise Multivariada: Matriz de Correlação Completa
print("\n******* MATRIZ DE CORRELAÇÃO *******")
print(df_energia.corr())

# 3. Regressão Linear Múltipla utilizando statsmodels
# Criando a fórmula dinamicamente com todas as preditoras
preditoras = " + ".join(df_energia.columns.drop('carga_resfriamento'))
formula = f'carga_resfriamento ~ {preditoras}'

regressao = smf.ols(formula, data=df_energia).fit()

print("\n******* SUMÁRIO DA REGRESSÃO LINEAR MÚLTIPLA *******")
print(regressao.summary())

'''
Análise: Na regressão linear múltipla, cada variável tenta explicar o alvo isoladamente.
Entretanto, variáveis físicas costumam ter interações (ex: altura potencializa efeito do telhado).
Vamos aplicar a regressão polinomial multivariada via Pipeline!
'''

# 4. Definição de X (todas as preditoras) e y (alvo)
X = df_energia.drop(columns=['carga_resfriamento'])
y = df_energia.comissao if 'comissao' in df_energia else df_energia.carga_resfriamento # Alvo Adaptado

print("\n******* MATRIZ X (TODAS AS PREDITORAS) *******")
print(X.head())

# 5. Implementação com PIPELINE e GRIDSEARCHCV
pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('model', LinearRegression())
])

# Testando graus menores para evitar explosão de colunas em modelos multivariados
param_grid = {
    'poly__degree': [1, 2, 3, 4, 5, 6]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)

grid.fit(X, y)

print("\nMelhor grau (Multivariado):", grid.best_params_)

melhor_modelo = grid.best_estimator_
y_pred = melhor_modelo.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("RMSE Final (Modelo Polinomial):", rmse)
print("R quadrado:", r2)

# 6. Visualização: Realidade vs Predição
plt.figure(figsize=(8,6))
plt.scatter(y, y_pred, alpha=0.5, c='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Realidade vs Predição (Modelo Polinomial Multivariado)')
plt.xlabel('Valores Reais (Carga)')
plt.ylabel('Valores Preditos (Carga)')
plt.grid(True)
plt.show()

# 7. Extração de Coeficientes
modelo_final = melhor_modelo.named_steps['model']
poly_features = melhor_modelo.named_steps['poly']

print("\n******* ESTRUTURA DO MODELO FINAL *******")
print(f"Intercepto: {modelo_final.intercept_}")
print(f"Quantidade de coeficientes gerados (incluindo interações): {len(modelo_final.coef_)}")

# 8. Previsão para um novo cenário (exemplo de um registro novo)
# Criando um registro com valores médios (0 na escala standard) e dummies
novo_predio = pd.DataFrame([[0.5, -0.2, 1.0, 0.0, 1, 0, 0]], columns=X.columns)
previsao = melhor_modelo.predict(novo_predio)

print("\nPrevisão para novo cenário:")
print(f"Características: {novo_predio.values}")
print(f"Carga de Resfriamento estimada: {previsao[0]:.2f}")