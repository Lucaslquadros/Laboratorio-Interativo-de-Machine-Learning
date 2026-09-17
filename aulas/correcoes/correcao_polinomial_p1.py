import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# leitura e conversão da base para um dataframe
df_icecream = pd.read_csv('Ice Cream.csv')

print("\n******* DADOS DO DATAFRAME *******")
print(df_icecream)

# Exibir a estrutura do DataFrame: quantidade de linhas e colunas, nomes das colunas, tipos de dados das colunas, etc...
print("\n******* ESTRUTURA DO DATAFRAME *******")
print(df_icecream.info())


X = df_icecream[['Temperature']]
y = df_icecream.Revenue

# OBS: é importante checar os dados de X e y
print("\n******* MATRIZ X - VARIÁVEL INDEPENDENTE *******")
print(X)

print("\n******* MATRIZ y - VARIÁVEL DEPENDENTE *******")
print(y)

'''
Implementação com PIPELINE

A forma mais elegante e profissional de fazer isso no Python é 
usando um Pipeline. Ele garante que o escalonamento, a criação do 
polinômio e a regressão aconteçam em uma sequência protegida, 
evitando que informações do treino "vazem" para o teste.

Por que usar Pipeline?
Organização: O código fica muito mais limpo.

Segurança: Se você escalar os dados de treino e teste separadamente 
de forma manual, pode cometer erros de cálculo. O Pipeline cuida 
disso para você.

Deploy: Quando você for usar esse modelo em produção (com dados 
reais novos), basta passar o dado bruto pelo modelo_final.predict(),
e ele aplicará todas as transformações sozinho.


Definicao de um pipeline para a utilizacao da valicao cruzada a fim 
de obter os hiperparametros.
'''
pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('model', LinearRegression())
])

param_grid = {
    'poly__degree': [1, 2, 3, 4, 5, 6]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,  # validação cruzada 5-fold
    scoring='neg_mean_squared_error'
)

grid.fit(X, y)

print("Melhor grau:", grid.best_params_)


melhor_modelo = grid.best_estimator_

y_pred = melhor_modelo.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))

print("RMSE:", rmse)


plt.scatter(X, y, c = "gray")
plt.xlabel("Temperature")
plt.ylabel("Revenue")
plt.plot(X, y_pred, color='r')
plt.show()

'''
Quando se usa o GridSearchCV, o objeto resultante não é o modelo 
final em si, mas um "maestros" que coordenou vários treinamentos. 
Para acessar os coeficientes e o intercepto, é necessário primeiro extrair 
o melhor estimador encontrado durante a busca.
'''

# 2. Acesse o passo da 'model' dentro do Pipeline
# (Use o nome que você deu ao passo no Pipeline, ex: 'model')
modelo_final = melhor_modelo.named_steps['model']

# 3. Agora você pode acessar os atributos padrão do Scikit-Learn
intercepto = modelo_final.intercept_
coeficientes = modelo_final.coef_

# Coeficientes
print(coeficientes)

# Intercepto
print(intercepto)

print("Equação: COMISSÃO = {:.1f} + {:.1f}*temperature".format(modelo_final.intercept_, modelo_final.coef_[1]))


# Previsao para uma nova quantidade
temperature = [[10]]
previsao = melhor_modelo.predict(temperature)
print("Temperatura {:.2f}: U$ {:.2f}".format(temperature[0][0], previsao[0]))

