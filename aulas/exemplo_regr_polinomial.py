import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

df = pd.read_csv('brazil_covid19.csv').groupby('date').sum()[27:].reset_index()
print(df)

# Preparação dos Dados
def_covid_brazil = pd.DataFrame({
    'date': pd.to_datetime(df['date']),
    'cases': df['cases'],
    'new_cases': df['cases'].diff().fillna(0).astype(int),
    'growth_cases': df['cases'].diff().fillna(0).astype(int)/df['cases'],
    'deaths': df['deaths'],
    'new_deaths': df['deaths'].diff().fillna(0).astype(int),
    'growth_deaths': df['deaths'].diff().fillna(0).astype(int)/df['deaths'],
    'mortality_rate': df['deaths']/df['cases']
})

print("\n******* DADOS DO DATAFRAME *******")
print(def_covid_brazil)


# Exibir a estrutura do DataFrame: quantidade de linhas e colunas, nomes das colunas, tipos de dados das colunas, etc...
print("\n******* ESTRUTURA DO DATAFRAME *******")
print(def_covid_brazil.info())

dates = pd.date_range(start=def_covid_brazil.iloc[0,0], end='2021-05-23')

print(def_covid_brazil.fillna(0).tail())

start = 17
end = len(def_covid_brazil)


X = np.asarray(range(start,end)).reshape(-1,1)
y = def_covid_brazil.iloc[start:,1]

# OBS: é importante checar os dados de X e y
print("\n******* MATRIZ X - VARIÁVEL INDEPENDENTE *******")
print(X)

print("\n******* MATRIZ y - VARIÁVEL DEPENDENTE *******")
print(y)

# Pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('model', LinearRegression())
])

param_grid = {
    'poly__degree': [1, 2, 3, 4, 5, 6]
}

# Validação Cruzada (GridSearch)
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,  # validação cruzada 5-fold
    scoring='neg_mean_squared_error'
)

grid.fit(X, y)

print("Melhor grau:", grid.best_params_)

melhor_modelo = grid.best_estimator_

y_pred = melhor_modelo.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))

print("RMSE:", rmse)

# geração do gráfico

fig, ax = plt.subplots(figsize=(14, 10))
plt.plot(dates[start:end], y, color='limegreen', linewidth=8, alpha=0.5)
#plt.plot(def_covid_brazil['date'][17:], def_covid_brazil['deaths'][17:], color='magenta', linewidth=8, alpha=0.5)

plt.plot(dates[start:len(dates)], y_pred, color='green', linestyle='None', marker='o')
#plt.plot(dates[start:len(dates)], yhat_deaths, color='darkorchid', linestyle='None', marker='o')

plt.title('COVID-19: previsão de casos no Brasil', fontsize=18, fontweight='bold', color='#333333')
plt.legend(labels=['casos','previsão de casos', 'novos casos'], fontsize=14)

plt.grid(which='major', color='#EEEEEE')
plt.grid(which='minor', color='#EEEEEE', linestyle=':')
plt.show()


# Previsao para os proximos 44 dias a partir de 23-05-2021
periodo = np.asarray(470).reshape(-1,1)
previsao = melhor_modelo.predict(periodo)
print("Para aproximadamente {} dias apos 23-05-2021, pode-se prever {:.2f} de casos de COVID".format(periodo[0][0] - 426, previsao[0]))
