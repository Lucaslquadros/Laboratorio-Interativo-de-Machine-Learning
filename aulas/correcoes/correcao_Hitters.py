import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("Hitters.csv")

# Pre-processamento dos dados

# Variaveis Dummy

league_dummy = pd.get_dummies(df.League,drop_first=True)
division_dummy = pd.get_dummies(df.Division,drop_first=True)
newleague_dummy = pd.get_dummies(df.NewLeague,drop_first=True)

df.drop(columns=['League','Division','NewLeague'],inplace=True)

df = pd.concat([df,league_dummy,division_dummy,newleague_dummy],axis=1)

# Dados ausentes
print(df.isnull().sum())

df.Salary = df['Salary'].fillna(value=df.Salary.median())

# Criacao do conjunto de atributos preditores (X) e o alvo (y)
X = df.drop(columns=['Salary'])
y = df.Salary
X_ = X

sc = StandardScaler()
X = sc.fit_transform(X)

# **************** REGRESSAO MULTIPLA ****************

# Treinamento do modelo considerando todas as variaveis preditoras
multiple_lr = LinearRegression().fit(X,y)


# ****************** VALIDACAO CRUZADA PARA ENCONTRAR O MELHOR MSE ******************
mse= cross_val_score(multiple_lr,X,y,scoring='neg_mean_squared_error',cv=10)

print(f"Media do MSE para a regressao multipla: {mse.mean()}")

multiple_lr_coeffs = multiple_lr.coef_
print(f"\nCoeficientes para a Regressao Multipla: {multiple_lr_coeffs}")

feature_names = df.drop('Salary',axis=1).columns


plt.figure(figsize=(10,6))
plt.plot(range(len(multiple_lr_coeffs)),multiple_lr_coeffs)
plt.axhline(0, color='r', linestyle='solid')
plt.xticks(range(len(feature_names)),feature_names,rotation=50)
plt.title("Coeficientes para a Regressao Multipla")
plt.ylabel("coeficientes")
plt.xlabel("features")
plt.show()


'''
OBS: Vemos que neste modelo, as características AtBat, Hits, CATBat e CRuns estão tendo um 
impacto considerável na progressão do diabetes, pois todas elas têm altos valores 
de coeficiente estimado (tanto para positivo quanto negativo).
'''

mask = np.triu(np.ones_like(X_.corr(),dtype=bool))
# Geracao do mapa de calor para observacao da correlacao entre as variaveis preditoras
sns.heatmap(X_.corr(),mask =mask, fmt='.2f', square=True, linecolor='white', annot=True, cmap="coolwarm")
plt.show()

'''
Nos modelos lineares múltiplos que construímos na seção anterior, ambas as 
características AtBat e Hits se mostraram importantes. No entanto, podemos observar 
que elas apresentam uma correlação positiva muito alta, de cerca de 0,97. Isso 
está claramente induzindo multicolinearidade no modelo. Além disso, várias outras variáveis, que
não têm tanto impacto no modelo, possuem multicolinearidade.
'''

# ****************** REGRESSAO RIDGE ********************

alpha_values = {'alpha':[0.001, 0.01,0.02,0.03,0.04, 0.05, 0.06, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100]}

ridge= GridSearchCV(Ridge(), alpha_values, scoring='neg_mean_squared_error', cv=10 )

print('\nO melhor valor de alpha para a regressao Ridge eh:',ridge.fit(X,y).best_params_)

print('\nMelhor score (Ridge):',ridge.fit(X,y).best_score_)




best_ridge_model= Ridge(alpha=3)

best_ridge_coeffs = best_ridge_model.fit(X,y).coef_
print(f"\nCoeficientes para a regressao Ridge: {best_ridge_coeffs}")

plt.figure(figsize=(10,6))
plt.plot(range(len(feature_names)),best_ridge_coeffs)
plt.axhline(0, color='r', linestyle='solid')
plt.xticks(range(len(feature_names)),feature_names,rotation=50)
plt.title("Coeficientes para a Regressao Ridge")
plt.ylabel("coeficientes")
plt.xlabel("features")
plt.show()

'''
Vemos que a regularização do nosso modelo de regressão linear múltipla usando a 
regressão Ridge aumenta o 'neg_mean_squared_error' médio de quase -107773.36 para 
cerca de -104816.07, o que é uma ótima melhoria.

Anteriormente, as características AtBat, Hits, CAtBat e CRuns se destacavam como características 
importantes na regressão linear múltipla; no entanto, seus valores de coeficientes 
são significativamente reduzidos após a regularização Ridge.

'''


# ****************** REGRESSAO LASSO ********************

alpha_values = {'alpha':[0.001, 0.01,0.02,0.03,0.04, 0.05, 0.06,0.07, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100]}

lasso= GridSearchCV(Lasso(), alpha_values, scoring='neg_mean_squared_error', cv=10 )

print('\nO melhor valor de alpha para a regressao Lasso eh:',lasso.fit(X,y).best_params_)

print('\nMelhor score (Lasso):',lasso.fit(X,y).best_score_)




best_lasso_model= Lasso(alpha=2)

best_lasso_coeffs = best_lasso_model.fit(X,y).coef_

plt.figure(figsize=(10,6))
plt.plot(range(len(feature_names)),best_lasso_coeffs)
plt.axhline(0, color='r', linestyle='solid')
plt.xticks(range(len(feature_names)),feature_names,rotation=50)
plt.title("Coeficientes para a Regressao Lasso")
plt.ylabel("coeficientes")
plt.xlabel("features")
plt.show()

print(f"\nCoeficientes para a regressao Lasso: {best_lasso_coeffs}")

'''
Vemos que o uso da regularização Lasso produz resultados um pouco melhores em 
comparação à regularização Ridge, ou seja, aumenta o 'neg_mean_squared_error' 
médio de quase -104816.07 para cerca de -104390.91.

A regularização Lasso elimina completamente os recursos Runs, CAtBat, CHits, CHmRun e N do modelo 
(já que seus coeficientes estimados são 0) e nos dá um modelo mais enxuto (com menos variáveis) 
com a melhor pontuação geral.
'''


# ****************** REGRESSAO ELASTICA ********************
alpha_values = {'alpha':[0.00005,0.0005,0.001, 0.01, 0.05, 0.06, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100],
 'l1_ratio':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1]}

elastic= GridSearchCV(ElasticNet(), alpha_values, scoring='neg_mean_squared_error', cv=10 )

print(f"\nMelhores valores para alpha e l1_ratio: {elastic.fit(X,y).best_params_}")

print('\nMelhor score (Elastica):',elastic.fit(X,y).best_score_)

'''
Neste caso, o melhor l1_ratio acaba sendo 1 , o que é o mesmo que uma regularização 
Lasso . Consequentemente, a melhor pontuação também permanece a mesma obtida com 
a regularização Lasso anteriormente.
'''

# Grafico para comparacao dos coeficientes para as regressoes

comparing_models = pd.DataFrame({'without_regularization':multiple_lr_coeffs,
 'Ridge':best_ridge_coeffs,
 'Lasso':best_lasso_coeffs},
 index=feature_names)
comparing_models.plot(figsize=(10, 6))
plt.axhline(0, color='r', linestyle='solid')
plt.title("Coeficientes para as regressões Linear, Ridge e Lasso")
plt.ylabel("coeficientes")
plt.xlabel("features")
plt.show()

'''
CONCLUSÃO:

A regularização Lasso é a claramente vencedora 
neste caso , pois produz a melhor pontuação média e também resulta em um modelo 
menor/mais simples.

'''