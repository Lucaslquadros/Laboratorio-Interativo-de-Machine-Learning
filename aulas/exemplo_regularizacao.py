import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import seaborn as sns


df = pd.read_csv("diabetes.csv")
# Criacao do conjunto de atributos preditores (X) e o alvo (y)
X = df.drop(columns=['diabetes_measure'])
y = df.diabetes_measure


# **************** REGRESSAO MULTIPLA ****************

# Treinamento do modelo considerando todas as variaveis preditoras
multiple_lr = LinearRegression().fit(X,y)

'''
Métrica de Avaliação do Modelo

O Scikit Learn fornece uma lista variada de métricas de pontuação para avaliar modelos. 
Como compararemos modelos de regressão linear hoje, o "erro_quadrado_médio_negativo" 
é o mais adequado para nós.
A convenção geral seguida por todas as métricas no scikit learn é que valores de retorno 
mais altos são melhores do que valores de retorno mais baixos .
Assim, métricas que medem a distância entre os valores previstos do modelo e 
os valores reais dos dados (como metrics.mean_squared_error) estão disponíveis 
como neg_mean_squared_error. Assim, por exemplo, um modelo com 
-100 neg_mean_squared_error é melhor do que um com -150 neg_mean_squared_error.

ALém disso, o MSE negativo é o mais compatível com os métodos de regularização, 
pois esses modelos minimizam uma função baseada em MSE. Então, usar MSE como métrica 
de avaliação mantém coerência matemática.

Outro fato é que o MSE penaliza erros grandes, importante em previsão médica.

Outro motivo é porque métodos de avaliação do Scikit-learn, 
como GridSearchCV e cross_val_score, assumem que maiores valores da métrica indicam 
melhores modelos. Como o MSE é uma métrica de erro (quanto menor melhor), o Scikit-learn 
utiliza o MSE negativo para manter a lógica de maximização.

'''

# ****************** VALIDACAO CRUZADA PARA ENCONTRAR O MELHOR MSE ******************
mse= cross_val_score(multiple_lr,X,y,scoring='neg_mean_squared_error',cv=10)

print(f"Media do MSE para a regressao multipla: {mse.mean()}")

multiple_lr_coeffs = multiple_lr.coef_
print(f"\nCoeficientes para a Regressao Multipla: {multiple_lr_coeffs}")

feature_names = df.drop('diabetes_measure',axis=1).columns


plt.figure(figsize=(10,6))
plt.plot(range(len(multiple_lr_coeffs)),multiple_lr_coeffs)
plt.axhline(0, color='r', linestyle='solid')
plt.xticks(range(len(feature_names)),feature_names,rotation=50)
plt.title("Coeficientes para a Regressao Multipla")
plt.ylabel("coeficientes")
plt.xlabel("features")
plt.show()

'''
OBS: Vemos que neste modelo, as características bmi, s1, s2 e s5 estão tendo um 
impacto considerável na progressão do diabetes, pois todas elas têm altos valores 
de coeficiente estimado (tanto para positivo quanto negativo).
'''

'''
Por que regularizar?

Um dos pressupostos básicos de um modelo de regressão linear múltipla é que não 
deve haver (ou deve haver muito pouca) multicolinearidade entre as variáveis
de características. Isso significa essencialmente que as variáveis de características
devem, idealmente, ter pouca ou nenhuma correlação entre si.

Uma alta correlação entre as duas variáveis de características gera vários problemas. 
Se incluirmos uma delas no modelo, adicionar a outra terá pouco ou nenhum impacto 
na melhoria do modelo, mas apenas o tornará mais complexo.
'''
mask = np.triu(np.ones_like(X.corr(),dtype=bool))
# Geracao do mapa de calor para observacao da correlacao entre as variaveis preditoras
sns.heatmap(X.corr(),mask =mask, fmt='.2f', square=True, linecolor='white', annot=True, cmap="coolwarm")
plt.show()

'''
Nos modelos lineares múltiplos que construímos na seção anterior, ambas as 
características s1 e s2 se mostraram importantes. No entanto, podemos observar 
que elas apresentam uma correlação positiva muito alta, de cerca de 0,896. Isso 
está claramente induzindo multicolinearidade no modelo.

Para combater esses problemas, podemos usar técnicas de regularização que nos 
permitem reduzir essa variância do modelo ao custo de adicionar algum viés , de 
modo que o erro total seja reduzido. Uma variância menor implica que o problema 
de sobreajuste é resolvido automaticamente, pois o modelo generaliza bem para 
dados não observados após a regularização.

As técnicas de regularização funcionam adicionando fatores de penalidade à 
função de custo MQO original, de modo que valores altos de coeficientes sejam 
penalizados , aproximando-os de zero.
'''

# ****************** REGRESSAO RIDGE ********************

alpha_values = {'alpha':[0.001, 0.01,0.02,0.03,0.04, 0.05, 0.06, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100]}

ridge= GridSearchCV(Ridge(), alpha_values, scoring='neg_mean_squared_error', cv=10 )

print('\nO melhor valor de alpha para a regressao Ridge eh:',ridge.fit(X,y).best_params_)

print('\nMelhor score (Ridge):',ridge.fit(X,y).best_score_)

'''
Vemos que a regularização do nosso modelo de regressão linear múltipla usando a 
regressão Ridge aumenta o 'neg_mean_squared_error' médio de quase -3000,38 para 
cerca de -2995,94, o que é uma melhoria moderada.
'''


best_ridge_model= Ridge(alpha=0.04)

best_ridge_coeffs = best_ridge_model.fit(X,y).coef_

plt.figure(figsize=(10,6))
plt.plot(range(len(feature_names)),best_ridge_coeffs)
plt.axhline(0, color='r', linestyle='solid')
plt.xticks(range(len(feature_names)),feature_names,rotation=50)
plt.title("Coeficientes para a Regressao Ridge")
plt.ylabel("coeficientes")
plt.xlabel("features")
plt.show()

'''
Anteriormente, as características s1 e s2 se destacavam como características 
importantes na regressão linear múltipla; no entanto, seus valores de coeficientes 
são significativamente reduzidos após a regularização Ridge. As características 
'bmi' e s5 continuam importantes.
'''

# ****************** REGRESSAO LASSO ********************

alpha_values = {'alpha':[0.001, 0.01,0.02,0.03,0.04, 0.05, 0.06,0.07, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100]}

lasso= GridSearchCV(Lasso(), alpha_values, scoring='neg_mean_squared_error', cv=10 )

print('\nO melhor valor de alpha para a regressao Lasso eh:',lasso.fit(X,y).best_params_)

print('\nMelhor score (Lasso):',lasso.fit(X,y).best_score_)

'''
Vemos que o uso da regularização Lasso produz resultados ligeiramente melhores em 
comparação à regularização Ridge, ou seja, aumenta o 'neg_mean_squared_error' 
médio de quase -3000,38 para cerca de -2986,37 (comparado a -2995,94 da regularização 
Ridge).
'''


best_lasso_model= Lasso(alpha=0.06)

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
A regularização Lasso elimina completamente os recursos 'idade', s2 e s4 do modelo 
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
Podemos observar claramente os seguintes pontos gerais sobre regularizações de 
Ridge e Lasso :

- Embora a regularização de Ridge reduza consideravelmente os valores das 
estimativas de coeficientes, aproximando-os de zero, ela não os torna exatamente zero.

- Por outro lado, a regularização Lasso elimina completamente as características 
('idade', 's2' e 's4' em nosso exemplo) do modelo, atribuindo valores zero aos seus 
coeficientes . Isso resulta em um modelo muito mais enxuto/simples.
Além disso, a regularização Lasso produz a melhor pontuação média (-2986,36) 
entre todos os três modelos. Portanto, a regularização Lasso é a claramente vencedora 
neste caso , pois produz a melhor pontuação média e também resulta em um modelo 
menor/mais simples. No entanto, isso nem sempre acontece.

A regularização Lasso tende a ter melhor desempenho em casos em que um 
número relativamente pequeno de recursos tem coeficientes substanciais 
(como bmi e s5 em nosso exemplo).

Por outro lado, a regressão de Ridge tem melhor desempenho em casos 
em que os coeficientes têm aproximadamente o mesmo tamanho, ou seja, 
todos os recursos impactam a variável de resposta de forma mais ou menos igual.


'''
# Predicao para novos dados utilizando o melhor modelo

# Treinamento do modelo Lasso com alpha = 0.06 (conforme o script de referência)
lasso_model = Lasso(alpha=0.06)
lasso_model.fit(X, y)

# 2. Carregamento da base não escalonada para ajustar o escalonador corretamente
df_raw = pd.read_csv("diabetes_nao_escalonado.csv")
X_raw = df_raw.drop(columns=['diabetes_measure'])
N_train = len(df_raw)

# 3. Definição de novos dados em valores reais (que não constam na base não escalonada)
new_data_raw = pd.DataFrame({
    'age': [62.0, 35.0],       # Idade em anos
    'sex': [1.0, 2.0],         # Gênero
    'bmi': [27.5, 23.5],       # IMC
    'bp': [92.0, 85.0],        # Pressão sanguínea
    's1': [170.0, 145.0],      # Exames laboratoriais (s1)
    's2': [90.0, 88.0],        # Exames laboratoriais (s2)
    's3': [48.0, 52.0],        # Exames laboratoriais (s3)
    's4': [3.8, 4.0],          # Exames laboratoriais (s4)
    's5': [4.4, 4.1],          # Exames laboratoriais (s5)
    's6': [88.0, 78.0]         # Exames laboratoriais (s6)
})

# 3. Escalonamento dos novos dados utilizando a média e o desvio padrão populacional (ddof=0) da base de treino, divididos por sqrt(N_train)
new_data_scaled = pd.DataFrame()
for col in X_raw.columns:
    mean_val = X_raw[col].mean()
    std_pop = X_raw[col].std(ddof=0) # Desvio padrão populacional do scikit-learn
    new_data_scaled[col] = (new_data_raw[col] - mean_val) / (std_pop * np.sqrt(N_train))

'''
IMPORTANTE: esse escalonamento "diferente" so foi aplicado a essa base de dados, pois a base "diabetes.csv" foi escalonada de forma diferente à tradicional (StandardScaler do scikit-learn). Para os outros exercícios, 
considere a base escalonada pelo scikit-learn.
'''

# 4. Execução da predição com o modelo Lasso
predictions = lasso_model.predict(new_data_scaled)

# Exibição dos resultados
for i, pred in enumerate(predictions, start=1):
    print(f"Predição para o Novo Paciente {i}: {pred:.2f}")