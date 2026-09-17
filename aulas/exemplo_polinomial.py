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
df_comissao = pd.read_excel('comissao.xlsx')

print("\n******* DADOS DO DATAFRAME *******")
print(df_comissao)

# Exibir a estrutura do DataFrame: quantidade de linhas e colunas, nomes das colunas, tipos de dados das colunas, etc...
print("\n******* ESTRUTURA DO DATAFRAME *******")
print(df_comissao.info())

df_comissao = df_comissao.astype({"comissao": float})


plt.scatter(df_comissao.quantidade, df_comissao.comissao)
plt.title('Correlação')
plt.xlabel('Quantidade')
plt.ylabel('Comissão')
plt.grid(True)
plt.show()


'''
Aparentemente, pelo gráfico, temos a impressão de ser uma curva. Entretanto,
vamos testar a regressão linear simples.
'''
# Gerar a tabela de correlação
print(df_comissao.corr())

# Aplicar a regressão linear utilizando o statsmodel

# Criação do modelo
regressao = smf.ols('comissao ~ quantidade', data = df_comissao).fit()

print(regressao.summary())

'''
IMPORTANTE: se analisarmos a equação da regressão (Comissão = -626 + 178.quantidade),
podemos observar algumas inconsistências como, por exemplo:

- quando a quantidade é igual a zero, a comissão é de R$-626.00
- quando a quantidade é igual a um, a comissão é de R$-448.00

A comissão continuará negativa até que o segundo termo da equação seja maior do que o primeiro.

Isso não faz sentido!!! :(
'''

plt.scatter(y=df_comissao.comissao, x=df_comissao.quantidade, color='blue', s=50, alpha=0.6)
X_plot = np.linspace(0, 70)
plt.plot(X_plot, X_plot*regressao.params[1] + regressao.params[0], color='r')
plt.title('Reta de regressão')
plt.ylabel('COMISSÃO')
plt.xlabel('QUANTIDADE')
plt.show()

'''
Analisando o gráfico, é possível observar que a regressão linear falha quando
forem realizadas previsões futuras, pois a tendência é que os pontos sigam outra
direção e não se ajustem mais à reta.

Ideia: Aplicar a regressão polinomial! ;-) 
'''

'''
Resumo simples: 
Grau 1 → regressão linear
Grau 2 → curva suave
Grau alto → risco de overfitting
'''
X = df_comissao[['quantidade']]
y = df_comissao.comissao

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
    cv=5,  # validação cruzada 5-fold
    scoring='neg_mean_squared_error'
)

grid.fit(X, y)

print("Melhor grau:", grid.best_params_)


melhor_modelo = grid.best_estimator_

y_pred = melhor_modelo.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))

print("RMSE:", rmse)


plt.scatter(X, y, c = "gray")
plt.xlabel("Quantidade")
plt.ylabel("Comissão")
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

print("Equação: COMISSÃO = {:.1f} + {:.1f}*quantidade + {:.1f}*quantidade^2".format(modelo_final.intercept_, modelo_final.coef_[1], modelo_final.coef_[2]))


# Previsao para uma nova quantidade
quantidade = [[72]]
previsao = melhor_modelo.predict(quantidade)
print("Se vender {}, irá ganhar {:.2f} reais de comissão".format(quantidade[0][0], previsao[0]))

