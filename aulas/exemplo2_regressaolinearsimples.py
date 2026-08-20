# **Regressão Linear com Uma Variável**

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

'''
Um agente imobiliário quer alguma ajuda para prever os preços das casas para as regiões nos EUA. Seria ótimo se você pudesse de alguma forma criar um modelo para ela que lhe permita colocar algumas características de uma casa e retornar uma estimativa de quanto a casa venderia.

Os dados contém as seguintes colunas:

'Avg. Area Income': Média da renda dos residentes de onde a casa está localizada
'Avg. Area House Age': Média de idade das casas da mesma cidade.
'Avg. Area Number of Rooms': Número médio de salas para casas na mesma cidade.
'Avg. Area Number of Bedrooms': Número médio de quartos para casas na mesma cidade
'Area Population': A população da cidade onde a casa está localizada.
'Price': Preço de venda da casa.
'Address': Endereço da casa;
'''

df_casas = pd.read_csv('USA_Housing.csv')

print("\n******* DADOS DO DATAFRAME *******")
print(df_casas)

# Exibir a estrutura do DataFrame: quantidade de linhas e colunas, nomes das colunas, tipos de dados das colunas, etc...
print("\n******* ESTRUTURA DO DATAFRAME *******")
print(df_casas.info())


# PASSO 0: DETERMINAR O OBJETIVO DA REGRESSÃO

'''
Nesse exemplo, será aplicada a regressão simples, pois o objetivo é prever
o valor de uma casa tendo como base somente UMA característica desse imóvel.

IMPORTANTE: É obrigatório saber o que se prever quando for utilizar um modelo
de regressão!! ;-)
'''

# PASSO 1: Gerar a tabela de correlação entre as variáveis numéricas

'''
A correlação de Pearson, também conhecida como coeficiente de correlação 
produto-momento de Pearson, é uma medida estatística que avalia a força e a direção 
da relação linear entre duas variáveis quantitativas. O coeficiente varia de -1 a +1, 
onde:

- 0.9 para mais ou para menos indica uma correlação muito forte.
- 0.7 a 0.9 positivo ou negativo indica uma correlação forte.
- 0.5 a 0.7 positivo ou negativo indica uma correlação moderada.
- 0.3 a 0.5 positivo ou negativo indica uma correlação fraca.

'''
# Matriz de correlação entre as variáveis numéridas do DataFrame
matriz_correlacao = df_casas.corr(numeric_only=True)
print("\n******* MAPA DE CALOR DAS CORRELAÇÕES ENTRE AS VARIÁVEIS *******")

sns.heatmap(matriz_correlacao, fmt='.2f', square=True, linecolor='white', annot=True, cmap="coolwarm")
plt.show()

'''
Deve-se analisar a matriz de correlação pelos valores das correlações
das variáveis independentes com a variável dependente (alvo - o que se quer prever).

Pela análise da matriz, pode-se afirmar que a maior correlação existente com a variável "Price" é
da variável "Avg. Area Income" (0.64 - correlação MODERADA). Assim, temos:

- variável INDEPENDENTE: "Avg. Area Income"
- variável DEPENDENTE: "Price".

'''


# PASSO 2: DIVIDIR o conjunto de dados em TREINAMENTO (70% dos dados) e TESTE (30% dos dados)

# ETAPA 1: Dividir os dados em uma matriz X que contém os dados da variável independente, e uma matriz y com os dados da variável dependente.

X = df_casas[['Avg. Area Income']] #atributo preditor (variável INDEPENDENTE)
y = df_casas['Price'] #alvo ou rótulo (variável DEPENDENTE)

# OBS: é importante checar os dados de X e y
print("\n******* MATRIZ X - VARIÁVEL INDEPENDENTE *******")
print(X)

print("\n******* MATRIZ y - VARIÁVEL DEPENDENTE *******")
print(y)

# ETAPA 2: Dividir os dados em um conjunto de TREINAMENTO e um conjunto de TESTE

'''
IMPORTANTE:

- O modelo de Regressão Simples será criado usando o conjunto de treinamento (70% dos dados)
- O conjunto de Teste (30% dos dados) será utilizado para a avaliação do modelo.

Resumindo:

- 70% dos dados: Conjunto de Treinamento
- 30% dos dados: Conjunto de Teste
'''

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.3,random_state=0)

# OBS: é importante checar os tamanhos (quantidade de linhas dos conjuntos)
print("\n******* TAMANHO DO CONJUNTO DE TREINAMENTO *******")
print(X_treinamento.shape[0])

print("\n******* TAMANHO DO CONJUNTO DE TESTE *******")
print(X_teste.shape[0])


# PASSO 3: CRIAÇÃO E TREINAMENTO DO MODELO DE REGRESSÃO SIMPLES

# Criação do modelo
modelo_regressao_simples = LinearRegression()

# Treinamento do modelo: cálculo do INTERCEPTO E DO COEFICIENTE
modelo_regressao_simples.fit(X_treinamento, y_treinamento)

# OBS: é importante verificar os valores do intercepto e do coeficiente

intercepto = modelo_regressao_simples.intercept_
print("\n******* INTERCEPTO *******")
print(intercepto)

coeficiente = modelo_regressao_simples.coef_
print("\n******* COEFICIENTE *******")
print(coeficiente)


# PASSO 4: AVALIAÇÃO DO MODELO DE REGRESSÃO SIMPLES


# ETAPA 1: Gerar as predições para o conjunto Teste
predicoes_modelo = modelo_regressao_simples.predict(X_teste)

# ETAPA 2: Calcular e analisar as métricas R-quadrado e MAE (Mean Absoluto Error)

r2 = r2_score(y_teste, predicoes_modelo)
mae = mean_absolute_error(y_teste, predicoes_modelo)
print("\n******* VALOR DO R-QUADRADO *******")
print(r2)
print("\n******* VALOR DO MAE *******")
print(mae)

'''
O que significa este valor? O r-quadrado é o coeficiente de determinação e ele expressa
a porcentagem da variação da variável dependente que a variável independente explica 
corretamente.

Em outros paralavras, podemos dizer que o r-quadrado expressa o quanto o nosso 
modelo conseguiu explicar os dados. Seu valor varia entre 0 (o modelo não consegue 
explicar a relação entre as variáveis) até 100% (o modelo conseguiu explicar o 
relacionamento entre as variáveis).

O resultado que nós obtivemos foi de 42,47%, ou seja, nosso modelo conseguiu explicar 
a variação da variável independente em 42,47% dos casos.
'''


# Vamos ver como nosso modelo se comporta e plotamos também o gráfico de dispersão do nosso dataset completo:

plt.scatter(X, y, color="blue")
plt.plot(X_teste, predicoes_modelo, color="red")
plt.title("Renda do residente x Valor da casa (Dados de Teste)")
plt.xlabel("Renda do residente")
plt.ylabel("Valor da casa")
plt.show()

# PASSO 5: PREDIÇÕES PARA NOVOS DADOS

# EXEMPLO 1: Prever o preçoç da casa para uma pessoa com renda de 80000.00

media_renda = {'Avg. Area Income':80000.00}

df = pd.DataFrame(data = media_renda,index=[0])

preco_casa = modelo_regressao_simples.predict(df)

print(f"\nO valor previsto da casa é {preco_casa[0]:.3f}")


# EXEMPLO 2: Prever o preçoç da casa para uma pessoa com renda de 30000.00

media_renda = {'Avg. Area Income':30000.00}

df = pd.DataFrame(data = media_renda,index=[0])

preco_casa = modelo_regressao_simples.predict(df)

print(f"O valor previsto da casa é {preco_casa[0]:.3f}")