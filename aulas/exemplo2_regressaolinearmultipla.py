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
'Avg. Area Number of Rooms': Número médio de quartos para casas na mesma cidade.
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
Nesse exemplo, será aplicada a regressão múltipla, pois o objetivo é prever
o valor de uma casa tendo como base algumas característica desse imóvel (várias variáveis).

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

Em um primeiro momento, vamos considerar como variáveis independentes todas as variáveis,
exceto a variável alvo (Price - Preço da Casa). Assim, temos:

- variáveis INDEPENDENTES: "Avg. Area Income", "Avg. Area House Age", 
"Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms" e "Area Population".
- variável DEPENDENTE: "Profit".

'''

# PASSO 2: DIVIDIR o conjunto de dados em TREINAMENTO (70% dos dados) e TESTE (30% dos dados)

# ETAPA 1: Dividir os dados em uma matriz X que contém os dados da variável independente, e uma matriz y com os dados da variável dependente.

X = df_casas[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']] #atributo preditor (variável INDEPENDENTE)
y = df_casas['Price'] #alvo ou rótulo (variável DEPENDENTE)

# OBS: é importante checar os dados de X e y
print("\n******* MATRIZ X - VARIÁVEIS INDEPENDENTES *******")
print(X)

print("\n******* MATRIZ y - VARIÁVEL DEPENDENTE *******")
print(y)

# ETAPA 2: Dividir os dados em um conjunto de TREINAMENTO e um conjunto de TESTE

'''
IMPORTANTE:

- O modelo de Regressão Múltipla será criado usando o conjunto de treinamento (70% dos dados)
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

# PASSO 3: CRIAÇÃO E TREINAMENTO DO MODELO DE REGRESSÃO MÚLTIPLA

# Criação do modelo
modelo_regressao_multipla = LinearRegression()

# Treinamento do modelo: cálculo do INTERCEPTO E DO COEFICIENTE
modelo_regressao_multipla.fit(X_treinamento, y_treinamento)

# OBS: é importante verificar os valores do intercepto e do coeficiente

intercepto = modelo_regressao_multipla.intercept_
print("\n******* INTERCEPTO *******")
print(intercepto)

coeficiente = modelo_regressao_multipla.coef_
print("\n******* COEFICIENTES *******")
print(coeficiente)


# PASSO 4: AVALIAÇÃO DO MODELO DE REGRESSÃO MÚLTIPLA


# ETAPA 1: Gerar as predições para o conjunto Teste
predicoes_modelo = modelo_regressao_multipla.predict(X_teste)

# ETAPA 2: Calcular e analisar as métricas R-quadrado e MAE (Mean Absoluto Error)

r2 = r2_score(y_teste, predicoes_modelo)
mae = mean_absolute_error(y_teste, predicoes_modelo)
print("\n******* VALOR DO R-QUADRADO *******")
print(r2)
print("\n******* VALOR DO MAE *******")
print(mae)

'''
O que significa este valor? O r-quadrado é o coeficiente de determinação e ele expressa
a porcentagem da variação da variável dependente que as variáveis independentes explicam 
corretamente.

Em outras paralavras, podemos dizer que o r-quadrado expressa o quanto o nosso 
modelo conseguiu explicar os dados. Seu valor varia entre 0 (o modelo não consegue 
explicar a relação entre as variáveis) até 100% (o modelo conseguiu explicar o 
relacionamento entre as variáveis).

O resultado que nós obtivemos foi de 0.9200, ou seja, nosso modelo conseguiu explicar 
a variação da variável independente em 92% dos casos.
'''


# PASSO 5: PREDIÇÕES PARA NOVOS DADOS

'''
EXEMPLO 1: Prever o preço da casa A, sendo que:
- Avg. Area Income = 100200.80
- Avg. Area House Age = 6.4
- Avg. Area Number of Rooms = 4.5
- Avg. Area Number of Bedrooms  = 3.5
- Area Population = 20748.30
'''
dados_casa = {'Avg. Area Income': 100200.80,
                  'Avg. Area House Age':6.4,
                  'Avg. Area Number of Rooms': 4.5,
                  'Avg. Area Number of Bedrooms': 3.5,
                  'Area Population': 20748.30
                   }

df = pd.DataFrame(data = dados_casa ,index=[0])

preco = modelo_regressao_multipla.predict(df)

print(f"\nO preço previsto para a casa A é R${preco[0]:.2f}")


'''
EXEMPLO 2: Prever o preço da casa B, sendo que:
- Avg. Area Income = 40750.50
- Avg. Area House Age = 8.3
- Avg. Area Number of Rooms = 5.2
- Avg. Area Number of Bedrooms  = 4.2
- Area Population = 90800.50
'''
dados_casa = {'Avg. Area Income': 40750.50,
                  'Avg. Area House Age':8.3,
                  'Avg. Area Number of Rooms': 5.2,
                  'Avg. Area Number of Bedrooms': 4.2,
                  'Area Population': 90800.50
                   }

df = pd.DataFrame(data = dados_casa ,index=[0])

preco = modelo_regressao_multipla.predict(df)

print(f"\nO preço previsto para a casa B é R${preco[0]:.2f}")