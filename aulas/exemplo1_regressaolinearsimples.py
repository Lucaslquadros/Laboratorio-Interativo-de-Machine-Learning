import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

df_salarios = pd.read_csv('base_salarios.csv')

print("\n******* DADOS DO DATAFRAME *******")
print(df_salarios)

# Exibir a estrutura do DataFrame: quantidade de linhas e colunas, nomes das colunas, tipos de dados das colunas, etc...
print("\n******* ESTRUTURA DO DATAFRAME *******")
print(df_salarios.info())


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
matriz_correlacao = df_salarios.corr(numeric_only=True)
print("\n******* MATRIZ DE CORRELAÇÃO ENTRE AS VARIÁVEIS *******")
print(matriz_correlacao)

'''
Deve-se analisar a matriz de correlação pelos valores das correlações
das variáveis independentes com a variável dependente (alvo - o que se quer prever).

Pela análise da matriz, pode-se afirmar que a correlação existente entre a variável "Salary"
e a "Experience Years" é de 0.98 - correlação MUIRO FORTE). Assim, temos:

- variável INDEPENDENTE: "Experience Years"
- variável DEPENDENTE: "Salary".

'''

# PASSO 2: DIVIDIR o conjunto de dados em TREINAMENTO (70% dos dados) e TESTE (30% dos dados)

# ETAPA 1: Dividir os dados em uma matriz X que contém os dados da variável independente, e uma matriz y com os dados da variável dependente.

X = df_salarios[['Experience Years']] #atributo preditor (variável INDEPENDENTE)
y = df_salarios['Salary'] #alvo ou rótulo (variável DEPENDENTE)

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

O resultado que nós obtivemos foi de 0.8943, ou seja, nosso modelo conseguiu explicar 
a variação da variável independente em 89,43% dos casos.
'''

# Vamos ver como nosso modelo se comporta e plotamos também o gráfico de dispersão do nosso dataset completo:

plt.scatter(X, y, color="blue")
plt.plot(X_teste, predicoes_modelo, color="red")
plt.title("Anos de experiência x Salário (Dados de Teste)")
plt.xlabel("Anos de experiência")
plt.ylabel("VSalário")
plt.show()


# PASSO 5: PREDIÇÕES PARA NOVOS DADOS

# EXEMPLO 1: Prever o salário tendo como 3.4 anos de experiência

anos_exper = {'Experience Years':3.4}

df = pd.DataFrame(data = anos_exper,index=[0])

salario = modelo_regressao_simples.predict(df)

print(f"\nO salário previsto para um funcionário com 3.4 anos de experiência é R${salario[0]:.2f}")


# EXEMPLO 2: Prever o salário tendo como 12 anos de experiência

anos_exper = {'Experience Years':12}

df = pd.DataFrame(data = anos_exper,index=[0])

salario = modelo_regressao_simples.predict(df)

print(f"\nO salário previsto para um funcionário com 12 anos de experiência é R${salario[0]:.2f}")


