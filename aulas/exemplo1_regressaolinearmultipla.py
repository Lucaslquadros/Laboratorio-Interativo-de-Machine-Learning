import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

'''
Imagine que você tem um fundo de investimento e recebe a lista de 50 empresas (startups).

Na lista é possível ver o quanto cada empresa teve de gastos administrativos, gastos 
com marketing, investimentos em P&D, o Estado (unidade federativa) que a empresa 
pertence e seus lucros.

Seu objetivo como cientista de dados é criar um modelo de machine learning que 
possa auxiliar o fundo de investimento na decisão se deve investir numa empresa ou não.

E para alcançar esse objetivo você deve construir um modelo de machine learning 
que seja capaz de prever o quão lucrativo é uma empresa a partir das informações 
de gastos de marketing, gastos administrativos etc.
'''

df_startups = pd.read_csv('50_Startups.csv')

print("\n******* DADOS DO DATAFRAME *******")
print(df_startups)

# Exibir a estrutura do DataFrame: quantidade de linhas e colunas, nomes das colunas, tipos de dados das colunas, etc...
print("\n******* ESTRUTURA DO DATAFRAME *******")
print(df_startups.info())



# PASSO 0: DETERMINAR O OBJETIVO DA REGRESSÃO

'''
Nesse exemplo, será aplicada a regressão simples, pois o objetivo é prever
o valor do LUCRO da startup tendo como base alguns gastos da empresa (várias variáveis).

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
matriz_correlacao = df_startups.corr(numeric_only=True)
print("\n******* MAPA DE CALOR DAS CORRELAÇÕES ENTRE AS VARIÁVEIS *******")

sns.heatmap(matriz_correlacao, fmt='.2f', square=True, linecolor='white', annot=True, cmap="coolwarm")
plt.show()



'''
Deve-se analisar a matriz de correlação pelos valores das correlações
das variáveis independentes com a variável dependente (alvo - o que se quer prever).

Pela análise da matriz, pode-se afirmar que as maiores correlações existentes com a variável "Profit" são
das variáveis "R&D Spend" (0.97 - correlação MUITO FORTE) e "Marketing Spend" (0.75 - correlação FORTE). Assim, temos:

- variáveis INDEPENDENTES: "R&D Spend" e "Marketing Spend"
- variável DEPENDENTE: "Profit".

'''

# PASSO 2: DIVIDIR o conjunto de dados em TREINAMENTO (70% dos dados) e TESTE (30% dos dados)

# ETAPA 1: Dividir os dados em uma matriz X que contém os dados da variável independente, e uma matriz y com os dados da variável dependente.

X = df_startups[['R&D Spend', 'Marketing Spend']] #atributo preditor (variável INDEPENDENTE)
y = df_startups['Profit'] #alvo ou rótulo (variável DEPENDENTE)

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


# PASSO 3: CRIAÇÃO E TREINAMENTO DO MODELO DE REGRESSÃO MULTIPLA

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


# EQUACAO DA REGRESSAO MULTIPLA
print("********* EQUACAO DA REGRESSAO LINEAR MULTIPLA *********")

print(f"Lucro = {intercepto:.2f} + ({coeficiente[0]:.2f} * R&D Spend) + ({coeficiente[1]:.2f} * Marketing Spend)")

# PASSO 4: AVALIAÇÃO DO MODELO DE REGRESSÃO MULTIPLA


# ETAPA 1: Gerar as predições para o conjunto Teste
predicoes_modelo = modelo_regressao_multipla.predict(X_teste)

# ETAPA 2: Calcular e analisar as métricas R-quadrado

r2 = r2_score(y_teste, predicoes_modelo)
print("\n******* VALOR DO R-QUADRADO *******")
print(r2)


'''
O que significa este valor? O r-quadrado é o coeficiente de determinação e ele expressa
a porcentagem da variação da variável dependente que a variável independente explica 
corretamente.

Em outras paralavras, podemos dizer que o r-quadrado expressa o quanto o nosso 
modelo conseguiu explicar os dados. Seu valor varia entre 0 (o modelo não consegue 
explicar a relação entre as variáveis) até 100% (o modelo conseguiu explicar o 
relacionamento entre as variáveis).

O resultado que nós obtivemos foi de 0.9431, ou seja, nosso modelo conseguiu explicar 
a variação da variável independente em 94,31% dos casos.

Para essa base de dados, nota-se que houve uma melhora no valor do R-quadrado quando
se fez a escolha das variáveis que apresentaram maior correlação com a variável "Profit".
'''

# Vamos ver como nosso modelo se comporta e plotamos também o gráfico de dispersão do nosso dataset completo:
sns.regplot(data=df_startups,x='R&D Spend', y='Profit')
plt.show()


# PASSO 5: PREDIÇÕES PARA NOVOS DADOS

'''
EXEMPLO 1: Prever o lucro da Startup A, sendo que:
- R&D Spend = 134560.30
- Marketing Spend = 207934.54
'''
gastos_startup = {'R&D Spend':134560.30,
                  'Marketing Spend':207934.54
                   }

df = pd.DataFrame(data = gastos_startup,index=[0])

lucro = modelo_regressao_multipla.predict(df)

print(f"\nO lucro previsto para a Startup A é R${lucro[0]:.2f}")


'''
EXEMPLO 2: Prever o lucro da Startup B, sendo que:
- R&D Spend = 203780.40
- Marketing Spend = 509870.33
'''

gastos_startup = {'R&D Spend':203780.40,
                  'Marketing Spend':509870.33
                   }

df = pd.DataFrame(data = gastos_startup,index=[0])

lucro = modelo_regressao_multipla.predict(df)

print(f"\nO lucro previsto para a Startup B é R${lucro[0]:.2f}")

