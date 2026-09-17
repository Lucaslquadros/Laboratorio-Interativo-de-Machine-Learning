import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import linear_model
import matplotlib.pyplot as plt
from statsmodels.stats import diagnostic as diag
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

# ==========================================================
# 1. Carregar a base de dados
# ==========================================================

df = pd.read_csv("weatherHistory.csv")



# Variáveis numéricas utilizadas como preditoras
X = df.drop(columns=['Humidity'])

# Variável alvo
y = df['Humidity']

sc = StandardScaler()
X = sc.fit_transform(X)

# ==========================================================
# 2. Divisão treino / teste
# ==========================================================


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.30)

# OBS: é importante checar os tamanhos (quantidade de linhas dos conjuntos)
print("\n******* TAMANHO DO CONJUNTO DE TREINAMENTO *******")
print(X_train.shape[0])

print("\n******* TAMANHO DO CONJUNTO DE TESTE *******")
print(X_test.shape[0])

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_train)

print("R quadrado: {}".format(r2_score(y_true=y_train, y_pred=y_pred)))

# ****** SUPOSIÇÃO #1: LINEARIDADE ******

sns.pairplot(df, x_vars=df.drop(columns=['Humidity']), y_vars='Humidity',
             aspect=0.7)
plt.show()

'''

INTERPRETAÇÃO: 

- Temperature (C) → relação aproximadamente linear

- Apparent Temperature (C) → relação aproximadamente linear

- Wind Speed (km/h) → relação fraca

- Wind Bearing (degrees) → relação fraca

- Visibility (km) → relação fraca

- Loud Cover → relação fraca

- Pressure (millibars) → relação fraca


✔ Suposição de linearidade parcialmente atendida
Existem relações aproximadamente lineares, mas não são fortes para todas as variáveis.
'''

# ****** SUPOSIÇÃO #2: MÉDIA DOS RESÍDUOS ******

'''
Resíduos são as diferenças entre o valor real e o valor previsto (resıduo=yreal−yprevisto). 
Uma das premissas da regressão linear é que a média dos resíduos deve ser zero.
'''
residuos = y_train.values - y_pred
mean_residuals = np.mean(residuos)
print("Média dos resíduos {}".format(mean_residuals))

'''
CONCLUSÃO: como o valor é muito próximo de 0 (zero). Isso significa que o modelo não 
tem viés sistemático (não está enviesado).

✔ Suposição atendida

'''

# ****** SUPOSIÇÃO #3: HOMOCEDASTICIDADE ******

print(y_pred.min(), y_pred.max())
print(residuos.min(), residuos.max())

sns.scatterplot(x=y_pred, y=residuos)
plt.xlabel('Valores preditos')
plt.ylabel('Residuos')
plt.ylim(-1.23, 0.38)
plt.xlim(0.28, 1.35)
sns.lineplot(x=[0.28, 1.35], y=[0, 0], color='blue')
plt.show()

'''
TESTE ESTATÍSTICO

Uma dica é ter em mente que, se quisermos 95% de confiança em nossas descobertas e testes, o valor de p deve ser menor  que 0,05 para poder rejeitar a hipótese nula. Lembre-se de que um pesquisador ou cientista de dados sempre tentará rejeitar a hipótese nula.

'''

# Teste de Goldfeld Quandt

'''
Verificando a heterocedasticidade: Usando o Teste de Goldfeld Quandt, testamos a heterocedasticidade.

Hipótese Nula (H0): Termos de erro são homocedásticos
Hipótese Alternativa (H1): Termos de erro são heterocedásticos.

Regra de decisão

Se

   * p-value < 0.05 → rejeita H0 → heterocedasticidade

   * p-value > 0.05 → não rejeita H0 → homocedasticidade

'''
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(residuos, X_train)
print(lzip(name, test))

'''

RESUMINDO: 

❌ Suposição não atendida: análise do gráfico e teste estatístico. 

'''

# ****** SUPOSIÇÃO #4: NORMALIDADE ******

sns.displot(residuos, kde=True)
plt.title('Normalidade dos resíduos')
plt.show()

'''
Pelo gráfico, podemos observar que não há uma curva normal.
A curva é assimétrica à esquerda.

❌ Suposição NÃO atendida.

'''

# ****** SUPOSIÇÃO #5: INDEPENDÊNCIA DOS RESÍDUOS ******

# Gráfico Resíduos x Valores Preditos

plt.figure(figsize=(10, 5))
sns.lineplot(x=y_pred, y=residuos, marker='o', color='blue')
plt.xlabel('Valores preditos')
plt.ylabel('Resíduos')
plt.ylim(-1.23, 0.38)
plt.xlim(0.28, 1.35)
sns.lineplot(x=[0.28, 1.35], y=[0, 0], color='red')
plt.title('Gráfico de resíduos x valores preditos')
plt.show()

# Gráfico PACF

sm.graphics.tsa.plot_pacf(residuos, lags=40)
plt.show()

# Teste estatístico

lb_test_results = diag.acorr_ljungbox(residuos, lags=40, return_df=True)

print(min(lb_test_results.lb_pvalue))

'''
Como o valor de p é menor que 0,05, rejeitamos a hipótese nula de que os termos de erro não são autocorrelacionados.

Por outro lado, podemos aplicar o teste de Durbin-Watson a fim de constatar se realmente a autocorrelação existente é significativa ou não.
 Caso o resultado do teste esteja dentro do intervalo de 1,5 e 2,5, consideramos que a autocorrelação não é problemática neste modelo de regressão.
'''

durbin_watson = sms.durbin_watson(residuos)

print(f"Estatística de Durbin-Watson: {durbin_watson}")

# Interpretação
if durbin_watson < 1.5:
    print("Há evidências de autocorrelação positiva.")
elif durbin_watson > 2.5:
    print("Há evidências de autocorrelação negativa.")
else:
    print("Não há evidências significativas de autocorrelação.")

'''
RESUMINDO:

 - Gráfico Valores Preditos x Residuos: não apresenta um padrão no
 esparsamento dos dados. Pela análise do gráfico, a suposição foi atendida.
 - Gráfico PACF: os resultados mostram que não há sinais de autocorrelação, visto que não há picos 
 fora da região azul do intervalo de confiança. Pela análise do gráfico, a suposição foi atendida.
 - Teste estatístico Ljungbox: como o valor de p é maior do que 0,05, então não há autocorrelação.
 - Teste de Durbin-Watson: não há evidências significativas de autocorrelação.

✔ Suposição atendida

'''

# ****** SUPOSIÇÃO #6: AUSÊNCIA DE COLINEARIDADE ******

sns.heatmap(df.drop(columns=['Humidity']).corr(), annot=True, cmap='RdYlGn', square=True)
plt.show()

'''
Esses dados contêm multicolinearidade muito forte entre as variáveis Temperature e
Apparent Temperature. Isso ocorre porque Apparent Temperature é calculada a 
partir da temperatura real.

❌ Suposição NÃO atendida.

'''

'''
CONCLUSÃO: Portanto, os principais pressupostos da Regressão Linear NÃO foram cumpridos com sucesso. 
Nesse caso, podemos afirmar que as variáveis independentes não conseguem explicar a dependente.

DICA: Para melhorar o modelo nessa base é recomendado remover uma variável altamente 
correlacionada. Por exemplo, remover Apparent Temperature. Isso reduz drasticamente a multicolinearidade.
'''




