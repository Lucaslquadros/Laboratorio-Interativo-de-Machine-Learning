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

df_planos = pd.read_csv('base_plano_saude_preparada.csv')

print("\n******* DADOS DO DATAFRAME *******")
print(df_planos.head(5))

print("\n******* ESTRUTURA DO DATAFRAME *******")
print(df_planos.info())



# PASSO 1: DIVIDIR o conjunto de dados em TREINAMENTO (70% dos dados) e TESTE (30% dos dados)

# ETAPA 1: Dividir os dados em uma matriz X que contém os dados da variável independente, e uma matriz y com os dados da variável dependente.

X = df_planos.drop(["gastos_plano"],axis=1)
y = df_planos.gastos_plano

# OBS: é importante checar os dados de X e y
print("\n******* MATRIZ X - VARIÁVEIS INDEPENDENTES *******")
print(X)

print("\n******* MATRIZ y - VARIÁVEL DEPENDENTE *******")
print(y)


sc = StandardScaler()
X[["idade","imc","filhos"]] = sc.fit_transform(X[["idade","imc","filhos"]])



# ETAPA 4: Dividir os dados em um conjunto de TREINAMENTO e um conjunto de TESTE

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0,test_size=0.30)

# OBS: é importante checar os tamanhos (quantidade de linhas dos conjuntos)
print("\n******* TAMANHO DO CONJUNTO DE TREINAMENTO *******")
print(X_train.shape[0])

print("\n******* TAMANHO DO CONJUNTO DE TESTE *******")
print(X_test.shape[0])

# PASSO 2: CRIAÇÃO E TREINAMENTO DO MODELO DE REGRESSÃO MÚLTIPLA

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)

# PASSO 4: AVALIAÇÃO DO MODELO DE REGRESSÃO MÚLTIPLA


# ETAPA 1: Gerar as predições para o conjunto Teste
y_pred = regr.predict(X_train)


print("R quadrado: {}".format(r2_score(y_true=y_train,y_pred=y_pred)))



'''
QUESTÃO: Será que é possível afirmar com certeza se, pela análise do R-quadrado ou qualquer 
outra métrica, o modelo é realmente BOM????

Em síntese, suposições (ou pressupostos estatísticos) são condições que assumimos — de forma 
implícita ou explícita — sobre os dados, a fim de aplicarmos uma determinada técnica estatística
'''


# ****** SUPOSIÇÃO #1: LINEARIDADE ******

'''
Em outras palavras, o modelo assume que a relação entre cada variável preditora (x) e o alvo (Y) é de natureza linear.

Caso a relação seja curvilinear, o modelo pode não captar bem os padrões dos dados. Por isso, 
é sempre recomendável verificar os gráficos de dispersão entre as variáveis antes de ajustar o 
modelo aos dados.

Uma forma de checar a linearidade entre as variáveis é visualizar a relação entre os variáveis preditoras e o alvo
usando gráficos de dispersão
'''
# visualize the relationship between the features and the response using scatterplots
sns.pairplot(df_planos, x_vars=df_planos.drop(["gastos_plano"],axis=1), y_vars='gastos_plano',aspect=0.7)
plt.show()

'''
Observando os gráficos, podemos ver que nenhuma das variáveis tem relação aproximadamente linear. 
Nesse caso, esta suposição nao foi comprovada.


'''

# ****** SUPOSIÇÃO #2: MÉDIA DOS RESÍDUOS ******

'''
Resíduos são as diferenças entre o valor real e o valor previsto (resıduo=yreal−yprevisto). 
Uma das premissas da regressão linear é que a média dos resíduos deve ser zero.
'''
residuos = y_train.values-y_pred
mean_residuals = np.mean(residuos)
print("Mean of Residuals {}".format(mean_residuals))

'''
CONCLUSÃO: como o valor é muito próximo de 0 (zero). Isso significa que o modelo não 
tem viés sistemático (não está enviesado). Portanto, podemos comprovar essa suposição!
'''

# ****** SUPOSIÇÃO #3: HOMOCEDASTICIDADE ******

'''
Homocedasticidade (também chamada de homogeneidade das variâncias), ou seja, consiste na constância da variância dos resíduos ao longo dos valores preditos. 
Essa suposição afirma que o desvio-padrão dos resíduos deve ser aproximadamente o mesmo para todos os níveis das variáveis preditoras.
Ou ainda, a dispersão dos resíduos deve ser semelhante para todos os valores das variáveis preditoras. Em outras palavras:
O erro do modelo deve ser igual para valores baixos e altos de Y

Se o nossos dados não atenderem essa suposição, estaremos diante da heterocedasticidade — o que pode afetar a precisão das estimativas e a validade dos testes estatísticos. 
Portanto, verificar essa suposição é essencial para garantir a robustez do modelo.
'''

print(residuos.max(),residuos.min())
sns.scatterplot(x=y_pred,y=residuos)
plt.xlabel('Valores preditos')
plt.ylabel('Residuos')
plt.xlim(-49842496,320541976)
plt.ylim(-265697303,1097930555)
sns.lineplot(x=[-265697303,1097930555],y=[0,0],color='blue')
plt.show()

'''
INTERPRETAÇÃO DO GRÁFICO

Interpretação

✔ nuvem aleatória → homocedasticidade

❌ formato de funil → heterocedasticidade

'''

# O gráfico caracteriza heterocedasticidade

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
Como o valor de p é maior que 0,05 no Teste de Quandt de Goldfeld, 
não podemos rejeitar sua hipótese nula de que os termos de erro 
são homoscedásticos. Por esse motivo, embora o teste tenha comprovado, nao
podemos comprovar totalmente a suposição pela análise do gráfico.

'''

# ****** SUPOSIÇÃO #4: NORMALIDADE ******

'''
É importante compreender que a regressão linear — seja ela simples ou múltipla — 
não exige que os preditores (xs) nem o resultado (Y) sigam uma distribuição normal. 
O que realmente importa é que os RESÍDUOS SEJAM NORMALMENTE DISTRIBUÍDOS.

Ou seja, mesmo com variáveis assimétricas, o modelo pode ser válido, desde que 
os resíduos (diferença entre valores observados e preditos) apresentem normalidade.
'''

sns.displot(residuos,kde=True)
plt.title('Normalidade dos resíduos')
plt.show()

'''
OBS: os termos residuais NÃO são distribuídos de forma normal para o número de pontos de teste que utilizamos. Lembre-se do teorema do limite central, 
que diz que, à medida que o tamanho da amostra aumenta, a distribuição tende a ser normal. 
O gráfico é assimétrico à direita. É muito difícil obter curvas e distribuições perfeitas 
em dados da vida real.

RESUMINDO: 

Interpretação do gráfico:

✔ curva aproximadamente normal → suposição atendida

❌ distribuição assimétrica ou multimodal → problema
'''

# ****** SUPOSIÇÃO #5: INDEPENDÊNCIA DOS RESÍDUOS ******

'''
Outra importante suposição da regressão linear é que os resíduos devem ser 
independentes entre si. Essa regra geral serve como uma “verificação final” para garantir que não 
há padrões ocultos ou efeitos não modelados nos dados.

A autocorrelação em regressão linear ocorre quando os resíduos (erros) do modelo não são independentes entre si, ou seja, 
quando o valor de um resíduo está relacionado ao valor de outro resíduo em momentos diferentes.

Se os resíduos se correlacionarem entre si — como pode ocorrer com dados 
temporais ou espaciais —, os resultados da regressão podem ser distorcidos. 
Por isso, é fundamental inspecionar gráficos de resíduos.

Quando os resíduos são autocorrelacionados, significa que o valor 
atual depende dos valores anteriores (históricos) e que há um 
padrão definido e inexplicável na variável Y que se manifesta nos 
termos de erro.

Assim, não deve haver autocorrelação nos dados e os 
termos de erro não devem formar nenhum padrão. 
Se houver padrões → autocorrelação.
'''

'''
Uma das formas de verificar se há independência ou não,
é pela análise dos gráficos:

* Resíduos x Valores Preditos

* Gráfico PACF (Partial Autocorrelation Function)

'''

'''
CONSIDERAÇÕES IMPORTANTES SOBRE O GRÁFICO 
RESIDUOS x VALORES PREDITOS


✔ Comportamento esperado (resíduos independentes)

Quando os resíduos são independentes, o gráfico mostra:

* pontos espalhados aleatoriamente

* sem padrão visível

* distribuídos em torno da linha zero

Exemplo conceitual:

Resíduos
   |
 2 |      .     .      .
 1 |   .      .      .
 0 |----------------------------
-1 |      .    .     .
-2 |   .      .    .
   |
   +----------------------------
          Valores preditos

Interpretação:

* os erros são aleatórios

* o modelo capturou os padrões principais

* não existe dependência entre observações

_______________________________

❌ Quando há autocorrelação

Se os resíduos apresentarem padrões, pode existir autocorrelação.

Exemplo 1 — padrão crescente ou decrescente
Resíduos
   |
 2 |        .
 1 |      .
 0 |    .
-1 |  .
-2 | .
   +----------------------

Interpretação:

* o modelo está subestimando ou superestimando sistematicamente

* indica dependência entre erros

Exemplo 2 — padrão ondulado
Resíduos
   |
 2 |    .     .
 1 |  .   . .
 0 | .      .
-1 |  .   . .
-2 |    .     .
   +----------------------

Interpretação:

* pode existir padrão temporal ou cíclico

* o modelo não capturou algum efeito

Exemplo 3 — blocos de resíduos
Resíduos
   |
 2 | . . . . .
 1 |
 0 |-----------
-1 |
-2 |      . . . . .
   +----------------

Interpretação:

* erros semelhantes em sequência

* indica dependência entre observações
'''

plt.figure(figsize=(10,5))
sns.lineplot(x=y_pred,y=residuos,marker='o',color='blue')
plt.xlabel('Valores preditos')
plt.ylabel('Resíduos')
plt.xlim(-49842496,320541976)
plt.ylim(-265697303,1097930555)
sns.lineplot(x=[-265697303,1097930555],y=[0,0],color='blue')
plt.title('Gráfico de resíduos x valores preditos')
plt.show()

# Nesse caso nao ha autocorrelacao

'''
CONSIDERAÇÕES IMPORTANTES SOBRE O GRAFICO PACF

O gráfico PACF (Partial Autocorrelation Function) é uma ferramenta usada
para verificar autocorrelação entre observações em diferentes 
defasagens (lags). Ele é usado para analisar se os resíduos do modelo 
de regressão são independentes, que é um dos pressupostos da regressão 
linear.

Lag: representa uma distância entre observações

*** Exemplo simplificado

Autocorrelação
 1.0 |        |
 0.8 |        |
 0.6 |        |
 0.4 |   |    |
 0.2 |   |    |
 0.0 |---|----|-------------------
-0.2 |
-0.4 |
      1  2  3  4  5
         Lag

Cada barra mostra:

quanto um valor está correlacionado com valores passados.



****** Intervalo de Confiança (Região Azul)

No gráfico existe uma faixa azul que representa o intervalo de confiança (geralmente 95%).

Interpretação:

Situação	                Interpretação
barra dentro da faixa	    correlação não significativa
barra fora da faixa	        autocorrelação significativa


Exemplo sem autocorrelação

Autocorrelação
  |
  |   |  |  |
  |   |  |  |
  |---|--|--|-------------------
  |
      1  2  3  4

Todas as barras estão dentro da região azul.

Conclusão:

✔ resíduos independentes (NAO existe autocorrelação).


Exemplo com autocorrelação

Autocorrelação
  |
  |   | 
  |   |        |
  |---|--------|--------------
  |
      1  2  3  4

A barra do lag 3 ultrapassa o intervalo.

Conclusão:

❌ existe autocorrelação.

'''

sm.graphics.tsa.plot_pacf(residuos, lags=40)
plt.show()

'''
Os resultados mostram sinais de autocorrelação, visto que há picos fora da região azul 
do intervalo de confiança. 
'''

# Teste estatístico
'''
Teste estatístico para garantir se há ausência de autocorrelação - Teste de Ljungbox

Hipótese Nula: Autocorrelação está ausente
Hipótese Alternativa: Autocorrelação está presente
'''

lb_test_results = diag.acorr_ljungbox(residuos, lags=40, return_df=True)

print(min(lb_test_results.lb_pvalue))

'''
Como o valor de p é maior que 0,05, não rejeitamos a hipótese nula de que os termos de erro não são autocorrelacionados.

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


# ****** SUPOSIÇÃO #6: AUSÊNCIA DE COLINEARIDADE ******

'''
Esse pressuposto se aplica particularmente para modelos de regressão linear múltipla. 
Em tais modelos, espera-se que os preditores não estejam fortemente correlacionados entre si. Quando nossos dados não respeitam essa condição, temos a chamada colinearidade.

Embora tecnicamente não seja uma suposição formal do modelo, a colinearidade pode tornar difícil a interpretação dos coeficientes, além de poder causar instabilidade na estimação 
dos parâmetros. Assim, é altamente recomendável avaliar a correlação entre os preditores antes de prosseguir com a análise.
'''

sns.heatmap(df_planos.corr(), annot=True,cmap='RdYlGn',square=True)
plt.show()

'''
Esses dados não contêm multicolinearidade perfeita entre as variáveis independentes. 
Caso houvesse, tentaríamos remover uma das variáveis correlacionadas, dependendo de qual fosse mais importante para o nosso modelo de regressão.
'''

'''
CONCLUSÃO: Portanto, alguns pressupostos da Regressão Linear não foram cumpridos com sucesso. Nesse caso, podemos afirmar que, as variáveis independentes não conseguem explicar a dependente.


O modelo não satisfaz completamente as suposições da regressão linear clássica.

Os principais problemas são:

1️. heterocedasticidade
2️. não normalidade dos resíduos

Esses problemas são muito comuns em dados de custos médicos, porque:

- distribuição de gastos é altamente assimétrica

- existem valores extremos (tratamentos caros).


Sugestão:

Algumas soluções comuns:

- Transformação logarítmica

'''
