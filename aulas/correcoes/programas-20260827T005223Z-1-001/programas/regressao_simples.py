import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregamento dos dados
df_agro = pd.read_csv('agro_tech.csv')

# 2. Calcular a matriz de correlação de Pearson
matriz_correlacao = df_agro.corr(numeric_only=True)
print("\n******* MATRIZ DE CORRELAÇÃO *******")
print(matriz_correlacao.round(2))

# 3. Gerar o mapa de calor (Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(
    matriz_correlacao,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    linecolor='white'
)
plt.title("Matriz de Correlação de Pearson - Dataset Agrotech", fontsize=14, pad=15)
plt.tight_layout()
plt.show()

# 4. Definição das variáveis (X: preditora com maior correlação, y: alvo)
X = df_agro[['precipitacao_anual']]
y = df_agro['toneladas_por_hectare']

# 5. Divisão Treino (70%) e Teste (30%)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

# 6. Criação e Treinamento do Modelo
modelo_simples = LinearRegression()
modelo_simples.fit(X_treino, y_treino)

intercepto = modelo_simples.intercept_
coeficiente = modelo_simples.coef_

print("\n******* PARÂMETROS DO MODELO SIMPLES *******")
print(f"Intercepto: {intercepto:.4f}")
print(f"Coeficiente: {coeficiente[0]:.4f}")
print(f"Equação: Produtividade = {intercepto:.2f} + ({coeficiente[0]:.4f} * precipitacao_anual)")

# 7. Avaliação do Modelo (R², MAE, MSE)
predicoes = modelo_simples.predict(X_teste)

r2 = r2_score(y_teste, predicoes)
mae = mean_absolute_error(y_teste, predicoes)
mse = mean_squared_error(y_teste, predicoes)

print("\n******* MÉTRICAS DE AVALIAÇÃO (TESTE) *******")
print(f"R-quadrado (R²): {r2:.4f} ({r2 * 100:.2f}%)")
print(f"MAE (Erro Médio Absoluto): {mae:.4f}")
print(f"MSE (Erro Quadrático Médio): {mse:.4f}")

# 8. Predição para Novos Dados
# Exemplo 1: Região com 800 mm de precipitação anual
cenario_1 = pd.DataFrame({'precipitacao_anual': [800.0]})
pred_1 = modelo_simples.predict(cenario_1)
print(f"\nProdutividade prevista para 800 mm de chuva: {pred_1[0]:.2f} ton/ha")

# Exemplo 2: Região com 1800 mm de precipitação anual
cenario_2 = pd.DataFrame({'precipitacao_anual': [1800.0]})
pred_2 = modelo_simples.predict(cenario_2)
print(f"Produtividade prevista para 1800 mm de chuva: {pred_2[0]:.2f} ton/ha")

# 9. Visualização da Reta de Regressão
plt.scatter(X, y, color="blue", alpha=0.5, label="Dados Reais")
plt.plot(X_teste, predicoes, color="red", linewidth=2, label="Reta de Regressão")
plt.title("Precipitação Anual x Toneladas por Hectare")
plt.xlabel("Precipitação Anual (mm)")
plt.ylabel("Produtividade (ton/ha)")
plt.legend()
plt.show()