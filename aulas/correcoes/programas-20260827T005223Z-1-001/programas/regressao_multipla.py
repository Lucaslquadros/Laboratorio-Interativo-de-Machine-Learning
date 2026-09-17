import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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

# 4. Definição das variáveis independentes (X) e dependente (y)
features = ['precipitacao_anual', 'fertilizante_kg_ha']
X = df_agro[features]
y = df_agro['toneladas_por_hectare']

# 5. Divisão Treino (70%) e Teste (30%)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

# 6. Treinamento do Modelo
modelo_multiplo = LinearRegression()
modelo_multiplo.fit(X_treino, y_treino)

intercepto = modelo_multiplo.intercept_
coeficientes = modelo_multiplo.coef_

print("\n******* PARÂMETROS DO MODELO MÚLTIPLO *******")
print(f"Intercepto: {intercepto:.4f}")
print(f"Coeficientes: {coeficientes}")
print(f"Equação: Produtividade = {intercepto:.2f} + ({coeficientes[0]:.4f} * precipitacao_anual) + ({coeficientes[1]:.4f} * fertilizante_kg_ha)")

# 7. Avaliação do Modelo (R², MAE, MSE)
predicoes = modelo_multiplo.predict(X_teste)

r2 = r2_score(y_teste, predicoes)
mae = mean_absolute_error(y_teste, predicoes)
mse = mean_squared_error(y_teste, predicoes)

print("\n******* MÉTRICAS DE AVALIAÇÃO (TESTE) *******")
print(f"R-quadrado (R²): {r2:.4f} ({r2 * 100:.2f}%)")
print(f"MAE (Erro Médio Absoluto): {mae:.4f}")
print(f"MSE (Erro Quadrático Médio): {mse:.4f}")

# 8. Predições para Novos Dados
cenario_novo = pd.DataFrame({
    'precipitacao_anual': [1200.0, 1600.0],
    'fertilizante_kg_ha': [250.0, 350.0]
})
predicoes_novas = modelo_multiplo.predict(cenario_novo)

for i, pred in enumerate(predicoes_novas, 1):
    print(f"\nCenário {i} - Produtividade prevista: {pred:.2f} ton/ha")

'''
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv('agro_tech (1).csv')

# Simple
X_s = df[['precipitacao_anual']]
y = df['toneladas_por_hectare']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y, test_size=0.3, random_state=0)
model_s = LinearRegression()
model_s.fit(X_train_s, y_train_s)
pred_s = model_s.predict(X_test_s)

mae_s = mean_absolute_error(y_test_s, pred_s)
mse_s = mean_squared_error(y_test_s, pred_s)
r2_s = r2_score(y_test_s, pred_s)

# Multiple (2 vars)
X_m = df[['precipitacao_anual', 'fertilizante_kg_ha']]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m, y, test_size=0.3, random_state=0)
model_m = LinearRegression()
model_m.fit(X_train_m, y_train_m)
pred_m = model_m.predict(X_test_m)

mae_m = mean_absolute_error(y_test_m, pred_m)
mse_m = mean_squared_error(y_test_m, pred_m)
r2_m = r2_score(y_test_m, pred_m)

print(f"Simples: R2={r2_s:.4f}, MAE={mae_s:.4f}, MSE={mse_s:.4f}")
print(f"Multipla: R2={r2_m:.4f}, MAE={mae_m:.4f}, MSE={mse_m:.4f}")


```

```text
Simples: R2=0.4847, MAE=2.3037, MSE=7.3525
Multipla: R2=0.8705, MAE=1.1035, MSE=1.8481


```

**Código Atualizado: Regressão Linear Simples**

Adicionadas as métricas **MAE** e **MSE**, além do passo de predição para novos dados no mesmo padrão dos arquivos de referência.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregamento dos dados
df_agro = pd.read_csv('agrotech.csv')

# 2. Definição das variáveis (X: preditora com maior correlação, y: alvo)
X = df_agro[['precipitacao_anual']]
y = df_agro['toneladas_por_hectare']

# 3. Divisão Treino (70%) e Teste (30%)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

# 4. Criação e Treinamento do Modelo
modelo_simples = LinearRegression()
modelo_simples.fit(X_treino, y_treino)

intercepto = modelo_simples.intercept_
coeficiente = modelo_simples.coef_

print("\n******* PARÂMETROS DO MODELO SIMPLES *******")
print(f"Intercepto: {intercepto:.4f}")
print(f"Coeficiente: {coeficiente[0]:.4f}")
print(f"Equação: Produtividade = {intercepto:.2f} + ({coeficiente[0]:.4f} * precipitacao_anual)")

# 5. Avaliação do Modelo (R², MAE, MSE)
predicoes = modelo_simples.predict(X_teste)

r2 = r2_score(y_teste, predicoes)
mae = mean_absolute_error(y_teste, predicoes)
mse = mean_squared_error(y_teste, predicoes)

print("\n******* MÉTRICAS DE AVALIAÇÃO (TESTE) *******")
print(f"R-quadrado (R²): {r2:.4f} ({r2 * 100:.2f}%)")
print(f"MAE (Erro Médio Absoluto): {mae:.4f}")
print(f"MSE (Erro Quadrático Médio): {mse:.4f}")

# 6. Predição para Novos Dados
# Exemplo 1: Região com 800 mm de precipitação anual
cenario_1 = pd.DataFrame({'precipitacao_anual': [800.0]})
pred_1 = modelo_simples.predict(cenario_1)
print(f"\nProdutividade prevista para 800 mm de chuva: {pred_1[0]:.2f} ton/ha")

# Exemplo 2: Região com 1800 mm de precipitação anual
cenario_2 = pd.DataFrame({'precipitacao_anual': [1800.0]})
pred_2 = modelo_simples.predict(cenario_2)
print(f"Produtividade prevista para 1800 mm de chuva: {pred_2[0]:.2f} ton/ha")

```

---

**Código Atualizado: Regressão Linear Múltipla**

Incluídas as métricas **MAE** e **MSE** para avaliação do modelo multivariado.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Carregamento dos dados
df_agro = pd.read_csv('agrotech.csv')

# 2. Definição das variáveis independentes (X) e dependente (y)
features = ['precipitacao_anual', 'fertilizante_kg_ha']
X = df_agro[features]
y = df_agro['toneladas_por_hectare']

# 3. Divisão Treino (70%) e Teste (30%)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

# 4. Treinamento do Modelo
modelo_multiplo = LinearRegression()
modelo_multiplo.fit(X_treino, y_treino)

intercepto = modelo_multiplo.intercept_
coeficientes = modelo_multiplo.coef_

print("\n******* PARÂMETROS DO MODELO MÚLTIPLO *******")
print(f"Intercepto: {intercepto:.4f}")
print(f"Coeficientes: {coeficientes}")
print(f"Equação: Produtividade = {intercepto:.2f} + ({coeficientes[0]:.4f} * precipitacao_anual) + ({coeficientes[1]:.4f} * fertilizante_kg_ha)")

# 5. Avaliação do Modelo (R², MAE, MSE)
predicoes = modelo_multiplo.predict(X_teste)

r2 = r2_score(y_teste, predicoes)
mae = mean_absolute_error(y_teste, predicoes)
mse = mean_squared_error(y_teste, predicoes)

print("\n******* MÉTRICAS DE AVALIAÇÃO (TESTE) *******")
print(f"R-quadrado (R²): {r2:.4f} ({r2 * 100:.2f}%)")
print(f"MAE (Erro Médio Absoluto): {mae:.4f}")
print(f"MSE (Erro Quadrático Médio): {mse:.4f}")

# 6. Predições para Novos Dados
cenario_novo = pd.DataFrame({
    'precipitacao_anual': [1200.0, 1600.0],
    'fertilizante_kg_ha': [250.0, 350.0]
})
predicoes_novas = modelo_multiplo.predict(cenario_novo)

for i, pred in enumerate(predicoes_novas, 1):
    print(f"\nCenário {i} - Produtividade prevista: {pred:.2f} ton/ha")

```

---

**Comparativo Completo das Métricas**

Regressão Simples: r2= 0.4847  MAE: 2.3037 ton/ha  MSE: 7.3525
Regressão Multipla: r2= 0.8705  MAE: 1.1035 ton/ha  MSE: 1.8481

* O **MAE** caiu mais da metade no modelo múltiplo (de $2.30$ para $1.10\text{ ton/ha}$), indicando que, em média, o erro da predição por hectare diminuiu expressivamente ao adicionar o fertilizante.
* O **MSE** teve uma redução drástica (de $7.35$ para $1.85$), mostrando que o modelo de regressão múltipla reduz erros grandes e melhora a estabilidade das estimativas.
'''