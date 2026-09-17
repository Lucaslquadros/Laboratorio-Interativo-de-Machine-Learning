import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv('finance_market.csv')
print("Columns:", df.columns.tolist())
print("\nHead:\n", df.head())
print("\nInfo:\n", df.info())

# 1. Correlação
corr_matrix = df.corr(numeric_only=True)
print("Correlação:\n", corr_matrix)

# 2. Definicao de X e y
X = df[['Indice_S&P500']]
y = df['ETF_Preco']

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Treinamento do Modelo
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nIntercepto (b0): {model.intercept_:.4f}")
print(f"Coeficiente angular (b1): {model.coef_[0]:.4f}")

# 5. Predição no conjunto de teste
y_pred = model.predict(X_test)

# 6. Validação do Modelo
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nR²: {r2:.6f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# 7. Predições para novos dados
new_data = pd.DataFrame({'Indice_S&P500': [4100.0, 4250.0, 4500.0]})
new_preds = model.predict(new_data)
new_data['ETF_Preco_Predito'] = new_preds
print("\nNovas predições:\n", new_data)