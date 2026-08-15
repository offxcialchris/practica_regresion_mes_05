import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar el dataset
df = pd.read_csv('../data/tips.csv')
print("La primeras filas del dataset:")
print(df.head())
print()
print("Info del dataset:")
print(df.shape)
print()
print("Valores nulos:")
print(df.isnull().sum())
print()

# Datos del modelo
X = df[['total_bill']]
y = df['tip']

# Creamos al modelo
modelo_simple = LinearRegression()
modelo_simple.fit(X, y)

print('Pendiente:', round(modelo_simple.coef_[0], 4))
print('Intercepto:', round(modelo_simple.intercept_, 4))
print()

# Codificar las 4 columnas de texto a la vez
df_enc = pd.get_dummies(df, columns=['sex', 'smoker', 'day', 'time'], drop_first=False)
print('Columnas antes de encoding:', df.shape[1])
print('Columnas despues de encoding:', df_enc.shape[1])

#Separar X y y
X = df_enc.drop(columns=['tip'])
y = df_enc['tip']

# Dividir train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('\nX_train:', X_train.shape)
print('X_test:', X_test.shape)

# Entrenar
modelo = LinearRegression()
modelo.fit(X_train, y_train)

for nombre, coef in zip(X.columns, modelo.coef_):
    print(f'{nombre}: {round(coef, 4)}')

#EVALÚA SOBRE EL CONJUNTO DE PRUEBA
y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print()
print("MAE (Error Absoluto Medio): ", round(mae,4))
print("MSE (Error Cuadratico Medio): ", round(mse,4))
print("RMSE (Raiz del Error Cuadratico Medio): ", round(rmse,4))
print("R2: (Coeficiente de Determinacion): ", round(r2,4))
print()

#Comparacion de modelo: simple vs. múltiple
modelo_simple = LinearRegression()
modelo_simple.fit(X_train[['total_bill']], y_train)
pred_simple = modelo_simple.predict(X_test[['total_bill']])
print("Simple R²:", round(r2_score(y_test, pred_simple), 4))
print("Múltiple R²:", round(r2, 4))






















































