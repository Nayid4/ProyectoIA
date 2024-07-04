
# src/model.py

from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def train_model(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = BayesianRidge()

    # Entrenar el modelo y obtener predicciones
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Obtener el error de iteración durante el entrenamiento
    iter_errors = []
    for i in range(1, 100):
        model.fit(X_train[:i], y_train[:i])  # Entrenar con un subconjunto de datos
        y_pred_iter = model.predict(X_train[:i])
        mse_iter = mean_squared_error(y_train[:i], y_pred_iter)
        iter_errors.append(mse_iter)

    return model, mse, r2, y_test, y_pred, iter_errors


def predict(model, data):
    # Definir explícitamente los nombres de características que se utilizaron en el entrenamiento
    feature_names = ['type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                     'sulphates', 'alcohol']

    # Crear un DataFrame con una sola fila de datos
    data_df = pd.DataFrame([data], columns=feature_names)

    # Realizar la predicción
    prediction = model.predict(data_df)[0]

    # Ajustar la predicción al rango [0, 10]
    prediction = max(0, min(10, prediction))

    return prediction


