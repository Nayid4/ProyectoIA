
# src/data_processing.py

import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def clean_data(df):
    # Eliminar filas con NaN y duplicados
    df_cleaned = df.dropna().drop_duplicates()

    # Convertir 'type' a 0 para 'white' y 1 para 'red'
    df_cleaned['type'] = df_cleaned['type'].apply(lambda x: 0 if x == 'white' else 1)

    # Asegurar que todas las columnas necesarias estén presentes
    # Añadir columnas faltantes con valores 0 si es necesario
    expected_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                        'sulphates', 'alcohol', 'type']

    for col in expected_columns:
        if col not in df_cleaned.columns:
            df_cleaned[col] = 0  # Añadir columna con valores 0 si falta

    return df_cleaned
