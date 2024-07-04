from src.data_processing import load_data, clean_data
from src.model import train_model, predict

# Cargar datos desde el archivo CSV
df = load_data('Data/winequalityN.csv')

# Limpiar y preparar los datos
df_cleaned = clean_data(df)

# Entrenar el modelo con los datos limpios
model, _, _, _, _ = train_model(df_cleaned)

# Ejemplo de datos para predecir, incluyendo el tipo de vino (0 para white, 1 para red)
data_to_predict = [0, 7, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3, 0.45, 8.8]  # Agregar el tipo de vino al final

# Realizar la predicción
predicted_quality = predict(model, data_to_predict)
print(f"Calidad predicha del vino: {predicted_quality}")
