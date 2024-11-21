import streamlit as st
import pandas as pd
import joblib

# Cargar los modelos previamente guardados
rf_model = joblib.load("randomforest_optimized_model.pkl")
gb_model = joblib.load("gradientboosting_optimized_model.pkl")
ada_model = joblib.load("adaboost_optimized_model.pkl")

# Opciones de modelo
modelos = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "AdaBoost": ada_model
}

# Título de la aplicación
st.title("Predicción con Modelos de Machine Learning")
st.write("Ingrese los valores manualmente para probar los modelos. Se sugieren rangos válidos para los datos.")

# Input de valores manuales con rangos sugeridos
open_val = st.slider("Open", min_value=0.0002, max_value=0.0005, value=0.0003, step=0.00001, format="%.5f")
high_val = st.slider("High", min_value=0.0002, max_value=0.0006, value=0.00035, step=0.00001, format="%.5f")
low_val = st.slider("Low", min_value=0.0001, max_value=0.0004, value=0.00025, step=0.00001, format="%.5f")
close_val = st.slider("Close", min_value=0.0002, max_value=0.0005, value=0.00029, step=0.00001, format="%.5f")
volume_val = st.slider("Volume", min_value=1e11, max_value=5e11, value=2e11, step=1e10, format="%.1e")
volatility_val = st.slider("Volatility", min_value=0.00001, max_value=0.0001, value=0.00004, step=0.00001, format="%.5f")
z_score_val = st.slider("z_score", min_value=-3.0, max_value=3.0, value=0.5, step=0.1, format="%.2f")

# Crear un DataFrame con los valores introducidos
valores_prueba = {
    "Open": open_val,
    "High": high_val,
    "Low": low_val,
    "Close": close_val,
    "Volume": volume_val,
    "Volatility": volatility_val,
    "z_score": z_score_val,
}
input_data = pd.DataFrame([valores_prueba])

# Seleccionar el modelo
modelo_seleccionado = st.selectbox("Seleccione el modelo", list(modelos.keys()))

# Botón para predecir
if st.button("Predecir"):
    # Obtener el modelo seleccionado
    modelo = modelos[modelo_seleccionado]
    # Realizar la predicción
    prediccion = modelo.predict(input_data)[0]
    # Mostrar el resultado
    st.success(f"Predicción con {modelo_seleccionado}: {prediccion:.4f}")
