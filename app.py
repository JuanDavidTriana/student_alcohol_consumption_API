from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

#Cargar el modelo
name_model = "modelo_entrenado.h5"
modelo = tf.keras.models.load_model("modelo_entrenado.h5", compile=False)

#FastAPI
app = FastAPI(
    title="Studen Performace Prediction API",
    description="API para predecir el rendimiento de estudiantes",
    version="1.0"
)

#Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Permite cualquier origen
    allow_credentials=True, # Permite las credenciales
    allow_methods=["*"],    # Permite cualquier método
    allow_headers=["*"]     # Permite cualquier cabecera
)

#Definir el esquema de entrada
class StudentData(BaseModel):
    features: list  # La lista de características del estudiante

#Ruta de estado
@app.get("/")
async def status():
    return {
            "status": True,
            "Modelo": name_model,
            }

#Ruta de prediccion
@app.post("/predict")
def predict(data: StudentData):
    try:
        # Convertir la lista de características en un array NumPy
        features = np.array(data.features, dtype=np.float32).reshape(1, -1)
        # Realizar la predicción
        prediction = modelo.predict(features)
        grade = float(prediction[0][0])
        # Devolver la predicción
        return {"predicted_grade": round(grade, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))