"""
API de predicción de churn con FastAPI - VERSIÓN MEJORADA.

La API carga un modelo serializado, valida los datos de entrada
y devuelve una predicción junto con su probabilidad.

MEJORAS IMPLEMENTADAS (Alternativa 4):
- Nivel de riesgo (ALTO/MEDIO/BAJO)
- Recomendación personalizada para el cliente
- Timestamp de la predicción
"""

from pathlib import Path
from datetime import datetime

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# ============================================
# PERSONALIZACIÓN OBLIGATORIA
# ============================================

# Múltiples opciones para encontrar el modelo (más robusto)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Posibles rutas donde podría estar el modelo
POSIBLES_RUTAS_MODELO = [
    PROJECT_ROOT / "models" / "modelo_churn_v1.joblib",           # Ruta normal
    PROJECT_ROOT / "modelo_churn_v1.joblib",                      # En la raíz
    Path.cwd() / "models" / "modelo_churn_v1.joblib",             # Desde directorio actual
    Path.cwd() / "modelo_churn_v1.joblib",                        # En raíz del proyecto actual
]

# Buscar el modelo en las rutas posibles
MODEL_PATH = None
for ruta in POSIBLES_RUTAS_MODELO:
    if ruta.exists():
        MODEL_PATH = ruta
        print(f"✅ Modelo encontrado en: {MODEL_PATH}")
        break

if MODEL_PATH is None:
    print("❌ ERROR: No se encontró el modelo en ninguna de estas rutas:")
    for ruta in POSIBLES_RUTAS_MODELO:
        print(f"   - {ruta}")
    raise RuntimeError(
        "No se encontró el modelo serializado. "
        "Ejecute primero: python src\\entrenar_modelo.py"
    )

VERSION_MODELO = "modelo_churn_v1"
AUTOR = "Monica Rodriguez Flores"

# Cargar el modelo
try:
    modelo = joblib.load(MODEL_PATH)
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    raise RuntimeError(f"No se pudo cargar el modelo: {e}")


# ============================================
# MODELO DE ENTRADA CON VALIDACIONES
# ============================================
class ClienteEntrada(BaseModel):
    antiguedad: int = Field(
        ...,
        ge=0,
        le=120,
        description="Antigüedad del cliente expresada en meses",
        examples=[12],
    )
    cargo_mensual: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Cargo mensual del cliente",
        examples=[95.5],
    )
    reclamos: int = Field(
        ...,
        ge=0,
        le=50,
        description="Cantidad de reclamos recientes",
        examples=[3],
    )

    # Validación adicional: clientes con alta antigüedad no deberían tener muchos reclamos
    @validator('reclamos')
    def validar_reclamos_antiguedad(cls, v, values):
        if 'antiguedad' in values and values['antiguedad'] > 60 and v > 10:
            raise ValueError(
                f"Clientes con {values['antiguedad']} meses de antigüedad "
                f"y {v} reclamos es un patrón inusual"
            )
        return v

    # Validación adicional: cargo bajo con muchos reclamos es inconsistente
    @validator('cargo_mensual')
    def validar_cargo_reclamos(cls, v, values):
        if 'reclamos' in values and values['reclamos'] > 5 and v < 30:
            raise ValueError(
                f"Cargo mensual bajo ({v}) con {values['reclamos']} reclamos "
                f"es un patrón inconsistente"
            )
        return v


# ============================================
# MODELO DE SALIDA MEJORADO (Alternativa 4)
# ============================================
class PrediccionSalida(BaseModel):
    prediccion: str
    probabilidad: float
    version_modelo: str
    autor: str
    # MEJORA: Nuevos campos
    nivel_riesgo: str
    recomendacion: str
    timestamp: str


# ============================================
# FUNCIÓN AUXILIAR PARA CÁLCULO DE RIESGO
# ============================================
def calcular_nivel_riesgo(antiguedad: int, cargo_mensual: float, reclamos: int, probabilidad: float) -> tuple:
    """
    Calcula el nivel de riesgo y genera una recomendación personalizada
    basada en múltiples factores del cliente.
    
    Retorna:
        nivel_riesgo: str - "ALTO", "MEDIO" o "BAJO"
        recomendacion: str - Mensaje personalizado
    """
    score_riesgo = 0

    # Factor 1: Reclamos (peso alto)
    if reclamos >= 5:
        score_riesgo += 40
    elif reclamos >= 3:
        score_riesgo += 25
    elif reclamos >= 1:
        score_riesgo += 10

    # Factor 2: Antigüedad baja (clientes nuevos)
    if antiguedad < 6:
        score_riesgo += 30
    elif antiguedad < 12:
        score_riesgo += 15
    elif antiguedad > 48:
        score_riesgo -= 10  # Clientes leales tienen menor riesgo

    # Factor 3: Cargo mensual alto (puede indicar insatisfacción si es caro)
    if cargo_mensual > 120:
        score_riesgo += 20
    elif cargo_mensual > 80:
        score_riesgo += 10

    # Factor 4: Probabilidad del modelo
    if probabilidad >= 0.7:
        score_riesgo += 25
    elif probabilidad >= 0.5:
        score_riesgo += 15

    # Determinar nivel y recomendación
    if score_riesgo >= 60:
        nivel = "ALTO"
        recomendacion = (
            "⚠️ ALERTA: Cliente con alta probabilidad de abandono. "
            "Acciones recomendadas: contacto inmediato, ofrecer beneficios de retención, "
            "analizar causas de reclamos, considerar descuento por fidelización."
        )
    elif score_riesgo >= 30:
        nivel = "MEDIO"
        recomendacion = (
            "📊 PRECAUCIÓN: Cliente con riesgo moderado de abandono. "
            "Acciones recomendadas: monitorear comportamiento, enviar encuesta de satisfacción, "
            "ofrecer programa de puntos o beneficios adicionales."
        )
    else:
        nivel = "BAJO"
        recomendacion = (
            "✅ TRANQUILO: Cliente con bajo riesgo de abandono. "
            "Acciones recomendadas: mantener estrategia actual, enviar contenido de valor, "
            "programa de referidos."
        )

    return nivel, recomendacion


# ============================================
# INICIALIZACIÓN DE FASTAPI
# ============================================
app = FastAPI(
    title="API de predicción de churn - MEJORADA",
    description=(
        "Servicio académico ML-Ops para estimar riesgo de abandono.\n"
        f"Autor: {AUTOR}\n\n"
        "## MEJORAS IMPLEMENTADAS\n"
        "- Nivel de riesgo (ALTO/MEDIO/BAJO)\n"
        "- Recomendación personalizada\n"
        "- Timestamp de la predicción\n"
        "- Validaciones cruzadas adicionales"
    ),
    version="2.0.0",
)


# ============================================
# ENDPOINTS
# ============================================
@app.get("/")
def inicio() -> dict:
    return {
        "mensaje": "Servicio ML-Ops activo - API MEJORADA",
        "estado": "ok",
        "autor": AUTOR,
        "version": "2.0.0",
        "mejoras": ["nivel_riesgo", "recomendacion", "timestamp"],
    }


@app.get("/health")
def health() -> dict:
    return {
        "estado": "ok",
        "modelo": VERSION_MODELO,
        "autor": AUTOR,
    }


@app.post("/predict", response_model=PrediccionSalida)
def predict(datos: ClienteEntrada) -> PrediccionSalida:
    try:
        # Preparar datos para el modelo
        X = [[
            datos.antiguedad,
            datos.cargo_mensual,
            datos.reclamos,
        ]]

        # Obtener probabilidad del modelo
        probabilidad = float(modelo.predict_proba(X)[0][1])
        etiqueta = "alto_riesgo" if probabilidad >= 0.50 else "bajo_riesgo"

        # MEJORA: Calcular nivel de riesgo y recomendación
        nivel_riesgo, recomendacion = calcular_nivel_riesgo(
            datos.antiguedad,
            datos.cargo_mensual,
            datos.reclamos,
            probabilidad
        )

        # MEJORA: Timestamp actual
        timestamp_actual = datetime.now().isoformat()

        return PrediccionSalida(
            prediccion=etiqueta,
            probabilidad=round(probabilidad, 4),
            version_modelo=VERSION_MODELO,
            autor=AUTOR,
            nivel_riesgo=nivel_riesgo,
            recomendacion=recomendacion,
            timestamp=timestamp_actual,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"No fue posible generar la predicción: {str(exc)}",
        ) from exc