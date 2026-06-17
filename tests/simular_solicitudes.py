"""
Simulación básica de solicitudes para observar el monitoreo de la API.

Este script permite generar tráfico controlado hacia la API predictiva
de churn desarrollada durante el laboratorio.

Objetivos:
1. Enviar solicitudes válidas al endpoint POST /predict.
2. Enviar una solicitud atípica para generar alertas de datos.
3. Enviar una solicitud inválida para comprobar el error HTTP 422.
4. Mostrar la latencia informada por el middleware de la API.
5. Consultar el resumen acumulado mediante el endpoint GET /metrics.
6. Verificar las mejoras de main6: nivel_riesgo, recomendacion, timestamp.

Importante:
- La API debe estar activa antes de ejecutar este archivo.
- Este script no entrena el modelo.
- Este script no modifica la API.
- Solamente simula solicitudes para observar su comportamiento.
"""

# ============================================================
# BLOQUE 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

from pprint import pprint
# pprint permite mostrar diccionarios JSON de forma ordenada y legible.

import requests
# requests permite enviar solicitudes HTTP desde Python.
# En este laboratorio se utiliza para comunicarse con la API local.

# ============================================================
# BLOQUE 2. CONFIGURACIÓN GENERAL
# ============================================================

# Dirección base donde se encuentra ejecutándose la API.
# El puerto predeterminado utilizado por Uvicorn es 8000.
BASE_URL = "http://127.0.0.1:8000"

# Tiempo máximo de espera para cada solicitud, expresado en segundos.
# Evita que el programa quede esperando indefinidamente si la API
# no responde o si existe algún problema de conexión.
TIMEOUT = 10

# ============================================================
# BLOQUE 3. CASOS DE PRUEBA
# ============================================================
# AQUÍ SE DEFINEN LAS SOLICITUDES QUE SE ENVIARÁN A POST /predict.
#
# Cada caso contiene:
# - nombre: etiqueta descriptiva para identificar la prueba;
# - datos: valores que recibirá la API;
# - descripcion: explicación del caso (opcional).
#
# Se incluyen:
# - tres casos válidos dentro de rangos razonables;
# - un caso atípico aceptado técnicamente, pero fuera del histórico;
# - un caso inválido que debe ser rechazado con código HTTP 422.
# - un caso con validación cruzada (antigüedad alta + muchos reclamos)

CASOS = [
    {
        "nombre": "cliente_estable",
        "descripcion": "Cliente con baja probabilidad de churn",
        "datos": {
            "antiguedad": 48,
            "cargo_mensual": 55.0,
            "reclamos": 0,
        },
    },
    {
        "nombre": "cliente_riesgo_medio",
        "descripcion": "Cliente con riesgo moderado",
        "datos": {
            "antiguedad": 18,
            "cargo_mensual": 110.0,
            "reclamos": 3,
        },
    },
    {
        "nombre": "cliente_alto_riesgo",
        "descripcion": "Cliente con alto riesgo de abandono",
        "datos": {
            "antiguedad": 4,
            "cargo_mensual": 145.0,
            "reclamos": 7,
        },
    },
    {
        "nombre": "cliente_atipico",
        "descripcion": "Valores técnicamente válidos pero fuera de rango histórico",
        "datos": {
            "antiguedad": 180,
            "cargo_mensual": 600.0,
            "reclamos": 35,
        },
        # Este caso es técnicamente válido porque los valores respetan
        # los límites generales definidos por Pydantic.
        #
        # Sin embargo, los valores se encuentran fuera de los rangos
        # históricos del entrenamiento. Por ello, la API debe generar
        # alertas_datos.
    },
    {
        "nombre": "cliente_invalido",
        "descripcion": "Cargo mensual negativo - debe ser rechazado",
        "datos": {
            "antiguedad": 12,
            "cargo_mensual": -50.0,
            "reclamos": 1,
        },
        # Este caso debe ser rechazado porque cargo_mensual es negativo.
        # La API devolverá un código HTTP 422 y aumentará el contador
        # errores_validacion.
    },
    {
        "nombre": "cliente_validacion_cruzada",
        "descripcion": "Cliente antiguo con muchos reclamos - validación cruzada",
        "datos": {
            "antiguedad": 72,
            "cargo_mensual": 85.0,
            "reclamos": 12,
        },
        # Este caso activa la validación cruzada de main6:
        # "Clientes con 72 meses de antigüedad y 12 reclamos es un patrón inusual"
        # Debe devolver error 422 con el mensaje de validación.
    },
]

# ============================================================
# BLOQUE 4. FUNCIÓN PARA MOSTRAR LA RESPUESTA DE CADA CASO
# ============================================================
# AQUÍ SE MUESTRA:
# - nombre del caso;
# - descripción del caso;
# - código HTTP recibido;
# - latencia medida por el middleware;
# - contenido JSON devuelto por la API;
# - campos mejorados (nivel_riesgo, recomendacion, timestamp).

def mostrar_respuesta(
    nombre: str,
    descripcion: str,
    respuesta: requests.Response,
) -> None:
    """
    Presenta de forma ordenada el resultado de una solicitud.

    Parámetros:
        nombre:
            Nombre descriptivo del caso evaluado.
        descripcion:
            Descripción del caso evaluado.
        respuesta:
            Objeto Response devuelto por la librería requests.
            Contiene el código HTTP, las cabeceras y el cuerpo JSON.
    """

    print("\n" + "=" * 70)
    print(f"Caso: {nombre}")
    print(f"Descripción: {descripcion}")

    # Mostrar el código HTTP devuelto por la API.
    #
    # Ejemplos:
    # - 200: solicitud procesada correctamente;
    # - 422: error de validación de datos;
    # - 500: error interno de la API.
    print(f"Estado HTTP: {respuesta.status_code}")

    # Recuperar la cabecera agregada por el middleware de la API.
    #
    # En api/main.py se incorporó:
    # response.headers["X-Process-Time-ms"] = ...
    #
    # Esta cabecera informa cuántos milisegundos tardó la solicitud.
    latencia = respuesta.headers.get("X-Process-Time-ms")

    if latencia is not None:
        print(f"Latencia informada por API: {latencia} ms")
    else:
        print("Latencia informada por API: no disponible")

    # Intentar convertir el cuerpo de la respuesta a JSON.
    #
    # pprint permite mostrar la respuesta de forma ordenada.
    # Si la respuesta no contiene JSON válido, se muestra el texto original.
    try:
        data = respuesta.json()
        print("\nDatos de respuesta:")
        pprint(data)
        
        # Verificar campos mejorados (desde main6)
        if respuesta.status_code == 200:
            print("\n📊 MEJORAS DE MAIN6:")
            if "nivel_riesgo" in data:
                print(f"  • Nivel de riesgo: {data['nivel_riesgo']}")
            if "recomendacion" in data:
                print(f"  • Recomendación: {data['recomendacion'][:80]}...")
            if "timestamp" in data:
                print(f"  • Timestamp: {data['timestamp']}")
            if "alertas_datos" in data and data['alertas_datos']:
                print(f"  • Alertas de datos: {len(data['alertas_datos'])}")
                for alerta in data['alertas_datos']:
                    print(f"    - {alerta}")

    except requests.exceptions.JSONDecodeError:
        print("La respuesta no contiene un JSON válido.")
        print(respuesta.text)

# ============================================================
# BLOQUE 5. FUNCIÓN PARA ENVIAR UNA SOLICITUD A POST /predict
# ============================================================
# AQUÍ SE IMPLEMENTA:
# - envío de cada caso de prueba;
# - comunicación con POST /predict;
# - manejo de problemas de conexión.

def enviar_caso(caso: dict) -> None:
    """
    Envía un caso de prueba al endpoint POST /predict.

    Parámetro:
        caso:
            Diccionario con un nombre descriptivo y los datos del cliente.
    """

    nombre = caso["nombre"]
    descripcion = caso.get("descripcion", "Sin descripción")
    datos = caso["datos"]

    try:
        # Enviar una solicitud HTTP POST.
        #
        # URL utilizada:
        # http://127.0.0.1:8000/predict
        #
        # El argumento json=datos convierte automáticamente el diccionario
        # de Python en el formato JSON esperado por la API.
        respuesta = requests.post(
            f"{BASE_URL}/predict",
            json=datos,
            timeout=TIMEOUT,
        )

        # Mostrar el resultado recibido.
        mostrar_respuesta(nombre, descripcion, respuesta)

    except requests.exceptions.ConnectionError:
        # Este error aparece normalmente cuando la API no está activa.
        print("\n" + "=" * 70)
        print(f"Caso: {nombre}")
        print("Error: no fue posible conectarse con la API.")
        print("Verifique que Uvicorn se encuentre activo en otra terminal.")

    except requests.exceptions.Timeout:
        # Este error aparece si la API demora más del tiempo configurado.
        print("\n" + "=" * 70)
        print(f"Caso: {nombre}")
        print(f"Error: la API no respondió en menos de {TIMEOUT} segundos.")

    except requests.exceptions.RequestException as exc:
        # Captura otros problemas relacionados con la solicitud HTTP.
        print("\n" + "=" * 70)
        print(f"Caso: {nombre}")
        print(f"Error inesperado durante la solicitud: {exc}")

# ============================================================
# BLOQUE 6. FUNCIÓN PARA CONSULTAR GET /metrics
# ============================================================
# AQUÍ SE IMPLEMENTA:
# - consulta del resumen acumulado de métricas;
# - comunicación con el endpoint GET /metrics.

def consultar_metricas() -> None:
    """
    Consulta y muestra las métricas acumuladas por la API.

    El endpoint GET /metrics resume:
    - cantidad total de solicitudes;
    - errores de validación;
    - errores internos;
    - predicciones válidas;
    - resultados de alto y bajo riesgo;
    - solicitudes con anomalías;
    - latencia promedio y máxima;
    - distribución de códigos HTTP.
    """

    print("\n" + "=" * 70)
    print("Resumen acumulado de métricas")

    try:
        # Enviar una solicitud HTTP GET al endpoint /metrics.
        respuesta_metricas = requests.get(
            f"{BASE_URL}/metrics",
            timeout=TIMEOUT,
        )

        print(f"Estado HTTP: {respuesta_metricas.status_code}")

        # Mostrar el resumen JSON de manera ordenada.
        data = respuesta_metricas.json()
        print("\n📈 MÉTRICAS ACUMULADAS:")
        
        # Mostrar métricas clave de forma más legible
        metricas_clave = [
            "solicitudes_totales",
            "errores_validacion",
            "errores_internos",
            "predicciones_validas",
            "predicciones_alto_riesgo",
            "predicciones_bajo_riesgo",
            "solicitudes_con_anomalias",
            "latencia_promedio_ms",
            "latencia_maxima_ms",
        ]
        
        for clave in metricas_clave:
            if clave in data:
                print(f"  • {clave}: {data[clave]}")
        
        if "codigos_http" in data:
            print(f"  • codigos_http: {data['codigos_http']}")
        
        print(f"\n  • version_modelo: {data.get('version_modelo', 'N/A')}")
        print(f"  • autor: {data.get('autor', 'N/A')}")

    except requests.exceptions.ConnectionError:
        print("Error: no fue posible consultar las métricas.")
        print("Verifique que la API se encuentre activa.")

    except requests.exceptions.Timeout:
        print(
            f"Error: la API no respondió en menos de {TIMEOUT} segundos."
        )

    except requests.exceptions.JSONDecodeError:
        print("Error: la respuesta de /metrics no contiene un JSON válido.")

    except requests.exceptions.RequestException as exc:
        print(f"Error inesperado durante la consulta: {exc}")

# ============================================================
# BLOQUE 7. FUNCIÓN PARA VERIFICAR EL ESTADO DE LA API
# ============================================================

def verificar_api() -> bool:
    """
    Verifica que la API esté activa antes de ejecutar las pruebas.

    Retorna:
        True si la API responde correctamente, False en caso contrario.
    """
    try:
        respuesta = requests.get(
            f"{BASE_URL}/health",
            timeout=5
        )
        
        if respuesta.status_code == 200:
            print("✅ API activa y funcionando correctamente")
            print(f"   Modelo: {respuesta.json().get('modelo', 'N/A')}")
            print(f"   Autor: {respuesta.json().get('autor', 'N/A')}")
            return True
        else:
            print(f"❌ API responde con código: {respuesta.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: No se pudo conectar con la API.")
        print("   Verifique que Uvicorn esté activo en otra terminal.")
        print("   Comando: python -m uvicorn api.main:app --reload")
        return False
    except requests.exceptions.Timeout:
        print("❌ ERROR: La API no respondió en el tiempo esperado.")
        return False
    except Exception as exc:
        print(f"❌ ERROR inesperado: {exc}")
        return False

# ============================================================
# BLOQUE 8. FUNCIÓN PRINCIPAL
# ============================================================
# AQUÍ SE DEFINE EL ORDEN COMPLETO DE EJECUCIÓN:
# 1. Verificar que la API esté activa.
# 2. Mostrar un encabezado.
# 3. Recorrer todos los casos de prueba.
# 4. Enviar cada solicitud a POST /predict.
# 5. Consultar GET /metrics al finalizar.

def main() -> None:
    """
    Ejecuta la simulación completa de solicitudes.
    """

    print("=" * 70)
    print("SIMULACIÓN DE SOLICITUDES PARA LA API PREDICTIVA")
    print("Versión con mejoras: nivel_riesgo, recomendacion, timestamp")
    print("=" * 70)

    # Paso 1: Verificar que la API esté activa
    if not verificar_api():
        print("\n❌ No se puede continuar con la simulación.")
        print("   Inicie la API en otra terminal y vuelva a ejecutar este script.")
        return

    # Paso 2: Ejecutar los casos de prueba
    print("\n" + "-" * 70)
    print("EJECUTANDO CASOS DE PRUEBA")
    print("-" * 70)

    # Recorrer secuencialmente todos los casos definidos anteriormente.
    for caso in CASOS:
        enviar_caso(caso)

    # Paso 3: Consultar el resumen acumulado después de procesar las solicitudes.
    consultar_metricas()

    # Paso 4: Mensaje final
    print("\n" + "=" * 70)
    print("✅ SIMULACIÓN COMPLETADA")
    print("=" * 70)
    print("\n📝 Resumen de lo observado:")
    print("  • Las predicciones incluyen nivel_riesgo, recomendacion y timestamp")
    print("  • Las alertas_datos indican valores fuera de rango histórico")
    print("  • Las métricas acumuladas están disponibles en /metrics")
    print("  • Los logs se registran en logs/monitor_api.log")

# ============================================================
# BLOQUE 9. PUNTO DE ENTRADA DEL PROGRAMA
# ============================================================
# Esta condición permite ejecutar main() únicamente cuando este archivo
# se inicia directamente desde PowerShell.
#
# Comando:
# python tests\simular_solicitudes.py
#
# Si el archivo fuera importado desde otro script, main() no se ejecutaría
# automáticamente.

if __name__ == "__main__":
    main()