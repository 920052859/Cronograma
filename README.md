# Predicción del Cronograma de Trabajo con IA – Estación E4-8

Aplicación académica en **Python + Streamlit + Machine Learning** para predecir el estado del cronograma del proyecto:

**Diseño, Procura y Construcción de la Infraestructura Civil de la Estación E4-8 de la Línea 4 del Metro de Lima y Callao.**

## Objetivo del ejercicio
Predecir si cada actividad planificada para enero y febrero de 2026 estará:

- A tiempo
- Con riesgo de retraso
- Retrasada

Además, identifica actividades críticas, desviaciones frente al plan original y prioridades de atención para el jefe de proyecto.

## Técnica utilizada
El modelo usa un `RandomForestClassifier` de scikit-learn entrenado con datos sintéticos calibrados para un proyecto EPC ferroviario urbano. Las variables utilizadas son:

- Duración planificada
- Avance planificado y avance real
- Desviación porcentual
- Rendimiento
- Ruta crítica
- Restricciones pendientes
- Disponibilidad de frentes
- Incidencia geotécnica
- Disponibilidad de materiales
- Permisos pendientes

## Estructura del repositorio

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── data/
    └── actividades_e08_ene_feb_2026.csv
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicación en Streamlit Community Cloud

1. Crear un repositorio en GitHub, por ejemplo: `prediccion-cronograma-e08`.
2. Subir todos los archivos de esta carpeta.
3. Ingresar a Streamlit Community Cloud.
4. Seleccionar **New app**.
5. Elegir el repositorio y colocar:
   - Branch: `main`
   - Main file path: `app.py`
6. Presionar **Deploy**.

## Resultado esperado
La aplicación muestra:

- KPIs ejecutivos del cronograma.
- Tabla de resultados solicitada por actividad.
- Predicción para enero 2026.
- Predicción para febrero 2026.
- Nivel de riesgo estimado.
- Gráficos de distribución de riesgo.
- Ranking de actividades que requieren atención del jefe de proyecto.
- Descarga de resultados en CSV.

## Nota académica
Este MVP puede conectarse posteriormente con MS Project, Primavera P6, Power BI o una base de datos real de avance para reemplazar el dataset sintético por información real de obra.
