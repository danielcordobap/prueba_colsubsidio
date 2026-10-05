# Prueba técnica Senior Data Scientist — Colsubsidio

## ¿De qué trata la prueba?

Colsubsidio administra la afiliación de empresas y ofrece servicios a sus empleadores afiliados. En los últimos periodos creció la **desafiliación** de algunas empresas, lo que preocupa por la sostenibilidad del negocio y la planeación comercial. Distintas gerencias piden análisis para entender el aporte esperado de cada empresa, los factores asociados a la desafiliación y el presupuesto de recaudo. La prueba plantea **tres casos de negocio independientes**, cada uno con su propio conjunto de datos; el objetivo es construir modelos que reduzcan la incertidumbre en la toma de decisiones y **sustentar las conclusiones desde la perspectiva técnica y de negocio**.

| Reto | Tipo | Pregunta de negocio | Datos | Resultado principal |
|---|---|---|---|---|
| **1. Aporte mensual** | Regresión | ¿Cuánto aporta cada empresa y qué factores explican las diferencias? | `empresas_afiliadas.csv` (10.000 empresas) | Modelo de árboles (XGBoost) con 8 factores; error típico de 4,1 % (≈ 432 mil pesos) en empresas que el modelo no vio |
| **2. Desafiliación** | Clasificación | ¿Qué empresas tienen mayor riesgo de retiro y qué las empuja? | `base_clasificacion.csv` (10.000 empresas, 30,3 % desafiliadas) | Regresión logística con 7 variables; detecta 466 de 606 desafiliaciones en test y, llamando al 20 % de mayor riesgo, se capturan 305 (50 %) |
| **3. Presupuesto** | Series de tiempo | ¿Cuánto recaudará la caja cada mes de 2026 y con qué margen? | `dataset_recaudo.xlsx` (132 meses, 2015-2025) | Suavizado exponencial (ETS) aditivo; 2,80 billones de pesos para 2026, rango probable al 80 % entre 2,68 y 2,92 |

Metodología común: **CRISP-DM**, comparación de al menos tres modelos, selección de variables con **algoritmo genético** para lograr modelos simples, interpretabilidad (efectos por factor, odds ratios, SHAP en el Reto 1) y presentación de hallazgos en lenguaje de negocio. Todas las cifras salen de código ejecutado.

## Estructura de la carpeta

Cada reto vive en su propia carpeta y es **autocontenido**: basta con descargar la carpeta (notebook + utilidades + datos) y ejecutarla, sin rutas fijas.

```
Jupyter_notebooks/
├── README.md                  Este archivo
├── requirements.txt           Versiones de las librerías usadas
├── problema_1/                Reto 1 — Aporte mensual (regresión)
├── problema_2/                Reto 2 — Riesgo de desafiliación (clasificación)
└── problema_3/                Reto 3 — Forecast y presupuesto de recaudo 2026
```

### `problema_1/`
| Archivo | Qué hace |
|---|---|
| `Problema_1.ipynb` | Notebook de la solución, por fases CRISP-DM: auditoría de datos, comparación de modelos, selección de factores con algoritmo genético, interpretabilidad (SHAP), validación final e intervalos |
| `utils_r1.py` | Funciones del Reto 1 (lectura, preparación, modelos, algoritmo genético, SHAP, métricas) para mantener limpio el notebook |
| `empresas_afiliadas.csv` | Datos del reto |
| `Colsubsidio_Problema1_Interpretabilidad.pptx` | Presentación de hallazgos (9 diapositivas) |

### `problema_2/`
| Archivo | Qué hace |
|---|---|
| `Problema_2.ipynb` | Notebook: auditoría de datos, comparación de 3 familias de modelos (regla 1-SE), algoritmo genético, desempate regresión logística vs XGBoost, calibración de probabilidades, odds ratios, matriz de confusión y deciles comerciales |
| `utils_r2.py` | Funciones del Reto 2 (preparación, validación cruzada, algoritmo genético, calibración, explicabilidad, deciles) |
| `base_clasificacion.csv` | Datos del reto |
| `Colsubsidio_Problema2_Desafiliacion.pptx` | Presentación de hallazgos (9 diapositivas) |

### `problema_3/`
| Archivo | Qué hace |
|---|---|
| `Problema_3.ipynb` | Notebook: auditoría y estacionalidad, comparación de 7 modelos en 5 orígenes de prueba (2021-2025), prueba con algoritmo genético de variables externas, análisis de factores macro, pronóstico 2026 con intervalos por simulación y dictamen presupuestal |
| `utils_r3.py` | Funciones del Reto 3 (carga, estacionalidad, validación rolling-origin, algoritmo genético, simulación de trayectorias, ajuste del modelo) |
| `dataset_recaudo.xlsx` | Datos del reto |
| `Colsubsidio_Problema3_Recaudo2026.pptx` | Presentación de hallazgos (9 diapositivas) |

## Cómo ejecutarlo

1. Instala las librerías de `requirements.txt` (principales: pandas, numpy, scikit-learn, xgboost, shap, statsmodels, matplotlib, openpyxl).
2. Abre el notebook del reto en Jupyter desde su carpeta; deben estar juntos el notebook, el `utils_rN.py` y el archivo de datos.
3. Ejecuta de arriba hacia abajo (*Restart & Run All*). Los tiempos son de segundos a pocos minutos; la semilla global es 42, por lo que los resultados son reproducibles.

## Notas

- Cada notebook empieza con un encabezado de fase que explica qué se hace y por qué; las conclusiones impresas se generan por código a partir de los resultados.
- Los efectos de los factores se leen como **asociaciones del modelo**, útiles para priorizar acciones; las palancas de negocio deben validarse con un piloto.
- Si Jupyter ya tenía abierto un `utils_rN.py`, reinicia el kernel después de reemplazarlo para que cargue la versión nueva.
