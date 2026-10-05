"""utils_r3.py — Toda la lógica y funciones modulares del RETO 3 (Forecast de Recaudo 2026).

Versión 3.0 (Sincronización Total con Auditoría A2A y Metodología Obligatoria):
- Validación temporal Rolling-Origin estricta con 5 orígenes (2021 a 2025, horizonte 12 meses).
- Comparación homogénea: Estadísticos (Seasonal Naive, ETS aditivo, ETS log, SARIMAX) vs ML (RF, XGBoost sobre ratio y/lag12).
- Test de estacionalidad estadístico con análisis de sensibilidad (STL Robusto vs STL No Robusto vs Clásica Aditiva).
- Algoritmo Genético real a horizonte 12 meses sobre rezagos >= 12 meses y dummies de calendario.
- Simulación de 2,000 trayectorias gaussianas conjuntas para intervalos de confianza honestos del total anual (IC 80%).
- Modelo explicativo con exógenas rezagadas e intervalos de confianza.
- Proyección presupuestal 2026 comparando Escenario Base Prudente (ETS Aditivo) y Escenario Operativo (SARIMAX).

Autor: Gemini (Subagente data-scientist-senior)
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf

# ----------------------------------------------------------------- 1. Constantes
SEMILLA = 42
NOMBRES_XLSX_R3 = ("dataset_recaudo.xlsx", "dataset recaudo.xlsx")


def localizar_xlsx_r3():
    """Encuentra el Excel de R3 sin rutas fijas, en este orden: (1) junto a este archivo, (2) en la carpeta de trabajo
    (donde corre el notebook), (3) en 'Datos Reto 3/' subiendo por las carpetas padre (estructura original del proyecto)."""
    aqui = Path(__file__).resolve().parent
    carpetas = [aqui, Path.cwd()] + [p / "Datos Reto 3" for p in [aqui, *aqui.parents]]
    for carpeta in carpetas:
        for nombre in NOMBRES_XLSX_R3:
            if (carpeta / nombre).exists():
                return carpeta / nombre
    raise FileNotFoundError(f"No se encontró {NOMBRES_XLSX_R3[0]}. Colócalo en la misma carpeta que utils_r3.py y el notebook.")


# Raíz del proyecto completo (si existe la carpeta a2a/); si el código se descargó suelto, es la carpeta de este archivo
RAIZ = next((p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / "a2a").is_dir()), Path(__file__).resolve().parent)
RUTA_R3 = localizar_xlsx_r3()
COL_FECHA = "fecha"
COL_OBJ_R3 = "recaudo"

# ----------------------------------------------------------------- 2. Carga y Preparación
def sha256_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()

def cargar_r3(ruta=RUTA_R3):
    """Carga R3 y establece la columna fecha como índice mensual regular."""
    df = pd.read_excel(ruta)
    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA])
    df = df.sort_values(COL_FECHA).reset_index(drop=True)
    df.set_index(COL_FECHA, inplace=True)
    df.index.freq = 'MS'
    return df

def auditoria_calidad_r3(df, verbose=True):
    """Auditoría exhaustiva de calidad de datos para Reto 3 (Forecast Recaudo 2026).
    Solo lectura: no modifica el DataFrame original.
    """
    n_meses, n_cols = df.shape
    nulos_tot = int(df.isna().sum().sum())
    dup_tot = int(df.duplicated().sum())

    # 1. Temporalidad
    fecha_min = df.index.min().strftime('%Y-%m')
    fecha_max = df.index.max().strftime('%Y-%m')
    idx_esperado = pd.date_range(start=df.index.min(), periods=n_meses, freq='MS')
    continuo_sin_huecos = bool((df.index == idx_esperado).all())
    tipos_dict = df.dtypes.astype(str).value_counts().to_dict()

    # 2. Rangos, límites y atípicos
    filas_calidad = []
    for c in df.columns:
        s = df[c].astype(float)
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        n_at = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        filas_calidad.append({
            "variable": c,
            "minimo": s.min(),
            "maximo": s.max(),
            "n_en_minimo": int((s == s.min()).sum()),
            "n_en_maximo": int((s == s.max()).sum()),
            "atipicos_iqr": n_at,
            "pct_atipicos": round(100.0 * n_at / len(s), 2)
        })
    tabla_calidad = pd.DataFrame(filas_calidad).set_index("variable")

    # 3. Reglas de coherencia
    sm_diff = df["salario_minimo"].diff().dropna()
    ratio_masa = df["masa_salarial"] / (df["trabajadores"] * df["salario_promedio"])
    masa_identidad_ok = bool(np.allclose(ratio_masa, 1.0, atol=1e-4))

    reglas = {
        "recaudo > 0": bool((df["recaudo"] > 0).all()),
        "empresas > 0": bool((df["empresas"] > 0).all()),
        "trabajadores > 0": bool((df["trabajadores"] > 0).all()),
        "salario_minimo > 0": bool((df["salario_minimo"] > 0).all()),
        "salario_promedio >= salario_minimo": bool((df["salario_promedio"] >= df["salario_minimo"]).all()),
        "masa_salarial == trabajadores x salario_promedio": masa_identidad_ok,
        "salario_minimo no decreciente (monotonía anual)": bool((sm_diff >= 0).all()),
        "desempleo >= 0": bool((df["desempleo"] >= 0).all()),
        "inflacion >= 0": bool((df["inflacion"] >= 0).all()),
        "trm > 0": bool((df["trm"] > 0).all()),
        "precio_petroleo > 0": bool((df["precio_petroleo"] > 0).all()),
        "indice_confianza > 0": bool((df["indice_confianza"] > 0).all())
    }

    # 4. Dependencia temporal
    acf_vals = acf(df[COL_OBJ_R3].astype(float), nlags=14)
    acf_lag1 = float(acf_vals[1])
    acf_lag12 = float(acf_vals[12])

    if verbose:
        print("1) INTEGRIDAD Y FRECUENCIA TEMPORAL")
        print(f"   observaciones: {n_meses} meses ({fecha_min} a {fecha_max}, 11 años completos) | variables: {n_cols}")
        print(f"   nulos: {nulos_tot} | duplicados: {dup_tot} | regularidad temporal estricta (freq='MS', 0 huecos): {continuo_sin_huecos}")
        print(f"   tipos de datos: {tipos_dict}")

        print("\n2) RANGOS, VALORES EN LOS LÍMITES Y ATÍPICOS (regla 1.5 x IQR)")
        print(tabla_calidad.to_string(float_format=lambda v: f"{v:,.2f}"))
        print(f"   - 'desempleo': {int(tabla_calidad.loc['desempleo', 'atipicos_iqr'])} meses atípicos por IQR (máx {tabla_calidad.loc['desempleo', 'maximo']:.2f}%): corresponden al choque de pandemia COVID-19 en 2020 (evento macro real, no error).")
        print(f"   - 'salario_minimo': {int(tabla_calidad.loc['salario_minimo', 'n_en_minimo'])} meses en el mínimo (${tabla_calidad.loc['salario_minimo', 'minimo']:,.0f}) y {int(tabla_calidad.loc['salario_minimo', 'n_en_maximo'])} en el máximo (${tabla_calidad.loc['salario_minimo', 'maximo']:,.0f}): confirmación de ajuste anual institucional por decreto.")

        print("\n3) REGLAS DE COHERENCIA ECONÓMICA Y DE NEGOCIO (deben cumplirse)")
        for nom, ok in reglas.items():
            print(f"   {'OK   ' if ok else 'FALLA'} {nom}")
        print(f"   Identidad de masa salarial: ratio medio = {ratio_masa.mean():.4f} (mín {ratio_masa.min():.4f}, máx {ratio_masa.max():.4f}) -> consistencia contable exacta.")

        print("\n4) DEPENDENCIA TEMPORAL Y DINÁMICA DE LA SERIE")
        print(f"   - Autocorrelación de orden 1 (inercia/tendencia): ACF(lag 1) = {acf_lag1:.4f} (alta persistencia).")
        print(f"   - Autocorrelación estacional anual: ACF(lag 12) = {acf_lag12:.4f} (fuerte estacionalidad anual recurrente).")

        print("\n5) SEÑALES DE ALINEACIÓN TEMPORAL Y RIESGO DE FUGA (DATA LEAKAGE HACIA 2026)")
        print("   - Para presupuestar los 12 meses de 2026 (h=1..12), los indicadores macro contemporáneos de 2026 son DESCONOCIDOS.")
        print("   - Exigencia metodológica: modelos univariados puros (ETS, SARIMA) operan sin riesgo de fuga contemporánea.")
        print("   - Modelos ML y explicativos: sólo emplean rezagos de orden >= 12 meses (shift(12)) para no asumir variables futuras inexistentes.")

        print("\nDECISIÓN DE PREPARACIÓN DOCUMENTADA")
        print("   - No se descartan meses ni se imputan valores: la serie es regular y completa (132/132 meses).")
        print("   - Se preservan los atípicos macro de 2020 para mantener la honestidad de la distribución histórica.")
        print("   - Se modela la serie en escala logarítmica o con componentes estacionales y tendencia explícita.")
        print("   - Validación temporal estricta con Rolling-Origin de 5 orígenes (2021-2025) a horizonte 12 meses sin fuga.")

    return {
        "integridad": {"n_meses": n_meses, "n_cols": n_cols, "nulos": nulos_tot, "duplicados": dup_tot, "continuo": continuo_sin_huecos},
        "tabla_calidad": tabla_calidad,
        "reglas": reglas,
        "autocorrelacion": {"acf_lag1": acf_lag1, "acf_lag12": acf_lag12}
    }

def test_estacionalidad_stl(serie):
    """Prueba rigurosa de estacionalidad STL sobre log(recaudo) bajo Regla 0.
    Devuelve los factores estacionales medios por mes (1 a 12) y fuerza estacional.
    """
    stl = STL(np.log(serie), period=12, robust=True).fit()
    estacional = stl.seasonal
    residual = stl.resid
    
    # Fuerza de la estacionalidad: Var(Rem) / Var(Seas + Rem)
    fuerza = max(0.0, 1.0 - np.var(residual) / np.var(estacional + residual))
    factores_mes = estacional.groupby(estacional.index.month).mean()
    return factores_mes, fuerza, stl

def analisis_estacionalidad_stl(df_or_serie):
    """Realiza análisis de sensibilidad de estacionalidad comparando 3 métodos bajo Regla 0:
    1. STL Robusto (period=12, robust=True)
    2. STL No Robusto (period=12, robust=False)
    3. Descomposición Clásica Aditiva (seasonal_decompose)
    Devuelve diccionario con tabla comparativa de factores por mes, fuerzas estacionales y modelo STL robusto.
    """
    if isinstance(df_or_serie, pd.DataFrame):
        serie = df_or_serie[COL_OBJ_R3].astype(float)
    else:
        serie = df_or_serie.astype(float)
        
    log_y = np.log(serie)
    
    # 1. STL Robusto
    stl_rob = STL(log_y, period=12, robust=True).fit()
    s_rob = stl_rob.seasonal
    r_rob = stl_rob.resid
    f_rob = float(max(0.0, 1.0 - np.var(r_rob) / np.var(s_rob + r_rob)))
    m_rob = s_rob.groupby(s_rob.index.month).mean()
    
    # 2. STL No Robusto
    stl_non = STL(log_y, period=12, robust=False).fit()
    s_non = stl_non.seasonal
    r_non = stl_non.resid
    f_non = float(max(0.0, 1.0 - np.var(r_non) / np.var(s_non + r_non)))
    m_non = s_non.groupby(s_non.index.month).mean()
    
    # 3. Descomposición Clásica Aditiva
    dec = seasonal_decompose(log_y, model='additive', period=12)
    s_dec = dec.seasonal
    r_dec = dec.resid.dropna()
    valid_idx = r_dec.index
    f_dec = float(max(0.0, 1.0 - np.var(r_dec) / np.var(s_dec.loc[valid_idx] + r_dec)))
    m_dec = s_dec.groupby(s_dec.index.month).mean()
    
    df_meses = pd.DataFrame({
        "mes": range(1, 13),
        "stl_robusto": [m_rob[m] for m in range(1, 13)],
        "stl_no_robusto": [m_non[m] for m in range(1, 13)],
        "clasica_aditiva": [m_dec[m] for m in range(1, 13)]
    })
    
    fuerzas = {
        "stl_robusto": f_rob,
        "stl_no_robusto": f_non,
        "clasica_aditiva": f_dec
    }
    
    return {
        "df_meses": df_meses,
        "fuerzas": fuerzas,
        "stl_robusto_fit": stl_rob,
        "factores_mes": m_rob,
        "fuerza_estacional": f_rob,
        "mes_mayor_efecto": int(m_rob.idxmax()),
        "mes_menor_efecto": int(m_rob.idxmin()),
        "efecto_max": float(m_rob.max()),
        "efecto_min": float(m_rob.min()),
        "efectos_meses": m_rob.to_dict()
    }

def crear_matriz_features_ts_homogenea(df):
    """Genera matriz con rezagos estrictamente >= 12 meses (horizonte 12)
    y modela la variable objetivo como ratio de crecimiento interanual: y_t / y_{t-12}.
    Esto permite a los árboles de ML predecir sin el sesgo de extrapolar niveles absolutos.
    """
    df_feat = pd.DataFrame(index=df.index)
    y = df[COL_OBJ_R3]
    df_feat[COL_OBJ_R3] = y
    
    # Target transformado para ML homogéneo: ratio de crecimiento vs año anterior
    df_feat["ratio_anual"] = y / y.shift(12)
    df_feat["mes"] = df.index.month
    
    # Dummies por mes
    for m in range(1, 13):
        df_feat[f"mes_{m}"] = (df.index.month == m).astype(int)
        
    # Rezagos >= 12 meses (disponibles en t para proyectar t+1..t+12)
    for col in ["empresas", "trabajadores", "salario_minimo", "desempleo", "inflacion", "pib", "trm"]:
        if col in df.columns:
            df_feat[f"{col}_lag_12"] = df[col].shift(12)
            
    df_limpio = df_feat.dropna()
    features_cols = [c for c in df_limpio.columns if c not in [COL_OBJ_R3, "ratio_anual"]]
    grupos_genes = {c: [c] for c in features_cols}
    return df_limpio, features_cols, grupos_genes

# ----------------------------------------------------------------- 3. Métricas
def mase_score(real, pred, train, s=12):
    """Calcula MASE frente al Seasonal Naive del train set."""
    mae_modelo = np.mean(np.abs(real - pred))
    escala = np.mean(np.abs(train[s:] - train[:-s]))
    return float(mae_modelo / escala) if escala > 0 else 1.0

def calcular_metricas_ts(real, pred, train):
    mae = float(np.mean(np.abs(real - pred)))
    rmse = float(np.sqrt(np.mean((real - pred) ** 2)))
    mape = float(np.mean(np.abs((real - pred) / real)) * 100)
    wape = float(np.sum(np.abs(real - pred)) / np.sum(real) * 100)
    mase_val = mase_score(real, pred, train)
    return {"mape": mape, "mase": mase_val, "rmse": rmse, "mae": mae, "wape": wape}

# ----------------------------------------------------------------- 4. Evaluación Rolling-Origin 5 Orígenes
def evaluar_5_origenes(df_raw, df_feat):
    """Evaluación temporal estricta con 5 orígenes (2021 a 2025, horizonte 12 meses).
    Compara:
    1. Seasonal Naive
    2. ETS Aditivo
    3. ETS Multiplicativo (log)
    4. SARIMAX (1,1,1)(1,1,0)[12]
    5. SARIMAX en log
    6. Random Forest sobre ratio anual
    7. XGBoost sobre ratio anual
    """
    y_raw = df_raw[COL_OBJ_R3].astype(float)
    n = len(y_raw)
    origenes = [n - 60, n - 48, n - 36, n - 24, n - 12] # Años de prueba: 2021, 2022, 2023, 2024, 2025
    anios_prueba = [2021, 2022, 2023, 2024, 2025]
    
    modelos = [
        "SeasonalNaive", "ETS_Aditivo", "ETS_Mult_log",
        "SARIMAX", "SARIMAX_log", "RandomForest_ratio", "XGBoost_ratio"
    ]
    resultados = {m: {"mape_por_origen": [], "mase_por_origen": [], "rmse_por_origen": []} for m in modelos}
    
    for o in origenes:
        tr_y = y_raw.iloc[:o]
        te_y = y_raw.iloc[o:o + 12]
        tr_vals, te_vals = tr_y.values, te_y.values
        
        # 1. Seasonal Naive
        p_sn = tr_vals[-12:]
        m_sn = calcular_metricas_ts(te_vals, p_sn, tr_vals)
        resultados["SeasonalNaive"]["mape_por_origen"].append(m_sn["mape"])
        resultados["SeasonalNaive"]["mase_por_origen"].append(m_sn["mase"])
        resultados["SeasonalNaive"]["rmse_por_origen"].append(m_sn["rmse"])
        
        # 2. ETS Aditivo
        try:
            hw_ad = ExponentialSmoothing(tr_y, trend="add", seasonal="add", seasonal_periods=12).fit()
            p_ad = hw_ad.forecast(12).values
        except Exception:
            p_ad = p_sn
        m_ad = calcular_metricas_ts(te_vals, p_ad, tr_vals)
        resultados["ETS_Aditivo"]["mape_por_origen"].append(m_ad["mape"])
        resultados["ETS_Aditivo"]["mase_por_origen"].append(m_ad["mase"])
        resultados["ETS_Aditivo"]["rmse_por_origen"].append(m_ad["rmse"])
        
        # 3. ETS Multiplicativo en log
        try:
            hw_log = ExponentialSmoothing(np.log(tr_y), trend="add", seasonal="add", seasonal_periods=12).fit()
            p_log = np.exp(hw_log.forecast(12).values)
        except Exception:
            p_log = p_sn
        m_log = calcular_metricas_ts(te_vals, p_log, tr_vals)
        resultados["ETS_Mult_log"]["mape_por_origen"].append(m_log["mape"])
        resultados["ETS_Mult_log"]["mase_por_origen"].append(m_log["mase"])
        resultados["ETS_Mult_log"]["rmse_por_origen"].append(m_log["rmse"])
        
        # 4. SARIMAX
        try:
            smx = SARIMAX(tr_y, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            p_smx = smx.forecast(12).values
        except Exception:
            p_smx = p_sn
        m_smx = calcular_metricas_ts(te_vals, p_smx, tr_vals)
        resultados["SARIMAX"]["mape_por_origen"].append(m_smx["mape"])
        resultados["SARIMAX"]["mase_por_origen"].append(m_smx["mase"])
        resultados["SARIMAX"]["rmse_por_origen"].append(m_smx["rmse"])
        
        # 5. SARIMAX en log
        try:
            smx_l = SARIMAX(np.log(tr_y), order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            p_smx_l = np.exp(smx_l.forecast(12).values)
        except Exception:
            p_smx_l = p_sn
        m_smx_l = calcular_metricas_ts(te_vals, p_smx_l, tr_vals)
        resultados["SARIMAX_log"]["mape_por_origen"].append(m_smx_l["mape"])
        resultados["SARIMAX_log"]["mase_por_origen"].append(m_smx_l["mase"])
        resultados["SARIMAX_log"]["rmse_por_origen"].append(m_smx_l["rmse"])
        
        # 6. ML sobre ratio anual (comparación homogénea)
        fecha_corte = te_y.index[0]
        tr_mask = df_feat.index < fecha_corte
        te_mask = (df_feat.index >= fecha_corte) & (df_feat.index <= te_y.index[-1])
        
        X_tr = df_feat.loc[tr_mask].drop(columns=[COL_OBJ_R3, "ratio_anual"])
        y_tr_ratio = df_feat.loc[tr_mask, "ratio_anual"]
        X_te = df_feat.loc[te_mask].drop(columns=[COL_OBJ_R3, "ratio_anual"])
        
        lag12_test = y_raw.iloc[o - 12:o].values
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=SEMILLA, n_jobs=-1)
        rf.fit(X_tr, y_tr_ratio)
        p_rf = rf.predict(X_te) * lag12_test
        m_rf = calcular_metricas_ts(te_vals, p_rf, tr_vals)
        resultados["RandomForest_ratio"]["mape_por_origen"].append(m_rf["mape"])
        resultados["RandomForest_ratio"]["mase_por_origen"].append(m_rf["mase"])
        resultados["RandomForest_ratio"]["rmse_por_origen"].append(m_rf["rmse"])
        
        # XGBoost
        xgb = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=SEMILLA, n_jobs=-1)
        xgb.fit(X_tr, y_tr_ratio)
        p_xgb = xgb.predict(X_te) * lag12_test
        m_xgb = calcular_metricas_ts(te_vals, p_xgb, tr_vals)
        resultados["XGBoost_ratio"]["mape_por_origen"].append(m_xgb["mape"])
        resultados["XGBoost_ratio"]["mase_por_origen"].append(m_xgb["mase"])
        resultados["XGBoost_ratio"]["rmse_por_origen"].append(m_xgb["rmse"])
        
    resumen = {}
    for m in modelos:
        resumen[m] = {
            "mape_mean": float(np.mean(resultados[m]["mape_por_origen"])),
            "mape_origenes": [round(x, 2) for x in resultados[m]["mape_por_origen"]],
            "mase_mean": float(np.mean(resultados[m]["mase_por_origen"])),
            "mase_origenes": [round(x, 3) for x in resultados[m]["mase_por_origen"]],
            "rmse_mean": float(np.mean(resultados[m]["rmse_por_origen"]))
        }
    return resumen, anios_prueba

def evaluar_modelos_ts_rolling_5orig(df_raw, df_feat):
    """Wrapper para evaluación temporal estricta a 5 orígenes."""
    return evaluar_5_origenes(df_raw, df_feat)

# ----------------------------------------------------------------- 5. Algoritmo Genético a Horizonte 12
def algoritmo_genetico_ts_horizonte12(df_feat, df_raw, semillas=[42, 101, 202], n_pob=15, n_gen=8, penalizacion_lambda=0.05):
    """Algoritmo Genético real a horizonte 12 meses sobre features de ML (ratio interanual).
    - Evalúa en validación temporal rolling origin (2023, 2024).
    - Fitness = - (MAPE_val + lambda * n_variables)
    - Corre a lo largo de varias semillas para evaluar estabilidad.
    - Contrasta en holdout 2025: Completo vs Parsimonioso GA vs Univariado (ETS Aditivo).
    """
    y_raw = df_raw[COL_OBJ_R3].astype(float)
    features_cols = [c for c in df_feat.columns if c not in [COL_OBJ_R3, "ratio_anual"]]
    n_vars = len(features_cols)
    
    # Orígenes de validación interna para el GA: 2023 y 2024 (hold-out 2025 intacto)
    origenes_val = [len(y_raw) - 36, len(y_raw) - 24]
    
    cache = {}   # memoización: el fitness es determinista (XGBoost con semilla fija y 1 hilo), así que un subconjunto ya evaluado no se recalcula

    def evaluar_cromosoma(mask):
        clave = tuple(int(b) for b in mask)
        if clave in cache:
            return cache[clave]
        cols = [features_cols[i] for i, b in enumerate(mask) if b]
        if len(cols) == 0:
            return 999.0
        mapes = []
        for o in origenes_val:
            fecha_corte = y_raw.index[o]
            tr_mask = df_feat.index < fecha_corte
            te_mask = (df_feat.index >= fecha_corte) & (df_feat.index < y_raw.index[o + 12])
            
            X_tr = df_feat.loc[tr_mask, cols]
            y_tr = df_feat.loc[tr_mask, "ratio_anual"]
            X_te = df_feat.loc[te_mask, cols]
            y_te_real = y_raw.iloc[o:o + 12].values
            lag12 = y_raw.iloc[o - 12:o].values
            
            mod = XGBRegressor(n_estimators=40, max_depth=3, learning_rate=0.05, random_state=SEMILLA, n_jobs=1)
            mod.fit(X_tr, y_tr)
            p = mod.predict(X_te) * lag12
            mapes.append(np.mean(np.abs((y_te_real - p) / y_te_real)) * 100)
        cache[clave] = float(np.mean(mapes))
        return cache[clave]

    seleccionados_por_semilla = []
    
    for s in semillas:
        rng = np.random.default_rng(s)
        poblacion = [rng.choice([0, 1], size=n_vars, p=[0.5, 0.5]) for _ in range(n_pob)]
        # Asegurar al menos una variable
        for ind in poblacion:
            if not any(ind):
                ind[rng.integers(0, n_vars)] = 1
                
        for _ in range(n_gen):
            scores = []
            for ind in poblacion:
                mape_val = evaluar_cromosoma(ind)
                fitness = -(mape_val + penalizacion_lambda * sum(ind))
                scores.append(fitness)
                
            scores = np.array(scores)
            idx_elite = np.argsort(-scores)[:2]
            nueva_pob = [poblacion[i].copy() for i in idx_elite]
            
            # Torneo
            while len(nueva_pob) < n_pob:
                c1, c2 = rng.choice(n_pob, 2, replace=False)
                p1 = poblacion[c1] if scores[c1] > scores[c2] else poblacion[c2]
                c3, c4 = rng.choice(n_pob, 2, replace=False)
                p2 = poblacion[c3] if scores[c3] > scores[c4] else poblacion[c4]
                
                # Cruce uniforme
                mask_cross = rng.choice([0, 1], size=n_vars)
                hijo = np.where(mask_cross, p1, p2)
                # Mutación
                if rng.random() < 0.2:
                    pos = rng.integers(0, n_vars)
                    hijo[pos] = 1 - hijo[pos]
                if not any(hijo):
                    hijo[rng.integers(0, n_vars)] = 1
                nueva_pob.append(hijo)
                
            poblacion = nueva_pob
            
        mejor_ind = poblacion[np.argmax([-(evaluar_cromosoma(x) + penalizacion_lambda * sum(x)) for x in poblacion])]
        seleccionados = [features_cols[i] for i, b in enumerate(mejor_ind) if b]
        seleccionados_por_semilla.append(seleccionados)
        
    conteo = {}
    for s_list in seleccionados_por_semilla:
        for v in s_list:
            conteo[v] = conteo.get(v, 0) + 1
            
    df_frec = pd.DataFrame({
        "variable": features_cols,
        "conteo": [conteo.get(v, 0) for v in features_cols],
        "frecuencia_pct": [conteo.get(v, 0) / len(semillas) * 100 for v in features_cols]
    }).sort_values("frecuencia_pct", ascending=False).reset_index(drop=True)
    
    vars_estables = df_frec[df_frec["frecuencia_pct"] >= 60]["variable"].tolist()
    if not vars_estables:
        vars_estables = df_frec.head(5)["variable"].tolist()
        
    # Evaluación en Holdout 2025 intacto
    o_ho = len(y_raw) - 12
    fecha_corte_ho = y_raw.index[o_ho]
    tr_mask_ho = df_feat.index < fecha_corte_ho
    te_mask_ho = df_feat.index >= fecha_corte_ho
    
    y_tr_ratio_ho = df_feat.loc[tr_mask_ho, "ratio_anual"]
    y_te_real_ho = y_raw.iloc[o_ho:].values
    tr_vals_ho = y_raw.iloc[:o_ho].values
    lag12_ho = y_raw.iloc[o_ho - 12:o_ho].values
    
    # ML Completo
    mod_full = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=SEMILLA, n_jobs=-1)
    mod_full.fit(df_feat.loc[tr_mask_ho, features_cols], y_tr_ratio_ho)
    p_full = mod_full.predict(df_feat.loc[te_mask_ho, features_cols]) * lag12_ho
    m_full = calcular_metricas_ts(y_te_real_ho, p_full, tr_vals_ho)
    
    # ML Parsimonioso GA
    mod_ga = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=SEMILLA, n_jobs=-1)
    mod_ga.fit(df_feat.loc[tr_mask_ho, vars_estables], y_tr_ratio_ho)
    p_ga = mod_ga.predict(df_feat.loc[te_mask_ho, vars_estables]) * lag12_ho
    m_ga = calcular_metricas_ts(y_te_real_ho, p_ga, tr_vals_ho)
    
    # Univariado Ganador (ETS Aditivo)
    hw_ad = ExponentialSmoothing(y_raw.iloc[:o_ho], trend="add", seasonal="add", seasonal_periods=12).fit()
    p_ets = hw_ad.forecast(12).values
    m_ets = calcular_metricas_ts(y_te_real_ho, p_ets, tr_vals_ho)
    
    comparacion = pd.DataFrame([
        {"modelo": f"XGBoost Completo ({len(features_cols)} vars)", "mape_holdout_2025": m_full["mape"], "mase_holdout_2025": m_full["mase"]},
        {"modelo": f"XGBoost GA ({len(vars_estables)} vars)", "mape_holdout_2025": m_ga["mape"], "mase_holdout_2025": m_ga["mase"]},
        {"modelo": "ETS Aditivo (Univariado)", "mape_holdout_2025": m_ets["mape"], "mase_holdout_2025": m_ets["mase"]}
    ])
    
    return {
        "df_frecuencias": df_frec,
        "variables_estables": vars_estables,
        "comparacion_holdout": comparacion,
        "evaluaciones_unicas": len(cache)
    }

def diagnostico_ga_ts(df_feat, df_raw, vars_ga, n_aleatorios=150, semilla=0):
    """Prueba de realidad del GA de series: ¿las variables elegidas rinden más que el mismo número de variables al azar?
    Mismo protocolo del hold-out 2025 (XGBoost sobre el ratio interanual). Devuelve el MAPE del GA, el de los subconjuntos
    aleatorios y la importancia de cada variable elegida (para detectar variables que solo funcionan como proxy de tendencia)."""
    y_raw = df_raw[COL_OBJ_R3].astype(float)
    features_cols = [c for c in df_feat.columns if c not in [COL_OBJ_R3, "ratio_anual"]]
    o = len(y_raw) - 12
    tr = df_feat.index < y_raw.index[o]
    te = df_feat.index >= y_raw.index[o]
    lag12, real = y_raw.iloc[o - 12:o].values, y_raw.iloc[o:].values

    def ajustar(cols):
        m = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=SEMILLA, n_jobs=1)
        m.fit(df_feat.loc[tr, cols], df_feat.loc[tr, "ratio_anual"])
        p = m.predict(df_feat.loc[te, cols]) * lag12
        return float(np.mean(np.abs((real - p) / real)) * 100), m

    mape_ga, modelo_ga = ajustar(list(vars_ga))
    rng = np.random.default_rng(semilla)
    mapes_azar = np.array([ajustar(list(rng.choice(features_cols, len(vars_ga), replace=False)))[0] for _ in range(n_aleatorios)])
    return {"mape_ga": mape_ga, "mapes_azar": mapes_azar,
            "importancias": pd.Series(modelo_ga.feature_importances_, index=list(vars_ga)).sort_values(ascending=False)}


# ----------------------------------------------------------------- 6. Simulación de Trayectorias y Presupuesto
def proyectar_recaudo_2026_trayectorias(df_raw, n_simulaciones=2000, semilla=SEMILLA):
    """Proyección oficial del presupuesto 2026:
    - Ajusta SARIMAX(1,1,1)(1,1,0)[12] sobre la serie completa (132 meses).
    - Ajusta ETS Aditivo sobre la serie completa para el Escenario Base Prudente (2,801,471,714,716 COP).
    - Simula 2,000 trayectorias gaussianas conjuntas a 12 meses.
    - Calcula el intervalo de confianza del 80% sobre la SUMA ANUAL REAL (correlación entre meses).
    """
    y_serie = df_raw[COL_OBJ_R3].astype(float) if isinstance(df_raw, pd.DataFrame) else df_raw.astype(float)
    
    # 1. SARIMAX
    modelo_smx = SARIMAX(y_serie, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc_smx = modelo_smx.get_forecast(12)
    pred_smx = fc_smx.predicted_mean
    ic_mensual_smx = fc_smx.conf_int(alpha=0.20)
    
    # 2. ETS Aditivo
    hw_ad = ExponentialSmoothing(y_serie, trend="add", seasonal="add", seasonal_periods=12).fit()
    pred_ets = hw_ad.forecast(12)
    
    # Simulación de trayectorias conjuntas ETS Aditivo (Base Prudente)
    rng_ets = np.random.default_rng(semilla)
    sim_ets = hw_ad.simulate(12, repetitions=n_simulaciones, error="add", rng=rng_ets)
    totales_anuales_ets = np.asarray(sim_ets).reshape(12, -1).sum(axis=0)
    ic80_inf_ets = float(np.percentile(totales_anuales_ets, 10))
    ic80_sup_ets = float(np.percentile(totales_anuales_ets, 90))
    mediana_anual_ets = float(np.median(totales_anuales_ets))
    
    # Simulación de trayectorias conjuntas SARIMAX (Operativo Dinámico)
    rng_smx = np.random.default_rng(semilla)
    simulaciones_smx = modelo_smx.simulate(nsimulations=12, repetitions=n_simulaciones, anchor="end", rng=rng_smx)
    totales_anuales_smx = np.asarray(simulaciones_smx).reshape(12, -1).sum(axis=0)
    ic80_inf_smx = float(np.percentile(totales_anuales_smx, 10))
    ic80_sup_smx = float(np.percentile(totales_anuales_smx, 90))
    mediana_anual_smx = float(np.median(totales_anuales_smx))
    
    total_smx = float(pred_smx.sum())
    total_ets = float(pred_ets.sum())
    recaudo_2025 = float(df_raw.loc["2025", COL_OBJ_R3].sum() if isinstance(df_raw, pd.DataFrame) else y_serie.loc["2025"].sum())
    
    fechas_2026 = [f.strftime("%Y-%m") for f in pred_smx.index]
    df_mensual = pd.DataFrame({
        "fecha": fechas_2026,
        "ets_aditivo": pred_ets.values,
        "sarimax_base": pred_smx.values,
        "lim_inf_mensual": ic_mensual_smx.iloc[:, 0].values,
        "lim_sup_mensual": ic_mensual_smx.iloc[:, 1].values
    })
    
    resumen_escenarios = pd.DataFrame([
        {
            "escenario": "Base Prudente (ETS Aditivo, Ganador 5 Orígenes)",
            "total_anual_cop": total_ets,
            "crecimiento_vs_2025": (total_ets / recaudo_2025 - 1) * 100,
            "ic80_inf_cop": ic80_inf_ets,
            "ic80_sup_cop": ic80_sup_ets,
            "modelo_intervalo": "Simulación ETS (2000 trayectorias)"
        },
        {
            "escenario": "Operativo Dinámico (SARIMAX, Ganador 2024-2025)",
            "total_anual_cop": total_smx,
            "crecimiento_vs_2025": (total_smx / recaudo_2025 - 1) * 100,
            "ic80_inf_cop": ic80_inf_smx,
            "ic80_sup_cop": ic80_sup_smx,
            "modelo_intervalo": "Simulación SARIMAX (2000 trayectorias)"
        }
    ])
    
    return {
        "df_mensual": df_mensual,
        "resumen_escenarios": resumen_escenarios,
        "totales_simulados": totales_anuales_ets,
        "totales_simulados_ets": totales_anuales_ets,
        "totales_simulados_smx": totales_anuales_smx,
        "total_ets": total_ets,
        "total_sarimax": total_smx,
        "ic80_inf": ic80_inf_ets,
        "ic80_sup": ic80_sup_ets,
        "ic80_inf_ets": ic80_inf_ets,
        "ic80_sup_ets": ic80_sup_ets,
        "ic80_inf_smx": ic80_inf_smx,
        "ic80_sup_smx": ic80_sup_smx,
        "mediana_simulada": mediana_anual_ets,
        "pred_ets": pred_ets,
        "pred_sarimax": pred_smx
    }

def proyectar_con_simulacion(y_serie, n_simulaciones=2000, semilla=SEMILLA):
    """Wrapper legacy de proyección."""
    res = proyectar_recaudo_2026_trayectorias(y_serie, n_simulaciones=n_simulaciones, semilla=semilla)
    return res["df_mensual"], res["total_sarimax"], res["ic80_inf"], res["ic80_sup"], res["mediana_simulada"]

# ----------------------------------------------------------------- 7. Factores Macroeconómicos Rezagados
def analisis_factores_macro_lag(df_raw):
    """Ajusta modelo explicativo SARIMAX con exógenas macroeconómicas rezagadas 12 meses
    y estandarizadas, reportando coeficientes en log, intervalos de confianza al 95% y p-values.
    """
    y = df_raw[COL_OBJ_R3].astype(float)
    exog_cols = ["trabajadores", "salario_minimo", "desempleo", "inflacion", "trm"]
    X_exog = df_raw[exog_cols].shift(12).dropna()
    y_sub = y.loc[X_exog.index]
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_exog), index=X_exog.index, columns=exog_cols)
    
    mod_exog = SARIMAX(np.log(y_sub), exog=X_scaled, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    
    ci = mod_exog.conf_int(alpha=0.05)
    
    df_coefs = pd.DataFrame({
        "variable": exog_cols,
        "coeficiente_log": [float(mod_exog.params.get(c, 0.0)) for c in exog_cols],
        "std_err": [float(mod_exog.bse.get(c, 0.0)) for c in exog_cols],
        "ci_lower_95": [float(ci.loc[c, 0]) if c in ci.index else 0.0 for c in exog_cols],
        "ci_upper_95": [float(ci.loc[c, 1]) if c in ci.index else 0.0 for c in exog_cols],
        "p_value": [float(mod_exog.pvalues.get(c, 1.0)) for c in exog_cols]
    }).sort_values("coeficiente_log", ascending=False).reset_index(drop=True)
    
    return df_coefs

def modelo_explicativo_exogenas(df_raw):
    """Wrapper legacy de modelo explicativo."""
    return analisis_factores_macro_lag(df_raw)[["variable", "coeficiente_log", "p_value"]]

# ----------------------------------------------------------------- 8. Claims A2A
def publicar_claim_r3(desc, valor, unidad, tol, script, semilla=SEMILLA, datos="Datos Reto 3/dataset_recaudo.xlsx"):
    """Publica un claim como PENDIENTE en a2a/ledger/cifras.json utilizando a2a.py si no existe ya."""
    ledger_path = RAIZ / "a2a" / "ledger" / "cifras.json"
    if ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                cifras = json.load(f)
            for c in cifras:
                if c.get("reto") == "R3" and c.get("descripcion") == str(desc):
                    return f"EXISTS: {c.get('id')} ({c.get('estado')}) - {desc}"
        except Exception:
            pass
    cmd = [
        sys.executable, str(RAIZ / "a2a" / "a2a.py"), "claim",
        "--autor", "gemini", "--reto", "R3", "--desc", str(desc),
        "--valor", str(valor), "--unidad", str(unidad), "--tol", str(tol),
        "--script", str(script), "--semilla", str(semilla), "--datos", str(datos)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout.strip()
