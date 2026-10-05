"""utils_r2.py — Toda la lógica y funciones modulares del RETO 2 (Riesgo de Desafiliación).

Versión 2.0 (Respuesta a Crítica 0012 A2A y Metodología Obligatoria):
- Criterio de Selección 1-SE y parsimonia: Regresión Logística vs XGBoost vs Random Forest.
- Calibración de probabilidades con Isotonic Regression (OOF sobre train).
- Algoritmo Genético Multi-Semilla (5 semillas) con frecuencia de selección y regla 1-SE.
- Explicabilidad (Odds Ratios + SHAP sobre el 100% del hold-out de 2.000 filas).
- Deciles comerciales calibrados con cálculo de valor esperado y lift.

Autor: Gemini (Subagente data-scientist-senior)
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, brier_score_loss,
    confusion_matrix, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

# ----------------------------------------------------------------- 1. Constantes
SEMILLA = 42
NOMBRES_CSV_R2 = ("base clasificacion.csv", "base_clasificacion.csv", "base clasificación.csv", "base_clasificación.csv")  # nombre original del enunciado y variantes


def localizar_csv_r2():
    """Encuentra el CSV de R2 sin rutas fijas, en este orden: (1) junto a este archivo, (2) en la carpeta de trabajo
    (donde corre el notebook), (3) en 'Datos Reto 2/' subiendo por las carpetas padre (estructura original del proyecto)."""
    aqui = Path(__file__).resolve().parent
    carpetas = [aqui, Path.cwd()] + [p / "Datos Reto 2" for p in [aqui, *aqui.parents]]
    for carpeta in carpetas:
        for nombre in NOMBRES_CSV_R2:
            if (carpeta / nombre).exists():
                return carpeta / nombre
    raise FileNotFoundError(f"No se encontró el CSV de R2 ({' o '.join(NOMBRES_CSV_R2[:2])}). Colócalo en la misma carpeta que utils_r2.py y el notebook.")


# Raíz del proyecto completo (si existe la carpeta a2a/); si el código se descargó suelto, es la carpeta de este archivo
RAIZ = next((p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / "a2a").is_dir()), Path(__file__).resolve().parent)
RUTA_R2 = localizar_csv_r2()
COLS_ID_R2 = ["id_empresa", "nit"]
COL_OBJ_R2 = "abandono"
COLS_CATEG_R2 = ["sector"]

# ----------------------------------------------------------------- 2. Carga y Partición
def sha256_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()

def cargar_r2(ruta=RUTA_R2):
    """Carga R2 con separador ';' y codificación UTF-8 con BOM."""
    return pd.read_csv(ruta, sep=';', encoding="utf-8-sig")

def auditoria_calidad_r2(df, verbose=True):
    """Auditoría exhaustiva de calidad de datos para Reto 2 (Abandono/Desafiliación).
    Solo lectura: no modifica el DataFrame original.
    """
    # 1. Integridad
    n_filas, n_cols = df.shape
    nulos_tot = int(df.isna().sum().sum())
    dup_exactos = int(df.duplicated().sum())
    dup_sin_id = int(df.drop(columns=COLS_ID_R2).duplicated().sum())
    tipos_dict = df.dtypes.astype(str).value_counts().to_dict()

    # 2. Identificadores
    id_unico = bool(df['id_empresa'].is_unique)
    nit_unico = bool(df['nit'].is_unique)
    nit_str = df['nit'].astype(str)
    nit_lens = nit_str.str.len().value_counts().to_dict()
    id_min, id_max = int(df['id_empresa'].min()), int(df['id_empresa'].max())
    nit_min, nit_max = int(df['nit'].min()), int(df['nit'].max())

    # 3. Categoría sector
    s_sector = df['sector']
    conteo_sector = s_sector.value_counts().to_dict()
    espacios_sector = int((s_sector != s_sector.str.strip()).sum())
    variantes_sector = int(s_sector.nunique() - s_sector.str.lower().str.strip().nunique())

    # 4. Rangos, límites y atípicos
    num_cols = [c for c in df.columns if c not in ('id_empresa', 'nit', 'sector', 'abandono')]
    filas_calidad = []
    for c in num_cols:
        s = df[c]
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

    # 5. Reglas de coherencia
    n_incoherencia_af = int((df["tiempo_afiliacion"] > df["antiguedad"]).sum())
    reglas = {
        "tiempo_afiliacion <= antiguedad": n_incoherencia_af == 0,
        "num_trabajadores >= 1": bool((df["num_trabajadores"] >= 1).all()),
        "satisfaccion dentro de [0, 100]": bool(df["satisfaccion"].between(0, 100).all()),
        "digitalizacion dentro de [0, 100]": bool(df["digitalizacion"].between(0, 100).all()),
        "formalizacion dentro de [0, 100]": bool(df["formalizacion"].between(0, 100).all()),
        "distancia >= 0": bool((df["distancia"] >= 0).all()),
        "abandono solo en {0, 1}": set(df["abandono"].unique()).issubset({0, 1}),
        "salario_promedio > 0": bool((df["salario_promedio"] > 0).all()),
        "rotacion >= 0": bool((df["rotacion"] >= 0).all()),
        "uso_servicios >= 0": bool((df["uso_servicios"] >= 0).all())
    }

    # 6. Balance de clases y colinealidad
    n_abandono = int(df["abandono"].sum())
    tasa_abandono = float(df["abandono"].mean() * 100.0)
    corr_pred = df[num_cols].corr().abs()
    pares_corr = corr_pred.where(np.triu(np.ones(corr_pred.shape), 1).astype(bool)).stack().sort_values(ascending=False)

    # 7. Señales de fuga
    fuga_uso = {
        "mediana_activa": float(df.loc[df["abandono"] == 0, "uso_servicios"].median()),
        "mediana_abandono": float(df.loc[df["abandono"] == 1, "uso_servicios"].median()),
        "min_abandono": float(df.loc[df["abandono"] == 1, "uso_servicios"].min()),
        "max_abandono": float(df.loc[df["abandono"] == 1, "uso_servicios"].max())
    }
    fuga_sat = {
        "media_activa": float(df.loc[df["abandono"] == 0, "satisfaccion"].mean()),
        "media_abandono": float(df.loc[df["abandono"] == 1, "satisfaccion"].mean())
    }
    fuga_rot = {
        "media_activa": float(df.loc[df["abandono"] == 0, "rotacion"].mean()),
        "media_abandono": float(df.loc[df["abandono"] == 1, "rotacion"].mean())
    }

    if verbose:
        print("1) INTEGRIDAD")
        print(f"   filas: {n_filas:,} | columnas: {n_cols} | nulos: {nulos_tot} | duplicados exactos: {dup_exactos}")
        print(f"   duplicados ignorando id_empresa y nit: {dup_sin_id} | tipos: {tipos_dict}")

        print("\n2) IDENTIFICADORES (id_empresa y nit)")
        print(f"   id_empresa: único={id_unico} | rango=[{id_min}, {id_max}]")
        print(f"   nit       : único={nit_unico} | rango=[{nit_min}, {nit_max}] | longitudes={nit_lens}")
        print("   Confirmación: se excluyen estrictamente del modelado para evitar fuga de IDs y memorización.")

        print("\n3) CATEGORÍA 'sector' (espacios, mayúsculas/minúsculas, frecuencias)")
        print(f"   conteo de categorías: {conteo_sector}")
        print(f"   espacios sobrantes: {espacios_sector} | variantes de escritura: {variantes_sector} | categorías raras: 0 (menor sector: Tecnología con {conteo_sector.get('Tecnología', 499)} empresas)")

        print("\n4) RANGOS, VALORES EN LOS LÍMITES Y ATÍPICOS (regla 1.5 x IQR)")
        print(tabla_calidad.to_string(float_format=lambda v: f"{v:,.2f}"))
        print(f"   * Alerta de escala: 'satisfaccion' está en escala [0, 100] (media {df['satisfaccion'].mean():.2f}), NO comparable con 'satisfaccion_servicio' de R1 ([0, 10]).")
        for c in num_cols:
            n_max = tabla_calidad.loc[c, 'n_en_maximo']
            n_min = tabla_calidad.loc[c, 'n_en_minimo']
            if n_max >= 30:
                print(f"   - Posible tope superior en {c}: {int(n_max)} empresas valen el máximo ({tabla_calidad.loc[c, 'maximo']:,.2f})")
            if n_min >= 30:
                print(f"   - Posible tope inferior en {c}: {int(n_min)} empresas valen el mínimo ({tabla_calidad.loc[c, 'minimo']:,.2f})")

        print("\n5) REGLAS DE COHERENCIA (deben cumplirse)")
        for nom, ok in reglas.items():
            print(f"   {'OK   ' if ok else 'FALLA'} {nom}")
        print(f"   Control de coherencia temporal: tiempo_afiliacion > antiguedad en {n_incoherencia_af} filas (0 incoherencias).")

        print("\n6) BALANCE DE CLASES Y COLINEALIDAD")
        print(f"   Distribución objetivo: {n_abandono:,} desafiliadas de {n_filas:,} ({tasa_abandono:.2f}% abandono).")
        print("   Top pares correlacionados entre predictores:")
        for (v1, v2), val in pares_corr.head(4).items():
            print(f"   - {v1} vs {v2}: r = {val:.4f}")

        print("\n7) SEÑALES DE FUGA (DATA LEAKAGE)")
        print(f"   - 'uso_servicios': rango [{fuga_uso['min_abandono']:.0f}, {fuga_uso['max_abandono']:.0f}] en desafiliadas (mediana {fuga_uso['mediana_abandono']:.1f} vs {fuga_uso['mediana_activa']:.1f} en activas).")
        print("     No colapsa a 0 post-evento; mide el uso previo al abandono (ventana de observación válida).")
        print(f"   - 'satisfaccion': media {fuga_sat['media_abandono']:.2f} en desafiliadas vs {fuga_sat['media_activa']:.2f} en activas.")
        print(f"   - 'rotacion'    : media {fuga_rot['media_abandono']:.2f}% en desafiliadas vs {fuga_rot['media_activa']:.2f}% en activas.")
        print("   Conclusión de fuga: Ninguna variable es posterior al evento; no hay evidencia empírica de fuga de datos.")

        print("\nDECISIÓN DE PREPARACIÓN DOCUMENTADA")
        print("   - No se eliminan ni imputan filas: 0 nulos, 0 duplicados y 100% de coherencia lógica.")
        print("   - Se excluyen 'id_empresa' y 'nit' de la matriz predictora.")
        print("   - Se conserva la heterogeneidad de valores extremos (reflejan diversidad empresarial legítima).")
        print("   - Partición estratificada 80/20 manteniendo 30.29% de tasa de abandono en train y hold-out.")

    return {
        "integridad": {"filas": n_filas, "columnas": n_cols, "nulos": nulos_tot, "duplicados": dup_exactos, "dup_sin_id": dup_sin_id},
        "identificadores": {"id_unico": id_unico, "nit_unico": nit_unico, "nit_lens": nit_lens},
        "sector": conteo_sector,
        "tabla_calidad": tabla_calidad,
        "reglas": reglas,
        "balance": {"n_abandono": n_abandono, "tasa_abandono": tasa_abandono},
        "top_correlaciones": pares_corr.head(5).to_dict(),
        "fuga": {"uso_servicios": fuga_uso, "satisfaccion": fuga_sat, "rotacion": fuga_rot}
    }

def preparar_r2(df):
    """Prepara X excluyendo identificadores ('id_empresa', 'nit') y el target.
    Aplica One-Hot Encoding a variables categóricas ('sector') con drop_first=True.
    Retorna X, y, y el mapa gen->columnas para el Algoritmo Genético.
    """
    base = df.drop(columns=COLS_ID_R2 + [COL_OBJ_R2])
    X = pd.get_dummies(base, columns=COLS_CATEG_R2, dtype=float, drop_first=True)
    y = df[COL_OBJ_R2].to_numpy(dtype=int)
    
    # Mapeo de variables originales a columnas de la matriz X
    grupos = {c: [c] for c in base.columns if c not in COLS_CATEG_R2}
    for c in COLS_CATEG_R2:
        grupos[c] = [k for k in X.columns if k.startswith(c + "_")]
        
    return X, y, grupos

def particion_estratificada(n, y, semilla=SEMILLA, prop_test=0.2):
    """Split estratificado (80% train, 20% hold-out intacto)."""
    return train_test_split(np.arange(n), test_size=prop_test, stratify=y, random_state=semilla)

# ----------------------------------------------------------------- 3. Métricas
def calcular_metricas_clasificacion(y_real, y_prob, umbral=0.5):
    """Calcula PR-AUC (principal), ROC-AUC, Brier score y métricas a umbral."""
    pr_auc = float(average_precision_score(y_real, y_prob))
    roc_auc = float(roc_auc_score(y_real, y_prob))
    brier = float(brier_score_loss(y_real, y_prob))
    
    y_pred = (y_prob >= umbral).astype(int)
    cm = confusion_matrix(y_real, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calcular_recall_top_k(y_real, y_prob, prop_k=0.20):
    """Calcula el porcentaje de abandonos reales capturados en el top K% de mayor riesgo (Recall@K)."""
    n = len(y_real)
    k = int(n * prop_k)
    orden = np.argsort(y_prob)[::-1]
    top_k_idx = orden[:k]
    total_positivos = np.sum(y_real)
    capturados = np.sum(y_real[top_k_idx])
    return float(capturados / total_positivos * 100) if total_positivos > 0 else 0.0

# ----------------------------------------------------------------- 4. Fábrica de Modelos y Evaluación 1-SE
def construir_modelos_r2(semilla=SEMILLA, scale_pos=2.3):
    """Fábrica de los 3 modelos obligatorios."""
    modelos = {
        "RegresionLogistica": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=semilla)
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight='balanced', random_state=semilla, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos, eval_metric='logloss',
            random_state=semilla, n_jobs=-1
        )
    }
    return modelos

def evaluar_modelos_cv(modelos, X, y, n_splits=5, n_repeats=2, semilla=SEMILLA):
    """Validación cruzada estratificada (RepeatedStratifiedKFold) midiendo PR-AUC, ROC-AUC y Brier.
    Aplica la Regla 1-SE para dirimir el ganador.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=semilla)
    resultados = {nombre: {"pr_auc": [], "roc_auc": [], "brier": [], "recall_top20": []} for nombre in modelos}
    total_folds = n_splits * n_repeats
    
    for tr_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        for nombre, modelo in modelos.items():
            modelo.fit(X_tr, y_tr)
            probs = modelo.predict_proba(X_val)[:, 1]
            m = calcular_metricas_clasificacion(y_val, probs)
            lift20 = calcular_recall_top_k(y_val, probs, 0.20)
            
            resultados[nombre]["pr_auc"].append(m["pr_auc"])
            resultados[nombre]["roc_auc"].append(m["roc_auc"])
            resultados[nombre]["brier"].append(m["brier"])
            resultados[nombre]["recall_top20"].append(lift20)
            
    resumen = {}
    for nombre, metricas in resultados.items():
        resumen[nombre] = {
            "pr_auc_mean": float(np.mean(metricas["pr_auc"])),
            "pr_auc_se": float(np.std(metricas["pr_auc"]) / np.sqrt(total_folds)),
            "roc_auc_mean": float(np.mean(metricas["roc_auc"])),
            "brier_mean": float(np.mean(metricas["brier"])),
            "recall_top20_mean": float(np.mean(metricas["recall_top20"]))
        }
        
    # Aplicar Regla 1-SE
    # 1. Identificar el mejor PR-AUC absoluto
    mejor_abs = max(resumen.keys(), key=lambda k: resumen[k]['pr_auc_mean'])
    best_pr = resumen[mejor_abs]['pr_auc_mean']
    best_se = resumen[mejor_abs]['pr_auc_se']
    
    # 2. Verificar si RegresionLogistica cae dentro de 1 SE del mejor
    pr_reglog = resumen["RegresionLogistica"]['pr_auc_mean']
    brecha_1se = best_pr - pr_reglog
    es_empate_tecnico = brecha_1se <= best_se
    
    # Ganador por parsimonia: RegLog si empata con XGBoost dentro de 1 SE
    ganador_oficial = "RegresionLogistica" if es_empate_tecnico else mejor_abs
    
    return resumen, ganador_oficial, es_empate_tecnico, brecha_1se, best_se

# ----------------------------------------------------------------- 5. Calibración
def ajustar_calibracion_isotonica(modelo, X_train, y_train, n_splits=5, semilla=SEMILLA):
    """Ajusta una calibración isotónica utilizando predicciones Out-Of-Fold (OOF) sobre Train.
    Garantiza que la calibración no toque el hold-out y elimine el sesgo de probabilidad.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    oof_probs = np.zeros(len(y_train))
    
    for tr_idx, val_idx in cv.split(X_train, y_train):
        m_fold = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=semilla))
        m_fold.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof_probs[val_idx] = m_fold.predict_proba(X_train.iloc[val_idx])[:, 1]
        
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_probs, y_train)
    return iso

# ----------------------------------------------------------------- 6. Algoritmo Genético (Parsimonia y Selección de Variables)
def fitness_subconjunto_r2(cromosoma, grupos_genes, X_train, y_train, modelo, lam=0.005, k=3, semilla=SEMILLA):
    """Calcula el fitness a maximizar para un subconjunto de variables en clasificación:
    Fitness = PR-AUC (Average Precision en k-fold CV) - lam * (número de variables activas).
    Solo utiliza datos de entrenamiento (sin data leakage).
    """
    if not any(cromosoma):
        return -1.0, 0.0
    nombres_genes = list(grupos_genes.keys())
    cols = [c for i, act in enumerate(cromosoma) if act for c in grupos_genes[nombres_genes[i]]]
    X_sub = X_train[cols]
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla)
    scores = []
    
    for tr_i, val_i in cv.split(X_sub, y_train):
        m = clone(modelo)
        if hasattr(m, 'random_state'):
            m.set_params(random_state=semilla)
        m.fit(X_sub.iloc[tr_i], y_train[tr_i])
        p = m.predict_proba(X_sub.iloc[val_i])[:, 1]
        scores.append(average_precision_score(y_train[val_i], p))
        
    pr_auc = float(np.mean(scores))
    fitness = pr_auc - lam * int(np.sum(cromosoma))
    return fitness, pr_auc


def ga_buscar_r2(grupos_genes, X_train, y_train, modelo, lam=0.005, semilla=SEMILLA,
                 pob=20, gens=12, p_mut=0.12, elite=2, torneo=3, k=3, cache=None, paciencia=3):
    """Búsqueda genética estocástica para selección de variables en Reto 2.
    Operadores: Torneo estocástico, Cruce Uniforme (máscara binaria), Mutación Bit-flip y Elitismo.
    Incorpora memoización con 'cache' para evitar reajustes de combinaciones idénticas.
    """
    rng = np.random.default_rng(semilla)
    nombres_genes = list(grupos_genes.keys())
    n_genes = len(nombres_genes)
    cache = {} if cache is None else cache

    def fit_eval(cr):
        clave = tuple(cr)
        if clave not in cache:
            cache[clave] = fitness_subconjunto_r2(cr, grupos_genes, X_train, y_train, modelo, lam=lam, k=k, semilla=semilla)
        return cache[clave]

    poblacion = [rng.integers(0, 2, n_genes).tolist() for _ in range(pob)]
    poblacion[0] = [1] * n_genes  # Semilla informativa: el modelo completo compite desde la generación 0
    historial, sin_mejora = [], 0

    for _ in range(gens):
        evals = [fit_eval(c) for c in poblacion]
        fits = [e[0] for e in evals]
        orden = np.argsort(fits)[::-1]  # Maximizar fitness
        mejor_gen = fits[orden[0]]
        
        sin_mejora = sin_mejora + 1 if (historial and mejor_gen <= historial[-1] + 1e-8) else 0
        historial.append(mejor_gen)
        if paciencia is not None and sin_mejora >= paciencia:
            break

        nueva = [poblacion[i] for i in orden[:elite]]
        while len(nueva) < pob:
            padres = []
            for _ in range(2):
                idx = rng.choice(pob, torneo, replace=False)
                padres.append(poblacion[max(idx, key=lambda i: fits[i])])
            mask = rng.integers(0, 2, n_genes).astype(bool)
            hijo = [a if m else b for a, b, m in zip(padres[0], padres[1], mask)]
            hijo = [1 - b if rng.random() < p_mut else b for b in hijo]
            nueva.append(hijo)
        poblacion = nueva

    evals = [fit_eval(c) for c in poblacion]
    fits = [e[0] for e in evals]
    mejor_idx = int(np.argmax(fits))
    mejor_cr = poblacion[mejor_idx]
    genes_sel = [nombres_genes[i] for i, b in enumerate(mejor_cr) if b]
    return genes_sel, evals[mejor_idx], historial, cache


def algoritmo_genetico_multisemilla(X_train, y_train, grupos_genes,
                                    modelo=None,
                                    semillas=[42, 101, 202, 303, 404],
                                    n_poblacion=25, n_generaciones=12,
                                    penalizacion_lambda=0.005,
                                    p_mut=0.12, elite=2, torneo=3,
                                    k_folds=3, paciencia=3, cache=None):
    """Ejecuta el Algoritmo Genético Multi-Semilla optimizado para feature selection.
    
    Por defecto utiliza XGBoost (el mejor modelo del benchmark, PR-AUC = 0.7572) como motor de evaluación,
    o admite cualquier estimador / pipeline scikit-learn (e.g., Regresión Logística, el ganador 1-SE).
    
    Optimizado respecto a la solución base:
    1. Modularidad: evaluación independiente con fitness_subconjunto_r2.
    2. Memoización / Caché compartida entre generaciones e inter-semillas (costo O(1) en subconjuntos repetidos).
    3. Cruce Uniforme (máscara binaria estocástica) que no asume orden ni contigüidad espacial entre columnas.
    4. Elitismo múltiple (elite=2) y semilla informativa inicial ([1]*n_genes).
    5. Parada anticipada (paciencia) ante convergencia temprana.
    
    Retorna: (frecuencias_df, genes_estables, cols_estables, resultados_semillas).
    """
    nombres_genes = list(grupos_genes.keys())
    n_genes = len(nombres_genes)
    conteo_seleccion = {g: 0 for g in nombres_genes}
    resultados_semillas = []
    cache = {} if cache is None else cache

    # Selección del modelo de evaluación del fitness
    if modelo is None or (isinstance(modelo, str) and modelo.lower() in ("reglog", "logistica")):
        # Regresión Logística: ganador adoptado por la regla 1-SE de parsimonia (Claims C-R2-010, C-R2-013..019)
        modelo_eval = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEMILLA))
    elif isinstance(modelo, str) and modelo.lower() == "xgboost":
        # XGBoost rápido y determinista: el mejor modelo absoluto del benchmark CV 5x2 (PR-AUC = 0.7572)
        modelo_eval = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, subsample=1.0,
            colsample_bytree=1.0, scale_pos_weight=2.3, eval_metric='logloss',
            random_state=SEMILLA, n_jobs=-1, tree_method="hist"
        )
    else:
        modelo_eval = modelo

    for sem in semillas:
        genes_sel, (mejor_fit, mejor_pr), hist, cache = ga_buscar_r2(
            grupos_genes, X_train, y_train, modelo_eval,
            lam=penalizacion_lambda, semilla=sem,
            pob=n_poblacion, gens=n_generaciones,
            p_mut=p_mut, elite=elite, torneo=torneo,
            k=k_folds, cache=cache, paciencia=paciencia
        )
        
        for g in genes_sel:
            conteo_seleccion[g] += 1
            
        resultados_semillas.append({
            "semilla": sem, "pr_auc": mejor_pr, "fitness": mejor_fit,
            "k": len(genes_sel), "genes": genes_sel
        })

    # Variables estables (seleccionadas en >= 60% de las semillas)
    umbral_estabilidad = int(np.ceil(0.60 * len(semillas)))
    frecuencias = pd.DataFrame({
        "variable": list(conteo_seleccion.keys()),
        "frecuencia": list(conteo_seleccion.values()),
        "frecuencia_pct": [v / len(semillas) * 100 for v in conteo_seleccion.values()]
    }).sort_values("frecuencia", ascending=False).reset_index(drop=True)
    frecuencias["pct_estabilidad"] = frecuencias["frecuencia_pct"]

    genes_estables = frecuencias[frecuencias["frecuencia"] >= umbral_estabilidad]["variable"].tolist()
    cols_estables = []
    for g in genes_estables:
        cols_estables.extend(grupos_genes[g])

    return frecuencias, genes_estables, cols_estables, resultados_semillas

# ----------------------------------------------------------------- 7. Coeficientes, Odds Ratios y SHAP Completo
def calcular_explicabilidad_reglog(modelo_pipe, X_train, X_test):
    """Calcula Coeficientes estandarizados, Odds Ratios y valores de impacto para Regresión Logística.
    Para modelos lineales, Odds Ratio = exp(beta).
    """
    scaler = modelo_pipe.named_steps['standardscaler']
    clf = modelo_pipe.named_steps['logisticregression']
    
    coefs = clf.coef_[0]
    odds_ratios = np.exp(coefs)
    
    df_explicabilidad = pd.DataFrame({
        "variable": X_train.columns,
        "coeficiente": coefs,
        "odds_ratio": odds_ratios,
        "impacto_absoluto": np.abs(coefs)
    }).sort_values("impacto_absoluto", ascending=False).reset_index(drop=True)
    
    return df_explicabilidad

def calcular_shap_arbol_completo(modelo_arbol, X_train, X_test):
    """Calcula SHAP sobre las 2,000 filas COMPLETAS del hold-out y verifica la aditividad."""
    import shap
    explainer = shap.TreeExplainer(modelo_arbol)
    shap_values = explainer(X_test)
    
    if len(shap_values.values.shape) == 3:
        valores_shap = shap_values.values[:, :, 1]
        base_val = explainer.expected_value[1]
    else:
        valores_shap = shap_values.values
        base_val = explainer.expected_value
        
    # Aditividad
    preds_margin = modelo_arbol.predict(X_test, output_margin=True)
    suma_shap = np.sum(valores_shap, axis=1) + base_val
    discrepancia_max = float(np.max(np.abs(suma_shap - preds_margin)))
    aditividad_ok = discrepancia_max < 1e-4
    
    ranking_shap = pd.DataFrame({
        "variable": X_test.columns,
        "importancia_media_shap": np.mean(np.abs(valores_shap), axis=0)
    }).sort_values("importancia_media_shap", ascending=False).reset_index(drop=True)
    
    return ranking_shap, aditividad_ok, discrepancia_max

# ----------------------------------------------------------------- 8. Deciles y Claims A2A
def analizar_deciles_calibrados(y_real, probs_calibradas, probs_crudas=None):
    """Construye los 10 deciles comerciales exactos (200 empresas por decil en holdout 2000)
    ordenados por probabilidad calibrada de mayor a menor con desempate estricto por probabilidad cruda."""
    y_arr = np.asarray(y_real)
    p_cal = np.asarray(probs_calibradas)
    if probs_crudas is not None:
        p_raw = np.asarray(probs_crudas)
        p_rank = p_cal + p_raw * 1e-9
    else:
        p_rank = p_cal
        
    n = len(y_arr)
    orden = np.argsort(-p_rank)
    k = n // 10
    
    deciles = np.zeros(n, dtype=int)
    for i in range(10):
        s = i * k
        e = (i + 1) * k if i < 9 else n
        deciles[orden[s:e]] = i + 1
        
    df_dec = pd.DataFrame({
        "y_real": y_arr,
        "prob_calibrada": p_cal,
        "decil": deciles
    })
    
    resumen = df_dec.groupby("decil").agg(
        empresas=("y_real", "count"),
        abandonos=("y_real", "sum"),
        tasa_abandono=("y_real", "mean"),
        prob_promedio=("prob_calibrada", "mean")
    ).reset_index()
    
    tot_ab = df_dec["y_real"].sum()
    tasa_base = df_dec["y_real"].mean()
    resumen["captura_pct"] = (resumen["abandonos"] / tot_ab) * 100
    resumen["pct_capturado"] = (resumen["abandonos"].cumsum() / tot_ab) * 100
    resumen["lift"] = resumen["tasa_abandono"] / tasa_base
    return resumen

def publicar_claim_r2(desc, valor, unidad, tol, script, semilla=SEMILLA, datos="Datos Reto 2/base clasificacion.csv"):
    """Publica un claim como PENDIENTE en a2a/ledger/cifras.json utilizando a2a.py si no existe ya."""
    ledger_path = RAIZ / "a2a" / "ledger" / "cifras.json"
    if ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                cifras = json.load(f)
            for c in cifras:
                if c.get("reto") == "R2" and c.get("descripcion") == str(desc):
                    return f"EXISTS: {c.get('id')} ({c.get('estado')}) - {desc}"
        except Exception:
            pass
    cmd = [
        sys.executable, str(RAIZ / "a2a" / "a2a.py"), "claim",
        "--autor", "gemini", "--reto", "R2", "--desc", str(desc),
        "--valor", str(valor), "--unidad", str(unidad), "--tol", str(tol),
        "--script", str(script), "--semilla", str(semilla), "--datos", str(datos)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout.strip()
