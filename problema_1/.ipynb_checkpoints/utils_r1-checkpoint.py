"""utils_r1.py — toda la lógica del RETO 1 (autocontenida: se lee sin abrir R2 ni R3).

r1_modelo.py es una bitácora lineal por fases CRISP-DM que solo llama estas funciones.
Secciones: 1 constantes · 2 datos y partición · 3 métricas · 4 validación cruzada · 5 modelos ·
6 algoritmo genético (parsimonia) · 7 incertidumbre conformal · 8 SHAP · 9 ledger A2A (solo PENDIENTE).
Autor: claude. Las funciones de GA/SHAP/ledger se duplican a propósito en los utils de los otros retos (decisión del usuario: código separado por reto).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedKFold, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------- 1. constantes
SEMILLA = 42
NOMBRES_CSV_R1 = ("empresas afiliadas.csv", "empresas_afiliadas.csv")  # nombre original del enunciado y variante con guion bajo


def localizar_csv_r1():
    """Encuentra el CSV de R1 sin rutas fijas, en este orden: (1) junto a este archivo, (2) en la carpeta de trabajo
    (donde corre el notebook), (3) en 'Datos Reto 1/' subiendo por las carpetas padre (estructura original del proyecto)."""
    aqui = Path(__file__).resolve().parent
    carpetas = [aqui, Path.cwd()] + [p / "Datos Reto 1" for p in [aqui, *aqui.parents]]
    for carpeta in carpetas:
        for nombre in NOMBRES_CSV_R1:
            if (carpeta / nombre).exists():
                return carpeta / nombre
    raise FileNotFoundError(f"No se encontró el CSV de R1 ({' o '.join(NOMBRES_CSV_R1)}). Colócalo en la misma carpeta que utils_r1.py y el notebook.")


# Raíz del proyecto completo (si existe la carpeta a2a/); si el código se descargó suelto, es la carpeta de este archivo
RAIZ = next((p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / "a2a").is_dir()), Path(__file__).resolve().parent)
RUTA_R1 = localizar_csv_r1()
COLS_CATEG_R1 = ["ciudad", "sector"]
COL_ID_R1, COL_OBJ_R1 = "id_empresa", "aporte_mensual"


# ----------------------------------------------------------------- 2. datos y partición
def sha256_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_r1(ruta=RUTA_R1):
    """Lee R1 (sep ',', BOM). El orden de filas del archivo es parte de la reproducibilidad del split."""
    return pd.read_csv(ruta, encoding="utf-8-sig")


def preparar_r1(df):
    """Devuelve X (one-hot de ciudad y sector, sin id ni objetivo), y_log y el mapa gen->columnas.

    Un 'gen' del algoritmo genético es una variable original; ciudad y sector agrupan sus dummies.
    """
    base = df.drop(columns=[COL_ID_R1, COL_OBJ_R1])
    X = pd.get_dummies(base, columns=COLS_CATEG_R1, dtype=float)
    grupos = {c: [c] for c in base.columns if c not in COLS_CATEG_R1}
    for c in COLS_CATEG_R1:
        grupos[c] = [k for k in X.columns if k.startswith(c + "_")]
    return X, np.log(df[COL_OBJ_R1].to_numpy(dtype=float)), grupos


def particion_holdout(n, semilla=SEMILLA, prop_test=0.2):
    """Índices (train, test) idénticos a train_test_split(range(n), test_size=0.2, random_state=42)."""
    return train_test_split(np.arange(n), test_size=prop_test, random_state=semilla)


# ----------------------------------------------------------------- 3. métricas (en COP)
def metricas_cop(y_log_real, y_log_pred):
    """MAE/RMSE en COP, WAPE y MAPE (%) tras volver de la escala log con exp()."""
    r, p = np.exp(y_log_real), np.exp(y_log_pred)
    err = p - r
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "wape": float(np.sum(np.abs(err)) / np.sum(r) * 100),
            "mape": float(np.mean(np.abs(err) / r) * 100)}


# ----------------------------------------------------------------- 5. modelos
class MedianaPorSector:
    """Baseline: predice (en log) la mediana del log-aporte del sector visto en el entrenamiento."""

    def __init__(self, columnas_sector):
        self.columnas_sector = columnas_sector

    def _sector(self, X):
        return X[self.columnas_sector].to_numpy().argmax(axis=1)

    def fit(self, X, y):
        s = self._sector(X)
        self.med_ = {k: float(np.median(y[s == k])) for k in np.unique(s)}
        self.global_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.array([self.med_.get(k, self.global_) for k in self._sector(X)])


def construir_modelos_r1(semilla=SEMILLA, columnas_sector=None, n_jobs=8):
    """Fábrica de modelos (hiperparámetros fijados a priori, NO afinados con el hold-out)."""
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    m = {
        "Ridge(log)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "RandomForest(log)": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                                   random_state=semilla, n_jobs=n_jobs),
        "XGBoost(log)": XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                                     colsample_bytree=0.8, random_state=semilla, n_jobs=n_jobs,
                                     tree_method="hist", verbosity=0),
        "LightGBM(log)": LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15, subsample=0.8,
                                       subsample_freq=1, colsample_bytree=0.8, random_state=semilla,
                                       n_jobs=n_jobs, verbose=-1),
    }
    return m


def clonar(modelo):
    from sklearn.base import clone
    return clone(modelo)


# ----------------------------------------------------------------- 4. validación cruzada
def cv_cop(modelo, X, y_log, repeticiones=2, k=5, semilla=SEMILLA, columnas=None):
    """CV repetida kxrepeticiones sobre X/y (ya restringidos al TRAIN). Devuelve media, error estándar y folds.

    El modelo se reajusta en cada fold (sin fuga). Para MedianaPorSector se pasa el objeto con fit/predict propio.
    """
    Xc = X if columnas is None else X[columnas]
    maes = []
    for tr, va in RepeatedKFold(n_splits=k, n_repeats=repeticiones, random_state=semilla).split(Xc):
        m = clonar(modelo) if hasattr(modelo, "get_params") else MedianaPorSector(modelo.columnas_sector)
        m.fit(Xc.iloc[tr], y_log[tr])
        maes.append(metricas_cop(y_log[va], m.predict(Xc.iloc[va]))["mae"])
    maes = np.array(maes)
    return {"mae_media": float(maes.mean()), "mae_sd": float(maes.std(ddof=1)),
            "mae_se": float(maes.std(ddof=1) / np.sqrt(len(maes))), "n_folds": int(len(maes))}


def elegir_ganador_1se(tabla, preferencia_simplicidad):
    """Regla 1-SE: entre los modelos dentro de 1 SE del mejor, gana el más simple según la lista de preferencia."""
    mejor = min(tabla, key=lambda k: tabla[k]["mae_media"])
    umbral = tabla[mejor]["mae_media"] + tabla[mejor]["mae_se"]
    candidatos = [k for k in tabla if tabla[k]["mae_media"] <= umbral]
    return min(candidatos, key=lambda k: preferencia_simplicidad.index(k)), mejor


# ----------------------------------------------------------------- 6. algoritmo genético (parsimonia)
def fitness_subconjunto(genes_activos, grupos, X, y_log, modelo, lam, k=3, semilla=SEMILLA):
    """Fitness a MINIMIZAR = MAE_CV (millones COP) + lam * (nº de variables). Solo usa datos de TRAIN."""
    cols = [c for g in genes_activos for c in grupos[g]]
    if not cols:
        return 1e9
    r = cv_cop(modelo, X, y_log, repeticiones=1, k=k, semilla=semilla, columnas=cols)
    return r["mae_media"] / 1e6 + lam * len(genes_activos)


def ga_buscar(grupos, X, y_log, modelo, lam, semilla, pob=20, gens=12, p_mut=0.12, elite=2, torneo=3,
              cache=None):
    """GA binario (variable dentro/fuera). Operadores: torneo, cruce uniforme, mutación bit-flip, elitismo.

    Devuelve (mejor_cromosoma_como_lista_de_genes, historial_mejor_fitness, cache). El cache evita recalcular
    subconjuntos repetidos (mismo X, y, modelo y lam => mismo fitness).
    """
    rng = np.random.default_rng(semilla)
    genes = list(grupos)
    cache = {} if cache is None else cache

    def fit(cr):
        clave = tuple(cr)
        if clave not in cache:
            act = [g for g, b in zip(genes, cr) if b]
            cache[clave] = fitness_subconjunto(act, grupos, X, y_log, modelo, lam)
        return cache[clave]

    poblacion = [rng.integers(0, 2, len(genes)).tolist() for _ in range(pob)]
    poblacion[0] = [1] * len(genes)  # semilla informativa: el modelo completo compite desde el inicio
    historial = []
    for _ in range(gens):
        puntajes = [fit(c) for c in poblacion]
        orden = np.argsort(puntajes)
        historial.append(float(puntajes[orden[0]]))
        nueva = [poblacion[i] for i in orden[:elite]]
        while len(nueva) < pob:
            padres = []
            for _ in range(2):
                idx = rng.choice(pob, torneo, replace=False)
                padres.append(poblacion[min(idx, key=lambda i: puntajes[i])])
            mask = rng.integers(0, 2, len(genes)).astype(bool)
            hijo = [a if m else b for a, b, m in zip(padres[0], padres[1], mask)]
            hijo = [1 - b if rng.random() < p_mut else b for b in hijo]
            nueva.append(hijo)
        poblacion = nueva
    puntajes = [fit(c) for c in poblacion]
    mejor = poblacion[int(np.argmin(puntajes))]
    return [g for g, b in zip(genes, mejor) if b], historial, cache


def ga_buscar_rapido(grupos, X, y_log, modelo, lam, semilla, pob=20, gens=12, p_mut=0.12, elite=2, torneo=3,
                     cache=None, n_jobs=8, paciencia=3):
    """Misma búsqueda que ga_buscar (mismo generador aleatorio => mismas trayectorias), más rápida:
    - n_jobs: evalúa en paralelo (hilos) los individuos nuevos de cada generación. XGBoost/LightGBM liberan el GIL; usar
      modelos con n_jobs=1 para no saturar el procesador.
    - paciencia: se detiene si el mejor fitness no mejora durante esas generaciones seguidas (None = sin parada anticipada).
    """
    from joblib import Parallel, delayed
    rng = np.random.default_rng(semilla)
    genes = list(grupos)
    cache = {} if cache is None else cache

    def evaluar_nuevos(poblacion):
        nuevos = {tuple(c): [g for g, b in zip(genes, c) if b] for c in poblacion if tuple(c) not in cache}
        claves = list(nuevos)
        if claves:
            res = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(fitness_subconjunto)(nuevos[k], grupos, X, y_log, modelo, lam) for k in claves)
            cache.update(zip(claves, res))
        return [cache[tuple(c)] for c in poblacion]

    poblacion = [rng.integers(0, 2, len(genes)).tolist() for _ in range(pob)]
    poblacion[0] = [1] * len(genes)
    historial, sin_mejora = [], 0
    for _ in range(gens):
        puntajes = evaluar_nuevos(poblacion)
        orden = np.argsort(puntajes)
        mejor_gen = float(puntajes[orden[0]])
        sin_mejora = sin_mejora + 1 if historial and mejor_gen >= historial[-1] - 1e-12 else 0
        historial.append(mejor_gen)
        if paciencia is not None and sin_mejora >= paciencia:
            break
        nueva = [poblacion[i] for i in orden[:elite]]
        while len(nueva) < pob:
            padres = []
            for _ in range(2):
                idx = rng.choice(pob, torneo, replace=False)
                padres.append(poblacion[min(idx, key=lambda i: puntajes[i])])
            mask = rng.integers(0, 2, len(genes)).astype(bool)
            hijo = [a if m else b for a, b, m in zip(padres[0], padres[1], mask)]
            hijo = [1 - b if rng.random() < p_mut else b for b in hijo]
            nueva.append(hijo)
        poblacion = nueva
    puntajes = evaluar_nuevos(poblacion)
    mejor = poblacion[int(np.argmin(puntajes))]
    return [g for g, b in zip(genes, mejor) if b], historial, cache


def filtro_correlacion(X, y_log, grupos, k):
    """Comparador simple del GA: los k genes con mayor |corr| (el máximo de sus dummies) con el log-objetivo."""
    puntaje = {g: max(abs(np.corrcoef(X[c], y_log)[0, 1]) for c in cols) for g, cols in grupos.items()}
    return sorted(puntaje, key=puntaje.get, reverse=True)[:k]


# ----------------------------------------------------------------- 7. incertidumbre conformal
def conformal_log(modelo, X_tr, y_tr, X_cal, y_cal, X_te, y_te, nivel=0.8, semilla=SEMILLA):
    """Split conformal sobre residuales absolutos en log. Devuelve cobertura empírica y ancho medio (COP) en TEST."""
    m = clonar(modelo)
    m.fit(X_tr, y_tr)
    res = np.abs(y_cal - m.predict(X_cal))
    n = len(res)
    q = float(np.quantile(res, min(1.0, np.ceil((n + 1) * nivel) / n), method="higher"))
    p = m.predict(X_te)
    inf, sup = np.exp(p - q), np.exp(p + q)
    real = np.exp(y_te)
    return {"cobertura": float(np.mean((real >= inf) & (real <= sup)) * 100), "ancho_medio": float(np.mean(sup - inf)),
            "q_log": q, "factor_inf": float(np.exp(-q)), "factor_sup": float(np.exp(q))}


# ----------------------------------------------------------------- 8. SHAP
def shap_arboles(modelo_ajustado, X_eval, grupos):
    """SHAP (TreeExplainer) en datos de evaluación. Devuelve valores, importancia por gen (suma de dummies),
    base y el error máximo de aditividad |base + sum(shap) - prediccion| en escala log."""
    import shap
    expl = shap.TreeExplainer(modelo_ajustado)
    sv = expl.shap_values(X_eval)
    base = float(np.ravel(expl.expected_value)[0])
    pred = modelo_ajustado.predict(X_eval)
    err = float(np.max(np.abs(base + sv.sum(axis=1) - pred)))
    imp_gen = pd.Series(dtype=float)
    for g, cols in grupos.items():
        cols = [c for c in cols if c in X_eval.columns]
        if cols:
            imp_gen[g] = float(np.abs(sv[:, [X_eval.columns.get_loc(c) for c in cols]].sum(axis=1)).mean())
    imp_gen = imp_gen[imp_gen > 0].sort_values(ascending=False)
    return sv, imp_gen, base, err


# ----------------------------------------------------------------- 9. ledger A2A (solo PENDIENTE)
def publicar_claims(candidatos, autor, reto, script, datos, semilla=SEMILLA):
    """Registra claims como PENDIENTE vía a2a.py (nunca VERIFIED; eso lo hace el otro agente). Omite los ya publicados."""
    ledger = RAIZ / "a2a" / "ledger" / "cifras.json"
    existentes = {c["descripcion"] for c in json.loads(ledger.read_text(encoding="utf-8"))} if ledger.exists() else set()
    for c in candidatos:
        if c["desc"] in existentes:
            continue
        subprocess.run([sys.executable, str(RAIZ / "a2a" / "a2a.py"), "claim", "--autor", autor, "--reto", reto,
                        "--desc", c["desc"], "--valor", repr(float(c["valor"])), "--unidad", c["unidad"],
                        "--tol", str(c["tol"]), "--script", script, "--semilla", str(semilla),
                        "--datos", datos], check=True, cwd=RAIZ)
