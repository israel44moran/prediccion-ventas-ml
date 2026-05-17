"""Sexta ronda: intentar batir al campeon actual (RF + Shrinkage 85/15).

Reglas anti-trampa:
- Pesos fijos con justificacion a priori.
- Paired t-test contra el campeon actual.
- Solo se declara ganador si p<0.10 sobre los 6 folds.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

from experimento_v2 import build_features_v2, col_features, evaluar, metricas, Receta
from modelo import cargar_csv


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def rf_constructor():
    return RandomForestRegressor(
        n_estimators=400, max_depth=14, min_samples_leaf=2,
        random_state=42, n_jobs=-1)


def evaluar_rf(train, ev):
    return evaluar(Receta("RF", mejorados=False, constructor=rf_constructor), train, ev)


def _lag_n(serie_full: np.ndarray, idx0: int, ev_len: int, n: int) -> np.ndarray:
    """Devuelve el lag-n alineado con el periodo de evaluacion."""
    arr = np.zeros(ev_len)
    for i in range(ev_len):
        pos = idx0 + i
        arr[i] = serie_full[pos - n] if pos - n >= 0 else serie_full[0]
    return arr


# ------------------------------------------------------------------
# Candidatos
# ------------------------------------------------------------------
def evaluar_campeon(train, ev):
    """RF + Shrinkage 85/15 con lag_7. El campeon actual."""
    _, pred_rf = evaluar_rf(train, ev)
    serie = pd.concat([train, ev])["ventas"].values
    idx0 = len(train)
    lag_7 = _lag_n(serie, idx0, len(ev), 7)
    pred = np.maximum(0.0, 0.85 * pred_rf + 0.15 * lag_7)
    return metricas(ev["ventas"].values, pred), pred


def evaluar_mediana_anchor(train, ev):
    """Ancla = mediana(lag_7, 14, 21, 28) en lugar de solo lag_7."""
    _, pred_rf = evaluar_rf(train, ev)
    serie = pd.concat([train, ev])["ventas"].values
    idx0 = len(train)
    lags = np.column_stack([_lag_n(serie, idx0, len(ev), k) for k in (7, 14, 21, 28)])
    ancla = np.median(lags, axis=1)
    pred = np.maximum(0.0, 0.85 * pred_rf + 0.15 * ancla)
    return metricas(ev["ventas"].values, pred), pred


def evaluar_doble_ancla(train, ev):
    """0.7 RF + 0.15 lag_7 + 0.15 mediana(lag_14, 21, 28)."""
    _, pred_rf = evaluar_rf(train, ev)
    serie = pd.concat([train, ev])["ventas"].values
    idx0 = len(train)
    lag_7 = _lag_n(serie, idx0, len(ev), 7)
    lags_mensual = np.column_stack([_lag_n(serie, idx0, len(ev), k) for k in (14, 21, 28)])
    ancla_mensual = np.median(lags_mensual, axis=1)
    pred = np.maximum(0.0, 0.7 * pred_rf + 0.15 * lag_7 + 0.15 * ancla_mensual)
    return metricas(ev["ventas"].values, pred), pred


def evaluar_decomposicion(train, ev):
    """Modelo estructural (dow_mean * tendencia) + RF sobre residuales.

    Estructura:
      mu_dow[dow]   = media historica de cada dia-de-semana en el train
      tend_t        = media movil 90d / media global del train
      pred_struct_t = mu_dow[dow(t)] * tend_t

    Luego RF predice y - pred_struct, y la prediccion final es la suma.
    """
    train_x = train.copy()
    train_x["dow"] = train_x["fecha"].dt.dayofweek
    mu_dow = train_x.groupby("dow")["ventas"].mean().to_dict()

    # Tendencia: razon de media movil 90d sobre media global
    media_global = float(train_x["ventas"].mean())
    if media_global < 1.0:
        media_global = 1.0
    train_x["tend"] = train_x["ventas"].rolling(90, min_periods=30).mean() / media_global
    train_x["tend"] = train_x["tend"].fillna(method="bfill").fillna(1.0)
    train_x["pred_struct"] = train_x["dow"].map(mu_dow) * train_x["tend"]
    train_x["residual"] = train_x["ventas"] - train_x["pred_struct"]

    # Entrenar RF sobre residuales
    # Reemplazo ventas por residual y construyo features estandar (los lags
    # se calculan sobre ventas reales, los conservamos)
    feat = build_features_v2(train, mejorados=False).dropna().reset_index(drop=True)
    cols = col_features(feat)
    # Alinear residual con feat (que perdio las primeras filas por NaN de lags)
    train_x_ali = train_x.set_index("fecha").reindex(feat["fecha"]).reset_index()
    y_res = train_x_ali["residual"].values
    modelo = rf_constructor()
    modelo.fit(feat[cols].fillna(0.0), y_res)

    # Predict recursivo sobre eval
    hist = train.copy()
    preds = []
    for fecha in pd.DatetimeIndex(ev["fecha"]):
        nuevo = pd.concat([hist, pd.DataFrame([{"fecha": fecha, "ventas": np.nan}])],
                          ignore_index=True)
        feat_n = build_features_v2(nuevo, mejorados=False).iloc[[-1]]
        X = feat_n[cols].ffill().fillna(0.0)
        res_pred = float(modelo.predict(X)[0])
        dow_t = fecha.dayofweek
        # tendencia ahora = media movil 90 sobre el hist actual
        ultimos_90 = hist["ventas"].tail(90)
        if len(ultimos_90) >= 30:
            tend_t = float(ultimos_90.mean() / media_global)
        else:
            tend_t = 1.0
        struct = mu_dow[dow_t] * tend_t
        pred = max(0.0, struct + res_pred)
        preds.append(pred)
        hist = pd.concat([hist, pd.DataFrame([{"fecha": fecha, "ventas": pred}])],
                         ignore_index=True)
    return metricas(ev["ventas"].values, np.array(preds)), np.array(preds)


def evaluar_decomposicion_shrinkage(train, ev):
    """Decomposicion (estructural + RF residual) y luego shrinkage 85/15."""
    _, pred_dec = evaluar_decomposicion(train, ev)
    serie = pd.concat([train, ev])["ventas"].values
    idx0 = len(train)
    lag_7 = _lag_n(serie, idx0, len(ev), 7)
    pred = np.maximum(0.0, 0.85 * pred_dec + 0.15 * lag_7)
    return metricas(ev["ventas"].values, pred), pred


def evaluar_campeon_mas_deltas(train, ev):
    """Campeon + features de deltas (cambios recientes) en el RF."""
    # Construir features extendidos con deltas
    def build_ext(df):
        out = build_features_v2(df, mejorados=False)
        out["delta_1_8"] = out["ventas"].shift(1) - out["ventas"].shift(8)
        out["delta_7_14"] = out["ventas"].shift(7) - out["ventas"].shift(14)
        out["delta_30"] = out["ventas"].shift(1) - out["ventas"].shift(31)
        return out

    feat = build_ext(train).dropna().reset_index(drop=True)
    cols = [c for c in feat.columns if c not in ("fecha", "ventas")]
    modelo = rf_constructor()
    modelo.fit(feat[cols].fillna(0.0), feat["ventas"])

    hist = train.copy()
    preds_rf = []
    for fecha in pd.DatetimeIndex(ev["fecha"]):
        nuevo = pd.concat([hist, pd.DataFrame([{"fecha": fecha, "ventas": np.nan}])],
                          ignore_index=True)
        feat_n = build_ext(nuevo).iloc[[-1]]
        X = feat_n[cols].ffill().fillna(0.0)
        p = max(0.0, float(modelo.predict(X)[0]))
        preds_rf.append(p)
        hist = pd.concat([hist, pd.DataFrame([{"fecha": fecha, "ventas": p}])],
                         ignore_index=True)
    preds_rf = np.array(preds_rf)

    serie = pd.concat([train, ev])["ventas"].values
    idx0 = len(train)
    lag_7 = _lag_n(serie, idx0, len(ev), 7)
    pred = np.maximum(0.0, 0.85 * preds_rf + 0.15 * lag_7)
    return metricas(ev["ventas"].values, pred), pred


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    df = cargar_csv("ventas_reales.csv")
    print(f"Dataset: {len(df)} dias")

    N_FOLDS = 6
    LARGO = 30
    folds = [(len(df) - (k - 1)*LARGO - LARGO, len(df) - (k - 1)*LARGO)
             for k in range(N_FOLDS, 0, -1)]

    candidatos = {
        "CAMPEON: RF + Shrinkage 85/15 (lag_7)":      evaluar_campeon,
        "D1. Ancla = mediana(lag_7,14,21,28)":         evaluar_mediana_anchor,
        "D2. Doble ancla: 0.7 RF + 0.15 lag_7 + 0.15 med": evaluar_doble_ancla,
        "D3. Decomposicion (estructural + RF residual)":   evaluar_decomposicion,
        "D4. Decomposicion + Shrinkage 85/15":             evaluar_decomposicion_shrinkage,
        "D5. RF + delta features + Shrinkage":             evaluar_campeon_mas_deltas,
    }

    print()
    print("Calculando MAE por fold (puede tardar ~5 min)...")
    print()

    maes_por_candidato = {}
    for nombre, fn in candidatos.items():
        maes = []
        for ini, fin in folds:
            train = df.iloc[:ini].reset_index(drop=True)
            ev = df.iloc[ini:fin].reset_index(drop=True)
            m, _ = fn(train, ev)
            maes.append(m["MAE"])
        maes_por_candidato[nombre] = np.array(maes)
        print(f"  {nombre:55s} per-fold: {[f'{x:>5,.0f}' for x in maes]}")
        print(f"  {'':55s} avg={np.mean(maes):>6,.0f}  std={np.std(maes):>5,.0f}")
        print()

    print("=" * 120)
    print(f"{'CANDIDATO':55s}  {'MAE_avg':>8s}  {'vs CAMP':>8s}  {'p(1-cola)':>9s}  {'wins':>5s}")
    print("=" * 120)
    base = maes_por_candidato["CAMPEON: RF + Shrinkage 85/15 (lag_7)"]
    base_avg = base.mean()
    print(f"{'CAMPEON: RF + Shrinkage 85/15 (lag_7)':55s}  {base_avg:>7,.0f}  {'-':>8s}  {'-':>9s}  {'-':>5s}")
    print()

    for nombre, maes in maes_por_candidato.items():
        if nombre.startswith("CAMPEON"):
            continue
        diff = base - maes
        avg = maes.mean()
        delta_pct = (avg - base_avg) / base_avg * 100
        wins = int((diff > 0).sum())
        if diff.std() < 1e-9:
            p_one = 1.0
        else:
            t, p_two = stats.ttest_rel(base, maes)
            p_one = p_two / 2 if t > 0 else 1 - p_two / 2
        marca = ""
        if p_one < 0.05:
            marca = "  *** SIG p<0.05"
        elif p_one < 0.10:
            marca = "  *  sig p<0.10"
        elif p_one < 0.20:
            marca = "  .  marginal p<0.20"
        print(f"{nombre:55s}  {avg:>7,.0f}  {delta_pct:>+6.1f}%  {p_one:>9.4f}  {wins:>2d}/6{marca}")

    print()
    print("=" * 120)
    print("CONCLUSION:")
    print("=" * 120)
    sig = [
        (n, m.mean())
        for n, m in maes_por_candidato.items()
        if not n.startswith("CAMPEON")
        and m.mean() < base_avg
        and stats.ttest_rel(base, m)[1] / 2 < 0.10
        and (stats.ttest_rel(base, m).statistic > 0)
    ]
    if not sig:
        print("Ningun candidato bate al campeon con significancia estadistica (p<0.10).")
        print("=> RF + Shrinkage 85/15 (lag_7) se queda como ganador.")
    else:
        sig.sort(key=lambda x: x[1])
        print(f"NUEVO CAMPEON: {sig[0][0]}  con MAE GBP {sig[0][1]:,.0f}")


if __name__ == "__main__":
    main()
