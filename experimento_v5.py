"""Quinta ronda: encontrar algo SIMPLE que sea estadisticamente mejor que RF.

Reglas anti-trampa:
- Pesos fijos con justificacion a priori (no se eligen mirando CV).
- Cambios minimos vs el modelo actual.
- Para cada candidato: paired t-test contra RF sobre los mismos 6 folds.
- Solo se declara ganador si p < 0.10 (significancia razonable con n=6).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

from experimento_v2 import (
    build_features_v2, col_features, evaluar, metricas, Receta, recursive_forecast,
)
from modelo import cargar_csv


# ------------------------------------------------------------------
# Candidatos
# ------------------------------------------------------------------
def rf_constructor():
    return RandomForestRegressor(
        n_estimators=400, max_depth=14, min_samples_leaf=2,
        random_state=42, n_jobs=-1)


def evaluar_rf(train, ev):
    r = Receta("RF", mejorados=False, constructor=rf_constructor)
    return evaluar(r, train, ev)


def evaluar_et(train, ev):
    r = Receta("ET", mejorados=False,
               constructor=lambda: ExtraTreesRegressor(
                   n_estimators=400, max_depth=14, min_samples_leaf=2,
                   random_state=42, n_jobs=-1))
    return evaluar(r, train, ev)


def evaluar_hgb_tuneado(train, ev):
    r = Receta("HGB", mejorados=False,
               constructor=lambda: HistGradientBoostingRegressor(
                   max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                   min_samples_leaf=30, l2_regularization=2.0,
                   max_depth=6, random_state=42))
    return evaluar(r, train, ev)


def evaluar_mediana_robusta(train, ev):
    """Mediana de (RF_pred, lag_7, lag_14, lag_21, lag_28).

    Idea: cuando RF se equivoca por mucho en un dia, los 4 lags del mismo
    dia-de-semana lo "anclan" hacia el patron historico. La mediana ignora
    el outlier (sea RF o sea un lag inusual). Cambio minimo, regla fija.
    """
    _, pred_rf = evaluar_rf(train, ev)
    serie_full = pd.concat([train, ev])["ventas"].values
    idx0 = len(train)
    finales = []
    for i in range(len(ev)):
        pos = idx0 + i
        lags = [serie_full[pos - k] for k in (7, 14, 21, 28) if pos - k >= 0]
        valores = [pred_rf[i]] + lags
        finales.append(np.median(valores))
    pred = np.maximum(0.0, np.array(finales))
    return metricas(ev["ventas"].values, pred), pred


def _naive_semanal(train, ev):
    serie_full = pd.concat([train, ev])["ventas"].values
    idx0 = len(train)
    return serie_full[idx0 - 7: idx0 - 7 + len(ev)]


def evaluar_shrinkage(train, ev, peso_rf):
    _, pred_rf = evaluar_rf(train, ev)
    pred_n = _naive_semanal(train, ev)
    pred = np.maximum(0.0, peso_rf * pred_rf + (1 - peso_rf) * pred_n)
    return metricas(ev["ventas"].values, pred), pred


def evaluar_shrinkage_67(train, ev):
    return evaluar_shrinkage(train, ev, 2/3)


def evaluar_shrinkage_75(train, ev):
    return evaluar_shrinkage(train, ev, 3/4)


def evaluar_shrinkage_85(train, ev):
    return evaluar_shrinkage(train, ev, 0.85)


def evaluar_mediana_3(train, ev):
    """Mediana de SOLO (RF, lag_7, mediana_lags_4). Mas simple que la de 5."""
    _, pred_rf = evaluar_rf(train, ev)
    serie_full = pd.concat([train, ev])["ventas"].values
    idx0 = len(train)
    finales = []
    for i in range(len(ev)):
        pos = idx0 + i
        lag_7 = serie_full[pos - 7]
        lags_4 = [serie_full[pos - k] for k in (7, 14, 21, 28) if pos - k >= 0]
        med_4 = float(np.median(lags_4))
        finales.append(float(np.median([pred_rf[i], lag_7, med_4])))
    pred = np.maximum(0.0, np.array(finales))
    return metricas(ev["ventas"].values, pred), pred


# ------------------------------------------------------------------
# Main: CV + paired t-test
# ------------------------------------------------------------------
def main() -> None:
    df = cargar_csv("ventas_reales.csv")
    print(f"Dataset: {len(df)} dias")

    N_FOLDS = 6
    LARGO = 30
    folds = [(len(df) - (k - 1)*LARGO - LARGO, len(df) - (k - 1)*LARGO)
             for k in range(N_FOLDS, 0, -1)]

    candidatos = {
        "REFERENCIA: RF actual":                      evaluar_rf,
        "C1. ExtraTrees (mismo set de features)":     evaluar_et,
        "C2. HistGB tuneado conservador":             evaluar_hgb_tuneado,
        "C3. Mediana(RF, lag_7,14,21,28)":            evaluar_mediana_robusta,
        "C4. Mediana(RF, lag_7, mediana_lags_4)":     evaluar_mediana_3,
        "C5. Shrinkage 2/3 RF + 1/3 naive":           evaluar_shrinkage_67,
        "C6. Shrinkage 3/4 RF + 1/4 naive":           evaluar_shrinkage_75,
        "C7. Shrinkage 0.85 RF + 0.15 naive":         evaluar_shrinkage_85,
    }

    print()
    print("Calculando MAE por fold para cada candidato (puede tardar ~3 min)...")
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
        print(f"  {nombre:50s} per-fold: {[f'{x:>5,.0f}' for x in maes]}")
        print(f"  {'':50s} avg={np.mean(maes):>6,.0f}  std={np.std(maes):>5,.0f}")
        print()

    # Tabla resumen y paired t-test vs RF
    print("=" * 110)
    print(f"{'CANDIDATO':50s}  {'MAE_avg':>8s}  {'vs RF':>8s}  {'p (1-cola)':>10s}  {'wins':>5s}")
    print("=" * 110)
    rf_maes = maes_por_candidato["REFERENCIA: RF actual"]
    rf_avg = rf_maes.mean()
    print(f"{'REFERENCIA: RF actual':50s}  {rf_avg:>7,.0f}  {'-':>8s}  {'-':>10s}  {'-':>5s}")
    print()

    resultados = []
    for nombre, maes in maes_por_candidato.items():
        if nombre.startswith("REFERENCIA"):
            continue
        diff = rf_maes - maes  # positivo = candidato gana
        avg = maes.mean()
        delta_pct = (avg - rf_avg) / rf_avg * 100
        wins = int((diff > 0).sum())
        if diff.std() < 1e-9:
            p_one = 1.0
            t = 0.0
        else:
            t, p_two = stats.ttest_rel(rf_maes, maes)
            p_one = p_two / 2 if t > 0 else 1 - p_two / 2  # H1: candidato < RF
        marca = ""
        if p_one < 0.05:
            marca = "  *** SIG p<0.05"
        elif p_one < 0.10:
            marca = "  *  sig p<0.10"
        resultados.append((nombre, avg, delta_pct, p_one, wins, marca))
        print(f"{nombre:50s}  {avg:>7,.0f}  {delta_pct:>+6.1f}%  {p_one:>10.4f}  {wins:>2d}/6{marca}")

    print()
    print("=" * 110)
    print("CONCLUSION:")
    print("=" * 110)
    sig = [r for r in resultados if r[3] < 0.10 and r[2] < 0]
    if not sig:
        print("Ningun candidato es estadisticamente mejor que RF (p<0.10).")
        print("=> RF se queda como ganador por simplicidad (navaja de Occam).")
    else:
        sig.sort(key=lambda x: x[3])
        nombre, avg, delta, p, wins, _ = sig[0]
        print(f"GANADOR: {nombre}")
        print(f"   MAE GBP {avg:,.0f}  ({delta:+.1f}% vs RF)")
        print(f"   p (1-cola) = {p:.4f}   gana en {wins}/6 folds")


if __name__ == "__main__":
    main()
