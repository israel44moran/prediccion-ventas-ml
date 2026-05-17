"""Cuarta ronda: blends sin tuning, target normalizado por dia de semana.

Reglas anti-trampa:
- Pesos del blend FIJOS (0.5 = promedio simple, no se eligen mirando CV).
- Target normalizado: el modelo predice ventas[t] / media_dow[t], y al final
  multiplicamos por la media de ese dia de la semana. Asi el modelo solo
  aprende DESVIACIONES del patron, lo cual estabiliza mucho.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

from experimento_v2 import (
    Receta, build_features_v2, col_features, evaluar, metricas, recursive_forecast,
)
from experimento_v3 import (
    hacer_h1_rf_baseline, hacer_h6_lgbm_v2, evaluar_blend_naive,
)
from modelo import cargar_csv


def evaluar_blend_3way(train_df, eval_df, recetas_y_pesos, peso_naive):
    """Blend de N recetas + naive semanal, todos con pesos fijos."""
    serie_full = pd.concat([train_df, eval_df])["ventas"].values
    idx_inicio = len(train_df)
    pred_naive = serie_full[idx_inicio - 7: idx_inicio - 7 + len(eval_df)]

    preds = [pred_naive * peso_naive]
    for receta, peso in recetas_y_pesos:
        _, pred = evaluar(receta, train_df, eval_df)
        preds.append(pred * peso)
    final = np.sum(preds, axis=0)
    return metricas(eval_df["ventas"].values, final), final


def evaluar_ratio_dow(train_df, eval_df):
    """Modelo que predice ventas[t] / media_dow_historica[t]."""
    # Calcular media por dia-de-semana SOLO sobre el train
    train = train_df.copy()
    train["dow"] = train["fecha"].dt.dayofweek
    media_dow = train.groupby("dow")["ventas"].mean().to_dict()
    # Evitar division por 0
    media_dow = {k: max(1.0, v) for k, v in media_dow.items()}

    # Target = ratio
    train["target_ratio"] = train["ventas"] / train["dow"].map(media_dow)

    # Features V2 sobre el train
    feat_train = build_features_v2(train, mejorados=True).dropna().reset_index(drop=True)
    cols = [c for c in feat_train.columns if c not in ("fecha", "ventas", "target_ratio", "dow")]
    y = feat_train["target_ratio"].values

    modelo = RandomForestRegressor(
        n_estimators=400, max_depth=14, min_samples_leaf=2,
        random_state=42, n_jobs=-1)
    modelo.fit(feat_train[cols].fillna(0.0), y)

    # Recursive forecast: predict ratio, multiply by media_dow
    hist = train_df.copy()
    preds = []
    for fecha in pd.DatetimeIndex(eval_df["fecha"]):
        nuevo = pd.concat([hist, pd.DataFrame([{"fecha": fecha, "ventas": np.nan}])],
                          ignore_index=True)
        feat = build_features_v2(nuevo, mejorados=True).iloc[[-1]]
        X = feat[cols].ffill().fillna(0.0)
        ratio = float(modelo.predict(X)[0])
        dow = fecha.dayofweek
        pred = max(0.0, ratio * media_dow[dow])
        preds.append(pred)
        hist = pd.concat([hist, pd.DataFrame([{"fecha": fecha, "ventas": pred}])],
                         ignore_index=True)
    return metricas(eval_df["ventas"].values, np.array(preds)), np.array(preds)


def main() -> None:
    df = cargar_csv("ventas_reales.csv")
    print(f"Dataset: {len(df)} dias")

    N_FOLDS = 6
    LARGO_FOLD = 30
    folds = []
    for k in range(N_FOLDS, 0, -1):
        idx_fin = len(df) - (k - 1) * LARGO_FOLD
        idx_ini = idx_fin - LARGO_FOLD
        folds.append((idx_ini, idx_fin))

    rf = hacer_h1_rf_baseline()
    lgbm = hacer_h6_lgbm_v2()

    candidatos = []

    # Referencias
    print("=" * 110)
    print("Recetas finalistas y ensembles con pesos FIJOS:")
    print("=" * 110)

    def correr(nombre, fn):
        maes, mapes = [], []
        for ini, fin in folds:
            train = df.iloc[:ini].reset_index(drop=True)
            ev = df.iloc[ini:fin].reset_index(drop=True)
            m, _ = fn(train, ev)
            maes.append(m["MAE"]); mapes.append(m["MAPE"])
        candidatos.append((nombre, float(np.mean(maes)), float(np.mean(mapes)),
                           float(np.std(maes))))
        print(f"  {nombre:55s} MAE GBP {np.mean(maes):>7,.0f} (+/- {np.std(maes):>5,.0f})  "
              f"MAPE {np.mean(mapes):>5.1f}%")

    correr("REF. RF actual (campeon)",
           lambda tr, ev: evaluar(rf, tr, ev))
    correr("REF. LGBM conservador + features V2",
           lambda tr, ev: evaluar(lgbm, tr, ev))
    correr("NEW. RF sobre ratio vs media_dow",
           evaluar_ratio_dow)
    correr("BLEND simple: 0.5 RF + 0.5 naive",
           lambda tr, ev: evaluar_blend_naive(rf, tr, ev, 0.5))
    correr("BLEND simple: 0.5 RF + 0.5 LGBM (sin naive)",
           lambda tr, ev: evaluar_blend_3way(tr, ev, [(rf, 0.5), (lgbm, 0.5)], 0.0))
    correr("BLEND 3-way: 1/3 RF + 1/3 LGBM + 1/3 naive",
           lambda tr, ev: evaluar_blend_3way(tr, ev, [(rf, 1/3), (lgbm, 1/3)], 1/3))
    correr("BLEND 3-way: 0.5 RF + 0.25 LGBM + 0.25 naive",
           lambda tr, ev: evaluar_blend_3way(tr, ev, [(rf, 0.5), (lgbm, 0.25)], 0.25))
    correr("BLEND 2-way: 0.5 ratio_dow + 0.5 RF",
           lambda tr, ev: (
               lambda r1, r2: (
                   metricas(ev["ventas"].values, (r1 + r2) / 2),
                   (r1 + r2) / 2
               )
           )(evaluar_ratio_dow(tr, ev)[1], evaluar(rf, tr, ev)[1]))

    print()
    print("=" * 110)
    print("RANKING:")
    print("=" * 110)
    candidatos.sort(key=lambda x: x[1])
    for i, (nombre, mae, mape, std) in enumerate(candidatos, 1):
        marca = " <- GANADOR" if i == 1 else ""
        print(f"  {i}. {nombre:55s} MAE GBP {mae:>7,.0f}  MAPE {mape:>5.1f}%  std {std:>5,.0f}{marca}")


if __name__ == "__main__":
    main()
