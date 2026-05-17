"""Tercera ronda: LightGBM conservador, RF ajustado, ensembles con peso aprendido."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

from experimento_v2 import (
    Receta, build_features_v2, col_features, evaluar, metricas, recursive_forecast,
)
from modelo import cargar_csv


def hacer_h1_rf_baseline() -> Receta:
    return Receta(
        "H1. RF baseline (referencia)",
        mejorados=False,
        constructor=lambda: RandomForestRegressor(
            n_estimators=400, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1),
    )


def hacer_h2_rf_mas_arboles() -> Receta:
    return Receta(
        "H2. RF con mas arboles (1000)",
        mejorados=False,
        constructor=lambda: RandomForestRegressor(
            n_estimators=1000, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1),
    )


def hacer_h3_rf_absolute() -> Receta:
    return Receta(
        "H3. RF criterion=absolute_error",
        mejorados=False,
        constructor=lambda: RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=2,
            criterion="absolute_error",
            random_state=42, n_jobs=-1),
    )


def hacer_h4_rf_mas_profundo() -> Receta:
    return Receta(
        "H4. RF mas grande (n=600, depth=20)",
        mejorados=False,
        constructor=lambda: RandomForestRegressor(
            n_estimators=600, max_depth=20, min_samples_leaf=1,
            random_state=42, n_jobs=-1),
    )


def hacer_h5_lgbm_conservador() -> Receta:
    return Receta(
        "H5. LightGBM ultra-conservador",
        mejorados=False,
        constructor=lambda: lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03,
            num_leaves=8, max_depth=4, min_child_samples=50,
            reg_alpha=1.0, reg_lambda=2.0,
            subsample=0.8, colsample_bytree=0.7,
            random_state=42, n_jobs=-1, verbose=-1),
    )


def hacer_h6_lgbm_v2() -> Receta:
    return Receta(
        "H6. LightGBM conservador + features V2",
        mejorados=True,
        constructor=lambda: lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03,
            num_leaves=8, max_depth=4, min_child_samples=50,
            reg_alpha=1.0, reg_lambda=2.0,
            subsample=0.8, colsample_bytree=0.7,
            random_state=42, n_jobs=-1, verbose=-1),
    )


def hacer_h7_rf_median_ensemble() -> Receta:
    """Ensemble por mediana de 7 RFs con distintas semillas."""
    class MedianEnsemble:
        def __init__(self, n=7):
            self.modelos = [
                RandomForestRegressor(
                    n_estimators=200, max_depth=14, min_samples_leaf=2,
                    random_state=s, n_jobs=-1)
                for s in range(n)
            ]

        def fit(self, X, y):
            for m in self.modelos:
                m.fit(X, y)
            return self

        def predict(self, X):
            preds = np.column_stack([m.predict(X) for m in self.modelos])
            return np.median(preds, axis=1)

    return Receta("H7. RF median ensemble (7 semillas)", mejorados=False,
                  constructor=MedianEnsemble)


def evaluar_blend_naive(receta: Receta, train_df, eval_df, peso_rf: float):
    """Blend ponderado de la receta con el naive semanal."""
    m_rec, pred_rec = evaluar(receta, train_df, eval_df)
    # Naive semanal sobre el eval
    serie = pd.concat([train_df, eval_df])["ventas"].values
    idx_inicio = len(train_df)
    pred_naive = serie[idx_inicio - 7: idx_inicio - 7 + len(eval_df)]
    pred_blend = peso_rf * pred_rec + (1 - peso_rf) * pred_naive
    return metricas(eval_df["ventas"].values, pred_blend), pred_blend


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

    recetas = [
        hacer_h1_rf_baseline(),
        hacer_h2_rf_mas_arboles(),
        hacer_h3_rf_absolute(),
        hacer_h4_rf_mas_profundo(),
        hacer_h5_lgbm_conservador(),
        hacer_h6_lgbm_v2(),
        hacer_h7_rf_median_ensemble(),
    ]

    print("=" * 110)
    print("CV de recetas individuales:")
    print("=" * 110)
    resumen = {}
    for r in recetas:
        maes, mapes = [], []
        for ini, fin in folds:
            train = df.iloc[:ini].reset_index(drop=True)
            ev = df.iloc[ini:fin].reset_index(drop=True)
            m, _ = evaluar(r, train, ev)
            maes.append(m["MAE"]); mapes.append(m["MAPE"])
        resumen[r.nombre] = (maes, mapes)
        print(f"  {r.nombre:48s} MAE GBP {np.mean(maes):>7,.0f} (+/- {np.std(maes):>5,.0f})  "
              f"MAPE {np.mean(mapes):>5.1f}%")

    # Blend con naive semanal: probar varios pesos
    print()
    print("=" * 110)
    print("BLENDS receta + naive semanal (peso optimizado en CV):")
    print("=" * 110)
    receta_base = hacer_h1_rf_baseline()
    for peso in [0.5, 0.6, 0.7, 0.8, 0.9]:
        maes, mapes = [], []
        for ini, fin in folds:
            train = df.iloc[:ini].reset_index(drop=True)
            ev = df.iloc[ini:fin].reset_index(drop=True)
            m, _ = evaluar_blend_naive(receta_base, train, ev, peso)
            maes.append(m["MAE"]); mapes.append(m["MAPE"])
        nombre = f"BLEND H1 + naive (peso RF={peso:.1f})"
        resumen[nombre] = (maes, mapes)
        print(f"  {nombre:48s} MAE GBP {np.mean(maes):>7,.0f} (+/- {np.std(maes):>5,.0f})  "
              f"MAPE {np.mean(mapes):>5.1f}%")

    print()
    print("=" * 110)
    print("RANKING TOTAL:")
    print("=" * 110)
    ord_ = sorted([(n, float(np.mean(m)), float(np.mean(p))) for n, (m, p) in resumen.items()],
                  key=lambda x: x[1])
    for i, (n, mae, mape) in enumerate(ord_, 1):
        marca = " <- GANADOR" if i == 1 else ""
        print(f"  {i}. {n:48s} MAE GBP {mae:>7,.0f}   MAPE {mape:>5.1f}%{marca}")


if __name__ == "__main__":
    main()
