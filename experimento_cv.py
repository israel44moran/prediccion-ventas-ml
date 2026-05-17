"""Cross-validation rolling para comparar modelos honestamente.

Con solo 30 dias de test final, una unica medicion tiene varianza alta.
Aqui evaluamos cada receta sobre 6 ventanas distintas de 30 dias
(rolling window forward) y promediamos. Eso da una estimacion mas estable.

Cada fold:
  Entrena con todo lo anterior al fold.
  Predice 30 dias del fold.
  Calcula MAE y MAPE.

Folds (ultimos 6 bloques de 30 dias):
  Fold 1: 2011-06-13 -> 2011-07-12
  Fold 2: 2011-07-13 -> 2011-08-11
  Fold 3: 2011-08-12 -> 2011-09-10
  Fold 4: 2011-09-11 -> 2011-10-10
  Fold 5: 2011-10-11 -> 2011-11-09
  Fold 6: 2011-11-10 -> 2011-12-09  (este es el test "oficial" del README)
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

from experimento import (
    evaluar,
    hacer_baseline_rf,
    hacer_gb_quantile,
    hacer_hgb,
    hacer_rf_log,
    hacer_rf_mejor_features,
    metricas,
)
from modelo import cargar_csv


def evaluar_naive_semanal(serie_completa, eval_real, idx_inicio_eval):
    pred = serie_completa[idx_inicio_eval - 7 : idx_inicio_eval - 7 + len(eval_real)]
    return metricas(eval_real, pred)


def main() -> None:
    df = cargar_csv("ventas_reales.csv")
    serie = df["ventas"].values
    print(f"Dataset: {len(df)} dias, {df['fecha'].min().date()} -> {df['fecha'].max().date()}")
    print()

    N_FOLDS = 6
    LARGO_FOLD = 30

    folds = []
    for k in range(N_FOLDS, 0, -1):
        idx_fin = len(df) - (k - 1) * LARGO_FOLD
        idx_ini = idx_fin - LARGO_FOLD
        folds.append((idx_ini, idx_fin))

    print(f"Folds (cada uno = 30 dias, entrenamiento = todo lo anterior):")
    for i, (ini, fin) in enumerate(folds, 1):
        f_ini = df["fecha"].iloc[ini].date()
        f_fin = df["fecha"].iloc[fin - 1].date()
        marca = "  <- test oficial" if i == N_FOLDS else ""
        print(f"  Fold {i}: {f_ini} -> {f_fin}  (train: {ini} dias){marca}")
    print()

    recetas = {
        "A. RF baseline (actual)": hacer_baseline_rf,
        "B. RF + lag365 + holidays + interacc": hacer_rf_mejor_features,
        "C. RF + (B) + log target": hacer_rf_log,
        "D. HistGB + (C)": hacer_hgb,
        "E. GB mediana + (C)": hacer_gb_quantile,
    }

    print("=" * 110)
    print("CROSS-VALIDATION ROLLING (6 folds)")
    print("=" * 110)

    resultados = {}
    for nombre, factory in recetas.items():
        maes, mapes = [], []
        for ini, fin in folds:
            train = df.iloc[:ini].reset_index(drop=True)
            eval_df = df.iloc[ini:fin].reset_index(drop=True)
            r = factory()
            m, _ = evaluar(r, train, eval_df)
            maes.append(m["MAE"]); mapes.append(m["MAPE"])
        resultados[nombre] = {"MAE": maes, "MAPE": mapes}
        print(f"  {nombre:42s}")
        print(f"    MAE  por fold: {[f'{m:>6,.0f}' for m in maes]}")
        print(f"    MAPE por fold: {[f'{m:>5.1f}%' for m in mapes]}")
        print(f"    PROMEDIO:  MAE GBP {np.mean(maes):>7,.0f} (+/- {np.std(maes):>6,.0f})   "
              f"MAPE {np.mean(mapes):>5.1f}% (+/- {np.std(mapes):>4.1f}%)")
        print()

    # Naive semanal sobre los mismos folds
    maes_n, mapes_n = [], []
    for ini, fin in folds:
        real = serie[ini:fin]
        m = evaluar_naive_semanal(serie, real, ini)
        maes_n.append(m["MAE"]); mapes_n.append(m["MAPE"])
    print(f"  (ref) Naive semanal")
    print(f"    MAE  por fold: {[f'{m:>6,.0f}' for m in maes_n]}")
    print(f"    MAPE por fold: {[f'{m:>5.1f}%' for m in mapes_n]}")
    print(f"    PROMEDIO:  MAE GBP {np.mean(maes_n):>7,.0f} (+/- {np.std(maes_n):>6,.0f})   "
          f"MAPE {np.mean(mapes_n):>5.1f}% (+/- {np.std(mapes_n):>4.1f}%)")
    print()

    print("=" * 110)
    print("RANKING POR MAE PROMEDIO:")
    print("=" * 110)
    todos = [(n, np.mean(r["MAE"]), np.mean(r["MAPE"])) for n, r in resultados.items()]
    todos.append(("(ref) Naive semanal", float(np.mean(maes_n)), float(np.mean(mapes_n))))
    todos.sort(key=lambda x: x[1])
    for i, (nombre, mae_avg, mape_avg) in enumerate(todos, 1):
        marca = " <- GANADOR" if i == 1 else ""
        print(f"  {i}. {nombre:42s} MAE GBP {mae_avg:>7,.0f}   MAPE {mape_avg:>5.1f}%{marca}")


if __name__ == "__main__":
    main()
