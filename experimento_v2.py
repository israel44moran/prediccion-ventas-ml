"""Segunda ronda de experimentacion. Mismo protocolo de 6-fold rolling CV.

Recetas nuevas a evaluar (todas honestas, sin tocar el test):

  G1. RF actual (campeon de la ronda anterior) — referencia
  G2. RF + features mejorados (EMA, lag_7/14/21/28 mean, pendiente, dias-a-feriado)
  G3. LightGBM con los mismos features que G2
  G4. LightGBM tuned via random search con TimeSeriesSplit interno
  G5. Stacking (RF + LightGBM) con ridge meta-learner
  G6. G4 + winsorizacion del target en entrenamiento (cap al p99)
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

import holidays
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from modelo import cargar_csv

UK_HOLIDAYS = holidays.UK()
UK_HOLIDAY_DATES = sorted(d for d in UK_HOLIDAYS.keys()
                          if 2008 <= d.year <= 2013)


# ---------------------------------------------------------------------------
# Features (V2 — agrega EMA, medias por mismo-dia-de-semana, pendiente, dist-feriado)
# ---------------------------------------------------------------------------
def _dias_a_feriado(ts) -> int:
    d = ts.date()
    pos = [(h - d).days for h in UK_HOLIDAY_DATES if (h - d).days >= 0]
    return min(pos) if pos else 365


def build_features_v2(df: pd.DataFrame, mejorados: bool = True) -> pd.DataFrame:
    out = df.copy()
    f = out["fecha"]

    # Calendario (igual que receta B)
    out["dow"] = f.dt.dayofweek
    out["dom"] = f.dt.day
    out["mes"] = f.dt.month
    out["semana"] = f.dt.isocalendar().week.astype(int)
    out["semana_mes"] = ((out["dom"] - 1) // 7 + 1).astype(int)
    out["es_finde"] = (out["dow"] >= 5).astype(int)
    out["es_sabado"] = (out["dow"] == 5).astype(int)
    out["es_quincena"] = out["dom"].isin([14, 15, 16, 29, 30, 31, 1]).astype(int)
    out["es_diciembre"] = (out["mes"] == 12).astype(int)
    out["es_pre_navidad"] = (
        (out["mes"] == 12) & (out["dom"] >= 15) & (out["dom"] <= 24)
    ).astype(int)
    out["dias_desde_inicio"] = (f - f.min()).dt.days
    out["es_feriado"] = f.dt.date.map(lambda d: d in UK_HOLIDAYS).astype(int)
    out["es_pre_feriado"] = (
        (f + pd.Timedelta(days=1)).dt.date.map(lambda d: d in UK_HOLIDAYS).astype(int)
    )
    out["mes_x_dow"] = out["mes"] * 7 + out["dow"]
    out["mes_x_finde"] = out["mes"] * out["es_finde"]

    # Rezagos cortos + anuales
    for k in [1, 7, 14, 30, 364, 365]:
        out[f"lag_{k}"] = out["ventas"].shift(k)

    # Estadisticos moviles
    for w in [7, 30]:
        out[f"media_{w}"] = out["ventas"].shift(1).rolling(w).mean()
        out[f"std_{w}"] = out["ventas"].shift(1).rolling(w).std()
        out[f"max_{w}"] = out["ventas"].shift(1).rolling(w).max()
    out["mediana_30"] = out["ventas"].shift(1).rolling(30).median()

    # === FEATURES NUEVOS DE V2 ===
    if mejorados:
        # Media de los ultimos 4 mismos-dias-de-semana (lag_7,14,21,28)
        out["media_4_dow"] = (
            out["ventas"].shift(7) + out["ventas"].shift(14)
            + out["ventas"].shift(21) + out["ventas"].shift(28)
        ) / 4.0
        # Mediana de los ultimos 4 mismos-dias-de-semana (robusto a outliers)
        out["mediana_4_dow"] = pd.concat(
            [out["ventas"].shift(s) for s in [7, 14, 21, 28]], axis=1
        ).median(axis=1)
        # Exponential moving averages
        ventas_shift = out["ventas"].shift(1)
        out["ema_7"] = ventas_shift.ewm(span=7, adjust=False).mean()
        out["ema_30"] = ventas_shift.ewm(span=30, adjust=False).mean()
        # Pendiente lineal sobre los ultimos 30 dias
        idx = np.arange(30)

        def _slope(serie):
            if serie.isna().any():
                return 0.0
            return float(np.polyfit(idx, serie.values, 1)[0])

        out["pendiente_30"] = ventas_shift.rolling(30).apply(_slope, raw=False)
        # Razon de volatilidad: std reciente vs std mensual
        out["vol_ratio"] = out["std_7"] / (out["std_30"] + 1e-6)
        # Dias hasta el proximo feriado UK
        out["dias_a_feriado"] = f.map(_dias_a_feriado).clip(upper=60)

    return out


def col_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("fecha", "ventas")]


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def metricas(real: np.ndarray, pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(real - pred)))
    rmse = float(np.sqrt(np.mean((real - pred) ** 2)))
    mask = real > 100
    mape = float(np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100) if mask.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ---------------------------------------------------------------------------
# Pronostico recursivo
# ---------------------------------------------------------------------------
def recursive_forecast(
    modelo, historico: pd.DataFrame, fechas_futuro: pd.DatetimeIndex,
    mejorados: bool, cols_train: list[str],
) -> np.ndarray:
    hist = historico.copy()
    preds = []
    for fecha in fechas_futuro:
        nuevo = pd.concat(
            [hist, pd.DataFrame([{"fecha": fecha, "ventas": np.nan}])], ignore_index=True
        )
        feat = build_features_v2(nuevo, mejorados=mejorados).iloc[[-1]]
        X = feat[cols_train].ffill().fillna(0.0)
        pred = max(0.0, float(modelo.predict(X)[0]))
        preds.append(pred)
        hist = pd.concat([hist, pd.DataFrame([{"fecha": fecha, "ventas": pred}])], ignore_index=True)
    return np.array(preds)


# ---------------------------------------------------------------------------
# Recetas
# ---------------------------------------------------------------------------
@dataclass
class Receta:
    nombre: str
    mejorados: bool
    constructor: Callable
    winsorizar: bool = False
    es_stacking: bool = False


def hacer_g1_rf_actual() -> Receta:
    return Receta(
        "G1. RF actual (campeon previo)",
        mejorados=False,
        constructor=lambda: RandomForestRegressor(
            n_estimators=400, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1),
    )


def hacer_g2_rf_features_v2() -> Receta:
    return Receta(
        "G2. RF + features V2",
        mejorados=True,
        constructor=lambda: RandomForestRegressor(
            n_estimators=400, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1),
    )


def hacer_g3_lgbm_default() -> Receta:
    return Receta(
        "G3. LightGBM (defaults razonables)",
        mejorados=True,
        constructor=lambda: lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=-1,
            num_leaves=31, min_child_samples=20,
            random_state=42, n_jobs=-1, verbose=-1),
    )


def hacer_g4_lgbm_huber() -> Receta:
    return Receta(
        "G4. LightGBM con objective=huber",
        mejorados=True,
        constructor=lambda: lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.04, num_leaves=24,
            min_child_samples=15, max_depth=-1,
            objective="huber", alpha=0.9,
            random_state=42, n_jobs=-1, verbose=-1),
    )


def hacer_g5_lgbm_winsor() -> Receta:
    return Receta(
        "G5. LightGBM + winsorizacion p99",
        mejorados=True,
        constructor=lambda: lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.04, num_leaves=24,
            min_child_samples=15, max_depth=-1,
            random_state=42, n_jobs=-1, verbose=-1),
        winsorizar=True,
    )


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------
def evaluar(receta: Receta, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    feat = build_features_v2(train_df, mejorados=receta.mejorados).dropna().reset_index(drop=True)
    cols = col_features(feat)
    y = feat["ventas"].values.copy()
    if receta.winsorizar:
        cap = np.percentile(y, 99)
        y = np.clip(y, 0, cap)
    modelo = receta.constructor()
    modelo.fit(feat[cols].fillna(0.0), y)
    pred = recursive_forecast(
        modelo, train_df, pd.DatetimeIndex(eval_df["fecha"]),
        mejorados=receta.mejorados, cols_train=cols,
    )
    return metricas(eval_df["ventas"].values, pred), pred


def evaluar_stacking(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """Stacking: RF + LightGBM como base, ridge como meta.

    Meta-learner se entrena con out-of-fold predictions de los 30 dias
    anteriores al eval (usados como nivel-1 train set).
    """
    serie = train_df["ventas"].values

    # Generar predicciones nivel-1 sobre el ultimo bloque del train (los ultimos 30 dias del train).
    n1_size = 30
    if len(train_df) < n1_size + 365 + 30:
        # Fallback: solo LightGBM
        r = hacer_g4_lgbm_huber()
        return evaluar(r, train_df, eval_df)

    train_n1_train = train_df.iloc[:-n1_size].reset_index(drop=True)
    train_n1_eval = train_df.iloc[-n1_size:].reset_index(drop=True)

    # Base 1: RF
    rf = hacer_g2_rf_features_v2()
    _, pred_rf_n1 = evaluar(rf, train_n1_train, train_n1_eval)
    # Base 2: LightGBM Huber
    lg = hacer_g4_lgbm_huber()
    _, pred_lg_n1 = evaluar(lg, train_n1_train, train_n1_eval)

    # Meta
    X_meta = np.column_stack([pred_rf_n1, pred_lg_n1])
    y_meta = train_n1_eval["ventas"].values
    meta = Ridge(alpha=1.0, positive=True)
    meta.fit(X_meta, y_meta)

    # Reentrenar base con TODO el train, predecir sobre eval
    _, pred_rf_eval = evaluar(rf, train_df, eval_df)
    _, pred_lg_eval = evaluar(lg, train_df, eval_df)
    X_eval = np.column_stack([pred_rf_eval, pred_lg_eval])
    pred_final = np.maximum(0.0, meta.predict(X_eval))
    return metricas(eval_df["ventas"].values, pred_final), pred_final


# ---------------------------------------------------------------------------
# Main: 6-fold rolling CV
# ---------------------------------------------------------------------------
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
        hacer_g1_rf_actual(),
        hacer_g2_rf_features_v2(),
        hacer_g3_lgbm_default(),
        hacer_g4_lgbm_huber(),
        hacer_g5_lgbm_winsor(),
    ]

    print("=" * 110)
    print("CROSS-VALIDATION ROLLING (6 folds) — V2")
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
        print(f"  {r.nombre:48s} MAE_avg GBP {np.mean(maes):>7,.0f} (+/- {np.std(maes):>5,.0f})  "
              f"MAPE_avg {np.mean(mapes):>5.1f}%")

    # G6: Stacking
    maes, mapes = [], []
    for ini, fin in folds:
        train = df.iloc[:ini].reset_index(drop=True)
        ev = df.iloc[ini:fin].reset_index(drop=True)
        m, _ = evaluar_stacking(train, ev)
        maes.append(m["MAE"]); mapes.append(m["MAPE"])
    resumen["G6. Stacking RF + LGBM (ridge)"] = (maes, mapes)
    print(f"  {'G6. Stacking RF + LGBM (ridge)':48s} MAE_avg GBP {np.mean(maes):>7,.0f} (+/- {np.std(maes):>5,.0f})  "
          f"MAPE_avg {np.mean(mapes):>5.1f}%")

    print()
    print("=" * 110)
    print("RANKING POR MAE PROMEDIO:")
    print("=" * 110)
    ord_ = sorted([(n, float(np.mean(m)), float(np.mean(p))) for n, (m, p) in resumen.items()],
                  key=lambda x: x[1])
    for i, (n, mae, mape) in enumerate(ord_, 1):
        marca = " <- GANADOR" if i == 1 else ""
        print(f"  {i}. {n:48s} MAE GBP {mae:>7,.0f}   MAPE {mape:>5.1f}%{marca}")


if __name__ == "__main__":
    main()
