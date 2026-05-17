"""Experimentacion honesta para mejorar el modelo.

Protocolo anti-trampa:
- Test final: ultimos 30 dias (NO se tocan hasta el ultimo paso).
- Validacion: 30 dias previos al test.
- Se prueban ideas SOLO sobre validacion.
- Se elige la mejor por MAE en validacion.
- Se reentrena con (train + val) y se reporta el numero sobre test.

Ideas evaluadas:
  A. RF actual (baseline)
  B. RF + lag_365 + UK holidays + interaccion mes*dow
  C. RF + (B) + log-transform del target
  D. HistGradientBoosting + (C)
  E. GradientBoosting quantile loss alpha=0.5 (mediana) + (C)
  F. Ensemble (mejor de los anteriores) + naive semanal
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

import holidays
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from modelo import cargar_csv

UK_HOLIDAYS = holidays.UK()


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, usar_lag365: bool, usar_holidays: bool,
                   usar_interacciones: bool) -> pd.DataFrame:
    out = df.copy()
    f = out["fecha"]
    out["dow"] = f.dt.dayofweek
    out["dom"] = f.dt.day
    out["mes"] = f.dt.month
    out["semana"] = f.dt.isocalendar().week.astype(int)
    out["es_finde"] = (out["dow"] >= 5).astype(int)
    out["es_sabado"] = (out["dow"] == 5).astype(int)
    out["es_quincena"] = out["dom"].isin([14, 15, 16, 29, 30, 31, 1]).astype(int)
    out["es_diciembre"] = (out["mes"] == 12).astype(int)
    out["es_pre_navidad"] = ((out["mes"] == 12) & (out["dom"] >= 15) & (out["dom"] <= 24)).astype(int)
    out["semana_mes"] = ((out["dom"] - 1) // 7 + 1).astype(int)
    out["dias_desde_inicio"] = (f - f.min()).dt.days

    if usar_holidays:
        out["es_feriado"] = f.dt.date.map(lambda d: d in UK_HOLIDAYS).astype(int)
        out["es_pre_feriado"] = (f + pd.Timedelta(days=1)).dt.date.map(lambda d: d in UK_HOLIDAYS).astype(int)

    if usar_interacciones:
        out["mes_x_dow"] = out["mes"] * 7 + out["dow"]  # 0..83
        out["mes_x_finde"] = out["mes"] * out["es_finde"]

    for k in [1, 7, 14, 30]:
        out[f"lag_{k}"] = out["ventas"].shift(k)
    if usar_lag365:
        out["lag_365"] = out["ventas"].shift(365)
        out["lag_364"] = out["ventas"].shift(364)  # mismo dia de la semana hace ~un anio
    for w in [7, 30]:
        out[f"media_{w}"] = out["ventas"].shift(1).rolling(w).mean()
        out[f"std_{w}"] = out["ventas"].shift(1).rolling(w).std()
        out[f"max_{w}"] = out["ventas"].shift(1).rolling(w).max()
    out["mediana_30"] = out["ventas"].shift(1).rolling(30).median()
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
    modelo,
    historico: pd.DataFrame,
    fechas_futuro: pd.DatetimeIndex,
    fb_kwargs: dict,
    target_log: bool,
) -> np.ndarray:
    """Predice dia por dia, alimentando la prediccion al siguiente paso."""
    hist = historico.copy()
    preds = []
    for fecha in fechas_futuro:
        nuevo = pd.concat(
            [hist, pd.DataFrame([{"fecha": fecha, "ventas": np.nan}])], ignore_index=True
        )
        feat = build_features(nuevo, **fb_kwargs).iloc[[-1]]
        cols = col_features(feat)
        X = feat[cols].ffill().fillna(0.0)
        pred = float(modelo.predict(X)[0])
        if target_log:
            pred = float(np.expm1(pred))
        pred = max(0.0, pred)
        preds.append(pred)
        hist = pd.concat([hist, pd.DataFrame([{"fecha": fecha, "ventas": pred}])], ignore_index=True)
    return np.array(preds)


# ---------------------------------------------------------------------------
# Wrappers de modelo
# ---------------------------------------------------------------------------
@dataclass
class Receta:
    nombre: str
    fb_kwargs: dict
    constructor: Callable
    target_log: bool


def hacer_baseline_rf() -> Receta:
    return Receta(
        nombre="A. RF baseline",
        fb_kwargs=dict(usar_lag365=False, usar_holidays=False, usar_interacciones=False),
        constructor=lambda: RandomForestRegressor(n_estimators=400, max_depth=14, min_samples_leaf=2,
                                                   random_state=42, n_jobs=-1),
        target_log=False,
    )


def hacer_rf_mejor_features() -> Receta:
    return Receta(
        nombre="B. RF + lag365 + holidays + interacciones",
        fb_kwargs=dict(usar_lag365=True, usar_holidays=True, usar_interacciones=True),
        constructor=lambda: RandomForestRegressor(n_estimators=400, max_depth=14, min_samples_leaf=2,
                                                   random_state=42, n_jobs=-1),
        target_log=False,
    )


def hacer_rf_log() -> Receta:
    return Receta(
        nombre="C. RF + (B) + log target",
        fb_kwargs=dict(usar_lag365=True, usar_holidays=True, usar_interacciones=True),
        constructor=lambda: RandomForestRegressor(n_estimators=400, max_depth=14, min_samples_leaf=2,
                                                   random_state=42, n_jobs=-1),
        target_log=True,
    )


def hacer_hgb() -> Receta:
    return Receta(
        nombre="D. HistGradientBoosting + (C)",
        fb_kwargs=dict(usar_lag365=True, usar_holidays=True, usar_interacciones=True),
        constructor=lambda: HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=8, min_samples_leaf=10, random_state=42
        ),
        target_log=True,
    )


def hacer_gb_quantile() -> Receta:
    return Receta(
        nombre="E. GradientBoosting (mediana) + (C)",
        fb_kwargs=dict(usar_lag365=True, usar_holidays=True, usar_interacciones=True),
        constructor=lambda: GradientBoostingRegressor(
            loss="quantile", alpha=0.5, n_estimators=400, learning_rate=0.05,
            max_depth=5, min_samples_leaf=10, random_state=42,
        ),
        target_log=True,
    )


# ---------------------------------------------------------------------------
# Evaluacion de una receta
# ---------------------------------------------------------------------------
def evaluar(receta: Receta, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    feat = build_features(train_df, **receta.fb_kwargs).dropna().reset_index(drop=True)
    cols = col_features(feat)
    y_train = np.log1p(feat["ventas"].values) if receta.target_log else feat["ventas"].values
    modelo = receta.constructor()
    modelo.fit(feat[cols].fillna(0.0), y_train)
    pred = recursive_forecast(
        modelo, train_df, pd.DatetimeIndex(eval_df["fecha"]), receta.fb_kwargs, receta.target_log
    )
    real = eval_df["ventas"].values
    return metricas(real, pred), pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = cargar_csv("ventas_reales.csv")
    print(f"Dataset: {len(df)} dias, {df['fecha'].min().date()} -> {df['fecha'].max().date()}")

    # Splits
    test = df.iloc[-30:].reset_index(drop=True)
    val = df.iloc[-60:-30].reset_index(drop=True)
    train = df.iloc[:-60].reset_index(drop=True)
    train_plus_val = df.iloc[:-30].reset_index(drop=True)

    print(f"Train : {train['fecha'].min().date()} -> {train['fecha'].max().date()}  ({len(train)} dias)")
    print(f"Val   : {val['fecha'].min().date()} -> {val['fecha'].max().date()}  ({len(val)} dias)  [se usa para elegir]")
    print(f"Test  : {test['fecha'].min().date()} -> {test['fecha'].max().date()}  ({len(test)} dias)  [INTACTO hasta el final]")
    print()

    recetas = [
        hacer_baseline_rf(),
        hacer_rf_mejor_features(),
        hacer_rf_log(),
        hacer_hgb(),
        hacer_gb_quantile(),
    ]

    print("=" * 95)
    print("FASE 1 — Evaluacion sobre VALIDACION (30 dias previos al test)")
    print("=" * 95)
    resultados = []
    preds_val = {}
    for r in recetas:
        m, p = evaluar(r, train, val)
        resultados.append((r, m))
        preds_val[r.nombre] = p
        print(f"  {r.nombre:50s} MAE GBP {m['MAE']:>8,.0f}  MAPE {m['MAPE']:>5.1f}%")

    # Baseline naive semanal sobre val
    serie = df["ventas"].values
    pred_naive_val = serie[-60-7:-30-7]
    m_naive = metricas(val["ventas"].values, pred_naive_val)
    print(f"  {'(ref) Naive semanal':50s} MAE GBP {m_naive['MAE']:>8,.0f}  MAPE {m_naive['MAPE']:>5.1f}%")

    # Ensemble: best receta + naive
    mejor = min(resultados, key=lambda x: x[1]["MAE"])
    pred_ensemble_val = (preds_val[mejor[0].nombre] + pred_naive_val) / 2
    m_ensemble = metricas(val["ventas"].values, pred_ensemble_val)
    print(f"  {'F. Ensemble (mejor + naive semanal)':50s} MAE GBP {m_ensemble['MAE']:>8,.0f}  MAPE {m_ensemble['MAPE']:>5.1f}%")

    print()
    candidatos = [(r.nombre, m["MAE"]) for r, m in resultados] + \
                 [("F. Ensemble (mejor + naive semanal)", m_ensemble["MAE"])]
    ganador = min(candidatos, key=lambda x: x[1])
    print(f"GANADOR EN VALIDACION: {ganador[0]} con MAE GBP {ganador[1]:,.0f}")
    print()

    # ---------- FASE 2: reentrenar ganador con train+val y evaluar sobre TEST ----------
    print("=" * 95)
    print("FASE 2 — Reentrenar ganador con (train + val) y reportar sobre TEST (jamas visto)")
    print("=" * 95)

    receta_ganadora = next(r for r in recetas if r.nombre == ganador[0]) if ganador[0] != "F. Ensemble (mejor + naive semanal)" \
                      else mejor[0]
    es_ensemble = ganador[0].startswith("F.")

    metricas_test, pred_test = evaluar(receta_ganadora, train_plus_val, test)
    if es_ensemble:
        pred_naive_test = serie[-30-7:-7]
        pred_test = (pred_test + pred_naive_test) / 2
        metricas_test = metricas(test["ventas"].values, pred_test)

    print(f"  {ganador[0]} sobre TEST:")
    print(f"     MAE   GBP {metricas_test['MAE']:>8,.0f}")
    print(f"     RMSE  GBP {metricas_test['RMSE']:>8,.0f}")
    print(f"     MAPE      {metricas_test['MAPE']:>5.1f}%")
    print()

    # Referencia: baseline naive semanal y RF baseline sobre TEST
    print("Para contexto, en el MISMO test:")
    pred_naive_test = serie[-30-7:-7]
    m = metricas(test["ventas"].values, pred_naive_test)
    print(f"  Naive semanal           MAE GBP {m['MAE']:>8,.0f}  MAPE {m['MAPE']:>5.1f}%")

    m_base, _ = evaluar(hacer_baseline_rf(), train_plus_val, test)
    print(f"  RF baseline original    MAE GBP {m_base['MAE']:>8,.0f}  MAPE {m_base['MAPE']:>5.1f}%")

    mejora = (m_base["MAE"] - metricas_test["MAE"]) / m_base["MAE"] * 100
    print()
    print(f"Mejora del ganador vs RF baseline en TEST: {mejora:+.1f}% sobre MAE")


if __name__ == "__main__":
    main()
