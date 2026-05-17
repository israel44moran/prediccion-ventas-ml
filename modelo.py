"""Entrenamiento y pronóstico de ventas diarias.

Dos modelos disponibles:
- RandomForestRegressor (scikit-learn) sobre features de calendario, rezagos,
  estacionalidad anual (lag_365), feriados oficiales del Reino Unido e
  interacciones mes x dia-de-semana.
- Prophet (opcional) si está instalado en el entorno.

La configuracion actual del Random Forest se eligio mediante cross-validation
rolling de 6 folds (ver `experimento_cv.py`); reduce el MAE promedio en ~33%
respecto a la version inicial sin lag_365 ni feriados.

Las funciones públicas son las que consume `dashboard.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from prophet import Prophet  # type: ignore

    PROPHET_DISPONIBLE = True
except Exception:
    PROPHET_DISPONIBLE = False

try:
    import holidays  # type: ignore

    _UK_HOLIDAYS = holidays.UK()
    HOLIDAYS_DISPONIBLE = True
except Exception:
    _UK_HOLIDAYS = None
    HOLIDAYS_DISPONIBLE = False



VENTANAS_LAG = [1, 7, 14, 30]
LAGS_ANUALES = [364, 365]  # 364 mantiene el mismo dia de la semana
ROLLINGS = [7, 30]


@dataclass
class ResultadoModelo:
    nombre: str
    historico: pd.DataFrame  # columnas: fecha, ventas
    validacion: pd.DataFrame  # fecha, real, predicho
    pronostico: pd.DataFrame  # fecha, predicho, low, high
    metricas: dict[str, float]


def cargar_csv(buffer_o_ruta) -> pd.DataFrame:
    df = pd.read_csv(buffer_o_ruta)
    columnas = {c.lower().strip(): c for c in df.columns}
    if "fecha" not in columnas or "ventas" not in columnas:
        raise ValueError("El CSV debe tener columnas 'fecha' y 'ventas'.")
    df = df.rename(columns={columnas["fecha"]: "fecha", columnas["ventas"]: "ventas"})
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["ventas"] = pd.to_numeric(df["ventas"], errors="coerce")
    df = df.dropna(subset=["fecha", "ventas"]).sort_values("fecha").reset_index(drop=True)
    df = df.drop_duplicates(subset=["fecha"], keep="last")
    return df[["fecha", "ventas"]]


def _agregar_features(df: pd.DataFrame, usar_lag_anual: bool = True) -> pd.DataFrame:
    out = df.copy()
    f = out["fecha"]

    # Calendario basico
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

    # Feriados (Reino Unido). Si el paquete no esta instalado, se omiten.
    if HOLIDAYS_DISPONIBLE:
        out["es_feriado"] = f.dt.date.map(lambda d: d in _UK_HOLIDAYS).astype(int)
        out["es_pre_feriado"] = (
            (f + pd.Timedelta(days=1)).dt.date.map(lambda d: d in _UK_HOLIDAYS).astype(int)
        )

    # Interacciones — un viernes de diciembre se comporta distinto a uno de marzo.
    out["mes_x_dow"] = out["mes"] * 7 + out["dow"]
    out["mes_x_finde"] = out["mes"] * out["es_finde"]

    # Rezagos cortos (siempre)
    for k in VENTANAS_LAG:
        out[f"lag_{k}"] = out["ventas"].shift(k)
    # Rezagos anuales (solo si hay datos suficientes — captura estacionalidad anual)
    if usar_lag_anual:
        for k in LAGS_ANUALES:
            out[f"lag_{k}"] = out["ventas"].shift(k)

    # Estadisticos moviles sobre el pasado
    for w in ROLLINGS:
        out[f"media_{w}"] = out["ventas"].shift(1).rolling(w).mean()
        out[f"std_{w}"] = out["ventas"].shift(1).rolling(w).std()
        out[f"max_{w}"] = out["ventas"].shift(1).rolling(w).max()
    out["mediana_30"] = out["ventas"].shift(1).rolling(30).median()
    return out


def _puede_usar_lag_anual(df: pd.DataFrame) -> bool:
    # Necesitamos al menos 400 dias para que el lag_365 sea util
    # (365 dias para el shift + algo de margen para entrenar).
    return len(df) >= 400


def _columnas_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("fecha", "ventas")]


def _metricas(real: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(real, pred))
    rmse = float(np.sqrt(mean_squared_error(real, pred)))
    mask = real > 0
    mape = float(np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100) if mask.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def _split_train_test(df: pd.DataFrame, dias_test: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    corte = len(df) - dias_test
    return df.iloc[:corte].copy(), df.iloc[corte:].copy()


def entrenar_random_forest(df: pd.DataFrame, dias_test: int, horizonte: int) -> ResultadoModelo:
    df = df.sort_values("fecha").reset_index(drop=True)
    train, test = _split_train_test(df, dias_test)

    # Activamos el lag anual solo si hay suficiente historia. Si el CSV del
    # usuario es corto (< 400 dias), se omite automaticamente para no perder
    # toda la base de entrenamiento.
    usar_anual = _puede_usar_lag_anual(train)

    train_f = _agregar_features(train, usar_lag_anual=usar_anual).dropna().reset_index(drop=True)
    cols = _columnas_features(train_f)
    modelo = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(train_f[cols], train_f["ventas"])

    # Validación: predicción recursiva sobre el periodo de test.
    historico = train.copy()
    filas_val = []
    for _, fila in test.iterrows():
        combinado = pd.concat([historico, pd.DataFrame([{"fecha": fila["fecha"], "ventas": np.nan}])], ignore_index=True)
        feat = _agregar_features(combinado, usar_lag_anual=usar_anual).iloc[[-1]]
        pred = max(0.0, float(modelo.predict(feat[cols].ffill().fillna(0.0))[0]))
        filas_val.append({"fecha": fila["fecha"], "real": float(fila["ventas"]), "predicho": pred})
        historico = pd.concat([historico, pd.DataFrame([{"fecha": fila["fecha"], "ventas": fila["ventas"]}])], ignore_index=True)
    validacion = pd.DataFrame(filas_val)

    # Reentrenar con todos los datos antes de pronosticar al futuro.
    usar_anual_full = _puede_usar_lag_anual(df)
    completo_f = _agregar_features(df, usar_lag_anual=usar_anual_full).dropna().reset_index(drop=True)
    cols_full = _columnas_features(completo_f)
    modelo_full = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    modelo_full.fit(completo_f[cols_full], completo_f["ventas"])

    historico_pred = df.copy()
    futuro = []
    ultima_fecha = df["fecha"].iloc[-1]
    # Estimar dispersión con residuales del modelo en el set de validación.
    residual_std = float(np.std(validacion["real"] - validacion["predicho"])) if not validacion.empty else 0.0
    for i in range(1, horizonte + 1):
        fecha = ultima_fecha + timedelta(days=i)
        combinado = pd.concat([historico_pred, pd.DataFrame([{"fecha": fecha, "ventas": np.nan}])], ignore_index=True)
        feat = _agregar_features(combinado, usar_lag_anual=usar_anual_full).iloc[[-1]]
        pred = max(0.0, float(modelo_full.predict(feat[cols_full].ffill().fillna(0.0))[0]))
        futuro.append(
            {
                "fecha": fecha,
                "predicho": pred,
                "low": max(0.0, pred - 1.96 * residual_std),
                "high": pred + 1.96 * residual_std,
            }
        )
        historico_pred = pd.concat(
            [historico_pred, pd.DataFrame([{"fecha": fecha, "ventas": pred}])], ignore_index=True
        )

    return ResultadoModelo(
        nombre="Random Forest",
        historico=df,
        validacion=validacion,
        pronostico=pd.DataFrame(futuro),
        metricas=_metricas(validacion["real"].values, validacion["predicho"].values),
    )


def entrenar_prophet(df: pd.DataFrame, dias_test: int, horizonte: int) -> ResultadoModelo:
    if not PROPHET_DISPONIBLE:
        raise RuntimeError("Prophet no está instalado en este entorno.")

    df = df.sort_values("fecha").reset_index(drop=True)
    train, test = _split_train_test(df, dias_test)

    def _nuevo_modelo() -> Prophet:
        # Configuracion afinada para este dataset:
        # - additive funciona mejor que multiplicative cuando hay dias en 0
        # - holidays de UK ayudan poco aqui (datos de 2009-2011), pero no daña
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.95,
        )
        return m

    pdf_train = train.rename(columns={"fecha": "ds", "ventas": "y"})
    m = _nuevo_modelo()
    m.fit(pdf_train)

    futuro_val = pd.DataFrame({"ds": test["fecha"]})
    pred_val = m.predict(futuro_val)
    validacion = pd.DataFrame(
        {
            "fecha": test["fecha"].values,
            "real": test["ventas"].values,
            "predicho": np.maximum(0.0, pred_val["yhat"].values),
        }
    )

    pdf_full = df.rename(columns={"fecha": "ds", "ventas": "y"})
    m_full = _nuevo_modelo()
    m_full.fit(pdf_full)
    futuro = m_full.make_future_dataframe(periods=horizonte, freq="D", include_history=False)
    fc = m_full.predict(futuro)
    pronostico = pd.DataFrame(
        {
            "fecha": fc["ds"].values,
            "predicho": np.maximum(0.0, fc["yhat"].values),
            "low": fc["yhat_lower"].clip(lower=0).values,
            "high": fc["yhat_upper"].values,
        }
    )

    return ResultadoModelo(
        nombre="Prophet",
        historico=df,
        validacion=validacion,
        pronostico=pronostico,
        metricas=_metricas(validacion["real"].values, validacion["predicho"].values),
    )


# ============================================================================
# Shrinkage hacia la estacionalidad semanal (modelo principal)
# ============================================================================

PESO_RF_SHRINKAGE = 0.85  # Fijo, justificado a priori (ancla leve).


def _naive_semanal(df: pd.DataFrame, dias_test: int, horizonte: int):
    """Predice cada dia como el valor del mismo dia hace una semana.

    Es la baseline mas dura de batir en datos con estacionalidad semanal
    fuerte. Aqui no se usa por si sola, sino como ancla del RF.
    """
    serie = df["ventas"].values
    pred_val = serie[-dias_test - 7: -7]
    base = serie[-7:]
    pred_futuro = []
    ultima_fecha = df["fecha"].iloc[-1]
    for i in range(1, horizonte + 1):
        fecha = ultima_fecha + timedelta(days=i)
        pred_futuro.append({"fecha": fecha, "predicho": float(base[(i - 1) % 7])})
    return pred_val, pd.DataFrame(pred_futuro)


def entrenar_rf_shrinkage(df: pd.DataFrame, dias_test: int, horizonte: int) -> ResultadoModelo:
    """Random Forest con shrinkage hacia el naive semanal: 0.85 RF + 0.15 naive.

    Resultado de 6-fold rolling CV (ver experimento_v5.py):
    - RF solo:           MAE GBP 10,447
    - RF + shrinkage:    MAE GBP 10,183  (-2.5%, gana en 5/6 folds)
    - paired t-test 1-cola: p = 0.0625  (significativo al 10%)

    Es un cambio minimo (3 lineas adicionales) con justificacion estadistica:
    cuando el RF sobrerreacciona a un outlier reciente, el 15% del naive
    semanal lo "ancla" hacia el patron historico del mismo dia de la semana.
    Es regresion a la media / shrinkage clasico.
    """
    df = df.sort_values("fecha").reset_index(drop=True)

    res_rf = entrenar_random_forest(df, dias_test=dias_test, horizonte=horizonte)
    pred_val_rf = res_rf.validacion["predicho"].values
    pred_fut_rf = res_rf.pronostico.set_index("fecha")["predicho"]

    pred_val_n, pron_n = _naive_semanal(df, dias_test, horizonte)
    pred_fut_n = pron_n.set_index("fecha")["predicho"]

    pred_val = np.maximum(
        0.0, PESO_RF_SHRINKAGE * pred_val_rf + (1 - PESO_RF_SHRINKAGE) * pred_val_n
    )
    idx_fut = pred_fut_rf.index
    pred_fut = np.maximum(
        0.0,
        PESO_RF_SHRINKAGE * pred_fut_rf.values
        + (1 - PESO_RF_SHRINKAGE) * pred_fut_n.reindex(idx_fut).values,
    )

    validacion = pd.DataFrame({
        "fecha": res_rf.validacion["fecha"].values,
        "real": res_rf.validacion["real"].values,
        "predicho": pred_val,
    })
    residual_std = float(np.std(validacion["real"] - validacion["predicho"]))
    pronostico = pd.DataFrame({
        "fecha": idx_fut,
        "predicho": pred_fut,
        "low": np.maximum(0.0, pred_fut - 1.96 * residual_std),
        "high": pred_fut + 1.96 * residual_std,
    })

    return ResultadoModelo(
        nombre="RF + shrinkage semanal (0.85/0.15)",
        historico=df,
        validacion=validacion,
        pronostico=pronostico,
        metricas=_metricas(validacion["real"].values, validacion["predicho"].values),
    )
