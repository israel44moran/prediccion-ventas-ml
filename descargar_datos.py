"""Descarga y procesa el dataset *Online Retail II* del UCI Machine Learning Repository.

Es un dataset real de un retailer en línea del Reino Unido entre el 1 de
diciembre de 2009 y el 9 de diciembre de 2011, distribuido públicamente por
UCI. Contiene ~1 millón de líneas de factura con `InvoiceDate`, `Quantity`,
`Price` y `Customer ID`.

Tras la descarga lo agregamos por fecha (sumando `Quantity * Price`) para
producir una serie diaria de ventas totales — el insumo que consume el modelo.

Salida: `ventas_reales.csv` con columnas `fecha` y `ventas` (libras esterlinas).
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
DIR_BASE = Path(__file__).parent
ARCHIVO_XLSX = DIR_BASE / "online_retail_II.xlsx"
ARCHIVO_CSV = DIR_BASE / "ventas_reales.csv"


def descargar() -> Path:
    if ARCHIVO_XLSX.exists():
        print(f"Ya existe {ARCHIVO_XLSX.name}, omito descarga.")
        return ARCHIVO_XLSX
    print(f"Descargando {URL} …")
    urlretrieve(URL, ARCHIVO_XLSX)  # noqa: S310
    print(f"OK — guardado en {ARCHIVO_XLSX.name} ({ARCHIVO_XLSX.stat().st_size/1e6:.1f} MB)")
    return ARCHIVO_XLSX


def procesar(ruta: Path) -> pd.DataFrame:
    print("Leyendo hojas del Excel (puede tardar 30–60 s)…")
    hojas = pd.read_excel(ruta, sheet_name=None, engine="openpyxl")
    print(f"Hojas encontradas: {list(hojas.keys())}")

    df = pd.concat(hojas.values(), ignore_index=True)
    print(f"Total de transacciones: {len(df):,}")

    # Normalizar nombres de columna (el archivo usa 'Price' en una hoja y
    # 'UnitPrice' en otra dependiendo de la versión).
    df.columns = [c.strip() for c in df.columns]
    columna_precio = "Price" if "Price" in df.columns else "UnitPrice"
    columna_factura = "Invoice" if "Invoice" in df.columns else "InvoiceNo"

    # Limpieza estándar de este dataset:
    # - quitar cancelaciones (la factura empieza con "C")
    # - quitar cantidades o precios <= 0 (devoluciones, ajustes)
    df = df[~df[columna_factura].astype(str).str.startswith("C", na=False)]
    df = df[(df["Quantity"] > 0) & (df[columna_precio] > 0)]
    df = df.dropna(subset=["InvoiceDate"])

    df["importe"] = df["Quantity"] * df[columna_precio]
    df["fecha"] = pd.to_datetime(df["InvoiceDate"]).dt.date

    diario = (
        df.groupby("fecha", as_index=False)["importe"].sum().rename(columns={"importe": "ventas"})
    )
    diario["fecha"] = pd.to_datetime(diario["fecha"])
    diario = diario.sort_values("fecha").reset_index(drop=True)

    # El dataset tiene unos pocos días sin operación (festivos UK). Para que
    # los modelos vean una serie continua, completamos con 0.
    rango = pd.date_range(diario["fecha"].min(), diario["fecha"].max(), freq="D")
    diario = diario.set_index("fecha").reindex(rango, fill_value=0.0).rename_axis("fecha").reset_index()

    diario["ventas"] = diario["ventas"].round(2)
    diario["fecha"] = diario["fecha"].dt.strftime("%Y-%m-%d")
    return diario


def main() -> None:
    try:
        ruta = descargar()
    except Exception as e:
        print(f"ERROR al descargar: {e}", file=sys.stderr)
        sys.exit(1)

    df = procesar(ruta)
    df.to_csv(ARCHIVO_CSV, index=False)
    print()
    print(f"Guardadas {len(df):,} filas diarias en {ARCHIVO_CSV.name}")
    print(f"Rango: {df['fecha'].iloc[0]} -> {df['fecha'].iloc[-1]}")
    print(f"Venta promedio diaria: £{df['ventas'].mean():,.2f}")
    print(f"Ventas acumuladas: £{df['ventas'].sum():,.2f}")


if __name__ == "__main__":
    main()
