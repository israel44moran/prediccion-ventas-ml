"""Descarga el dataset 'Bread Basket' — una panaderia real de Edimburgo (Escocia).

Es un dataset publico de 21,293 transacciones reales entre el 30 de octubre
de 2016 y el 9 de abril de 2017, distribuido en multiples repositorios de
GitHub. Originalmente compilado para analisis de canasta de mercado, sirve
perfectamente como serie de tiempo diaria de una panaderia pequena.

Como el dataset no incluye precios (solo items vendidos), agregamos por dia
contando el numero de transacciones unicas — es decir, **cuantos clientes
pasaron por la caja cada dia**. Para una panaderia esa es la metrica mas
relevante de actividad (proxy de ventas y de carga de trabajo).

Salida: `ventas_panaderia.csv` con columnas `fecha` y `ventas` (donde
'ventas' es el numero de transacciones del dia).
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

URLS = [
    "https://raw.githubusercontent.com/viktree/curly-octo-chainsaw/master/BreadBasket_DMS.csv",
    "https://raw.githubusercontent.com/reisanar/datasets/master/BreadBasket.csv",
    "https://raw.githubusercontent.com/yortos/bakery/master/BreadBasket_DMS.csv",
]

DIR_BASE = Path(__file__).parent
ARCHIVO_CRUDO = DIR_BASE / "panaderia_edinburgh_raw.csv"
ARCHIVO_CSV = DIR_BASE / "ventas_panaderia.csv"


def descargar() -> Path:
    if ARCHIVO_CRUDO.exists():
        print(f"Ya existe {ARCHIVO_CRUDO.name}, omito descarga.")
        return ARCHIVO_CRUDO
    for url in URLS:
        try:
            print(f"Intentando: {url}")
            urlretrieve(url, ARCHIVO_CRUDO)  # noqa: S310
            print(f"OK — descargado ({ARCHIVO_CRUDO.stat().st_size/1024:.1f} KB)")
            return ARCHIVO_CRUDO
        except Exception as e:
            print(f"  fallo: {e}")
    raise RuntimeError("No pude descargar desde ningun mirror.")


def procesar(ruta: Path) -> pd.DataFrame:
    print("Leyendo CSV...")
    df = pd.read_csv(ruta)
    print(f"Columnas crudas: {list(df.columns)}")
    print(f"Filas crudas: {len(df):,}")

    df.columns = [c.strip() for c in df.columns]

    # Renombrar a estandar
    col_fecha = "Date" if "Date" in df.columns else "date"
    col_trans = "Transaction" if "Transaction" in df.columns else "transaction"

    df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
    df = df.dropna(subset=[col_fecha])
    df = df[df[col_trans] > 0]

    # Quitar items que son 'NONE' (transacciones vacias del dataset)
    if "Item" in df.columns:
        df = df[df["Item"].astype(str).str.upper() != "NONE"]

    # Numero de transacciones unicas por dia
    diario = (
        df.groupby(df[col_fecha].dt.date)[col_trans]
          .nunique()
          .reset_index()
    )
    diario.columns = ["fecha", "ventas"]
    diario["fecha"] = pd.to_datetime(diario["fecha"])

    # Rellenar dias faltantes con 0 (panaderia cerrada o no opero)
    rango = pd.date_range(diario["fecha"].min(), diario["fecha"].max(), freq="D")
    diario = (
        diario.set_index("fecha")
              .reindex(rango, fill_value=0)
              .rename_axis("fecha")
              .reset_index()
    )

    diario["ventas"] = diario["ventas"].astype(float)
    diario["fecha"] = diario["fecha"].dt.strftime("%Y-%m-%d")
    return diario


def main() -> None:
    try:
        ruta = descargar()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    df = procesar(ruta)
    df.to_csv(ARCHIVO_CSV, index=False)
    print()
    print(f"Guardadas {len(df):,} filas diarias en {ARCHIVO_CSV.name}")
    print(f"Rango: {df['fecha'].iloc[0]} -> {df['fecha'].iloc[-1]}")
    print(f"Transacciones promedio por dia: {df['ventas'].mean():.1f}")
    print(f"Mejor dia: {df['ventas'].max():.0f} transacciones")
    print(f"Peor dia:  {df['ventas'].min():.0f} transacciones")
    print(f"Dias con 0 transacciones (cerrado): {(df['ventas']==0).sum()}")


if __name__ == "__main__":
    main()
