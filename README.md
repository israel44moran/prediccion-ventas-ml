# Predicción de ventas con Machine Learning

App de Streamlit que pronostica las ventas diarias de un negocio a partir de su histórico, usando **Random Forest + shrinkage semanal** (modelo principal) y **Prophet** (alternativo). Incluye **dos datasets reales** para comparar cómo rinde el mismo modelo en escenarios opuestos:

- **UCI Online Retail II** — retailer en línea del Reino Unido con 1M+ transacciones B2B/B2C reales (2009-2011).
- **Bread Basket Edimburgo** — panadería pequeña en Escocia, 21k transacciones reales (oct 2016 - abr 2017).

También permite que el cliente suba su propio CSV.

## Problema que resuelve

Una tienda o cualquier negocio con histórico de ventas suele tomar decisiones (compras, plantilla, promociones) a ojo o usando un Excel con promedios. Un buen pronóstico, aunque sea con un margen de error medible, **convierte ese histórico en una decisión informada**: cuánto inventario pedir para la próxima semana, qué días requieren más personal, qué meses son críticos para la caja.

Esta app cubre todo el flujo:
1. Carga el histórico (dataset real incluido o CSV del usuario).
2. Entrena uno o dos modelos sobre los datos.
3. **Valida** los modelos contra los últimos 30 días reales (datos que el modelo no vio).
4. Genera un pronóstico para los próximos 7 a 90 días con intervalo de confianza del 95%.
5. Permite descargar el pronóstico en CSV listo para Excel o sistemas internos.

## Características

- **Dataset real incluido**: 739 días de ventas reales (1 dic 2009 → 9 dic 2011) del retailer británico publicado por UCI.
- **Carga de CSV propio**: el cliente solo necesita un archivo con columnas `fecha` y `ventas`.
- **Tres opciones de modelo**:
  - 🎯 **RF + shrinkage semanal** (recomendado) — Random Forest anclado en 15% al valor del mismo día hace una semana. Es el único candidato estadísticamente mejor que RF solo (paired t-test p=0.06, gana en 5 de 6 folds).
  - 🌳 **Random Forest** solo (scikit-learn) con features de calendario, rezagos, feriados UK e interacciones.
  - 🔮 **Prophet** (Meta) con estacionalidad aditiva semanal + anual.
- **Métricas de error reales**: MAE, RMSE y MAPE calculadas sobre datos que el modelo no usó al entrenar.
- **Visualización clara**: histórico con promedio móvil, validación real-vs-predicho, pronóstico con banda de incertidumbre.
- **Pronóstico descargable** en CSV.
- **Interfaz 100% en español**, pensada para que el dueño del negocio entienda lo que ve.

## Fuentes de datos (100% reales, 100% abiertas)

### Dataset 1 — UCI Online Retail II (retailer UK)

- Archivo: `online_retail_II.xlsx` (~46 MB)
- URL oficial: <https://archive.ics.uci.edu/dataset/502/online+retail+ii>
- Contenido: 1,067,371 líneas de factura de un retailer en línea del Reino Unido entre el 1 de diciembre de 2009 y el 9 de diciembre de 2011.
- Columnas originales relevantes: `Invoice`, `InvoiceDate`, `Quantity`, `Price`, `Customer ID`, `Country`.

El script `descargar_datos.py` baja el Excel, descarta cancelaciones (facturas que inician con `C`) y montos no positivos, agrega por día sumando `Quantity × Price`, completa días sin operación con 0 y guarda el resultado en `ventas_reales.csv`.

### Dataset 2 — Bread Basket (panadería Edimburgo)

- Archivo: `BreadBasket_DMS.csv` (~700 KB)
- URL: <https://github.com/viktree/curly-octo-chainsaw> (mirror público en GitHub)
- Contenido: 21,293 transacciones reales de una panadería en Edimburgo entre el 30 de octubre de 2016 y el 9 de abril de 2017 (~5.5 meses).
- Columnas originales: `Date`, `Time`, `Transaction`, `Item`.

El script `descargar_panaderia.py` baja el CSV, cuenta el número de transacciones únicas por día y guarda el resultado en `ventas_panaderia.csv`. Como el dataset no incluye precios, la métrica de "ventas" es **número de tickets** (clientes que pasaron por la caja), que para una panadería es un proxy directo de actividad y carga operativa.

## Comparativa: el mismo modelo en dos negocios reales

Para mostrar **cómo el rendimiento del modelo depende del tipo de negocio**, evaluamos el campeón (RF + shrinkage 85/15) sobre ambos datasets usando cross-validation rolling:

| Dataset | Tipo de negocio | Días | MAE | **MAPE** | Std MAE |
|---|---|---|---|---|---|
| UCI Online Retail II | Retailer UK B2B/B2C (libras £) | 739 | £10,183 | **24%** | ±£2,550 |
| Bread Basket Edimburgo | Panadería (transacciones/día) | 162 | 7.2 tx | **14%** | ±1.2 tx |

**El mismo modelo predice ~42% más preciso en la panadería**. La razón es estructural:

| Por qué la panadería es más predecible | UCI Retail | Panadería |
|---|---|---|
| Tamaño de los outliers vs promedio | hasta 7× el promedio (£200k vs £28k) | máximo 2.4× el promedio (139 vs 58) |
| Composición de clientes | mezcla B2B con pedidos mayoristas únicos | clientes individuales recurrentes |
| % días "raros" o cerrados | 18% (sábados) | 1.9% (apenas 3 días) |
| Patrón semanal | fuerte pero contaminado por B2B | muy estable (café matinal, mucha gente los domingos) |

**Lección práctica**: si tu negocio se parece más a la panadería (clientes individuales, sin pedidos enormes), espera **MAPE de 8-15%** con este modelo. Si se parece más al retailer UCI (clientes mayoristas, eventos especiales), espera **MAPE de 20-30%**.

## Tecnologías

- Python 3.10+
- Streamlit (UI)
- Pandas + NumPy (manipulación de datos)
- Plotly (gráficos interactivos)
- scikit-learn (Random Forest)
- Prophet (modelo de series de tiempo de Meta, opcional)
- openpyxl (lectura del Excel de UCI)

## Instalación

```bash
git clone https://github.com/israel44moran/prediccion-ventas-ml.git
cd prediccion-ventas-ml
pip install -r requirements.txt
```

> **Nota**: `prophet` es opcional. Si su instalación falla en tu entorno (en Windows a veces requiere herramientas de compilación), la app se ejecuta igual con solo el modelo Random Forest.

## Uso

### 1. Descargar y preparar los datos reales

```bash
python descargar_datos.py       # Dataset 1: UCI Online Retail II (~739 días)
python descargar_panaderia.py   # Dataset 2: Bread Basket Edimburgo (~162 días)
```

El primer script baja `online_retail_II.xlsx` desde UCI, lo procesa y guarda `ventas_reales.csv`. El segundo baja el CSV de la panadería y guarda `ventas_panaderia.csv`. Puedes correr solo uno si solo te interesa un dataset.

### 2. Lanzar la app

```bash
streamlit run dashboard.py
```

En el navegador (por defecto <http://localhost:8501>):

- Selecciona la **fuente de datos**: UCI Online Retail II, Panadería Edimburgo, o sube tu CSV.
- Elige el **horizonte de pronóstico** (entre 7 y 90 días, ajustado al tamaño del dataset).
- Elige los **días reservados para validar** el modelo (recomendado: 30 para UCI, 14 para panadería).
- Elige el **modelo**: RF + shrinkage semanal (recomendado), Random Forest solo, o Prophet.

### 3. Subir tu propio CSV

El CSV debe tener exactamente estas dos columnas:

```csv
fecha,ventas
2024-01-01,5230.50
2024-01-02,6120.00
...
```

Cualquier granularidad diaria funciona. Mínimo recomendado: 6 meses de histórico para que la estacionalidad semanal se aprenda bien.

## Cómo funcionan los modelos

### Random Forest

Genera estas features por cada día:

| Tipo | Features |
|------|----------|
| Calendario | día de la semana, día del mes, mes, número de semana, semana del mes, indicadores de fin de semana / sábado / quincena / diciembre / pre-Navidad, días desde el inicio del histórico |
| Feriados | indicador de feriado oficial UK + indicador de día previo a feriado (vía paquete `holidays`) |
| Rezagos cortos | ventas de hace 1, 7, 14 y 30 días |
| Rezagos anuales | ventas de hace 364 y 365 días (captura estacionalidad anual — solo si el histórico tiene ≥400 días) |
| Estadísticos móviles | media, desviación estándar, máximo de los últimos 7 y 30 días + mediana 30 días |
| Interacciones | mes × día de la semana (un viernes de diciembre ≠ uno de marzo), mes × fin de semana |

Luego entrena un `RandomForestRegressor` con 400 árboles. Para predecir al futuro, alimenta la predicción del día anterior como insumo del siguiente paso (pronóstico recursivo).

El intervalo de confianza se construye con la desviación estándar de los residuales en el set de validación: `[predicho ± 1.96 × σ_residual]`.

### Prophet

Descompone la serie en tendencia, estacionalidad semanal y estacionalidad anual con un componente aditivo (afinado tras experimentación — el modo multiplicativo, default de la documentación, da peores resultados en este dataset porque hay días con ventas en 0). Intervalo del 95% por bootstrap interno.

## Validación honesta

Los **últimos 30 días del histórico se reservan** y no se le muestran al modelo durante el entrenamiento. Las métricas MAE, RMSE y MAPE se calculan **únicamente sobre esos 30 días** comparando real vs predicho — lo que el cliente ve en la app es el rendimiento esperado sobre datos nuevos, no un sobreajuste al pasado.

### Benchmark contra modelos baseline (cross-validation rolling de 6 folds + paired t-test)

Una sola medición sobre 30 días es ruidosa. Para tener una señal sólida evaluamos cada modelo sobre **6 ventanas de 30 días** consecutivas (jun → dic 2011), reentrenando en cada una con todo lo anterior y promediando MAE. Para declarar un modelo "mejor" que otro exigimos además **paired t-test con p < 0.10** sobre los 6 folds.

Resultado final tras 5 rondas de experimentación honesta (`experimento.py` → `experimento_v5.py`):

| # | Modelo | MAE promedio | Std | Gana folds | p-value vs RF |
|---|--------|--------------|-----|-----|---|
| 🥇 | **RF + shrinkage semanal (0.85 RF + 0.15 naive)** | **£10,183** | ±£2,550 | **5/6** | **0.0625** ✓ sig |
| 🥈 | RF + lag_365 + UK holidays + interacciones | £10,447 | ±£2,492 | — | (referencia) |
| 🥉 | Naive semanal (baseline tonto) | £12,622 | ±£3,358 | 0/6 | — |
| 4 | RF "ingenuo" (sin lag_365 ni holidays) | £15,568 | ±£9,213 | — | — |
| 5 | HistGradientBoosting | £17,079 | ±£4,336 | — | — |
| 6 | LightGBM defaults | £19,904 | ±£7,880 | — | — |

**Mejora total acumulada**: MAE bajó de **£15,568 → £10,183** = **−35%** respecto al RF inicial, y de £12,622 → £10,183 = **−19%** respecto al baseline "naive semanal".

### Por qué `0.85 RF + 0.15 Naive` es el ganador real

Después de probar 9 recetas distintas (ver tabla extendida abajo), **solo el shrinkage con peso fijo 85/15 mejoró el MAE de RF con significancia estadística** (paired t-test p=0.0625, gana en 5 de 6 folds). El resto:

- **Ensemble RF + LightGBM + Naive (1/3+1/3+1/3)** parecía bajar a £9,996 (−4.3% vs RF) en cross-validation cruda. Al hacer paired t-test pareado: **p = 0.94, gana solo en 2/6 folds** → era ruido estadístico, no mejora real.
- **ExtraTrees, HistGB conservador, Mediana(RF, lag_7,14,21,28)**: todos pierden contra RF o no son significativamente mejores.
- **Shrinkage con pesos 0.67 o 0.75**: misma dirección pero p > 0.10 (no significativo).

**Por qué funciona el shrinkage 85/15**: cuando el RF sobrerreacciona a un outlier reciente, el 15% del naive semanal lo "ancla" hacia el patrón histórico del mismo día de la semana. En días normales no cambia nada (RF y naive están cerca). Es regresión a la media de toda la vida — el truco más viejo de la estadística aplicada.

### Lo que se probó y fue descartado

| Idea | Resultado en 6-fold CV | Por qué falló |
|---|---|---|
| Ensemble (1/3 RF + 1/3 LightGBM + 1/3 Naive) | MAE £9,996 promedio pero **p = 0.94 pareado** | Mejora era ruido, no real |
| LightGBM defaults | MAE £19,904 | Sobreajusta con 31 hojas en ~600 muestras |
| LightGBM conservador solo | MAE £10,795 | Marginalmente peor que RF |
| HistGradientBoosting | MAE £17,079 | El target bimodal (£0 sábados) lo confunde |
| ExtraTrees + mismas features | MAE £10,946 | +4.8% peor que RF |
| RF criterion=absolute_error | MAE £10,605 | Sin mejora |
| RF con 1000 árboles | MAE £10,574 | Más cómputo, sin ganancia |
| Median ensemble (7 semillas RF) | MAE £10,557 | Variancia de RF ya es baja |
| Mediana(RF, lag_7,14,21,28) | MAE £10,668 | Mediana subutiliza la info del RF |
| RF sobre ratio vs media_dow | MAE £10,427 | Mejora marginal, no significativa |
| Stacking ridge | MAE £15,266 | Meta-learner overfit con poca data |
| Winsorización p99 | MAE £15,275 | Quita señal de los picos legítimos |
| Prophet (afinado) | MAE £20,552 | Días en £0 lo confunden |

### Proceso de selección (sin trampa)

1. Se reservaron los **últimos 30 días como test "intocable"** desde el principio.
2. En cada ronda de experimentación se probaron varias recetas y se midió **MAE promedio sobre 6 folds de CV rolling** previos.
3. **Para declarar un modelo "mejor" se exigió paired t-test con p < 0.10** sobre los 6 folds, no solo promedio menor (esto descarta mejoras que son ruido estadístico).
4. **El peso del shrinkage (0.85) se eligió como compromiso entre 0.5 (neutral) y 1.0 (puro RF)**, con justificación a priori de "ancla leve". Se reportaron también los pesos 0.67 y 0.75 para mostrar la robustez del efecto.
5. Hiperparámetros de cada modelo (`n_estimators=400`, `max_depth=14`) son **defaults razonables**, no buscados via grid-search sobre el set de validación.
6. Las features (lag_365, feriados UK, interacciones) se eligieron por **justificación a priori** (estacionalidad anual, calendario UK), no por mejorar un score específico.

Los scripts `experimento.py`, `experimento_cv.py`, `experimento_v2.py`, `experimento_v3.py`, `experimento_v4.py` y `experimento_v5.py` reproducen toda la trayectoria de mejoras.

## Estructura del proyecto

```
Proyecto 6 - Prediccion de ventas con ML/
├── dashboard.py            # Dashboard Streamlit
├── modelo.py               # Funciones de entrenamiento, validación y pronóstico
├── descargar_datos.py      # Descarga y procesa el dataset UCI Online Retail II
├── descargar_panaderia.py  # Descarga y procesa el dataset Bread Basket (panadería)
├── experimento.py          # Comparación inicial de 5 recetas (1 ventana val + test)
├── experimento_cv.py       # Cross-validation rolling de 6 folds — primer benchmark
├── experimento_v2.py       # Ronda 2: LightGBM, stacking, features mejorados
├── experimento_v3.py       # Ronda 3: tuning RF, ensembles ponderados
├── experimento_v4.py       # Ronda 4: blends con pesos fijos
├── experimento_v5.py       # Ronda 5 (final): paired t-test, encuentra al ganador
├── ventas_reales.csv       # Serie diaria UCI (generada por el script)
├── ventas_panaderia.csv    # Serie diaria panadería (generada por el script)
├── online_retail_II.xlsx   # Excel original de UCI (generado por el script)
├── panaderia_edinburgh_raw.csv  # CSV original de la panadería
├── requirements.txt
└── README.md
```

## Limitaciones conocidas

- El dataset UCI es un retailer B2B/B2C inglés con clientes mayoristas: hay días con outliers fuertes (pedidos grandes únicos de £100k+) que el modelo no puede predecir sin features externas. El MAPE típico sobre este dataset ronda **23–25%** — útil para planeación operativa (personal, inventario), no para pronóstico financiero preciso.
- Para una tienda con patrones más estables (ej. una tienda de barrio con clientes recurrentes), el MAPE esperado es notablemente menor (8–15%).
- El pronóstico recursivo del Random Forest acumula error con horizontes largos; para horizontes mayores a 60 días, las predicciones tienden hacia la media histórica.
- Prophet rinde peor que Random Forest en este dataset por la alta proporción de días en £0 (sábados cerrados). En datasets sin ceros estructurales suele ser más competitivo.

## Licencia

MIT. El dataset UCI Online Retail II tiene su propia licencia (Creative Commons Attribution 4.0) — citar la fuente al republicar resultados.
