"""
Prediccion de ventas con Machine Learning
Dashboard interactivo que entrena modelos sobre ventas diarias reales
del retailer UCI Online Retail II (~1M transacciones, 2009-2011).
Ejecutar: streamlit run dashboard.py
"""

import os
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modelo import (
    PROPHET_DISPONIBLE,
    cargar_csv,
    entrenar_prophet,
    entrenar_random_forest,
    entrenar_rf_shrinkage,
)

# ============================================================
# CONFIGURACION
# ============================================================
st.set_page_config(
    page_title="Prediccion de ventas con ML",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# TOKENS DE DISENO
# ============================================================
INK         = "#0E1218"
SURFACE     = "#171C24"
SURFACE_2   = "#1F2530"
BORDER      = "#2A3140"
RULE        = "#3A4258"
CREAM       = "#F2EDE3"
COOL        = "#9AA3B5"
MUTED       = "#5A6478"

AMBER       = "#D4A574"
AMBER_HI    = "#E8C9A0"
AMBER_DIM   = "#8B6E47"
AMBER_FILL  = "rgba(212, 165, 116, 0.18)"

GREEN       = "#7A9B7E"
RED         = "#C26B5E"
BLUE        = "#7088A8"

DATASETS = {
    "UCI Online Retail II (UK 2009-2011)": {
        "ruta": "ventas_reales.csv",
        "unidad": "£",
        "decimales": 0,
        "descripcion": "Retailer en linea del Reino Unido, ~1M transacciones B2B/B2C.",
    },
    "Panaderia Edimburgo (2016-2017)": {
        "ruta": "ventas_panaderia.csv",
        "unidad": "tx",
        "decimales": 1,
        "descripcion": "Panaderia real en Edimburgo, 21k transacciones en 5.5 meses.",
    },
}

# ============================================================
# CARGA DE DATOS
# ============================================================
@st.cache_data
def cargar_dataset(ruta: str):
    if not os.path.exists(ruta):
        return None
    return cargar_csv(ruta)


@st.cache_data
def cargar_subido(contenido_bytes: bytes):
    return cargar_csv(BytesIO(contenido_bytes))


# ============================================================
# ESTILOS (consistente con el resto del portafolio)
# ============================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,300..700,0..100&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'DM Sans', sans-serif;
        background-color: {INK};
        color: {CREAM};
    }}
    #MainMenu, footer {{visibility: hidden;}}
    .block-container {{
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1340px;
    }}

    [data-testid="stSidebar"] > div:first-child {{
        background-color: {SURFACE};
        padding-top: 2rem;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {{
        color: {CREAM};
    }}
    [data-testid="stSidebar"] label {{
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        color: {MUTED} !important;
    }}

    .hero-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; color: {AMBER};
        letter-spacing: 2px; text-transform: uppercase;
        margin: 0 0 0.5rem 0;
    }}
    .hero-title {{
        font-family: 'Fraunces', serif;
        font-weight: 500; font-size: 2.4rem;
        line-height: 1.05; letter-spacing: -1px;
        color: {CREAM}; margin: 0 0 0.6rem 0;
    }}
    .hero-deck {{
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem; color: {COOL};
        margin: 0; max-width: 760px; line-height: 1.5;
    }}

    .meta-bar {{
        display: flex; gap: 3rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem; color: {MUTED};
        text-transform: uppercase; letter-spacing: 1.5px;
        padding: 1rem 0;
        border-top: 1px solid {BORDER};
        border-bottom: 1px solid {BORDER};
        margin: 2rem 0;
    }}
    .meta-bar strong {{ color: {CREAM}; font-weight: 500; }}

    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0;
        border-top: 1px solid {BORDER};
        border-bottom: 1px solid {BORDER};
        margin: 2rem 0;
    }}
    .kpi-cell {{
        padding: 1.75rem 1.5rem 1.75rem 0;
        border-right: 1px solid {BORDER};
    }}
    .kpi-cell:first-child {{ padding-left: 0; }}
    .kpi-cell:last-child  {{ border-right: none; padding-right: 0; }}
    .kpi-cell-padded {{ padding-left: 1.5rem; }}
    .kpi-label {{
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 2px;
        color: {MUTED}; margin: 0 0 0.75rem 0;
    }}
    .kpi-number {{
        font-family: 'Fraunces', serif;
        font-weight: 400; font-size: 2.4rem;
        line-height: 1; letter-spacing: -1.5px;
        color: {CREAM}; margin: 0;
        font-feature-settings: 'tnum';
    }}
    .kpi-unit {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; color: {MUTED};
        text-transform: uppercase; letter-spacing: 1.5px;
        margin: 0.75rem 0 0 0;
        display: flex; align-items: center; gap: 6px;
    }}
    .kpi-tick {{
        display: inline-block; width: 8px; height: 1px;
        background: {AMBER};
    }}

    .section-block {{
        margin: 3rem 0 1.5rem 0;
        padding-top: 2rem;
        border-top: 1px solid {BORDER};
    }}
    .section-meta {{
        display: flex; align-items: baseline; gap: 1rem;
        margin-bottom: 0.5rem;
    }}
    .section-id {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem; color: {AMBER};
        letter-spacing: 1.5px; text-transform: uppercase;
    }}
    .section-divider {{ flex: 1; height: 1px; background: {BORDER}; }}
    .section-headline {{
        font-family: 'Fraunces', serif;
        font-weight: 500; font-size: 1.75rem;
        line-height: 1.1; letter-spacing: -0.5px;
        color: {CREAM}; margin: 0.5rem 0 0 0;
    }}
    .section-deck {{
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem; color: {COOL};
        margin: 0.5rem 0 0 0;
    }}

    .insight-callout {{
        background: {SURFACE};
        border: 1px solid {AMBER_DIM};
        border-left: 3px solid {AMBER};
        border-radius: 6px;
        padding: 1rem 1.3rem;
        margin: 1.5rem 0;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        color: {CREAM};
        line-height: 1.5;
    }}
    .insight-callout strong {{ color: {AMBER}; font-weight: 600; }}

    .metric-row {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0;
        border-top: 1px solid {BORDER};
        border-bottom: 1px solid {BORDER};
        margin: 1.5rem 0;
    }}
    .metric-cell {{
        padding: 1.3rem 1.5rem 1.3rem 0;
        border-right: 1px solid {BORDER};
    }}
    .metric-cell:first-child {{ padding-left: 0; }}
    .metric-cell:last-child  {{ border-right: none; padding-right: 0; }}
    .metric-cell-padded {{ padding-left: 1.5rem; }}

    .colophon {{
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 2px solid {RULE};
        display: flex; justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; color: {MUTED};
        text-transform: uppercase; letter-spacing: 1.5px;
    }}

    .stSelectbox > div > div, .stMultiSelect > div > div,
    .stRadio > div, .stFileUploader > div {{
        background-color: {INK} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 4px !important;
        color: {CREAM} !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    .stSlider > div {{ color: {CREAM} !important; }}
    span[data-baseweb="tag"] {{
        background-color: {AMBER_DIM} !important;
        color: {CREAM} !important;
        border-radius: 2px !important;
    }}
    .stDownloadButton > button, .stButton > button {{
        background-color: {SURFACE} !important;
        color: {CREAM} !important;
        border: 1px solid {AMBER_DIM} !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }}
    .stDownloadButton > button:hover, .stButton > button:hover {{
        border-color: {AMBER} !important;
        color: {AMBER} !important;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# UTILIDADES DE GRAFICA
# ============================================================
def apply_chart_style(fig, height=380):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=CREAM, size=12),
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
                   tickfont=dict(color=COOL, size=10, family="JetBrains Mono, monospace"),
                   linecolor=BORDER, showgrid=False),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
                   tickfont=dict(color=COOL, size=10, family="JetBrains Mono, monospace"),
                   linecolor=BORDER, showgrid=True, gridwidth=1),
        legend=dict(font=dict(color=CREAM, size=11), bgcolor="rgba(0,0,0,0)",
                    orientation="h", y=1.08, x=0),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER,
                        font=dict(color=CREAM, family="JetBrains Mono, monospace", size=11)),
    )
    return fig


def section_header(section_id, title, subtitle):
    st.markdown(f"""
    <div class="section-block">
        <div class="section-meta">
            <span class="section-id">— {section_id}</span>
            <div class="section-divider"></div>
        </div>
        <p class="section-headline">{title}</p>
        <p class="section-deck">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


UNIDAD_ACTUAL = "£"
DECIMALES_ACTUAL = 0


def fmt_money(v: float) -> str:
    return f"{UNIDAD_ACTUAL}{v:,.{DECIMALES_ACTUAL}f}"


# ============================================================
# PANEL LATERAL
# ============================================================
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 0 0 1rem 0; border-bottom: 1px solid {BORDER}; margin-bottom: 1rem;">
        <p style="font-family: 'JetBrains Mono'; font-size: 0.65rem; color: {AMBER}; letter-spacing: 2px; text-transform: uppercase; margin: 0;">— Controles</p>
        <p style="font-family: 'Fraunces'; font-style: italic; font-size: 1.2rem; color: {CREAM}; margin: 0.3rem 0 0 0;">Configuracion</p>
    </div>
    """, unsafe_allow_html=True)

    opciones_fuente = list(DATASETS.keys()) + ["Subir mi propio CSV"]
    origen = st.radio("Fuente de datos", options=opciones_fuente, index=0)

    df = None
    nombre_dataset = origen
    if origen in DATASETS:
        cfg = DATASETS[origen]
        df = cargar_dataset(cfg["ruta"])
        if df is None:
            scripts = {
                "ventas_reales.csv": "descargar_datos.py",
                "ventas_panaderia.csv": "descargar_panaderia.py",
            }
            st.error(f"Falta `{cfg['ruta']}`. Corre primero:\n\npython {scripts.get(cfg['ruta'], '')}")
        else:
            st.markdown(
                f'<p style="font-family: \'DM Sans\'; font-size: 0.75rem; color: {COOL}; '
                f'margin: 0.5rem 0 1rem 0; line-height: 1.4;">{cfg["descripcion"]}</p>',
                unsafe_allow_html=True,
            )
    else:
        archivo = st.file_uploader("Subir CSV (columnas: fecha, ventas)", type=["csv"])
        if archivo is not None:
            try:
                df = cargar_subido(archivo.getvalue())
            except Exception as e:
                st.error(f"Error en CSV: {e}")
        else:
            st.info("Sube un CSV para continuar.")

    # Ajustar parametros segun tamano del dataset
    if df is not None:
        max_horiz = min(90, max(7, len(df) // 4))
        default_horiz = min(30, max(7, len(df) // 8))
        max_test = min(60, max(14, len(df) // 5))
        default_test = min(30, max(14, len(df) // 8))
        horizonte = st.slider("Horizonte de pronostico (dias)", 7, max_horiz, default_horiz, step=1)
        dias_test = st.slider("Dias reservados para validar", 14, max_test, default_test, step=1)
    else:
        horizonte, dias_test = 30, 30

    opciones_modelo = ["RF + shrinkage semanal (recomendado)", "Random Forest"]
    if PROPHET_DISPONIBLE:
        opciones_modelo += ["Prophet"]
    modelo_elegido = st.selectbox("Modelo", opciones_modelo, index=0)

    if not PROPHET_DISPONIBLE:
        st.markdown(
            f'<p style="font-family: \'DM Sans\'; font-size: 0.75rem; color: {COOL}; '
            f'margin: 0.5rem 0 0 0; line-height: 1.4;">'
            f"Instala <code>prophet</code> para habilitarlo como tercera opcion.</p>",
            unsafe_allow_html=True,
        )


# ============================================================
# HERO
# ============================================================
st.markdown(f"""
<p class="hero-eyebrow">— PREDICCION DE VENTAS · MACHINE LEARNING · DATASETS REALES</p>
<h1 class="hero-title">Pronostico de ventas diarias con modelos de Machine Learning</h1>
<p class="hero-deck">
    Entrenamiento y validacion del mismo modelo sobre dos datasets reales
    para mostrar como rinde en escenarios distintos: un <strong>retailer en linea
    UK con 1M de transacciones B2B/B2C</strong> y una <strong>panaderia pequena
    en Edimburgo con clientes recurrentes</strong>. El cliente tambien puede subir
    su propio CSV para obtener pronosticos a 7-90 dias con intervalo de confianza.
</p>
""", unsafe_allow_html=True)


if df is None or df.empty:
    st.stop()

# Ajustar formato de moneda segun el dataset elegido
if origen in DATASETS:
    UNIDAD_ACTUAL = DATASETS[origen]["unidad"]
    DECIMALES_ACTUAL = DATASETS[origen]["decimales"]
else:
    UNIDAD_ACTUAL = ""
    DECIMALES_ACTUAL = 0


# ============================================================
# BARRA META
# ============================================================
st.markdown(f"""
<div class="meta-bar">
    <div>EMITIDO &nbsp;·&nbsp; <strong>{datetime.now().strftime("%d %b %Y").upper()}</strong></div>
    <div>FUENTE &nbsp;·&nbsp; <strong>{nombre_dataset.upper()}</strong></div>
    <div>MODELO ACTIVO &nbsp;·&nbsp; <strong>{modelo_elegido.upper()}</strong></div>
    <div>HORIZONTE &nbsp;·&nbsp; <strong>{horizonte} DIAS</strong></div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# KPIs DEL HISTORICO
# ============================================================
total = float(df["ventas"].sum())
prom = float(df["ventas"].mean())
mejor = df.loc[df["ventas"].idxmax()]
peor = df.loc[df["ventas"].idxmin()]
fecha_min = df["fecha"].min().strftime("%d %b %Y").upper()
fecha_max = df["fecha"].max().strftime("%d %b %Y").upper()

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-cell">
        <p class="kpi-label">Dias en el historico</p>
        <p class="kpi-number">{len(df):,}</p>
        <p class="kpi-unit"><span class="kpi-tick"></span> {fecha_min} → {fecha_max}</p>
    </div>
    <div class="kpi-cell kpi-cell-padded">
        <p class="kpi-label">Venta promedio diaria</p>
        <p class="kpi-number">{fmt_money(prom)}</p>
        <p class="kpi-unit"><span class="kpi-tick"></span> Acumulado {fmt_money(total)}</p>
    </div>
    <div class="kpi-cell kpi-cell-padded">
        <p class="kpi-label">Mejor dia</p>
        <p class="kpi-number">{fmt_money(mejor['ventas'])}</p>
        <p class="kpi-unit"><span class="kpi-tick"></span> {mejor['fecha'].strftime('%d %b %Y').upper()}</p>
    </div>
    <div class="kpi-cell kpi-cell-padded">
        <p class="kpi-label">Peor dia</p>
        <p class="kpi-number">{fmt_money(peor['ventas'])}</p>
        <p class="kpi-unit"><span class="kpi-tick"></span> {peor['fecha'].strftime('%d %b %Y').upper()}</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# GRAFICA 01 — HISTORICO
# ============================================================
section_header(
    "GRAFICA 01 / SERIE",
    "Historico de ventas diarias",
    "Linea azul: venta diaria real. Linea ambar: promedio movil de 30 dias para suavizar el ruido. "
    "Esta es la serie que los modelos veran para aprender la estacionalidad y la tendencia."
)

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=df["fecha"], y=df["ventas"],
    mode="lines", name="Venta diaria real",
    line=dict(color=BLUE, width=1.3),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>£%{y:,.0f}<extra></extra>",
))
fig_hist.add_trace(go.Scatter(
    x=df["fecha"], y=df["ventas"].rolling(30).mean(),
    mode="lines", name="Promedio movil 30 dias",
    line=dict(color=AMBER, width=2.4),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>Media 30d: £%{y:,.0f}<extra></extra>",
))
fig_hist = apply_chart_style(fig_hist, height=400)
fig_hist.update_layout(
    yaxis=dict(title=f"Ventas ({UNIDAD_ACTUAL})", title_font=dict(size=10, color=MUTED),
               showgrid=True, gridcolor=BORDER, tickformat=","),
)
st.plotly_chart(fig_hist, use_container_width=True)


# ============================================================
# ENTRENAR MODELOS
# ============================================================
if dias_test >= len(df):
    st.error("No hay suficientes datos para reservar tantos dias de validacion.")
    st.stop()


def render_validacion(resultado):
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-cell">
            <p class="kpi-label">MAE — error absoluto medio</p>
            <p class="kpi-number">{fmt_money(resultado.metricas['MAE'])}</p>
            <p class="kpi-unit"><span class="kpi-tick"></span> Cuanto se equivoca en promedio, en £</p>
        </div>
        <div class="metric-cell metric-cell-padded">
            <p class="kpi-label">RMSE — raiz del error cuadratico</p>
            <p class="kpi-number">{fmt_money(resultado.metricas['RMSE'])}</p>
            <p class="kpi-unit"><span class="kpi-tick"></span> Penaliza mas los dias muy fallados</p>
        </div>
        <div class="metric-cell metric-cell-padded">
            <p class="kpi-label">MAPE — error porcentual</p>
            <p class="kpi-number">{resultado.metricas['MAPE']:.1f}%</p>
            <p class="kpi-unit"><span class="kpi-tick"></span> Magnitud relativa del error</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    val = resultado.validacion
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(
        x=val["fecha"], y=val["real"],
        mode="lines+markers", name="Real",
        line=dict(color=GREEN, width=2),
        marker=dict(size=6, color=GREEN),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Real: £%{y:,.0f}<extra></extra>",
    ))
    fig_v.add_trace(go.Scatter(
        x=val["fecha"], y=val["predicho"],
        mode="lines+markers", name=f"Predicho ({resultado.nombre})",
        line=dict(color=AMBER, width=2, dash="dash"),
        marker=dict(size=6, color=AMBER),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Predicho: £%{y:,.0f}<extra></extra>",
    ))
    fig_v = apply_chart_style(fig_v, height=360)
    fig_v.update_layout(
        yaxis=dict(title=f"Ventas ({UNIDAD_ACTUAL})", title_font=dict(size=10, color=MUTED),
                   showgrid=True, gridcolor=BORDER, tickformat=","),
    )
    st.plotly_chart(fig_v, use_container_width=True)


def render_pronostico(resultado):
    historico = resultado.historico.tail(120)
    pron = resultado.pronostico
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=historico["fecha"], y=historico["ventas"],
        mode="lines", name="Historico reciente",
        line=dict(color=BLUE, width=1.6),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Real: £%{y:,.0f}<extra></extra>",
    ))
    fig_p.add_trace(go.Scatter(
        x=pron["fecha"], y=pron["high"], name="Banda alta",
        mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig_p.add_trace(go.Scatter(
        x=pron["fecha"], y=pron["low"], name="Intervalo 95%",
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        fill="tonexty", fillcolor=AMBER_FILL,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Limite inferior: £%{y:,.0f}<extra></extra>",
    ))
    fig_p.add_trace(go.Scatter(
        x=pron["fecha"], y=pron["predicho"],
        mode="lines", name=f"Pronostico ({resultado.nombre})",
        line=dict(color=AMBER, width=2.6),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Pronostico: £%{y:,.0f}<extra></extra>",
    ))
    fig_p = apply_chart_style(fig_p, height=440)
    fig_p.update_layout(
        yaxis=dict(title=f"Ventas ({UNIDAD_ACTUAL})", title_font=dict(size=10, color=MUTED),
                   showgrid=True, gridcolor=BORDER, tickformat=","),
    )
    st.plotly_chart(fig_p, use_container_width=True)

    tabla = resultado.pronostico.copy()
    tabla["fecha"] = pd.to_datetime(tabla["fecha"]).dt.date
    tabla = tabla.rename(columns={
        "predicho": "ventas_predichas",
        "low": "limite_inferior",
        "high": "limite_superior",
    })
    tabla[["ventas_predichas", "limite_inferior", "limite_superior"]] = tabla[
        ["ventas_predichas", "limite_inferior", "limite_superior"]
    ].round(2)
    with st.expander("Abrir tabla de pronostico"):
        st.dataframe(tabla, use_container_width=True, hide_index=True, height=360)
        csv = tabla.to_csv(index=False).encode("utf-8")
        st.download_button(
            "DESCARGAR PRONOSTICO CSV",
            csv,
            f"pronostico_{resultado.nombre.lower().replace(' ', '_')}.csv",
            "text/csv",
        )


with st.spinner("Entrenando modelo(s) sobre datos reales..."):
    resultados = []
    if modelo_elegido.startswith("RF + shrinkage"):
        resultados.append(entrenar_rf_shrinkage(df, dias_test=dias_test, horizonte=horizonte))
    elif modelo_elegido == "Random Forest":
        resultados.append(entrenar_random_forest(df, dias_test=dias_test, horizonte=horizonte))
    elif modelo_elegido == "Prophet" and PROPHET_DISPONIBLE:
        resultados.append(entrenar_prophet(df, dias_test=dias_test, horizonte=horizonte))


# ============================================================
# SECCION POR MODELO
# ============================================================
for i, resultado in enumerate(resultados, start=2):
    section_header(
        f"GRAFICA 0{i} / VALIDACION {resultado.nombre.upper()}",
        f"Real vs predicho en los ultimos {dias_test} dias",
        f"El modelo {resultado.nombre} fue entrenado sin ver estos {dias_test} dias. "
        "Las metricas miden que tan cerca quedaron las predicciones del valor real."
    )
    render_validacion(resultado)

    mape = resultado.metricas["MAPE"]
    calidad = (
        "excelente para este tipo de dato"
        if mape < 10
        else "buena y aceptable para tomar decisiones operativas"
        if mape < 20
        else "aceptable; este dataset tiene outliers grandes por clientes mayoristas"
    )
    st.markdown(f"""
    <div class="insight-callout">
        <strong>Lectura:</strong> el modelo {resultado.nombre} se equivoca en promedio
        <strong>{fmt_money(resultado.metricas['MAE'])}</strong> por dia
        (MAPE <strong>{mape:.1f}%</strong>) sobre el set de validacion. Esto es {calidad}.
    </div>
    """, unsafe_allow_html=True)

    section_header(
        f"GRAFICA 0{i+len(resultados)} / PRONOSTICO {resultado.nombre.upper()}",
        f"Pronostico {horizonte} dias hacia adelante",
        "Linea azul: ultimos 120 dias reales. Linea ambar: pronostico del modelo. "
        "Banda sombreada: intervalo de confianza del 95%."
    )
    render_pronostico(resultado)


# ============================================================
# COMPARATIVO (solo si se eligen ambos)
# ============================================================
if len(resultados) == 2:
    section_header(
        "GRAFICA 0X / COMPARATIVO",
        "Random Forest vs Prophet — quien predice mejor",
        "Misma metrica, mismo set de validacion. Comparativa lado a lado de los dos enfoques."
    )
    comp = pd.DataFrame([
        {"modelo": r.nombre, **r.metricas} for r in resultados
    ])
    fig_c = go.Figure()
    fig_c.add_trace(go.Bar(
        x=comp["modelo"], y=comp["MAPE"],
        marker=dict(color=[AMBER, BLUE], line=dict(width=0)),
        text=[f"{v:.1f}%" for v in comp["MAPE"]],
        textposition="outside",
        textfont=dict(color=CREAM, family="JetBrains Mono, monospace", size=11),
        hovertemplate="<b>%{x}</b><br>MAPE: %{y:.2f}%<extra></extra>",
    ))
    fig_c = apply_chart_style(fig_c, height=300)
    fig_c.update_layout(
        yaxis=dict(title="MAPE (%)", title_font=dict(size=10, color=MUTED),
                   showgrid=True, gridcolor=BORDER),
        showlegend=False,
    )
    st.plotly_chart(fig_c, use_container_width=True)

    mejor = comp.loc[comp["MAPE"].idxmin()]
    st.markdown(f"""
    <div class="insight-callout">
        <strong>Veredicto:</strong> el modelo con menor error porcentual en validacion es
        <strong>{mejor['modelo']}</strong> con MAPE de <strong>{mejor['MAPE']:.1f}%</strong>.
        Para horizontes largos (60+ dias) Prophet suele degradarse menos; para horizontes cortos
        Random Forest aprovecha mejor los rezagos recientes.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# METODOLOGIA
# ============================================================
section_header(
    "ANEXO / METODO",
    "Como funcionan los modelos",
    "Detalle tecnico de las features y supuestos. Util para auditar o replicar el pipeline."
)

st.markdown(f"""
<div style="font-family: 'DM Sans'; font-size: 0.92rem; color: {CREAM}; line-height: 1.65;">
<p><strong style="color: {AMBER};">Random Forest (scikit-learn).</strong>
Construye features de calendario por cada dia (dia de la semana, dia del mes, mes, semana ISO,
indicador de fin de semana, indicador de quincena, indicador de diciembre), <strong>rezagos</strong>
de 1, 7, 14 y 30 dias, y <strong>promedios moviles</strong> de 7 y 30 dias. Entrena un ensemble de
400 arboles (max_depth=14). El pronostico al futuro se hace de forma <em>recursiva</em>: la
prediccion del dia N se reusa como insumo para el dia N+1. El intervalo de confianza se construye
con la desviacion estandar de los residuales en validacion: <code>[predicho ± 1.96σ]</code>.</p>

<p><strong style="color: {AMBER};">Prophet (Meta).</strong>
Descompone la serie en tendencia, estacionalidad semanal y estacionalidad anual con un componente
multiplicativo, con intervalos del 95% por bootstrap interno. Es el baseline estandar para
forecasting de demanda porque captura patrones repetitivos sin feature engineering manual.</p>

<p><strong style="color: {AMBER};">Validacion honesta.</strong>
Los ultimos <strong>{dias_test} dias</strong> del historico se reservan y <em>no</em> se le
muestran al modelo durante el entrenamiento. Las metricas MAE/RMSE/MAPE se calculan solo sobre
ese tramo — es la mejor estimacion del error futuro real, no del sobreajuste al pasado.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PIE
# ============================================================
st.markdown(f"""
<div class="colophon">
    <div>PROYECTO 6 · FORECASTING</div>
    <div>FUENTES · UCI ONLINE RETAIL II + BREAD BASKET EDIMBURGO</div>
    <div>PYTHON · STREAMLIT · SCIKIT-LEARN · PROPHET</div>
</div>
""", unsafe_allow_html=True)
