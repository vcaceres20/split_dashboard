# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Dashboard Cumplimiento Plan", layout="wide")

from shared import (
    LABELS_INTERVALO,
    MES_MAP,
    SIN_DATO_LABEL,
    tabla_histograma,
    matriz_nivel_x_dimension,
    multiselect_con_nulos,
    load_df_cus,
)

# Ahora `app.py` contiene la Curva de Evolución (la página principal se renombra funcionalmente)
st.title("Seguimiento de Cumplimiento - Canal Directo")

# Cargar datos
df = load_df_cus()

# ======================================================
# 3. SIDEBAR – FILTROS
# ======================================================
st.sidebar.title("Filtros")

tipo = st.sidebar.radio("Tipo", ["Soles", "Volumen"])
tipo_key = "sol" if tipo == "Soles" else "vol"

anios = sorted(df["año"].dropna().unique())
anio_sel = st.sidebar.multiselect("Año", anios, default=anios)

mes_sel, mask_mes = multiselect_con_nulos("Mes", df["mes_nombre"])
abc_sel, mask_abc = multiselect_con_nulos("ABC", df["ABC"])
region_sel, mask_region = multiselect_con_nulos("Región", df["des_oficina_venta_alicorp"])
canal_sel, mask_canal = multiselect_con_nulos("Canal", df["des_grupo_precio_alicorp"])
zona_sel, mask_zona = multiselect_con_nulos("Zona", df["des_grupo_vendedor_alicorp"])

# Aplicar filtros
df_filt = df[
    df["año"].isin(anio_sel)
    & mask_mes
    & mask_abc
    & mask_region
    & mask_canal
    & mask_zona
].copy()

# KPIs estáticos: último periodo disponible (Soles y Volumen)
last_periodo = df["periodo_mes"].max()
df_last = df[df["periodo_mes"] == last_periodo]

# Totales Soles
sol_real = df_last["venta_real"].sum()
sol_plan = df_last["venta_plan"].sum()
sol_cumpl = sol_real / sol_plan if sol_plan > 0 else np.nan

# Totales Volumen
vol_real = df_last["peso_real"].sum()
vol_plan = df_last["peso_plan"].sum()
vol_cumpl = vol_real / vol_plan if vol_plan > 0 else np.nan

# Mostrar KPIs en cajas blancas con sombra
st.caption(f"KPIs para: {MES_MAP[last_periodo.month]} {last_periodo.year}")
kpicol1, kpicol2 = st.columns(2)

card_style = (
    "background:white;padding:12px;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.08);"
)

def render_card(title, value, subtitle=""):
    subtitle_html = f"<div style='font-size:12px;color:#8a8a8a;margin-top:6px'>{subtitle}</div>" if subtitle else ""
    return (
        f"<div style=\"{card_style}\">"
        f"<div style='font-size:13px;color:#6b6b6b'>{title}</div>"
        f"<div style='font-size:20px;font-weight:700;margin-top:6px'>{value}</div>"
        f"{subtitle_html}"
        f"</div>"
    )

with kpicol1:
    c1, c2, c3 = st.columns(3)
    c1.markdown(render_card("Soles - Real", f"S/. {sol_real:,.0f}"), unsafe_allow_html=True)
    c2.markdown(render_card("Soles - Plan", f"S/. {sol_plan:,.0f}"), unsafe_allow_html=True)
    c3.markdown(render_card("Soles - Cumplimiento", f"{sol_cumpl:.1%}" if not np.isnan(sol_cumpl) else "NA"), unsafe_allow_html=True)

with kpicol2:
    c1, c2, c3 = st.columns(3)
    c1.markdown(render_card("Volumen - Real", f"{vol_real:,.0f} Ton"), unsafe_allow_html=True)
    c2.markdown(render_card("Volumen - Plan", f"{vol_plan:,.0f} Ton"), unsafe_allow_html=True)
    c3.markdown(render_card("Volumen - Cumplimiento", f"{vol_cumpl:.1%}" if not np.isnan(vol_cumpl) else "NA"), unsafe_allow_html=True)

# ======================================================
# HISTOGRAMA
# ======================================================
st.markdown("---")
st.subheader(f"Distribución de cumplimiento ({'Facturación' if tipo_key=='sol' else 'Volumen'})")

tabla_hist = tabla_histograma(df_filt, tipo=tipo_key)

if not tabla_hist.empty:
    inter_col = f"intervalo_{tipo_key}"
    y_title = "Monto real (MM S/.)" if tipo_key == "sol" else "Volumen real (miles)"

    def asignar_color(intervalo):
        if intervalo in ["Menor a 0%", "0-15%", "15-25%", "25-35%", "35-45%", "45-55%", "55-65%", "65-75%", "75-85%"]:
            return "#736867"
        elif intervalo in ["85-95%"]:
            return "#EFFF1C"
        elif intervalo in ["95-105%"]:
            return "#A4FF4A"
        elif intervalo in ["105-115%"]:
            return "#FFBF9C"
        else:
            return "#FF430F"

    tabla_hist["color"] = tabla_hist[inter_col].apply(asignar_color)

    chart_hist = (
        alt.Chart(tabla_hist)
        .mark_bar()
        .encode(
            x=alt.X(inter_col, sort=LABELS_INTERVALO,
                    title="Intervalo de cumplimiento"),
            y=alt.Y("valor", title=y_title),
            color=alt.Color(
                "color:N",
                scale=alt.Scale(
                    domain=tabla_hist["color"].unique().tolist(),
                    range=tabla_hist["color"].unique().tolist()
                ),
                legend=None
            ),
            tooltip=[inter_col, "valor"]
        )
    )
    st.altair_chart(chart_hist, use_container_width=True)
else:
    st.info("No hay datos para los filtros seleccionados.")


# ======================================================
# 7. HEATMAPS NIVEL x REGIÓN / CANAL (según tipo)
# ======================================================
st.subheader(f"Distribución por nivel y Región / Canal ({tipo})")

col_r, col_c = st.columns(2)

with col_r:
    st.caption("Nivel vs Región")
    mat_reg = matriz_nivel_x_dimension(df_filt, "des_oficina_venta_alicorp", tipo_key)
    if not mat_reg.empty:
        st.dataframe(
            mat_reg.style
            .format("{:.0%}"),
            use_container_width=True
        )
    else:
        st.write("Sin datos.")

with col_c:
    st.caption("Nivel vs Canal")
    mat_can = matriz_nivel_x_dimension(df_filt, "des_grupo_precio_alicorp", tipo_key)
    if not mat_can.empty:
        st.dataframe(
            mat_can.style
            .format("{:.0%}"),
            use_container_width=True
        )
    else:
        st.write("Sin datos.")


# ======================================================
# Curva de Evolución
# ======================================================
st.markdown("---")

# Leyenda de niveles (colores coincidentes con el histograma)
colores_leyenda = ["#736867", "#EFFF1C", "#A4FF4A", "#FFBF9C", "#FF430F"]
etiquetas_leyenda = [
    "Nivel 1 — 0-85%",
    "Nivel 2 — 85-95%",
    "Nivel 3 — 95-105%",
    "Nivel 4 — 105-115%",
    "Nivel 5 — Más de 115%",
]

cols_ley = st.columns(len(colores_leyenda))
for c, color, label in zip(cols_ley, colores_leyenda, etiquetas_leyenda):
    c.markdown(
        f"<div style='display:flex;align-items:center'><div style='background:{color};width:18px;height:18px;border-radius:3px;margin-right:8px;'></div><div>{label}</div></div>",
        unsafe_allow_html=True,
    )

df_niv = df_filt.dropna(subset=["nivel_cum_sol"]).copy()

if df_niv.empty:
    st.info("No hay datos para la evolución de niveles con los filtros actuales.")
else:
    evo = (
        df_niv.groupby(
            ["periodo_mes", "año", "mes", "nivel_cum_sol"],
            observed=True,
            dropna=False,
        )
        .agg(venta_real=("venta_real", "sum"))
        .reset_index()
    )

    evo["periodo_label"] = evo["mes"].map(MES_MAP) + " " + evo["año"].astype(str)
    periodos_ordenados = [f"{MES_MAP[p.month]} {p.year}" for p in sorted(evo["periodo_mes"].unique())]
    evo["pct"] = evo["venta_real"] / evo.groupby("periodo_mes")["venta_real"].transform("sum")
    evo = evo.dropna(subset=["nivel_cum_sol"]).copy()
    evo["Nivel"] = evo["nivel_cum_sol"].astype(int).astype(str)

    colores_nivel = colores_leyenda

    chart_evo = (
        alt.Chart(evo)
        .mark_area()
        .encode(
            x=alt.X("periodo_label:N", sort=periodos_ordenados, title="Periodo"),
            y=alt.Y("pct:Q", stack="zero", axis=alt.Axis(format="%", title="% venta real")),
            color=alt.Color(
                "Nivel:N",
                title="Nivel",
                scale=alt.Scale(domain=["1", "2", "3", "4", "5"], range=colores_nivel),
            ),
            tooltip=[
                alt.Tooltip("periodo_label:N", title="Periodo"),
                alt.Tooltip("Nivel:N"),
                alt.Tooltip("pct:Q", format=".1%", title="% venta"),
            ],
        )
    )
    st.altair_chart(chart_evo, use_container_width=True)

    tabla_niv = evo.pivot_table(index="periodo_label", columns="Nivel", values="pct", aggfunc="sum")
    orden_presentes = [p for p in periodos_ordenados if p in tabla_niv.index]
    tabla_niv = tabla_niv.reindex(orden_presentes).fillna(0)
    tabla_niv.index.name = "Periodo"

    st.caption("Tabla % venta real por nivel y mes")
    st.dataframe(tabla_niv.style.format("{:.1%}"), use_container_width=True)