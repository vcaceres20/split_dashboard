# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Dashboard de Cumplimiento Plan", layout="wide")

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

mes_opciones = df[df["año"].isin(anio_sel)]["mes_nombre"] if anio_sel else df["mes_nombre"].iloc[0:0]
mes_sel, mask_mes = multiselect_con_nulos("Mes", df["mes_nombre"], opciones_override=mes_opciones)
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

# KPIs: totales según filtros del sidebar
if df_filt.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    df_kpi = df_filt.copy()
else:
    df_kpi = df_filt.copy()

# Totales Soles
sol_real = df_kpi["venta_real"].sum()
sol_plan = df_kpi["venta_plan"].sum()
sol_cumpl = sol_real / sol_plan if sol_plan > 0 else np.nan

# Totales Volumen
vol_real = df_kpi["vol_ton_real"].sum()
vol_plan = df_kpi["vol_ton_plan"].sum()
vol_cumpl = vol_real / vol_plan if vol_plan > 0 else np.nan

# Mostrar KPIs en cajas blancas con sombra
if df_kpi.empty:
    st.caption("KPIs para: Sin datos")
else:
    st.caption("KPIs según filtros seleccionados")
kpicol1, kpicol2, kpicol3 = st.columns([3, 3, 1])

card_style = (
    "background:white;padding:12px;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.08);text-align:center;"
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
    c1.markdown(render_card("Soles - Real", f"S/. {sol_real/1_000_000:,.1f} MM"), unsafe_allow_html=True)
    c2.markdown(render_card("Soles - Plan", f"S/. {sol_plan/1_000_000:,.1f} MM"), unsafe_allow_html=True)
    c3.markdown(render_card("Soles - Cumplimiento", f"{sol_cumpl:.1%}" if not np.isnan(sol_cumpl) else "NA"), unsafe_allow_html=True)

with kpicol2:
    c1, c2, c3 = st.columns(3)
    c1.markdown(render_card("Volumen - Real", f"{vol_real:,.0f} Ton"), unsafe_allow_html=True)
    c2.markdown(render_card("Volumen - Plan", f"{vol_plan:,.0f} Ton"), unsafe_allow_html=True)
    c3.markdown(render_card("Volumen - Cumplimiento", f"{vol_cumpl:.1%}" if not np.isnan(vol_cumpl) else "NA"), unsafe_allow_html=True)

# KPI: clientes distintos según filtros
clientes_distintos = df_kpi["cod_cliente_alicorp_actual"].nunique() if not df_kpi.empty else 0
with kpicol3:
    st.markdown(render_card("N° clientes", f"{clientes_distintos:,.0f}"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TABLA DE AN?LISIS DE REBATES POR RANGO (SEG?N FILTROS)
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.subheader(f"Análisis de Rebates por Rango de Cumplimiento - {tipo}")

# Usar datos segun filtros del sidebar
df_rebate = df_filt.copy()

# Definir rangos de cumplimiento y calcular datos por rango según tipo seleccionado
def asignar_rango_rebate(cumpl):
    if pd.isna(cumpl) or cumpl < 0.85:
        return "0-85%"
    elif cumpl < 0.95:
        return "85-95%"
    elif cumpl < 1.05:
        return "95-105%"
    elif cumpl < 1.15:
        return "105-115%"
    else:
        return "más de 115%"

# Usar el cumplimiento según el tipo seleccionado
if tipo_key == "sol":
    df_rebate["rango_rebate"] = df_rebate["cumplimiento_sol"].apply(asignar_rango_rebate)
    valor_col = "venta_real"
    col_label = "Venta (S/.)"
    pct_label = "% Venta"
else:
    df_rebate["rango_rebate"] = df_rebate["cumplimiento_vol"].apply(asignar_rango_rebate)
    valor_col = "vol_ton_real"
    col_label = "Vol (Ton)"
    pct_label = "% Vol"

# Agrupar por rango y calcular
rebate_data = df_rebate.groupby("rango_rebate", as_index=False).agg(
    valor=(valor_col, "sum"),
    clientes=("cod_cliente_alicorp_actual", "nunique")
)

# Calcular porcentaje
total_valor = rebate_data["valor"].sum()
rebate_data["pct_valor"] = rebate_data["valor"] / total_valor if total_valor > 0 else 0

# Definir Desc y % Desc vs. optimo según la imagen
desc_valores = {
    "0-85%": 0.0,
    "85-95%": 0.0,
    "95-105%": 0.009,
    "105-115%": 0.014,
    "más de 115%": 0.015
}

desc_vs_optimo = {
    "0-85%": None,  # -S/
    "85-95%": None,  # -S/
    "95-105%": None,
    "105-115%": 0.0052,
    "más de 115%": 0.0058
}

rebate_data["Desc"] = rebate_data["rango_rebate"].map(desc_valores)
rebate_data["pct_desc_vs_optimo"] = rebate_data["rango_rebate"].map(desc_vs_optimo)

# Calcular Rebate vs. optimo (siempre basado en volumen para el cálculo financiero)
def calcular_rebate(row):
    rango = row["rango_rebate"]

    if tipo_key == "sol":
        monto = row["valor"]
        if rango in ["0-85%", "85-95%"]:
            return -0.01 * monto
        elif rango == "95-105%":
            return 0
        else:  # "105-115%" o "mas de 115%"
            pct_desc = row["pct_desc_vs_optimo"]
            if pd.notna(pct_desc):
                return pct_desc * monto
            return 0

    # Volumen: mantener calculo financiero original
    vol = row["valor"]
    if rango in ["0-85%", "85-95%"]:
        return -0.01 * vol * 7000
    elif rango == "95-105%":
        return 0  # No hay descuento adicional
    else:  # "105-115%" o "mas de 115%"
        pct_desc = row["pct_desc_vs_optimo"]
        if pd.notna(pct_desc):
            return vol * (pct_desc / 100) * 7000
        return 0

rebate_data["rebate_vs_optimo"] = rebate_data.apply(calcular_rebate, axis=1)

# Ordenar por rango
orden_rangos = ["0-85%", "85-95%", "95-105%", "105-115%", "más de 115%"]
rebate_data["rango_rebate"] = pd.Categorical(rebate_data["rango_rebate"], categories=orden_rangos, ordered=True)
rebate_data = rebate_data.sort_values("rango_rebate")

# Preparar tabla para display
tabla_rebate = rebate_data[["rango_rebate", "clientes", "valor", "pct_valor", "Desc", "pct_desc_vs_optimo", "rebate_vs_optimo"]].copy()
tabla_rebate.columns = ["Cumplimiento", "N° Clientes", col_label, pct_label, "Desc", "% Desc vs. optimo", "Rebate vs. optimo"]

# Función para colorear filas según el rango
def color_fila_rebate(row):
    if row["Cumplimiento"] in ["0-85%"]:
        return ['background-color: #736867; color: white'] * len(row)
    elif row["Cumplimiento"] == "85-95%":
        return ['background-color: #EFFF1C; color: black'] * len(row)
    elif row["Cumplimiento"] == "95-105%":
        return ['background-color: #A4FF4A; color: black'] * len(row)
    elif row["Cumplimiento"] == "105-115%":
        return ['background-color: #FFBF9C; color: black'] * len(row)
    else:
        return ['background-color: #FF430F; color: white'] * len(row)

# Centrar la tabla usando columnas (dejar espacio a los lados)
col_izq, col_centro, col_der = st.columns([1, 3, 1])

with col_centro:
    # Formatear y mostrar tabla
    st.dataframe(
        tabla_rebate.style
        .format({
            col_label: "{:,.0f}",
            pct_label: "{:.0%}",
            "Desc": lambda x: f"{x:.1%}" if pd.notna(x) else "-",
            "% Desc vs. optimo": lambda x: f"{x:.2%}" if pd.notna(x) else "-S/",
            "Rebate vs. optimo": lambda x: f"S/. {x:,.0f}" if pd.notna(x) else "S/. 0"
        })
        .apply(color_fila_rebate, axis=1)
        .set_properties(**{'text-align': 'center'}),
        use_container_width=True,
        hide_index=True
    )

    st.caption(f"**Total Rebate vs. Óptimo:** S/. {rebate_data['rebate_vs_optimo'].sum():,.0f}")

# ======================================================
# HISTOGRAMA
# ======================================================
st.markdown("---")
st.subheader(f"Distribución de cumplimiento ({'Facturación' if tipo_key=='sol' else 'Volumen'})")

tabla_hist = tabla_histograma(df_filt, tipo=tipo_key)

if not tabla_hist.empty:
    inter_col = f"intervalo_{tipo_key}"
    y_title = "Monto real (MM S/.)" if tipo_key == "sol" else "Volumen real (Ton)"

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
st.subheader(f"Distribución por Nivel y Región / Canal ({tipo})")

col_r, col_c = st.columns(2)

with col_r:
    st.caption("Nivel vs Región")
    mat_reg = matriz_nivel_x_dimension(df_filt, "des_oficina_venta_alicorp", tipo_key)
    if not mat_reg.empty:
        st.dataframe(
            mat_reg.style
            .format("{:.0%}")
            .set_properties(**{'text-align': 'center'}),
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
            .format("{:.0%}")
            .set_properties(**{'text-align': 'center'}),
            use_container_width=True
        )
    else:
        st.write("Sin datos.")


# ======================================================
# Curva de Evolución
# ======================================================
st.markdown("---")
st.subheader(f"Curva de Evolución - {tipo}")

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

nivel_col = "nivel_cum_sol" if tipo_key == "sol" else "nivel_cum_vol"
valor_col = "venta_real" if tipo_key == "sol" else "vol_ton_real"
metrica_label = "venta real" if tipo_key == "sol" else "volumen real"

df_niv = df_filt.dropna(subset=[nivel_col]).copy()

if df_niv.empty:
    st.info("No hay datos para la evolución de niveles con los filtros actuales.")
else:
    evo = (
        df_niv.groupby(
            ["periodo_mes", "año", "mes", nivel_col],
            observed=True,
            dropna=False,
        )
        .agg(valor=(valor_col, "sum"))
        .reset_index()
    )

    evo["periodo_label"] = evo["mes"].map(MES_MAP) + " " + evo["año"].astype(str)
    periodos_ordenados = [f"{MES_MAP[p.month]} {p.year}" for p in sorted(evo["periodo_mes"].unique())]
    evo["pct"] = evo["valor"] / evo.groupby("periodo_mes")["valor"].transform("sum")
    evo = evo.dropna(subset=[nivel_col]).copy()
    evo["Nivel"] = evo[nivel_col].astype(int).astype(str)

    colores_nivel = colores_leyenda

    chart_evo = (
        alt.Chart(evo)
        .mark_area()
        .encode(
            x=alt.X("periodo_label:N", sort=periodos_ordenados, title="Periodo"),
            y=alt.Y("pct:Q", stack="zero", axis=alt.Axis(format="%", title=f"% {metrica_label}")),
            color=alt.Color(
                "Nivel:N",
                title="Nivel",
                scale=alt.Scale(domain=["1", "2", "3", "4", "5"], range=colores_nivel),
            ),
            tooltip=[
                alt.Tooltip("periodo_label:N", title="Periodo"),
                alt.Tooltip("Nivel:N"),
                alt.Tooltip("pct:Q", format=".1%", title=f"% {metrica_label}"),
            ],
        )
    )
    st.altair_chart(chart_evo, use_container_width=True)

    tabla_niv = evo.pivot_table(index="periodo_label", columns="Nivel", values="pct", aggfunc="sum")
    orden_presentes = [p for p in periodos_ordenados if p in tabla_niv.index]
    tabla_niv = tabla_niv.reindex(orden_presentes).fillna(0)

    # Renombrar columnas para que digan "Nivel 1", "Nivel 2", etc.
    tabla_niv.columns = [f"Nivel {col}" for col in tabla_niv.columns]
    tabla_niv.index.name = "Periodo"

    st.caption(f"Tabla % {metrica_label} por nivel y mes")
    st.dataframe(
        tabla_niv.style
        .format("{:.1%}")
        .set_properties(**{'text-align': 'center'}),
        use_container_width=True
    )
