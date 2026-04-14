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
    styler_map_compat,
    multiselect_con_nulos,
    load_df_cus,
)

# Ahora `app.py` contiene la Curva de Evolución (la página principal se renombra funcionalmente)
st.title("Seguimiento de Cumplimiento - Canal Directo")

# Cargar datos
df = load_df_cus()


def _restore_radio(widget_key, memory_key, options, default):
    opts = list(options)
    if widget_key not in st.session_state:
        prev = st.session_state.get(memory_key, default)
        st.session_state[widget_key] = prev if prev in opts else (default if default in opts else opts[0])
    elif st.session_state.get(widget_key) not in opts:
        st.session_state[widget_key] = default if default in opts else opts[0]


def _restore_multiselect(widget_key, memory_key, options, default):
    opts = list(options)
    default_vals = list(default) if isinstance(default, (list, tuple, set)) else [default]
    default_vals = [v for v in default_vals if v in opts]

    if widget_key not in st.session_state:
        prev = st.session_state.get(memory_key, default_vals)
        if not isinstance(prev, list):
            prev = [prev]
        st.session_state[widget_key] = [v for v in prev if v in opts]
    else:
        cur = st.session_state.get(widget_key, [])
        if not isinstance(cur, list):
            cur = [cur]
        st.session_state[widget_key] = [v for v in cur if v in opts]

    if not st.session_state.get(widget_key) and default_vals:
        st.session_state[widget_key] = default_vals

# ======================================================
# 3. SIDEBAR – FILTROS
# ======================================================
st.sidebar.title("Filtros")

_restore_radio("ov_tipo", "ov_mem_tipo", ["Soles", "Volumen"], "Soles")
tipo = st.sidebar.radio("Tipo", ["Soles", "Volumen"], key="ov_tipo")
st.session_state["ov_mem_tipo"] = tipo
tipo_key = "sol" if tipo == "Soles" else "vol"

anios = sorted(df["a?o"].dropna().unique())
default_anios = [a for a in [2025, 2026] if a in anios]
if not default_anios:
    default_anios = anios
_restore_multiselect("ov_anio", "ov_mem_anio", anios, default_anios)
anio_sel = st.sidebar.multiselect("Año", anios, key="ov_anio")
st.session_state["ov_mem_anio"] = anio_sel

df_periodos = df[df["a?o"].isin(anio_sel)] if anio_sel else df.iloc[0:0]
periodos_ordenados = sorted(df_periodos["periodo_mes"].dropna().unique().tolist())
periodo_labels = [f"{MES_MAP.get(p.month, p.strftime('%m'))} {p.year}" for p in periodos_ordenados]
periodo_map = dict(zip(periodo_labels, periodos_ordenados))
default_labels = periodo_labels[-7:] if len(periodo_labels) > 7 else periodo_labels
_restore_multiselect("ov_periodo", "ov_mem_periodo", periodo_labels, default_labels)
periodo_sel = st.sidebar.multiselect("Mes-Año", periodo_labels, key="ov_periodo")
st.session_state["ov_mem_periodo"] = periodo_sel
st.sidebar.checkbox(
    "Seleccionar todo Mes-Año",
    key="sel_all_periodo_overview",
    on_change=lambda: st.session_state.__setitem__("ov_periodo", periodo_labels),
)
periodo_sel_dt = [periodo_map[p] for p in periodo_sel]
mask_periodo = df["periodo_mes"].isin(periodo_sel_dt)

abc_opciones = sorted(df["ABC"].dropna().unique().tolist())
if df["ABC"].isna().any():
    abc_opciones.append(SIN_DATO_LABEL)
_restore_multiselect("ov_abc", "ov_mem_abc", abc_opciones, abc_opciones)
abc_sel, mask_abc = multiselect_con_nulos("ABC", df["ABC"], key="ov_abc")
st.session_state["ov_mem_abc"] = abc_sel

region_opciones = sorted(df["des_oficina_venta_alicorp"].dropna().unique().tolist())
if df["des_oficina_venta_alicorp"].isna().any():
    region_opciones.append(SIN_DATO_LABEL)
_restore_multiselect("ov_region", "ov_mem_region", region_opciones, region_opciones)
region_sel, mask_region = multiselect_con_nulos("Región", df["des_oficina_venta_alicorp"], key="ov_region")
st.session_state["ov_mem_region"] = region_sel

canal_opciones = sorted(df["des_grupo_precio_alicorp"].dropna().unique().tolist())
if df["des_grupo_precio_alicorp"].isna().any():
    canal_opciones.append(SIN_DATO_LABEL)
_restore_multiselect("ov_canal", "ov_mem_canal", canal_opciones, canal_opciones)
canal_sel, mask_canal = multiselect_con_nulos("Canal", df["des_grupo_precio_alicorp"], key="ov_canal")
st.session_state["ov_mem_canal"] = canal_sel

zona_opciones = sorted(df["des_grupo_vendedor_alicorp"].dropna().unique().tolist())
if df["des_grupo_vendedor_alicorp"].isna().any():
    zona_opciones.append(SIN_DATO_LABEL)
_restore_multiselect("ov_zona", "ov_mem_zona", zona_opciones, zona_opciones)
zona_sel, mask_zona = multiselect_con_nulos(
    "Zona", df["des_grupo_vendedor_alicorp"], opciones_override=zona_opciones, key="ov_zona"
)
st.session_state["ov_mem_zona"] = zona_sel
st.sidebar.checkbox(
    "Seleccionar todo Zona",
    key="sel_all_zona_overview",
    on_change=lambda: st.session_state.__setitem__("ov_zona", zona_opciones),
)

jcc_opciones = sorted(df["JCC"].dropna().unique().tolist())
if df["JCC"].isna().any():
    jcc_opciones.append(SIN_DATO_LABEL)
_restore_multiselect("ov_jcc", "ov_mem_jcc", jcc_opciones, jcc_opciones)
jcc_sel, mask_jcc = multiselect_con_nulos("JCC", df["JCC"], opciones_override=jcc_opciones, key="ov_jcc")
st.session_state["ov_mem_jcc"] = jcc_sel
st.sidebar.checkbox(
    "Seleccionar todo JCC",
    key="sel_all_jcc_overview",
    on_change=lambda: st.session_state.__setitem__("ov_jcc", jcc_opciones),
)

# Aplicar filtros
df_filt = df[
    df["año"].isin(anio_sel)
    & mask_periodo
    & mask_abc
    & mask_region
    & mask_canal
    & mask_zona
    & mask_jcc
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
            tooltip=[
                alt.Tooltip(f"{inter_col}:N", title="Intervalo"),
                alt.Tooltip("valor:Q", title="Valor", format=",.0f"),
            ]
        )
    )
    st.altair_chart(chart_hist, width='stretch')
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

colores_leyenda_rev = list(reversed(colores_leyenda))
etiquetas_leyenda_rev = list(reversed(etiquetas_leyenda))

cols_ley = st.columns(len(colores_leyenda_rev))
for c, color, label in zip(cols_ley, colores_leyenda_rev, etiquetas_leyenda_rev):
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

    colores_nivel = colores_leyenda_rev
    niveles_orden = ["5", "4", "3", "2", "1"]

    chart_evo = (
        alt.Chart(evo)
        .mark_area()
        .encode(
            x=alt.X("periodo_label:N", sort=periodos_ordenados, title="Periodo"),
            y=alt.Y("pct:Q", stack="zero", axis=alt.Axis(format="%", title=f"% {metrica_label}")),
            color=alt.Color(
                "Nivel:N",
                title="Nivel",
                scale=alt.Scale(domain=niveles_orden, range=colores_nivel),
                sort=niveles_orden,
            ),
            tooltip=[
                alt.Tooltip("periodo_label:N", title="Periodo"),
                alt.Tooltip("Nivel:N"),
                alt.Tooltip("pct:Q", format=".1%", title=f"% {metrica_label}"),
            ],
        )
    )

    chart_text = (
        alt.Chart(evo)
        .transform_stack(
            stack="pct",
            groupby=["periodo_label"],
            sort=[alt.SortField("Nivel", order="descending")],
            as_=["y0", "y1"],
        )
        .mark_text(size=10, color="black", dy=-6)
        .encode(
            x=alt.X("periodo_label:N", sort=periodos_ordenados),
            y=alt.Y("y0:Q"),
            text=alt.Text("pct:Q", format=".0%"),
            detail="Nivel:N",
            tooltip=[],
        )
    )
    st.altair_chart(chart_evo + chart_text, width='stretch')

    tabla_niv = evo.pivot_table(index="periodo_label", columns="Nivel", values="pct", aggfunc="sum")
    orden_presentes = [p for p in periodos_ordenados if p in tabla_niv.index]
    tabla_niv = tabla_niv.reindex(orden_presentes).fillna(0)

    # Renombrar columnas para que digan "Nivel 1", "Nivel 2", etc.
    tabla_niv.columns = [f"Nivel {col}" for col in tabla_niv.columns]
    # Reordenar columnas de nivel (5 a 1)
    orden_cols = [f"Nivel {n}" for n in ["5", "4", "3", "2", "1"] if f"Nivel {n}" in tabla_niv.columns]
    tabla_niv = tabla_niv[orden_cols]
    tabla_niv.index.name = "Periodo"

    estilos_por_nivel = {
        "Nivel 5": "background-color: #FF430F; color: white",
        "Nivel 4": "background-color: #FFBF9C; color: black",
        "Nivel 3": "background-color: #A4FF4A; color: black",
        "Nivel 2": "background-color: #EFFF1C; color: black",
        "Nivel 1": "background-color: #736867; color: white",
    }

    styled_niv = (
        tabla_niv.style
        .format("{:.0%}")
        .set_properties(**{"text-align": "center"})
    )
    for col in tabla_niv.columns:
        if col in estilos_por_nivel:
            color_css = estilos_por_nivel[col]
            styled_niv = styler_map_compat(styled_niv, lambda _, css=color_css: css, subset=[col])

    st.caption(f"Tabla % {metrica_label} por nivel y mes")
    st.dataframe(
        styled_niv,
        use_container_width=True
    )

