import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from shared import (
    MES_MAP,
    SIN_DATO_LABEL,
    ensure_multiselect_state,
    multiselect_con_nulos,
    load_df_base_raw,
    load_df_cus,
)

st.set_page_config(page_title="Resumen", layout="wide")
st.title("Resumen")

# Cargar datos
df_query = load_df_base_raw()
df = load_df_cus()

st.download_button(
    "Descargar base",
    df_query.to_csv(index=False),
    file_name="base_query.csv",
    mime="text/csv",
)

st.sidebar.title("Filtros")

anios = sorted(df["año"].dropna().unique())
default_anios = [a for a in [2025, 2026] if a in anios]
if not default_anios:
    default_anios = anios
ensure_multiselect_state("res_anio", anios, default=default_anios)
anio_sel = st.sidebar.multiselect("Año", anios, key="res_anio")

df_periodos = df[df["año"].isin(anio_sel)] if anio_sel else df.iloc[0:0]
periodos_ordenados = sorted(df_periodos["periodo_mes"].dropna().unique().tolist())
periodo_labels = [f"{MES_MAP.get(p.month, p.strftime('%m'))} {p.year}" for p in periodos_ordenados]
periodo_map = dict(zip(periodo_labels, periodos_ordenados))
default_labels = periodo_labels[-7:] if len(periodo_labels) > 7 else periodo_labels
ensure_multiselect_state("res_periodo", periodo_labels, default=default_labels)
periodo_sel = st.sidebar.multiselect("Mes-Año", periodo_labels, key="res_periodo")
st.sidebar.checkbox(
    "Seleccionar todo Mes-Año",
    key="sel_all_periodo_resumen",
    on_change=lambda: st.session_state.__setitem__("res_periodo", periodo_labels),
)
periodo_sel_dt = [periodo_map[p] for p in periodo_sel]
mask_periodo = df["periodo_mes"].isin(periodo_sel_dt)

abc_sel, mask_abc = multiselect_con_nulos("ABC", df["ABC"], key="res_abc")
region_sel, mask_region = multiselect_con_nulos("Región", df["des_oficina_venta_alicorp"], key="res_region")
canal_sel, mask_canal = multiselect_con_nulos("Canal", df["des_grupo_precio_alicorp"], key="res_canal")
zona_opciones = sorted(df["des_grupo_vendedor_alicorp"].dropna().unique().tolist())
if df["des_grupo_vendedor_alicorp"].isna().any():
    zona_opciones.append(SIN_DATO_LABEL)
zona_sel, mask_zona = multiselect_con_nulos(
    "Zona", df["des_grupo_vendedor_alicorp"], opciones_override=zona_opciones, key="res_zona"
)
st.sidebar.checkbox(
    "Seleccionar todo Zona",
    key="sel_all_zona_resumen",
    on_change=lambda: st.session_state.__setitem__("res_zona", zona_opciones),
)

jcc_opciones = sorted(df["JCC"].dropna().unique().tolist())
if df["JCC"].isna().any():
    jcc_opciones.append(SIN_DATO_LABEL)
jcc_sel, mask_jcc = multiselect_con_nulos("JCC", df["JCC"], opciones_override=jcc_opciones, key="res_jcc")
st.sidebar.checkbox(
    "Seleccionar todo JCC",
    key="sel_all_jcc_resumen",
    on_change=lambda: st.session_state.__setitem__("res_jcc", jcc_opciones),
)

df_filt = df[
    df["año"].isin(anio_sel)
    & mask_periodo
    & mask_abc
    & mask_region
    & mask_canal
    & mask_zona
    & mask_jcc
].copy()

if df_filt.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

st.markdown("---")
st.subheader("Matriz Zona x Nivel (Volumen Real %)")

df_zona_nivel = df_filt.copy()
df_zona_nivel["zona"] = df_zona_nivel["des_grupo_vendedor_alicorp"].fillna(SIN_DATO_LABEL)
df_zona_nivel["nivel_vol_num"] = pd.to_numeric(df_zona_nivel["nivel_cum_vol"], errors="coerce")
df_zona_nivel = df_zona_nivel[df_zona_nivel["nivel_vol_num"].between(1, 5, inclusive="both")].copy()

if df_zona_nivel.empty:
    st.info("No hay datos de volumen para construir la matriz Zona x Nivel.")
else:
    mtx = (
        df_zona_nivel.groupby(["zona", "nivel_vol_num"], as_index=False)
        .agg(vol_real=("vol_ton_real", "sum"))
    )
    pt = (
        mtx.pivot_table(
            index="zona",
            columns="nivel_vol_num",
            values="vol_real",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
    )

    row_total = pt.sum(axis=1).replace(0, np.nan)
    pt_pct = pt.div(row_total, axis=0).fillna(0)
    pt_pct = pt_pct.sort_values(by=5, ascending=False)

    zonas_ordenadas = pt_pct.index.tolist()
    heat = (
        pt_pct.reset_index()
        .melt(id_vars="zona", var_name="Nivel", value_name="Pct")
    )
    heat["Nivel"] = heat["Nivel"].astype(int)

    colores_nivel = ["#736867", "#EFFF1C", "#A4FF4A", "#FFBF9C", "#FF430F"]
    base = alt.Chart(heat).encode(
        x=alt.X(
            "Nivel:O",
            title="Nivel",
            sort=[1, 2, 3, 4, 5],
            axis=alt.Axis(labelColor="black", titleColor="black"),
        ),
        y=alt.Y(
            "zona:N",
            title="Zona",
            sort=zonas_ordenadas,
            axis=alt.Axis(labelColor="black", titleColor="black"),
        ),
    )
    rect = base.mark_rect(stroke="white").encode(
        color=alt.Color(
            "Nivel:O",
            scale=alt.Scale(domain=[1, 2, 3, 4, 5], range=colores_nivel),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("zona:N", title="Zona"),
            alt.Tooltip("Nivel:O", title="Nivel"),
            alt.Tooltip("Pct:Q", title="% Vol Real", format=".0%"),
        ],
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("Pct:Q", format=".0%"),
        color=alt.condition(
            "datum.Nivel == 1 || datum.Nivel == 5",
            alt.value("white"),
            alt.value("black"),
        ),
    )
    st.altair_chart((rect + text).properties(height=max(260, 28 * len(zonas_ordenadas))), use_container_width=True)

# Tabla mensual: año, mes, volumen plan/real, cumplimiento
resumen = (
    df_filt.groupby(["periodo_mes", "año", "mes"], as_index=False)
    .agg(
        vol_plan=("vol_ton_plan", "sum"),
        vol_real=("vol_ton_real", "sum"),
    )
)
resumen["periodo_label"] = resumen["mes"].map(MES_MAP) + " " + resumen["año"].astype(str)
resumen["cumplimiento"] = np.where(resumen["vol_plan"] > 0, resumen["vol_real"] / resumen["vol_plan"], np.nan)
resumen = resumen.sort_values("periodo_mes")

tabla = resumen[["año", "periodo_label", "vol_plan", "vol_real", "cumplimiento"]].copy()
tabla.columns = ["Año", "Mes", "Sum Vol Plan", "Sum Vol Real", "Cumplimiento"]

def _color_cumpl(val):
    if pd.isna(val):
        return "background-color: white; color: black"
    try:
        val_float = float(val)
    except Exception:
        return "background-color: white; color: black"
    if val_float < 0.85:
        return "background-color: #736867; color: white"
    elif val_float < 0.95:
        return "background-color: #EFFF1C; color: black"
    elif val_float < 1.05:
        return "background-color: #A4FF4A; color: black"
    elif val_float < 1.15:
        return "background-color: #FFBF9C; color: black"
    else:
        return "background-color: #FF430F; color: white"

col_tabla, col_chart = st.columns([1.1, 1.5])

with col_tabla:
    st.subheader("Resumen Mensual - Volumen")
    st.dataframe(
        tabla.style
        .format({
            "Sum Vol Plan": "{:,.0f}",
            "Sum Vol Real": "{:,.0f}",
            "Cumplimiento": "{:.1%}",
        })
        .applymap(_color_cumpl, subset=["Cumplimiento"])
        .set_properties(**{"text-align": "center"}),
        use_container_width=True,
        hide_index=True,
        height=max(220, 36 + len(tabla) * 24),
    )

with col_chart:
    st.subheader("Evolución Plan vs Real (Volumen)")
    chart_df = resumen[["periodo_label", "vol_plan", "vol_real"]].copy()
    chart_df = chart_df.melt(
        id_vars=["periodo_label"],
        value_vars=["vol_plan", "vol_real"],
        var_name="Serie",
        value_name="Valor",
    )
    chart_df["Serie"] = chart_df["Serie"].map({"vol_plan": "Plan", "vol_real": "Real"})

    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("periodo_label:N", sort=periodo_labels, title="Periodo"),
            y=alt.Y("Valor:Q", title=""),
            color=alt.Color(
                "Serie:N",
                title="Serie",
                scale=alt.Scale(
                    domain=["Plan", "Real"],
                    range=["#2E7D32", "#EF6C00"],
                ),
            ),
            tooltip=["periodo_label", "Serie", "Valor"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.subheader("Resumen Mensual - Soles")

resumen_sol = (
    df_filt.groupby(["periodo_mes", "año", "mes"], as_index=False)
    .agg(
        sol_plan=("venta_plan", "sum"),
        sol_real=("venta_real", "sum"),
    )
)
resumen_sol["periodo_label"] = resumen_sol["mes"].map(MES_MAP) + " " + resumen_sol["año"].astype(str)
resumen_sol["cumplimiento"] = np.where(resumen_sol["sol_plan"] > 0, resumen_sol["sol_real"] / resumen_sol["sol_plan"], np.nan)
resumen_sol = resumen_sol.sort_values("periodo_mes")

tabla_sol = resumen_sol[["año", "periodo_label", "sol_plan", "sol_real", "cumplimiento"]].copy()
tabla_sol.columns = ["Año", "Mes", "Sum Sol Plan", "Sum Sol Real", "Cumplimiento"]

col_tabla_sol, col_chart_sol = st.columns([1.1, 1.5])

with col_tabla_sol:
    st.dataframe(
        tabla_sol.style
        .format({
            "Sum Sol Plan": "S/. {:,.0f}",
            "Sum Sol Real": "S/. {:,.0f}",
            "Cumplimiento": "{:.1%}",
        })
        .applymap(_color_cumpl, subset=["Cumplimiento"])
        .set_properties(**{"text-align": "center"}),
        use_container_width=True,
        hide_index=True,
        height=max(220, 36 + len(tabla_sol) * 24),
    )

with col_chart_sol:
    st.subheader("Evolución Plan vs Real (Soles)")
    chart_df_sol = resumen_sol[["periodo_label", "sol_plan", "sol_real"]].copy()
    chart_df_sol = chart_df_sol.melt(
        id_vars=["periodo_label"],
        value_vars=["sol_plan", "sol_real"],
        var_name="Serie",
        value_name="Valor",
    )
    chart_df_sol["Serie"] = chart_df_sol["Serie"].map({"sol_plan": "Plan", "sol_real": "Real"})

    chart_sol = (
        alt.Chart(chart_df_sol)
        .mark_line(point=True)
        .encode(
            x=alt.X("periodo_label:N", sort=periodo_labels, title="Periodo"),
            y=alt.Y("Valor:Q", title=""),
            color=alt.Color(
                "Serie:N",
                title="Serie",
                scale=alt.Scale(
                    domain=["Plan", "Real"],
                    range=["#2E7D32", "#EF6C00"],
                ),
            ),
            tooltip=["periodo_label", "Serie", "Valor"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart_sol, use_container_width=True)
