import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import random

from shared import (
    MES_MAP,
    SIN_DATO_LABEL,
    ensure_multiselect_state,
    ensure_radio_state,
    multiselect_con_nulos,
    load_df_cus,
    load_df_with_categoria,
    load_df_sugeridos,
)

st.set_page_config(page_title="Evolución Clientes JCC", layout="wide")
st.title("Evolución de Cumplimiento - Clientes por JCC")

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


# Cargar datos
df = load_df_cus()
df_cat = load_df_with_categoria()
df_sug = load_df_sugeridos()

# Sidebar filtros
st.sidebar.title("Filtros")

# Filtro de Tipo (Soles o Volumen)
_restore_radio("evo_tipo", "evo_mem_tipo", ["Soles", "Volumen"], "Soles")
tipo = st.sidebar.radio("Tipo", ["Soles", "Volumen"], key="evo_tipo")
st.session_state["evo_mem_tipo"] = tipo
tipo_key = "sol" if tipo == "Soles" else "vol"

# Filtro de Año - por defecto 2025 y 2026 (si existen)
anios = sorted(df["año"].dropna().unique(), reverse=True)
default_anios = [a for a in [2025, 2026] if a in anios]
if not default_anios:
    default_anios = [anios[0]] if anios else []
_restore_multiselect("evo_anio", "evo_mem_anio", anios, default_anios)
anio_sel = st.sidebar.multiselect("Año", anios, key="evo_anio")
st.session_state["evo_mem_anio"] = anio_sel

# Filtro de Periodo (Mes-Año) - por defecto últimos 7 periodos
df_periodos = df[df["año"].isin(anio_sel)] if anio_sel else df.iloc[0:0]
periodos_ordenados = sorted(df_periodos["periodo_mes"].dropna().unique().tolist())
default_periodos = periodos_ordenados[-7:] if len(periodos_ordenados) > 7 else periodos_ordenados
periodo_labels = [f"{MES_MAP.get(p.month, p.strftime('%m'))} {p.year}" for p in periodos_ordenados]
periodo_map = dict(zip(periodo_labels, periodos_ordenados))
default_labels = [f"{MES_MAP.get(p.month, p.strftime('%m'))} {p.year}" for p in default_periodos]
_restore_multiselect("evo_periodo", "evo_mem_periodo", periodo_labels, default_labels)
periodo_sel = st.sidebar.multiselect("Mes-Año", periodo_labels, key="evo_periodo")
st.session_state["evo_mem_periodo"] = periodo_sel
periodo_sel_dt = [periodo_map[p] for p in periodo_sel]
mask_periodo = df["periodo_mes"].isin(periodo_sel_dt)

# Filtros adicionales usando multiselect_con_nulos
_restore_multiselect("evo_abc", "evo_mem_abc", sorted(df["ABC"].dropna().unique().tolist()), sorted(df["ABC"].dropna().unique().tolist()))
abc_sel, mask_abc = multiselect_con_nulos("ABC", df["ABC"], key="evo_abc")
st.session_state["evo_mem_abc"] = abc_sel

_restore_multiselect(
    "evo_region",
    "evo_mem_region",
    sorted(df["des_oficina_venta_alicorp"].dropna().unique().tolist()),
    sorted(df["des_oficina_venta_alicorp"].dropna().unique().tolist()),
)
region_sel, mask_region = multiselect_con_nulos("Región", df["des_oficina_venta_alicorp"], key="evo_region")
st.session_state["evo_mem_region"] = region_sel

_restore_multiselect(
    "evo_canal",
    "evo_mem_canal",
    sorted(df["des_grupo_precio_alicorp"].dropna().unique().tolist()),
    sorted(df["des_grupo_precio_alicorp"].dropna().unique().tolist()),
)
canal_sel, mask_canal = multiselect_con_nulos("Canal", df["des_grupo_precio_alicorp"], key="evo_canal")
st.session_state["evo_mem_canal"] = canal_sel

zona_opciones = sorted(df["des_grupo_vendedor_alicorp"].dropna().unique().tolist())
if df["des_grupo_vendedor_alicorp"].isna().any():
    zona_opciones.append(SIN_DATO_LABEL)
_restore_multiselect("evo_zona", "evo_mem_zona", zona_opciones, zona_opciones)
zona_sel, mask_zona = multiselect_con_nulos(
    "Zona", df["des_grupo_vendedor_alicorp"], opciones_override=zona_opciones, key="evo_zona"
)
st.session_state["evo_mem_zona"] = zona_sel
st.sidebar.checkbox(
    "Seleccionar todo Zona",
    key="sel_all_zona_evo",
    on_change=lambda: st.session_state.__setitem__("evo_zona", zona_opciones),
)

jcc_opciones = sorted(df["JCC"].dropna().unique().tolist())
if df["JCC"].isna().any():
    jcc_opciones.append(SIN_DATO_LABEL)
base_jcc = sorted([j for j in jcc_opciones if j != SIN_DATO_LABEL])
default_jcc = base_jcc[:3] if base_jcc else []
_restore_multiselect("evo_jcc", "evo_mem_jcc", jcc_opciones, default_jcc)
jcc_sel, mask_jcc = multiselect_con_nulos("JCC", df["JCC"], opciones_override=jcc_opciones, key="evo_jcc")
st.session_state["evo_mem_jcc"] = jcc_sel

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

# Filtrar categorias por los anos seleccionados para el detalle
# (si la columna existe en la base de categorias)
df_cat_filt = df_cat.copy()
anio_col_cat = next((c for c in df_cat_filt.columns if c in ["a?o", "año", "ano"]), None)
if anio_col_cat is not None:
    df_cat_filt = df_cat_filt[df_cat_filt[anio_col_cat].isin(anio_sel)].copy()


# Datos ya filtrados por JCC desde el sidebar
df_jcc = df_filt.copy()

if df_jcc.empty:
    st.info("No hay datos para los JCC seleccionados.")
    st.stop()


# Cruce con consolidado de clientes sugeridos (cliente + periodo)
if not df_sug.empty:
    df_jcc_keys = df_jcc[["cod_cliente_alicorp_actual", "periodo_mes"]].copy()
    df_jcc_keys["cod_cliente_alicorp_actual"] = df_jcc_keys["cod_cliente_alicorp_actual"].astype(str).str.strip()
    df_jcc_keys["periodo_mes"] = pd.to_datetime(df_jcc_keys["periodo_mes"], errors="coerce")

    df_sug_keys = df_sug[["cod_cliente_alicorp_actual", "periodo_mes"]].copy()
    df_sug_keys["cod_cliente_alicorp_actual"] = df_sug_keys["cod_cliente_alicorp_actual"].astype(str).str.strip()
    df_sug_keys["periodo_mes"] = pd.to_datetime(df_sug_keys["periodo_mes"], errors="coerce")

    cruces_sugeridos = df_jcc_keys.merge(
        df_sug_keys.drop_duplicates(),
        on=["cod_cliente_alicorp_actual", "periodo_mes"],
        how="inner",
    )
    clientes_sugeridos_set = set(cruces_sugeridos["cod_cliente_alicorp_actual"].dropna().unique().tolist())
else:
    clientes_sugeridos_set = set()

# Función para colorear celdas segÃºn cumplimiento
def color_cumplimiento(val):
    if pd.isna(val) or val == "N/A":
        return 'background-color: white; color: black; font-size: 14px'

    # Si es string (puede ser "#¡DIV/0!" o similar)
    if isinstance(val, str):
        return 'background-color: white; color: black; font-size: 14px'

    # Convertir a float si es necesario
    try:
        val_float = float(val)
    except:
        return 'background-color: white; color: black; font-size: 14px'

    if val_float < 0.85:
        return 'background-color: #736867; color: white; font-size: 14px'
    elif val_float < 0.95:
        return 'background-color: #EFFF1C; color: black; font-size: 14px'
    elif val_float < 1.05:
        return 'background-color: #A4FF4A; color: black; font-size: 14px'
    elif val_float < 1.15:
        return 'background-color: #FFBF9C; color: black; font-size: 14px'
    else:
        return 'background-color: #FF430F; color: white; font-size: 14px'



def render_detalle_cliente(df_all, df_cat_all, df_sug_all, cliente_cod, nombre_cliente):
    cliente_cod_str = str(cliente_cod).strip()
    df_cliente = df_all[df_all["cod_cliente_alicorp_actual"].astype(str).str.strip() == cliente_cod_str].copy()
    if df_cliente.empty:
        st.info("No hay datos para este cliente con los filtros seleccionados.")
        return

    df_cliente["periodo_mes"] = pd.to_datetime(df_cliente["periodo_mes"], errors="coerce")
    df_cliente = df_cliente.dropna(subset=["periodo_mes"]).copy()
    if df_cliente.empty:
        st.info("No hay periodos validos para este cliente.")
        return

    # Consolidar por periodo para mostrar todos los meses segun filtros activos.
    df_det = (
        df_cliente.groupby("periodo_mes", as_index=False)
        .agg(
            venta_plan=("venta_plan", "sum"),
            venta_real=("venta_real", "sum"),
            vol_ton_plan=("vol_ton_plan", "sum"),
            vol_ton_real=("vol_ton_real", "sum"),
        )
        .sort_values("periodo_mes")
    )

    df_det["cumplimiento_sol"] = np.where(df_det["venta_plan"] > 0, df_det["venta_real"] / df_det["venta_plan"], np.nan)
    df_det["cumplimiento_vol"] = np.where(df_det["vol_ton_plan"] > 0, df_det["vol_ton_real"] / df_det["vol_ton_plan"], np.nan)
    df_det["periodo_label"] = df_det["periodo_mes"].apply(lambda x: f"{MES_MAP.get(x.month, x.strftime('%m'))} {x.year}")
    periodos_ordenados = df_det["periodo_label"].tolist()

    # Traer volumen sugerido del consolidado por cliente y periodo.
    df_sug_cliente = df_sug_all.copy()
    if not df_sug_cliente.empty:
        df_sug_cliente["cod_cliente_alicorp_actual"] = df_sug_cliente["cod_cliente_alicorp_actual"].astype(str).str.strip()
        df_sug_cliente["periodo_mes"] = pd.to_datetime(df_sug_cliente["periodo_mes"], errors="coerce")
        df_sug_cliente = df_sug_cliente[
            df_sug_cliente["cod_cliente_alicorp_actual"] == cliente_cod_str
        ].copy()
        df_sug_cliente = (
            df_sug_cliente.groupby("periodo_mes", as_index=False)
            .agg(vol_sugerido=("vol_sugerido", "sum"))
        )
    else:
        df_sug_cliente = pd.DataFrame(columns=["periodo_mes", "vol_sugerido"])

    df_det = df_det.merge(df_sug_cliente, on="periodo_mes", how="left")

    st.subheader(f"{nombre_cliente}")

    # Facturacion (Soles)
    st.markdown("---")
    st.subheader("Facturación (Soles)")
    col_tabla_sol, col_grafico_sol = st.columns([1, 2])

    with col_tabla_sol:
        tabla_sol = df_det[["periodo_label", "venta_plan", "venta_real", "cumplimiento_sol"]].copy()
        tabla_sol.columns = ["Mes", "Suma de Sol Plan", "Suma de Sol Real", "Cump"]

        total_plan_sol = tabla_sol["Suma de Sol Plan"].sum()
        total_real_sol = tabla_sol["Suma de Sol Real"].sum()
        total_cump_sol = total_real_sol / total_plan_sol if total_plan_sol > 0 else np.nan

        total_row_sol = pd.DataFrame({
            "Mes": ["Total general"],
            "Suma de Sol Plan": [total_plan_sol],
            "Suma de Sol Real": [total_real_sol],
            "Cump": [total_cump_sol]
        })

        tabla_sol_display = pd.concat([tabla_sol, total_row_sol], ignore_index=True)

        def color_cumpl_sol(val):
            if pd.isna(val):
                return "background-color: white"
            if val < 0.85:
                return "background-color: #736867; color: white"
            elif val < 0.95:
                return "background-color: #EFFF1C; color: black"
            elif val < 1.05:
                return "background-color: #A4FF4A; color: black"
            elif val < 1.15:
                return "background-color: #FFBF9C; color: black"
            else:
                return "background-color: #FF430F; color: white"

        styled_sol = tabla_sol_display.style.format({
            "Suma de Sol Plan": "{:,.0f}",
            "Suma de Sol Real": "{:,.0f}",
            "Cump": lambda x: f"{x:.0%}" if pd.notna(x) else "N/A"
        }).applymap(color_cumpl_sol, subset=["Cump"]).set_properties(**{"text-align": "center"})

        st.dataframe(styled_sol, use_container_width=True, hide_index=True)

    with col_grafico_sol:
        chart_data_sol = df_det[["periodo_label", "venta_plan", "venta_real"]].copy()
        chart_data_sol = chart_data_sol.melt(
            id_vars=["periodo_label"],
            value_vars=["venta_plan", "venta_real"],
            var_name="Tipo",
            value_name="Valor"
        )
        chart_data_sol["Tipo"] = chart_data_sol["Tipo"].map({
            "venta_plan": "Suma de Sol Plan",
            "venta_real": "Suma de Sol Real"
        })

        chart_sol = alt.Chart(chart_data_sol).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("periodo_label:N", title="", sort=periodos_ordenados, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Valor:Q", title="Monto (S/.)"),
            color=alt.Color("Tipo:N",
                scale=alt.Scale(
                    domain=["Suma de Sol Plan", "Suma de Sol Real"],
                    range=["#2E7D32", "#EF6C00"]
                ),
                legend=alt.Legend(title="")
            ),
            tooltip=[
                alt.Tooltip("periodo_label:N", title="Periodo"),
                alt.Tooltip("Tipo:N", title="Tipo"),
                alt.Tooltip("Valor:Q", title="Valor", format=",.0f")
            ]
        ).properties(height=300)

        st.altair_chart(chart_sol, use_container_width=True)

    # Volumen (Toneladas)
    st.markdown("---")
    st.subheader("Volumen (Toneladas)")
    col_tabla_vol, col_grafico_vol = st.columns([1, 2])

    with col_tabla_vol:
        tabla_vol = df_det[["periodo_label", "vol_ton_plan", "vol_ton_real", "vol_sugerido", "cumplimiento_vol"]].copy()
        tabla_vol.columns = ["Mes", "Suma de Vol Plan", "Suma de Vol Real", "Suma de Vol Sugerido", "Cump"]

        total_plan_vol = tabla_vol["Suma de Vol Plan"].sum()
        total_real_vol = tabla_vol["Suma de Vol Real"].sum()
        total_sug_vol = tabla_vol["Suma de Vol Sugerido"].sum(min_count=1)
        total_cump_vol = total_real_vol / total_plan_vol if total_plan_vol > 0 else np.nan

        total_row_vol = pd.DataFrame({
            "Mes": ["Total general"],
            "Suma de Vol Plan": [total_plan_vol],
            "Suma de Vol Real": [total_real_vol],
            "Suma de Vol Sugerido": [total_sug_vol],
            "Cump": [total_cump_vol]
        })

        tabla_vol_display = pd.concat([tabla_vol, total_row_vol], ignore_index=True)

        styled_vol = tabla_vol_display.style.format({
            "Suma de Vol Plan": "{:,.0f}",
            "Suma de Vol Real": "{:,.0f}",
            "Suma de Vol Sugerido": lambda x: f"{x:,.0f}" if pd.notna(x) else "-",
            "Cump": lambda x: f"{x:.0%}" if pd.notna(x) else "N/A"
        }).applymap(color_cumpl_sol, subset=["Cump"]).set_properties(**{"text-align": "center"})

        st.dataframe(styled_vol, use_container_width=True, hide_index=True)

    with col_grafico_vol:
        chart_data_vol = df_det[["periodo_label", "vol_ton_plan", "vol_ton_real", "vol_sugerido"]].copy()
        chart_data_vol = chart_data_vol.melt(
            id_vars=["periodo_label"],
            value_vars=["vol_ton_plan", "vol_ton_real", "vol_sugerido"],
            var_name="Tipo",
            value_name="Valor"
        )
        chart_data_vol = chart_data_vol.dropna(subset=["Valor"]).copy()
        chart_data_vol["Tipo"] = chart_data_vol["Tipo"].map({
            "vol_ton_plan": "Suma de Vol Plan",
            "vol_ton_real": "Suma de Vol Real",
            "vol_sugerido": "Suma de Vol Sugerido",
        })

        chart_vol = alt.Chart(chart_data_vol).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("periodo_label:N", title="", sort=periodos_ordenados, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Valor:Q", title="Volumen (Ton)"),
            color=alt.Color("Tipo:N",
                scale=alt.Scale(
                    domain=["Suma de Vol Plan", "Suma de Vol Real", "Suma de Vol Sugerido"],
                    range=["#2E7D32", "#EF6C00", "#1E88E5"]
                ),
                legend=alt.Legend(title="")
            ),
            tooltip=[
                alt.Tooltip("periodo_label:N", title="Periodo"),
                alt.Tooltip("Tipo:N", title="Tipo"),
                alt.Tooltip("Valor:Q", title="Valor", format=",.0f")
            ]
        ).properties(height=300)

        st.altair_chart(chart_vol, use_container_width=True)

    # Mix por Categoria
    st.markdown("---")
    st.subheader("Mix por Categoría")

    df_cat_cliente = df_cat_all[df_cat_all["cod_cliente_alicorp_actual"] == cliente_cod].copy()

    if df_cat_cliente.empty:
        st.info("No hay datos de categoria para este cliente con los filtros seleccionados.")
        return

    tipo_mix = st.radio(
        "Selecciona metrica para mix:",
        ["Soles", "Volumen"],
        key=f"mix_tipo_{cliente_cod}",
        horizontal=True
    )

    categorias = (
        df_cat_cliente["des_categoria"]
        .dropna()
        .unique()
        .tolist()
    )
    categorias = sorted(categorias)
    default_categorias = categorias[:5] if len(categorias) >= 5 else categorias

    if tipo_mix == "Soles":
        categoria_sel = st.multiselect(
            "Selecciona Categorias:",
            categorias,
            default=default_categorias,
            key=f"mix_cat_soles_{cliente_cod}",
            help="Selecciona las categorias que quieres visualizar en los pie charts"
        )

        cat_filtrado = df_cat_cliente.copy()
        if categoria_sel:
            cat_filtrado = cat_filtrado[cat_filtrado["des_categoria"].isin(categoria_sel)].copy()

        mix_plan = (
            cat_filtrado.groupby("des_categoria", as_index=False)
            .agg(valor=("venta_plan", "sum"))
        )

        mix_real = (
            cat_filtrado.groupby("des_categoria", as_index=False)
            .agg(valor=("venta_real", "sum"))
        )

        titulo_plan = "Plan - Soles"
        titulo_real = "Real - Soles"
    else:
        categoria_sel = st.multiselect(
            "Selecciona Categorias:",
            categorias,
            default=default_categorias,
            key=f"mix_cat_vol_{cliente_cod}",
            help="Selecciona las categorias que quieres visualizar en los pie charts"
        )

        cat_filtrado = df_cat_cliente.copy()
        if categoria_sel:
            cat_filtrado = cat_filtrado[cat_filtrado["des_categoria"].isin(categoria_sel)].copy()

        mix_plan = (
            cat_filtrado.groupby("des_categoria", as_index=False)
            .agg(valor=("vol_ton_plan", "sum"))
        )

        mix_real = (
            cat_filtrado.groupby("des_categoria", as_index=False)
            .agg(valor=("vol_ton_real", "sum"))
        )

        titulo_plan = "Plan - Volumen (Ton)"
        titulo_real = "Real - Volumen (Ton)"

    mix_plan = mix_plan[mix_plan["valor"] > 0].copy()
    mix_real = mix_real[mix_real["valor"] > 0].copy()

    if not mix_plan.empty or not mix_real.empty:
        col_pie1, col_pie2 = st.columns(2)

        with col_pie1:
            st.caption(f"**{titulo_plan}**")
            if not mix_plan.empty:
                pie_plan = alt.Chart(mix_plan).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="valor", type="quantitative"),
                    color=alt.Color(
                        field="des_categoria",
                        type="nominal",
                        legend=alt.Legend(title="Categoria", orient="bottom")
                    ),
                    tooltip=[
                        alt.Tooltip("des_categoria:N", title="Categoria"),
                        alt.Tooltip("valor:Q", title="Valor", format=",.0f")
                    ]
                ).properties(height=300)
                st.altair_chart(pie_plan, use_container_width=True)
            else:
                st.info("No hay datos de Plan para este cliente")

        with col_pie2:
            st.caption(f"**{titulo_real}**")
            if not mix_real.empty:
                pie_real = alt.Chart(mix_real).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="valor", type="quantitative"),
                    color=alt.Color(
                        field="des_categoria",
                        type="nominal",
                        legend=alt.Legend(title="Categoria", orient="bottom")
                    ),
                    tooltip=[
                        alt.Tooltip("des_categoria:N", title="Categoria"),
                        alt.Tooltip("valor:Q", title="Valor", format=",.0f")
                    ]
                ).properties(height=300)
                st.altair_chart(pie_real, use_container_width=True)
            else:
                st.info("No hay datos de Real para este cliente")
    else:
        st.info("No hay datos de categorias para este cliente")

# Seleccionar la columna de cumplimiento segÃºn el tipo
cumpl_col = f"cumplimiento_{tipo_key}"

# Crear pivot table unica con TODOS los clientes de TODOS los JCC seleccionados
# Usar periodo (mes + anio) para evitar mezclar meses de anos distintos.
df_pivot = df_jcc.copy()
df_pivot["periodo_mes"] = pd.to_datetime(df_pivot["periodo_mes"], errors="coerce")
df_pivot = df_pivot.dropna(subset=["periodo_mes"])

def _label_periodo(ts):
    return f"{MES_MAP.get(ts.month, ts.strftime('%m'))} {ts.year}"

df_pivot["periodo_label"] = df_pivot["periodo_mes"].apply(_label_periodo)
periodos_ordenados_dt = sorted(df_pivot["periodo_mes"].unique())
meses_ordenados = [_label_periodo(pd.Timestamp(p)) for p in periodos_ordenados_dt]

# Pares (cliente, periodo_label) sugeridos para pintar celdas en morado.
sugeridos_pairs = set()
if not df_sug.empty:
    df_sug_mark = df_sug.copy()
    df_sug_mark["cod_cliente_alicorp_actual"] = df_sug_mark["cod_cliente_alicorp_actual"].astype(str).str.strip()
    df_sug_mark["periodo_mes"] = pd.to_datetime(df_sug_mark["periodo_mes"], errors="coerce")
    df_sug_mark = df_sug_mark.dropna(subset=["periodo_mes"]).copy()
    df_sug_mark["periodo_label"] = df_sug_mark["periodo_mes"].apply(_label_periodo)
    sugeridos_pairs = set(
        zip(
            df_sug_mark["cod_cliente_alicorp_actual"],
            df_sug_mark["periodo_label"],
        )
    )

pivot_data = df_pivot.pivot_table(
    index=["JCC", "cod_cliente_alicorp_actual", "nom_cliente_alicorp_actual"],
    columns="periodo_label",
    values=cumpl_col,
    aggfunc="mean"  # Promedio si hay multiples registros
)

# Asegurar orden cronologico de columnas
meses_ordenados = [m for m in meses_ordenados if m in pivot_data.columns]
pivot_data = pivot_data[meses_ordenados]

# Resetear index para tener JCC, codigo y nombre como columnas
pivot_data = pivot_data.reset_index()

# Renombrar columnas
pivot_data.columns.name = None
pivot_data = pivot_data.rename(columns={
    "cod_cliente_alicorp_actual": "Código Cliente",
    "nom_cliente_alicorp_actual": "Nombre Cliente"
})

# Reordenar columnas: JCC, Código Cliente, Nombre Cliente, luego meses
cols = ["JCC", "Código Cliente", "Nombre Cliente"] + meses_ordenados
pivot_data = pivot_data[cols]

# Quitar filas sin datos en meses
if meses_ordenados:
    mask_vacias = pivot_data[meses_ordenados].isna().all(axis=1)
    pivot_data = pivot_data[~mask_vacias].copy()

# Quitar filas sin código de cliente
pivot_data = pivot_data[pivot_data["Código Cliente"].astype(str).str.strip() != ""].copy()

# Calcular conteos de sobre/sub cumplimiento por cliente (todos los meses)
valores_mes = pivot_data[meses_ordenados]
pivot_data["cnt_rojo"] = (valores_mes > 1.15).sum(axis=1)
pivot_data["cnt_marron"] = (valores_mes < 0.85).sum(axis=1)

# Clasificacion de los ultimos 2 meses
if len(meses_ordenados) >= 2:
    ultimos_2_meses = meses_ordenados[-2:]

    def clasificar_ultimos(row):
        v1, v2 = row[ultimos_2_meses[0]], row[ultimos_2_meses[1]]
        if pd.isna(v1) or pd.isna(v2):
            return "otro"
        rojo1 = v1 > 1.15
        rojo2 = v2 > 1.15
        marron1 = v1 < 0.85
        marron2 = v2 < 0.85
        if rojo1 and rojo2:
            return "sobre"
        if marron1 and marron2:
            return "sub"
        if (rojo1 and marron2) or (marron1 and rojo2):
            return "mixto"
        return "otro"

    pivot_data["clas_ult2"] = pivot_data.apply(clasificar_ultimos, axis=1)
else:
    pivot_data["clas_ult2"] = "otro"

clientes_sobre = (pivot_data["clas_ult2"] == "sobre").sum()
clientes_sub = (pivot_data["clas_ult2"] == "sub").sum()

total_clientes = len(pivot_data)
pct_sobre = (clientes_sobre / total_clientes) if total_clientes > 0 else 0
pct_sub = (clientes_sub / total_clientes) if total_clientes > 0 else 0

# Ordenar: no mixto primero, luego mas rojos, luego mas marron
pivot_data["orden_mixto"] = pivot_data["clas_ult2"] == "mixto"
pivot_data = pivot_data.sort_values(
    ["JCC", "orden_mixto", "cnt_rojo", "cnt_marron", "Nombre Cliente"],
    ascending=[True, True, False, False, True]
)

# KPIs de sobre/sub cumplimiento
card_style = (
    "background:white;padding:12px;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.08);text-align:center;"
)

def render_card(title, value):
    return (
    f'<div style="{card_style}">'
    f'<div style="font-size:13px;color:#6b6b6b">{title}</div>'
    f'<div style="font-size:20px;font-weight:700;margin-top:6px">{value}</div>'
    f'</div>'
)

sp_l, sp_c, sp_r = st.columns([1, 3, 1])
with sp_c:
    c_k1, c_k2, c_k3, c_k4 = st.columns(4)
    with c_k1:
        st.markdown(render_card("N° Clientes con sobrecumplimiento", f"{clientes_sobre:,.0f}"), unsafe_allow_html=True)
    with c_k2:
        st.markdown(render_card("% Clientes sobrecumplimiento", f"{pct_sobre:.1%}"), unsafe_allow_html=True)
    with c_k3:
        st.markdown(render_card("N° Clientes con subcumplimiento", f"{clientes_sub:,.0f}"), unsafe_allow_html=True)
    with c_k4:
        st.markdown(render_card("% Clientes subcumplimiento", f"{pct_sub:.1%}"), unsafe_allow_html=True)

filtrar_criticos = st.checkbox(
    "Mostrar solo clientes criticos",
    value=False,
    help="Clientes con cumplimiento < 85% o > 115% en los ultimos 2 meses"
)

filtrar_sugeridos = st.checkbox(
    "Mostrar solo clientes sugeridos",
    value=False,
    help="Clientes del archivo Consolidado Sugeridos para los meses y años filtrados"
)


# Aplicar filtro de clientes criticos si esta activado
if filtrar_criticos and len(meses_ordenados) >= 2:
    ultimos_2_meses = meses_ordenados[-2:]

    def es_critico(row):
        valores = [row[mes] for mes in ultimos_2_meses]
        criticos = 0
        for val in valores:
            if pd.notna(val):
                if val < 0.85 or val > 1.15:
                    criticos += 1
        periodos_activos = row[meses_ordenados].notna().sum()
        return criticos == 2 and periodos_activos > 3

    mask_criticos = pivot_data.apply(es_critico, axis=1)
    pivot_data = pivot_data[mask_criticos].copy()

    if pivot_data.empty:
        st.warning("No hay clientes criticos con los filtros seleccionados.")
        st.stop()


# Aplicar filtro de clientes sugeridos (cruce cliente + periodo)
if filtrar_sugeridos:
    cod_col_sug = [
        c for c in pivot_data.columns if "Cliente" in c and c not in ["Nombre Cliente", "JCC"]
    ][0]
    mask_sugeridos = pivot_data[cod_col_sug].astype(str).str.strip().isin(clientes_sugeridos_set)
    pivot_data = pivot_data[mask_sugeridos].copy()

    if pivot_data.empty:
        st.warning("No hay clientes sugeridos con los filtros seleccionados.")
        st.stop()
# Aplicar filtro de clientes criticos si esta activado
# Formatear la tabla
def format_cumpl(val):
    if pd.isna(val):
        return "#DIV/0!"
    return f"{val:.0%}"

# Aplicar formato solo a las columnas de meses
formato_dict = {col: format_cumpl for col in meses_ordenados}

cod_col = [c for c in pivot_data.columns if "Cliente" in c and c not in ["Nombre Cliente", "JCC"]][0]
cols_display = ["JCC", cod_col, "Nombre Cliente"] + meses_ordenados
pivot_data = pivot_data[cols_display]

styled_table = pivot_data.style.format(formato_dict)

# Aplicar colores a las celdas de cumplimiento (solo meses)
styled_table = styled_table.applymap(
    color_cumplimiento,
    subset=meses_ordenados
)

# Resaltar meses sugeridos con azul de alto contraste.
def _pintar_sugeridos(row):
    estilos = [""] * len(row)
    cod_val = str(row[cod_col]).strip()
    for i, col in enumerate(row.index):
        if col in meses_ordenados and (cod_val, col) in sugeridos_pairs:
            estilos[i] = "font-weight: 800; color: #1565C0"
    return estilos

styled_table = styled_table.apply(_pintar_sugeridos, axis=1)

# Centrar todas las celdas
styled_table = styled_table.set_properties(**{'text-align': 'center'})

# Tabla principal con colores y seleccion de filas
pivot_display = pivot_data.reset_index(drop=True)

table_height = max(220, min(600, 36 + len(pivot_display) * 24))
selection = st.dataframe(
    styled_table,
    use_container_width=True,
    hide_index=True,
    height=table_height,
    on_select="rerun",
    selection_mode="multi-row",
    key="tabla_clientes",
)

st.markdown(
    "<div style='display:flex;align-items:center;margin-top:6px;'>"
    "<div style='background:#1E88E5;width:16px;height:16px;border-radius:3px;margin-right:8px;'></div>"
    "<div style='font-size:12px;color:#666;'>Mes sugerido</div>"
    "</div>",
    unsafe_allow_html=True,
)

# Procesar seleccion de filas
selected_rows = []
if selection is not None:
    try:
        selected_rows = selection.selection.rows
    except Exception:
        selected_rows = []

selected_clientes_orden = st.session_state.get("evo_selected_clientes_orden", [])
if selected_rows:
    cod_col_sel = [c for c in pivot_display.columns if "Cliente" in c and c not in ["Nombre Cliente", "JCC"]][0]
    selected_clientes_orden = []
    for idx in selected_rows:
        cliente_cod = pivot_display.iloc[idx][cod_col_sel]
        cod_sel = str(cliente_cod).strip()
        if cod_sel and cod_sel not in selected_clientes_orden:
            selected_clientes_orden.append(cod_sel)
    st.session_state["evo_selected_clientes_orden"] = selected_clientes_orden

selected_clientes = set(selected_clientes_orden)

st.markdown("---")
st.subheader("Evolución por JCC")

df_jcc_evo = df_jcc.copy()
df_jcc_evo["periodo_mes"] = pd.to_datetime(df_jcc_evo["periodo_mes"], errors="coerce")
df_jcc_evo = df_jcc_evo.dropna(subset=["periodo_mes"]).copy()
df_jcc_evo["cod_cliente_alicorp_actual"] = df_jcc_evo["cod_cliente_alicorp_actual"].astype(str).str.strip()
if df_jcc_evo.empty:
    st.info("No hay datos para mostrar evolución por JCC con los filtros actuales.")
else:
    df_jcc_evo["periodo_label"] = df_jcc_evo["periodo_mes"].apply(_label_periodo)
    periodos_ordenados_dt = sorted(df_jcc_evo["periodo_mes"].unique())
    periodos_ordenados = [_label_periodo(pd.Timestamp(p)) for p in periodos_ordenados_dt]

    if tipo_key == "sol":
        plan_col = "venta_plan"
        real_col = "venta_real"
    else:
        plan_col = "vol_ton_plan"
        real_col = "vol_ton_real"

    base = (
        df_jcc_evo.groupby(
            ["JCC", "cod_cliente_alicorp_actual", "nom_cliente_alicorp_actual", "periodo_label"],
            as_index=False,
        )
        .agg(plan=(plan_col, "sum"), real=(real_col, "sum"))
    )
    base["cumplimiento"] = np.where(base["plan"] > 0, base["real"] / base["plan"], np.nan)

    jcc_lista = sorted([j for j in jcc_sel if j != SIN_DATO_LABEL])
    for jcc in jcc_lista:
        df_j = base[base["JCC"] == jcc].copy()
        if df_j.empty:
            continue

        with st.expander(f"JCC: {jcc}", expanded=False):
            idx_cols = ["cod_cliente_alicorp_actual", "nom_cliente_alicorp_actual"]

            def _pivot(valor_col):
                pt = df_j.pivot_table(
                    index=idx_cols,
                    columns="periodo_label",
                    values=valor_col,
                    aggfunc="sum",
                )
                pt = pt.reindex(columns=[p for p in periodos_ordenados if p in pt.columns])
                pt.index.set_names(["cod_cliente_alicorp_actual", "nom_cliente_alicorp_actual"], inplace=True)
                return pt

            tabla_plan = _pivot("plan")
            tabla_real = _pivot("real")
            tabla_cump = _pivot("cumplimiento")

            def _color_cumpl(val):
                if pd.isna(val):
                    return "background-color: white; color: black; font-size: 14px"
                try:
                    val_float = float(val)
                except Exception:
                    return "background-color: white; color: black; font-size: 14px"
                if val_float < 0.85:
                    return "background-color: #736867; color: white; font-size: 14px"
                elif val_float < 0.95:
                    return "background-color: #EFFF1C; color: black; font-size: 14px"
                elif val_float < 1.05:
                    return "background-color: #A4FF4A; color: black; font-size: 14px"
                elif val_float < 1.15:
                    return "background-color: #FFBF9C; color: black; font-size: 14px"
                else:
                    return "background-color: #FF430F; color: white; font-size: 14px"

            tabla_comb = pd.concat(
                [tabla_plan, tabla_real, tabla_cump],
                axis=1,
                keys=["Plan", "Real", "Cumplimiento"],
            )

            tabla_comb = tabla_comb.reset_index()
            cols = []
            for col in tabla_comb.columns:
                if isinstance(col, tuple):
                    base_col = col[0]
                    if base_col == "cod_cliente_alicorp_actual":
                        cols.append(("Cliente", "Código Cliente"))
                    elif base_col == "nom_cliente_alicorp_actual":
                        cols.append(("Cliente", "Nombre Cliente"))
                    else:
                        cols.append(col)
                else:
                    if col == "cod_cliente_alicorp_actual":
                        cols.append(("Cliente", "Código Cliente"))
                    elif col == "nom_cliente_alicorp_actual":
                        cols.append(("Cliente", "Nombre Cliente"))
                    else:
                        cols.append(("Cliente", str(col)))
            tabla_comb.columns = pd.MultiIndex.from_tuples(cols)

            # Quitar filas sin datos en Plan/Real/Cumplimiento
            cols_val = [c for c in tabla_comb.columns if isinstance(c, tuple) and c[0] in ["Plan", "Real", "Cumplimiento"]]
            if cols_val:
                mask_vacias = tabla_comb[cols_val].isna().all(axis=1)
                tabla_comb = tabla_comb[~mask_vacias].copy()

            # Resolver columna de código cliente (MultiIndex o plano)
            cod_col = next(
                (c for c in tabla_comb.columns if isinstance(c, tuple) and c[1] == "Código Cliente"),
                None,
            )
            if cod_col is None:
                cod_col = next((c for c in tabla_comb.columns if c == "Código Cliente"), None)

            if cod_col is not None:
                mask_cod = tabla_comb[cod_col].astype(str).str.strip() != ""
                tabla_comb = tabla_comb[mask_cod].copy()

            # Matriz de estilos para resaltar meses sugeridos en Plan y Cumplimiento
            estilos_sug = pd.DataFrame("", index=tabla_comb.index, columns=tabla_comb.columns)
            if cod_col is not None:
                for i in tabla_comb.index:
                    cod_val = tabla_comb.loc[i, cod_col]
                    cod_key = str(cod_val).strip()
                    for periodo in periodos_ordenados:
                        if (cod_key, periodo) in sugeridos_pairs:
                            col_plan = ("Plan", periodo)
                            col_cump = ("Cumplimiento", periodo)
                            if col_plan in estilos_sug.columns:
                                estilos_sug.loc[i, col_plan] = "background-color: #1E88E5; color: white; font-weight: 900"
                            if col_cump in estilos_sug.columns:
                                estilos_sug.loc[i, col_cump] = "color: #1565C0; font-weight: 800"

            def _style_sug(data):
                return estilos_sug

            idx = pd.IndexSlice
            cols_cump = tabla_comb.loc[:, idx["Cumplimiento", :]].columns
            format_plan_real = "{:,.0f}" if tipo_key == "vol" else "S/. {:,.0f}"
            fmt = {}
            for col in tabla_comb.columns:
                if col[0] == "Cumplimiento":
                    fmt[col] = "{:.0%}"
                elif col[0] in ["Plan", "Real"]:
                    fmt[col] = format_plan_real

            styled = (
                tabla_comb.style.format(fmt)
                .applymap(_color_cumpl, subset=cols_cump)
                .apply(_style_sug, axis=None)
            )

            # Resaltar celda de nombre del cliente si fue seleccionado en el cuadro principal
            if ("Cliente", "Nombre Cliente") in tabla_comb.columns and selected_clientes:
                def _resaltar_nombre(row):
                    estilos = [""] * len(row)
                    cod = row[cod_col] if cod_col is not None else ""
                    if str(cod).strip() in selected_clientes:
                        try:
                            idx_nom = list(row.index).index(("Cliente", "Nombre Cliente"))
                            estilos[idx_nom] = "background-color: #FFD6D6; color: #8B0000; font-weight: 700"
                        except ValueError:
                            pass
                    return estilos
                styled = styled.apply(_resaltar_nombre, axis=1)

            if st.button("Añadir plan", key=f"btn_add_plan_{jcc}"):
                st.session_state[f"show_plan_{jcc}"] = True

            if f"plan_inputs_{jcc}" not in st.session_state:
                st.session_state[f"plan_inputs_{jcc}"] = {}

            if st.session_state.get(f"show_plan_{jcc}", False):
                st.caption("Plan ingresado por cliente")
                with st.form(key=f"form_plan_{jcc}"):
                    base_inputs = tabla_comb.loc[:, [("Cliente", "Código Cliente"), ("Cliente", "Nombre Cliente")]].copy()
                    base_inputs.columns = ["Código Cliente", "Nombre Cliente"]
                    st.session_state[f"plan_names_{jcc}"] = dict(
                        zip(
                            base_inputs["Código Cliente"].astype(str).str.strip(),
                            base_inputs["Nombre Cliente"].astype(str),
                        )
                    )
                    codigos_base = base_inputs["Código Cliente"].astype(str).str.strip()
                    base_inputs["Plan Ingresado"] = (
                        codigos_base.map(st.session_state[f"plan_inputs_{jcc}"]).fillna("")
                    )
                    if f"plan_obs_{jcc}" not in st.session_state:
                        st.session_state[f"plan_obs_{jcc}"] = {}
                    base_inputs["Observaciones"] = (
                        codigos_base.map(st.session_state[f"plan_obs_{jcc}"]).fillna("")
                    )

                    edit_height = max(220, 36 + len(base_inputs) * 24)
                    edited = st.data_editor(
                        base_inputs,
                        use_container_width=True,
                        hide_index=True,
                        height=edit_height,
                        key=f"editor_plan_{jcc}",
                        column_config={
                            "Plan Ingresado": st.column_config.TextColumn("Plan Ingresado"),
                            "Observaciones": st.column_config.TextColumn("Observaciones"),
                        },
                    )

                    submitted = st.form_submit_button("Guardar planes")

                if submitted:
                    for _, row in edited.iterrows():
                        cod = str(row["Código Cliente"]).strip()
                        val = row["Plan Ingresado"]
                        if val is None or str(val).strip() == "":
                            st.session_state[f"plan_inputs_{jcc}"].pop(cod, None)
                        else:
                            st.session_state[f"plan_inputs_{jcc}"][cod] = str(val).strip()
                        obs = row.get("Observaciones", "")
                        if obs is None or str(obs).strip() == "":
                            st.session_state[f"plan_obs_{jcc}"].pop(cod, None)
                        else:
                            st.session_state[f"plan_obs_{jcc}"][cod] = str(obs).strip()

                    st.session_state["plan_descarga"] = st.session_state.get("plan_descarga", [])
                    for _, row in edited.iterrows():
                        cod = str(row["Código Cliente"]).strip()
                        val = row["Plan Ingresado"]
                        if val is None or str(val).strip() == "":
                            continue
                        st.session_state["plan_descarga"].append(
                            {
                                "JCC": jcc,
                                "cod_cliente_alicorp_actual": cod,
                                "nom_cliente_alicorp_actual": row["Nombre Cliente"],
                                "plan_ingresado": str(val).strip(),
                                "observaciones": str(row.get("Observaciones", "")).strip(),
                            }
                        )
                    st.success("Planes guardados.")


            exp_height = max(320, 36 + len(tabla_comb) * 24)
            st.dataframe(styled, use_container_width=True, hide_index=True, height=exp_height)

            # Graficos de evolucion por cliente (2 por fila)
            df_chart = df_j.copy()
            df_chart["cod_cliente_alicorp_actual"] = df_chart["cod_cliente_alicorp_actual"].astype(str).str.strip()
            df_chart["periodo_label"] = pd.Categorical(df_chart["periodo_label"], categories=periodos_ordenados, ordered=True)

            df_sug_j = pd.DataFrame()
            if tipo_key == "vol" and not df_sug.empty:
                df_sug_j = df_sug.copy()
                df_sug_j["cod_cliente_alicorp_actual"] = df_sug_j["cod_cliente_alicorp_actual"].astype(str).str.strip()
                df_sug_j["periodo_mes"] = pd.to_datetime(df_sug_j["periodo_mes"], errors="coerce")
                df_sug_j = df_sug_j.dropna(subset=["periodo_mes"]).copy()
                df_sug_j["periodo_label"] = df_sug_j["periodo_mes"].apply(_label_periodo)

            # Ordenar graficos segun el orden del cuadro del JCC
            cods_orden = []
            if cod_col is not None:
                cods_orden = (
                    tabla_comb.loc[:, cod_col]
                    .astype(str)
                    .str.strip()
                    .dropna()
                    .tolist()
                )
            nombres_map = (
                df_chart[["cod_cliente_alicorp_actual", "nom_cliente_alicorp_actual"]]
                .drop_duplicates()
                .set_index("cod_cliente_alicorp_actual")["nom_cliente_alicorp_actual"]
                .to_dict()
            )
            cods_sel_jcc = []
            if selected_clientes_orden:
                cods_disponibles = set(df_chart["cod_cliente_alicorp_actual"].astype(str).str.strip().tolist())
                cods_sel_jcc = [c for c in selected_clientes_orden if c in cods_disponibles]

            if cods_sel_jcc:
                clientes_jcc = [(c, nombres_map.get(c, "")) for c in cods_sel_jcc]
            elif cods_orden:
                clientes_jcc = [(c, nombres_map.get(c, "")) for c in cods_orden]
            else:
                clientes_jcc = (
                    df_chart[["cod_cliente_alicorp_actual", "nom_cliente_alicorp_actual"]]
                    .drop_duplicates()
                    .sort_values("nom_cliente_alicorp_actual")
                    .values.tolist()
                )

            if clientes_jcc:
                st.markdown("**Evolución por cliente**")
            cols = None
            for idx_cli, (cod_cli, nom_cli) in enumerate(clientes_jcc):
                if idx_cli % 2 == 0:
                    cols = st.columns(2)
                col = cols[0] if idx_cli % 2 == 0 else cols[1]

                df_c = df_chart[df_chart["cod_cliente_alicorp_actual"] == str(cod_cli)].copy()
                serie = (
                    df_c.groupby("periodo_label", as_index=False)
                    .agg(plan=("plan", "sum"), real=("real", "sum"))
                )
                serie = serie.sort_values("periodo_label")
                serie_melt = serie.melt(
                    id_vars=["periodo_label"],
                    value_vars=["plan", "real"],
                    var_name="Serie",
                    value_name="Valor",
                )

                if not df_sug_j.empty:
                    df_sug_c = df_sug_j[df_sug_j["cod_cliente_alicorp_actual"] == str(cod_cli)].copy()
                    if not df_sug_c.empty:
                        sug = (
                            df_sug_c.groupby("periodo_label", as_index=False)
                            .agg(sugerido=("vol_sugerido", "sum"))
                        )
                        sug = sug.sort_values("periodo_label")
                        sug_melt = sug.rename(columns={"sugerido": "Valor"})
                        sug_melt["Serie"] = "sugerido"
                        serie_melt = pd.concat([serie_melt, sug_melt], ignore_index=True)

                chart = (
                    alt.Chart(serie_melt)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("periodo_label:N", sort=periodos_ordenados, title="Periodo"),
                        y=alt.Y("Valor:Q", title=""),
                        color=alt.Color(
                            "Serie:N",
                            title="Serie",
                            scale=alt.Scale(
                                domain=["plan", "real", "sugerido"],
                                range=["#2E7D32", "#EF6C00", "#1E88E5"],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("periodo_label:N", title="Periodo"),
                            alt.Tooltip("Serie:N", title="Serie"),
                            alt.Tooltip("Valor:Q", title="Valor", format=",.0f"),
                        ],
                    )
                    .properties(height=220, title="")
                )
                with col:
                    st.markdown(f"**{nom_cli}**")
                    st.altair_chart(chart, use_container_width=True)
                    with st.expander(f"Mix categoría - {nom_cli}", expanded=False):
                        df_mix = df_cat_filt.copy()
                        df_mix["cod_cliente_alicorp_actual"] = df_mix["cod_cliente_alicorp_actual"].astype(str).str.strip()
                        df_mix = df_mix[df_mix["cod_cliente_alicorp_actual"] == str(cod_cli)].copy()
                        if periodo_sel_dt:
                            df_mix["periodo_mes"] = pd.to_datetime(df_mix["periodo_mes"], errors="coerce")
                            df_mix = df_mix[df_mix["periodo_mes"].isin(periodo_sel_dt)].copy()

                        if df_mix.empty or "des_categoria" not in df_mix.columns:
                            st.info("No hay datos de mix por categoría.")
                        else:
                            tipo_mix = st.radio(
                                "Selecciona métrica para mix:",
                                ["Soles", "Volumen"],
                                key=f"mix_tipo_{jcc}_{cod_cli}",
                                horizontal=True,
                            )

                            categorias = (
                                df_mix["des_categoria"]
                                .dropna()
                                .unique()
                                .tolist()
                            )
                            categorias = sorted(categorias)
                            default_categorias = categorias[:5] if len(categorias) >= 5 else categorias

                            if tipo_mix == "Soles":
                                categoria_sel = st.multiselect(
                                    "Selecciona Categorías:",
                                    categorias,
                                    default=default_categorias,
                                    key=f"mix_cat_soles_{jcc}_{cod_cli}",
                                    help="Selecciona las categorías que quieres visualizar en los pie charts",
                                )
                                cat_filtrado = df_mix.copy()
                                if categoria_sel:
                                    cat_filtrado = cat_filtrado[cat_filtrado["des_categoria"].isin(categoria_sel)].copy()
                                mix_plan = (
                                    cat_filtrado.groupby("des_categoria", as_index=False)
                                    .agg(valor=("venta_plan", "sum"))
                                )
                                mix_real = (
                                    cat_filtrado.groupby("des_categoria", as_index=False)
                                    .agg(valor=("venta_real", "sum"))
                                )
                                titulo_plan = "Plan - Soles"
                                titulo_real = "Real - Soles"
                            else:
                                categoria_sel = st.multiselect(
                                    "Selecciona Categorías:",
                                    categorias,
                                    default=default_categorias,
                                    key=f"mix_cat_vol_{jcc}_{cod_cli}",
                                    help="Selecciona las categorías que quieres visualizar en los pie charts",
                                )
                                cat_filtrado = df_mix.copy()
                                if categoria_sel:
                                    cat_filtrado = cat_filtrado[cat_filtrado["des_categoria"].isin(categoria_sel)].copy()
                                mix_plan = (
                                    cat_filtrado.groupby("des_categoria", as_index=False)
                                    .agg(valor=("vol_ton_plan", "sum"))
                                )
                                mix_real = (
                                    cat_filtrado.groupby("des_categoria", as_index=False)
                                    .agg(valor=("vol_ton_real", "sum"))
                                )
                                titulo_plan = "Plan - Volumen (Ton)"
                                titulo_real = "Real - Volumen (Ton)"

                            mix_plan = mix_plan[mix_plan["valor"] > 0].copy()
                            mix_real = mix_real[mix_real["valor"] > 0].copy()

                            if not mix_plan.empty or not mix_real.empty:
                                col_pie1, col_pie2 = st.columns(2)
                                with col_pie1:
                                    st.caption(f"**{titulo_plan}**")
                                    if not mix_plan.empty:
                                        pie_plan = alt.Chart(mix_plan).mark_arc(innerRadius=50).encode(
                                            theta=alt.Theta(field="valor", type="quantitative"),
                                            color=alt.Color(
                                                field="des_categoria",
                                                type="nominal",
                                                legend=alt.Legend(title="Categoría", orient="bottom"),
                                            ),
                                            tooltip=[
                                                alt.Tooltip("des_categoria:N", title="Categoría"),
                                                alt.Tooltip("valor:Q", title="Valor", format=",.0f"),
                                            ],
                                        ).properties(height=260)
                                        st.altair_chart(pie_plan, use_container_width=True)
                                    else:
                                        st.info("No hay datos de Plan para este cliente")

                                with col_pie2:
                                    st.caption(f"**{titulo_real}**")
                                    if not mix_real.empty:
                                        pie_real = alt.Chart(mix_real).mark_arc(innerRadius=50).encode(
                                            theta=alt.Theta(field="valor", type="quantitative"),
                                            color=alt.Color(
                                                field="des_categoria",
                                                type="nominal",
                                                legend=alt.Legend(title="Categoría", orient="bottom"),
                                            ),
                                            tooltip=[
                                                alt.Tooltip("des_categoria:N", title="Categoría"),
                                                alt.Tooltip("valor:Q", title="Valor", format=",.0f"),
                                            ],
                                        ).properties(height=260)
                                        st.altair_chart(pie_real, use_container_width=True)
                                    else:
                                        st.info("No hay datos de Real para este cliente")
                            else:
                                st.info("No hay datos de categorías para este cliente")

st.markdown("---")
st.subheader("Descarga de Plan Ingresado")
rows_descarga = []
if isinstance(st.session_state.get("plan_descarga"), list):
    rows_descarga = st.session_state["plan_descarga"]

if rows_descarga:
    df_descarga = pd.DataFrame(rows_descarga)
    st.download_button(
        "Descargar CSV",
        df_descarga.to_csv(index=False),
        file_name="planes_ingresados.csv",
        mime="text/csv",
    )
else:
    st.info("No hay planes ingresados para descargar.")

# Leyenda de colores

st.markdown("---")
st.caption("**Leyenda de Cumplimiento:**")
colores_leyenda = ["#736867", "#EFFF1C", "#A4FF4A", "#FFBF9C", "#FF430F"]
etiquetas_leyenda = [
    "< 85%",
    "85-95%",
    "95-105%",
    "105-115%",
    "> 115%"
]

cols_ley = st.columns(len(colores_leyenda))
for c, color, label in zip(cols_ley, colores_leyenda, etiquetas_leyenda):
    c.markdown(
        f"<div style='display:flex;align-items:center;justify-content:center'>"
        f"<div style='background:{color};width:20px;height:20px;border-radius:3px;margin-right:8px;'></div>"
        f"<div>{label}</div></div>",
        unsafe_allow_html=True,
    )


