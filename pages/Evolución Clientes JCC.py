import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from shared import (
    MES_MAP,
    multiselect_con_nulos,
    load_df_cus,
    load_df_with_categoria,
    load_df_sugeridos,
)

st.set_page_config(page_title="Evolución Clientes JCC", layout="wide")
st.title("Evolución de Cumplimiento - Clientes por JCC")

# Cargar datos
df = load_df_cus()
df_cat = load_df_with_categoria()
df_sug = load_df_sugeridos()

# Sidebar filtros
st.sidebar.title("Filtros")

# Filtro de Tipo (Soles o Volumen)
tipo = st.sidebar.radio("Tipo", ["Soles", "Volumen"])
tipo_key = "sol" if tipo == "Soles" else "vol"

# Filtro de Año - por defecto el último año
anios = sorted(df["año"].dropna().unique(), reverse=True)
anio_sel = st.sidebar.multiselect("Año", anios, default=[anios[0]] if anios else [])

# Filtros adicionales usando multiselect_con_nulos
abc_sel, mask_abc = multiselect_con_nulos("ABC", df["ABC"])
region_sel, mask_region = multiselect_con_nulos("Región", df["des_oficina_venta_alicorp"])
canal_sel, mask_canal = multiselect_con_nulos("Canal", df["des_grupo_precio_alicorp"])
zona_sel, mask_zona = multiselect_con_nulos("Zona", df["des_grupo_vendedor_alicorp"])

# Aplicar filtros
df_filt = df[
    df["año"].isin(anio_sel)
    & mask_abc
    & mask_region
    & mask_canal
    & mask_zona
].copy()

# Filtrar categorias por los anos seleccionados para el detalle
# (si la columna existe en la base de categorias)
df_cat_filt = df_cat.copy()
anio_col_cat = next((c for c in df_cat_filt.columns if c in ["a?o", "año", "ano"]), None)
if anio_col_cat is not None:
    df_cat_filt = df_cat_filt[df_cat_filt[anio_col_cat].isin(anio_sel)].copy()


# Obtener todos los JCC disponibles
jcc_vals = sorted(df_filt["JCC"].dropna().unique().tolist())
jcc_sel = jcc_vals

# Filtrar por JCC seleccionados
df_jcc = df_filt[df_filt["JCC"].isin(jcc_sel)].copy()

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

# Función para colorear celdas según cumplimiento
def color_cumplimiento(val):
    if pd.isna(val) or val == "N/A":
        return 'background-color: white; color: black'

    # Si es string (puede ser "#¡DIV/0!" o similar)
    if isinstance(val, str):
        return 'background-color: white; color: black'

    # Convertir a float si es necesario
    try:
        val_float = float(val)
    except:
        return 'background-color: white; color: black'

    if val_float < 0.85:
        return 'background-color: #736867; color: white'
    elif val_float < 0.95:
        return 'background-color: #EFFF1C; color: black'
    elif val_float < 1.05:
        return 'background-color: #A4FF4A; color: black'
    elif val_float < 1.15:
        return 'background-color: #FFBF9C; color: black'
    else:
        return 'background-color: #FF430F; color: white'



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
                    range=["#1f77b4", "#ff7f0e"]
                ),
                legend=alt.Legend(title="")
            ),
            tooltip=[
                alt.Tooltip("periodo_label:N", title="Mes"),
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
                    range=["#1f77b4", "#ff7f0e", "#7E57C2"]
                ),
                legend=alt.Legend(title="")
            ),
            tooltip=[
                alt.Tooltip("periodo_label:N", title="Mes"),
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

# Seleccionar la columna de cumplimiento según el tipo
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
    ["orden_mixto", "cnt_rojo", "cnt_marron", "Nombre Cliente"],
    ascending=[True, False, False, True]
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

# Resaltar en morado los meses sugeridos por cliente.
def _pintar_sugeridos(row):
    estilos = [""] * len(row)
    cod_val = str(row[cod_col]).strip()
    for i, col in enumerate(row.index):
        if col in meses_ordenados and (cod_val, col) in sugeridos_pairs:
            estilos[i] = "background-color: #7E57C2; color: white; font-weight: 700"
    return estilos

styled_table = styled_table.apply(_pintar_sugeridos, axis=1)

# Centrar todas las celdas
styled_table = styled_table.set_properties(**{'text-align': 'center'})

# Tabla principal con colores y seleccion de filas
pivot_display = pivot_data.reset_index(drop=True)

selection = st.dataframe(
    styled_table,
    use_container_width=True,
    hide_index=True,
    height=600,
    on_select="rerun",
    selection_mode="multi-row",
    key="tabla_clientes",
)

st.markdown(
    "<div style='display:flex;align-items:center;margin-top:6px;'>"
    "<div style='background:#7E57C2;width:16px;height:16px;border-radius:3px;margin-right:8px;'></div>"
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

if selected_rows:
    st.markdown("---")
    cod_col = [c for c in pivot_display.columns if "Cliente" in c and c not in ["Nombre Cliente", "JCC"]][0]

    for idx in selected_rows:
        nombre_cliente = pivot_display.iloc[idx]["Nombre Cliente"]
        cliente_cod = pivot_display.iloc[idx][cod_col]
        with st.expander(f"Detalle: {nombre_cliente}", expanded=False):
            render_detalle_cliente(df_filt, df_cat_filt, df_sug, cliente_cod, nombre_cliente)

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
