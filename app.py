# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Dashboard Cumplimiento Plan", layout="wide")

# -------------------------------
# Bins y labels globales
# -------------------------------
BINS_INTERVALO = [-np.inf, 0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
                  0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65,
                  1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35, 2.45, 2.55,
                  2.65, 2.75, 2.85, 2.95, 3, np.inf]

LABELS_INTERVALO = ["Menor a 0%", "0-15%", "15-25%", "25-35%", "35-45%", "45-55%",
                    "55-65%", "65-75%", "75-85%", "85-95%", "95-105%", "105-115%",
                    "115-125%", "125-135%", "135-145%", "145-155%", "155-165%",
                    "165-175%", "175-185%", "185-195%", "195-205%", "205-215%",
                    "215-225%", "225-235%", "235-245%", "245-255%", "255-265%",
                    "265-275%", "275-285%", "285-295%", "295-300%", "Más de 300%"]

BINS_NIVEL = [0, 0.85, 0.95, 1.05, 1.15, float("inf")]
LABELS_NIVEL = [1, 2, 3, 4, 5]

MES_MAP = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
           7: "Jul", 8: "Ago", 9: "Set", 10: "Oct", 11: "Nov", 12: "Dic"}

SIN_DATO_LABEL = "(Sin dato)"


# ======================================================
# 1. FUNCIONES DE CÁLCULO
# ======================================================
def calcular_campos_cumplimiento(df, real_col, plan_col,
                                 prefix, escala_real):
    """
    df          : DataFrame
    real_col    : venta_real o peso_real
    plan_col    : venta_plan o peso_plan
    prefix      : 'sol' o 'vol'
    escala_real : 1_000_000 (soles) o 1_000 (volumen)
    """
    df = df.copy()

    cum_col = f"cumplimiento_{prefix}"
    inter_col = f"intervalo_{prefix}"
    nivel_col = f"nivel_cum_{prefix}"
    esc_col = f"real_{prefix}_esc"

    df[cum_col] = np.where(
        df[plan_col] == 0,
        np.nan,
        df[real_col] / df[plan_col]
    )

    df[inter_col] = pd.cut(
        df[cum_col],
        bins=BINS_INTERVALO,
        labels=LABELS_INTERVALO,
        include_lowest=True,
        right=True
    )

    df[nivel_col] = pd.cut(
        df[cum_col],
        bins=BINS_NIVEL,
        labels=LABELS_NIVEL,
        include_lowest=True
    )

    df[esc_col] = df[real_col] / escala_real

    return df


def tabla_histograma(df, tipo="sol"):
    """Tabla para histograma: intervalo vs valor escalado."""
    if tipo == "sol":
        inter_col = "intervalo_sol"
        val_col = "real_sol_esc"
    else:
        inter_col = "intervalo_vol"
        val_col = "real_vol_esc"

    tabla = (
        df.groupby(inter_col, observed=True)[val_col]
          .sum()
          .reset_index()
          .rename(columns={val_col: "valor"})
    )
    return tabla


def matriz_nivel_x_dimension(df, dim_col, tipo_key):
    """
    Matriz Nivel (1–5) x dimensión (Región / Canal) con % por columna.
    tipo_key: 'sol' (usa venta_real, nivel_cum_sol) o 'vol' (usa peso_real, nivel_cum_vol)
    """
    if tipo_key == "sol":
        nivel_col = "nivel_cum_sol"
        value_col = "venta_real"
    else:
        nivel_col = "nivel_cum_vol"
        value_col = "peso_real"

    if nivel_col not in df.columns:
        return pd.DataFrame()

    t = (
        df.groupby([nivel_col, dim_col], dropna=True)
          .agg(valor=(value_col, "sum"))
          .reset_index()
    )
    if t.empty:
        return pd.DataFrame()

    t["Nivel"] = t[nivel_col].astype(int)
    pt = t.pivot_table(index="Nivel", columns=dim_col, values="valor",
                       aggfunc="sum", fill_value=0)

    col_tot = pt.sum(axis=0)
    pt_pct = pt.div(col_tot, axis=1)
    pt_pct = pt_pct.sort_index()
    return pt_pct


def multiselect_con_nulos(label, serie):
    opciones = sorted(serie.dropna().unique().tolist())
    meses_orden = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                   "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]
    if label == "Mes":
        opciones = [m for m in meses_orden if m in opciones]

    if serie.isna().any():
        opciones.append(SIN_DATO_LABEL)
    seleccion = st.sidebar.multiselect(label, opciones, default=opciones)
    valores_validos = [val for val in seleccion if val != SIN_DATO_LABEL]
    mask = serie.isin(valores_validos)
    if SIN_DATO_LABEL in seleccion:
        mask = mask | serie.isna()
    return seleccion, mask


# ======================================================
# 2. CARGA Y TRANSFORMACIÓN INICIAL
# ======================================================
def load_df_cus():
    # 1) leer base.csv
    df = pd.read_parquet("inputs/base.parquet")

    # 2) transformaciones iniciales
    df["peso_real"] = df["peso_real"].astype(float)
    df["peso_plan"] = df["peso_plan"].astype(float)
    df["venta_real"] = df["venta_real"].astype(float)
    df["venta_plan"] = df["venta_plan"].astype(float)

    df["periodo"] = pd.to_datetime(df["periodo"])

    # 3) pivot a nivel cliente-periodo
    df_cus = df.pivot_table(
        index=[
            "periodo",
            "cod_cliente_alicorp_actual",
            "nom_cliente_alicorp_actual",
            "des_grupo_precio_alicorp",
            "JCC",
            "des_oficina_venta_alicorp",
            "des_grupo_vendedor_alicorp",
        ],
        aggfunc={
            "peso_real": "sum",
            "peso_plan": "sum",
            "venta_real": "sum",
            "venta_plan": "sum",
        },
    ).reset_index()

    # 5) año / mes / periodo_mes
    df_cus["año"] = df_cus["periodo"].dt.year
    df_cus["mes"] = df_cus["periodo"].dt.month
    df_cus["mes_nombre"] = df_cus["mes"].map(MES_MAP)
    # primer día del mes para series de tiempo
    df_cus["periodo_mes"] = df_cus["periodo"].dt.to_period("M").dt.to_timestamp()

    # 6) ABC sobre 2025
    abc = (
        df_cus[df_cus["año"] == 2025]
        .groupby("cod_cliente_alicorp_actual", as_index=False)
        .agg({"venta_real": "sum"})
        .sort_values("venta_real", ascending=False)
    )
    abc["pct_individual"] = abc["venta_real"] / abc["venta_real"].sum()
    abc["pct_acum"] = abc["pct_individual"].cumsum()
    abc["ABC"] = pd.cut(
        abc["pct_acum"],
        bins=[0, 0.80, 0.95, float("inf")],
        labels=["A", "B", "C"],
        include_lowest=True,
    )

    # 7) merge ABC a df_cus
    df_cus = df_cus.merge(
        abc[["cod_cliente_alicorp_actual", "ABC"]],
        on="cod_cliente_alicorp_actual",
        how="left",
    )

    # 8) calcular cumplimiento soles y volumen
    df_cus = calcular_campos_cumplimiento(
        df_cus, real_col="venta_real", plan_col="venta_plan",
        prefix="sol", escala_real=1_000_000
    )
    df_cus = calcular_campos_cumplimiento(
        df_cus, real_col="peso_real", plan_col="peso_plan",
        prefix="vol", escala_real=1_000
    )

    return df_cus


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

# ======================================================
# 4. APLICAR FILTROS
# ======================================================
df_filt = df[
    df["año"].isin(anio_sel)
    & mask_mes
    & mask_abc
    & mask_region
    & mask_canal
    & mask_zona
].copy()

# columnas dinámicas por tipo
if tipo_key == "sol":
    total_real = df_filt["venta_real"].sum()
    total_plan = df_filt["venta_plan"].sum()
    unidad = "S/."
else:
    total_real = df_filt["peso_real"].sum()
    total_plan = df_filt["peso_plan"].sum()
    unidad = "Ton"

cumpl_global = total_real / total_plan if total_plan > 0 else np.nan

# ======================================================
# 5. KPIs
# ======================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Real", f"{total_real:,.0f} {unidad}")
with col2:
    st.metric("Plan", f"{total_plan:,.0f} {unidad}")
with col3:
    st.metric(
        "Cumplimiento",
        f"{cumpl_global:.1%}" if not np.isnan(cumpl_global) else "NA"
    )

st.markdown("---")

# ======================================================
# 6. HISTOGRAMA
# ======================================================
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

    chart = (
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
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No hay datos para los filtros seleccionados.")

st.markdown("---")

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
# 8. EVOLUCIÓN DE NIVELES (1–5) – % VENTA REAL
# ======================================================
st.markdown("---")
st.subheader("Evolución del mix de niveles (venta real)")

df_niv = df_filt#.dropna(subset=["nivel_cum_sol"]).copy()

if df_niv.empty:
    st.info("No hay datos para la evolución de niveles con los filtros actuales.")
else:
    # agregamos por mes y nivel (usar periodo_mes y etiqueta legible)
    evo = (
        df_niv.groupby(
            ["periodo_mes", "año", "mes", "nivel_cum_sol"],
            observed=True,
            dropna=False,
        )
        .agg(venta_real=("venta_real", "sum"))
        .reset_index()
    )

    # etiqueta legible para el eje x (p. ej. "Jun 2025" con meses en español)
    evo["periodo_label"] = evo["mes"].map(MES_MAP) + " " + evo["año"].astype(str)

    # ordenar los periodos según la fecha real
    periodos_ordenados = [
        f"{MES_MAP[p.month]} {p.year}" for p in sorted(evo["periodo_mes"].unique())
    ]

    # % de venta por nivel dentro de cada periodo
    evo["pct"] = evo["venta_real"] / evo.groupby("periodo_mes")["venta_real"].transform("sum")

    # Nivel como categoría (evita valores NaN en la leyenda/tooltip)
    evo["Nivel"] = evo["nivel_cum_sol"].astype("Int64").astype("string").fillna("Sin nivel")

    # área apilada 100% usando la etiqueta como eje x (ordenada)
    chart_evo = (
        alt.Chart(evo)
        .mark_area()
        .encode(
            x=alt.X("periodo_label:N", sort=periodos_ordenados, title="Periodo"),
            y=alt.Y("pct:Q", stack="zero", axis=alt.Axis(format="%", title="% venta real")),
            color=alt.Color("Nivel:N", title="Nivel"),
            tooltip=[
                alt.Tooltip("periodo_label:N", title="Periodo"),
                alt.Tooltip("Nivel:N"),
                alt.Tooltip("pct:Q", format=".1%", title="% venta"),
            ],
        )
    )
    st.altair_chart(chart_evo, use_container_width=True)

    # tabla tipo pivote (mes x nivel) usando periodo_mes para ordenar y periodo_label para mostrar
    tabla_niv = (
        evo.pivot_table(index="periodo_label", columns="Nivel", values="pct", aggfunc="sum")
        .reindex(periodos_ordenados)
        .fillna(0)
    ).sort_index()

    st.caption("Tabla % venta real por nivel y mes")
    st.dataframe(tabla_niv.style.format("{:.1%}"), use_container_width=True)
# ...existing code...
# ======================================================
# 9. JCC – CLIENTES: PLAN, REAL y CUMPLIMIENTO
# ======================================================
st.markdown("---")
st.subheader("Clientes por JCC – Plan, Real y Cumplimiento")

# JCC disponibles con los filtros actuales
jcc_vals = sorted(df_filt["JCC"].dropna().unique().tolist())

if not jcc_vals:
    st.info("No hay JCC disponibles con los filtros actuales.")
else:
    jcc_sel = st.selectbox("Selecciona JCC", jcc_vals)

    df_jcc = df_filt[df_filt["JCC"] == jcc_sel].copy()

    # Clientes de ese JCC (sin selección por defecto)
    clientes_vals = sorted(
        df_jcc["nom_cliente_alicorp_actual"].dropna().unique().tolist()
    )
    cliente_sel = st.multiselect(
        "Selecciona clientes", clientes_vals, default=[]
    )

    if not cliente_sel:
        st.info("Selecciona al menos un cliente para ver el detalle.")
        st.stop()

    df_jcc = df_jcc[df_jcc["nom_cliente_alicorp_actual"].isin(cliente_sel)]

    if df_jcc.empty:
        st.info("No hay datos para ese JCC / clientes con los filtros actuales.")
    else:
        # -----------------------------------------
        # 9.1 Agregado mensual por cliente
        # -----------------------------------------
        agg = (
            df_jcc.groupby(["nom_cliente_alicorp_actual", "mes_nombre"], as_index=False)
                  .agg(
                      venta_real=("venta_real", "sum"),
                      venta_plan=("venta_plan", "sum"),
                      peso_real=("peso_real", "sum"),
                      peso_plan=("peso_plan", "sum"),
                  )
        )

        # columnas de trabajo según tipo
        if tipo_key == "sol":
            col_plan = "venta_plan"
            col_real = "venta_real"
            titulo_tipo = "Vol Plan / Vol Real (S/.)"
        else:
            col_plan = "peso_plan"
            col_real = "peso_real"
            titulo_tipo = "Vol Plan / Vol Real (Volumen)"

        # valores base sin escalas (si quieres escalar, divide aquí)
        agg["Plan"] = agg[col_plan]
        agg["Real"] = agg[col_real]

        # cumplimiento
        agg["Cumplimiento"] = np.where(
            agg["Plan"] == 0,
            np.nan,
            agg["Real"] / agg["Plan"]
        )

        # ordenar columnas de meses
        orden_meses = [m for m in MES_MAP.values() if m in agg["mes_nombre"].unique()]

        # -----------------------------------------
        # 9.2 Tablas pivot: PLAN y REAL
        # -----------------------------------------
        pivot_plan = pd.pivot_table(
            agg,
            index="nom_cliente_alicorp_actual",
            columns="mes_nombre",
            values="Plan",
            aggfunc="sum",
            margins=True,
            margins_name="Total general",
        )

        pivot_real = pd.pivot_table(
            agg,
            index="nom_cliente_alicorp_actual",
            columns="mes_nombre",
            values="Real",
            aggfunc="sum",
            margins=True,
            margins_name="Total general",
        )

        # reordenar meses en columnas
        pivot_plan = pivot_plan.reindex(columns=orden_meses)
        pivot_real = pivot_real.reindex(columns=orden_meses)

        # -----------------------------------------
        # 9.3 Tabla pivot: CUMPLIMIENTO (Real / Plan)
        # -----------------------------------------
        # calculamos cumplimiento a partir de los pivots
        pivot_cumpl = pivot_real / pivot_plan.replace(0, np.nan)

        # función de color según umbral (igual que histograma)
        def color_cumpl(val):
            if pd.isna(val):
                return ""
            if val < 0.85:
                color = "#4B0082"   # plomo/morado
            elif val < 0.95:
                color = "#FFD700"   # amarillo
            elif val < 1.05:
                color = "#00B050"   # verde
            elif val < 1.15:
                color = "#FF6B6B"   # rojo claro
            else:
                color = "#DC143C"   # rojo fuerte
            return f"background-color: {color}; color: white;"

        # -----------------------------------------
        # 9.4 Mostrar tablas lado a lado
        # -----------------------------------------
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### PLAN")
            st.dataframe(
                pivot_plan.style.format("{:,.0f}"),
                use_container_width=True
            )

        with c2:
            st.markdown("#### REAL")
            st.dataframe(
                pivot_real.style.format("{:,.0f}"),
                use_container_width=True
            )

        with c3:
            st.markdown("#### CUMPLIMIENTO")
            st.dataframe(
                pivot_cumpl.style
                .format("{:.0%}")
                .applymap(color_cumpl),
                use_container_width=True
            )

        st.caption(f"Vista basada en {titulo_tipo} por cliente y mes para el JCC: {jcc_sel}")