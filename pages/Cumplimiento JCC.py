import streamlit as st
import pandas as pd
import numpy as np

from shared import (
    MES_MAP,
    multiselect_con_nulos,
    load_df_cus,
)

st.set_page_config(page_title="Cumplimiento JCC", layout="wide")
st.title("Cumplimiento JCC")

# Cargar datos
df = load_df_cus()

# Sidebar filtros (recrea los mismos filtros para esta página)
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
        # 9.1 Agregado mensual por cliente
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

        agg["Plan"] = agg[col_plan]
        agg["Real"] = agg[col_real]

        agg["Cumplimiento"] = np.where(
            agg["Plan"] == 0,
            np.nan,
            agg["Real"] / agg["Plan"]
        )

        orden_meses = [m for m in MES_MAP.values() if m in agg["mes_nombre"].unique()]

        # Tablas pivot: PLAN y REAL
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

        pivot_plan = pivot_plan.reindex(columns=orden_meses)
        pivot_real = pivot_real.reindex(columns=orden_meses)

        # CUMPLIMIENTO (Real / Plan)
        pivot_cumpl = pivot_real / pivot_plan.replace(0, np.nan)

        # Colores alineados con histograma (niveles 1..5)
        colores_nivel = ["#736867", "#EFFF1C", "#A4FF4A", "#FFBF9C", "#FF430F"]

        def color_cumpl(val):
            if pd.isna(val):
                return ""
            if val < 0.85:
                color = colores_nivel[0]
            elif val < 0.95:
                color = colores_nivel[1]
            elif val < 1.05:
                color = colores_nivel[2]
            elif val < 1.15:
                color = colores_nivel[3]
            else:
                color = colores_nivel[4]
            return f"background-color: {color}; color: black;"

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
