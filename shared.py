import streamlit as st
import pandas as pd
import numpy as np

# Bins y labels globales
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


def calcular_campos_cumplimiento(df, real_col, plan_col, prefix, escala_real):
    df = df.copy()
    cum_col = f"cumplimiento_{prefix}"
    inter_col = f"intervalo_{prefix}"
    nivel_col = f"nivel_cum_{prefix}"
    esc_col = f"real_{prefix}_esc"

    df[cum_col] = np.where(df[plan_col] == 0, np.nan, df[real_col] / df[plan_col])

    df[inter_col] = pd.cut(df[cum_col], bins=BINS_INTERVALO, labels=LABELS_INTERVALO,
                           include_lowest=True, right=True)

    df[nivel_col] = pd.cut(df[cum_col], bins=BINS_NIVEL, labels=LABELS_NIVEL, include_lowest=True)
    df[esc_col] = df[real_col] / escala_real
    return df


def tabla_histograma(df, tipo="sol"):
    if tipo == "sol":
        inter_col = "intervalo_sol"
        val_col = "real_sol_esc"
    else:
        inter_col = "intervalo_vol"
        val_col = "real_vol_esc"

    tabla = df.groupby(inter_col, observed=True)[val_col].sum().reset_index().rename(columns={val_col: "valor"})
    return tabla


def matriz_nivel_x_dimension(df, dim_col, tipo_key):
    if tipo_key == "sol":
        nivel_col = "nivel_cum_sol"
        value_col = "venta_real"
    else:
        nivel_col = "nivel_cum_vol"
        value_col = "peso_real"

    if nivel_col not in df.columns:
        return pd.DataFrame()

    t = df.groupby([nivel_col, dim_col], dropna=True).agg(valor=(value_col, "sum")).reset_index()
    if t.empty:
        return pd.DataFrame()
    t["Nivel"] = t[nivel_col].astype(int)
    pt = t.pivot_table(index="Nivel", columns=dim_col, values="valor", aggfunc="sum", fill_value=0)
    col_tot = pt.sum(axis=0)
    pt_pct = pt.div(col_tot, axis=1).sort_index()
    return pt_pct


def multiselect_con_nulos(label, serie):
    opciones = sorted(serie.dropna().unique().tolist())
    meses_orden = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]
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


def load_df_cus():
    # lee el parquet y aplica transformaciones
    df = pd.read_parquet("inputs/base.parquet")
    df["peso_real"] = df["peso_real"].astype(float)
    df["peso_plan"] = df["peso_plan"].astype(float)
    df["venta_real"] = df["venta_real"].astype(float)
    df["venta_plan"] = df["venta_plan"].astype(float)
    df["periodo"] = pd.to_datetime(df["periodo"]) 

    df_cus = df.pivot_table(index=["periodo","cod_cliente_alicorp_actual","nom_cliente_alicorp_actual",
                                  "des_grupo_precio_alicorp","JCC","des_oficina_venta_alicorp",
                                  "des_grupo_vendedor_alicorp"],
                            aggfunc={"peso_real":"sum","peso_plan":"sum","venta_real":"sum","venta_plan":"sum"}).reset_index()

    df_cus["año"] = df_cus["periodo"].dt.year
    df_cus["mes"] = df_cus["periodo"].dt.month
    df_cus["mes_nombre"] = df_cus["mes"].map(MES_MAP)
    df_cus["periodo_mes"] = df_cus["periodo"].dt.to_period("M").dt.to_timestamp()

    abc = df_cus[df_cus["año"] == 2025].groupby("cod_cliente_alicorp_actual", as_index=False).agg({"venta_real":"sum"}).sort_values("venta_real", ascending=False)
    abc["pct_individual"] = abc["venta_real"] / abc["venta_real"].sum()
    abc["pct_acum"] = abc["pct_individual"].cumsum()
    abc["ABC"] = pd.cut(abc["pct_acum"], bins=[0,0.80,0.95,float("inf")], labels=["A","B","C"], include_lowest=True)

    df_cus = df_cus.merge(abc[["cod_cliente_alicorp_actual","ABC"]], on="cod_cliente_alicorp_actual", how="left")

    df_cus = calcular_campos_cumplimiento(df_cus, real_col="venta_real", plan_col="venta_plan", prefix="sol", escala_real=1_000_000)
    df_cus = calcular_campos_cumplimiento(df_cus, real_col="peso_real", plan_col="peso_plan", prefix="vol", escala_real=1_000)

    return df_cus
