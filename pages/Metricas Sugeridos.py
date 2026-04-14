import streamlit as st
import pandas as pd
import numpy as np

from shared import MES_MAP, ensure_multiselect_state, load_df_cus, load_df_sugeridos

st.set_page_config(page_title="Métricas Sugeridos", layout="wide")
st.title("Métricas Sugeridos")

# Carga de datos
_df_base = load_df_cus()
_df_sug = load_df_sugeridos()

if _df_sug.empty:
    st.warning("No se encontro data de sugeridos en inputs/Consolidado Sugeridos.xlsx (hoja Consolidado).")
    st.stop()

# Base mensual por cliente (plan y real)
anio_col = next((c for c in _df_base.columns if c in ["año", "a?o", "ano"]), None)
if anio_col is None:
    anio_col = "año"

base_cli_mes = (
    _df_base.groupby(
        [
            "cod_cliente_alicorp_actual",
            "nom_cliente_alicorp_actual",
            "periodo_mes",
            anio_col,
            "mes",
            "mes_nombre",
        ],
        as_index=False,
    )
    .agg(
        vol_plan=("vol_ton_plan", "sum"),
        vol_real=("vol_ton_real", "sum"),
    )
)

base_cli_mes["cod_cliente_alicorp_actual"] = base_cli_mes["cod_cliente_alicorp_actual"].astype(str).str.strip()
_df_sug["cod_cliente_alicorp_actual"] = _df_sug["cod_cliente_alicorp_actual"].astype(str).str.strip()
_df_sug["periodo_mes"] = pd.to_datetime(_df_sug["periodo_mes"], errors="coerce")

# Cruce sugeridos vs base
sug_det = _df_sug.merge(
    base_cli_mes,
    on=["cod_cliente_alicorp_actual", "periodo_mes"],
    how="left",
)

# Completar campos para no perder clientes sugeridos que no esten en base
sug_det["anio"] = pd.to_numeric(sug_det.get(anio_col), errors="coerce")
sug_det["anio"] = sug_det["anio"].fillna(sug_det["periodo_mes"].dt.year)
sug_det["mes"] = sug_det["mes"].fillna(sug_det["periodo_mes"].dt.month)
sug_det["mes_nombre"] = sug_det["mes"].map(MES_MAP)
sug_det["nom_cliente_alicorp_actual"] = sug_det["nom_cliente_alicorp_actual"].fillna("(Sin nombre en base)")
sug_det["vol_plan"] = sug_det["vol_plan"].fillna(0.0)
sug_det["vol_real"] = sug_det["vol_real"].fillna(0.0)

# Sidebar filtros
st.sidebar.title("Filtros")
anios = sorted(sug_det["anio"].dropna().astype(int).unique().tolist(), reverse=True)
ensure_multiselect_state("met_anio", anios, default=anios)
anio_sel = st.sidebar.multiselect("Año", anios, key="met_anio")

meses_disponibles = (
    sug_det[sug_det["anio"].isin(anio_sel)]["mes_nombre"].dropna().unique().tolist()
    if anio_sel
    else []
)
orden_meses = [MES_MAP[m] for m in range(1, 13)]
meses_sel_default = [m for m in orden_meses if m in meses_disponibles]
ensure_multiselect_state("met_mes", meses_sel_default, default=meses_sel_default)
meses_sel = st.sidebar.multiselect("Mes", meses_sel_default, key="met_mes")

filtro = sug_det.copy()
if anio_sel:
    filtro = filtro[filtro["anio"].isin(anio_sel)].copy()
if meses_sel:
    filtro = filtro[filtro["mes_nombre"].isin(meses_sel)].copy()

if filtro.empty:
    st.info("No hay datos de sugeridos con los filtros seleccionados.")
    st.stop()

filtro["periodo_label"] = filtro["periodo_mes"].apply(lambda x: f"{MES_MAP.get(x.month, x.strftime('%m'))} {x.year}")

# Resumen adherencia por mes
resumen = (
    filtro.groupby(["periodo_mes", "periodo_label"], as_index=False)
    .agg(
        clientes_sugeridos=("cod_cliente_alicorp_actual", "nunique"),
        vol_plan_total=("vol_plan", "sum"),
        dif_abs_total=("vol_sugerido", lambda s: 0.0),
    )
)
# calcular sumatoria de abs(sugerido-plan) por mes
_dif = filtro.copy()
_dif["dif_abs"] = (_dif["vol_sugerido"] - _dif["vol_plan"]).abs()
_dif_mes = _dif.groupby("periodo_mes", as_index=False).agg(dif_abs_total=("dif_abs", "sum"))
resumen = resumen.drop(columns=["dif_abs_total"]).merge(_dif_mes, on="periodo_mes", how="left")
resumen["adherencia_sugerido"] = np.where(
    resumen["vol_plan_total"] > 0,
    1 - (resumen["dif_abs_total"] / resumen["vol_plan_total"]),
    np.nan,
)
resumen = resumen.sort_values("periodo_mes")

st.subheader("Resumen de Adherencia por Mes")
st.dataframe(
    resumen[["periodo_label", "clientes_sugeridos", "vol_plan_total", "dif_abs_total", "adherencia_sugerido"]]
    .rename(
        columns={
            "periodo_label": "Periodo",
            "clientes_sugeridos": "N° Clientes Sugeridos",
            "vol_plan_total": "Vol Plan Total",
            "dif_abs_total": "Sum Abs(Sugerido - Plan)",
            "adherencia_sugerido": "Adherencia Sugerido",
        }
    )
    .style.format(
        {
            "N° Clientes Sugeridos": "{:,.0f}",
            "Vol Plan Total": "{:,.0f}",
            "Sum Abs(Sugerido - Plan)": "{:,.0f}",
            "Adherencia Sugerido": "{:.1%}",
        }
    )
    .set_properties(**{"text-align": "center"}),
    width='stretch',
    hide_index=True,
)

# Detalle clientes sugeridos
st.markdown("---")
st.subheader("Clientes Sugeridos: Plan vs Real vs Sugerido (Mensual)")

detalle_cols = [
    "periodo_mes",
    "periodo_label",
    "cod_cliente_alicorp_actual",
    "nom_cliente_alicorp_actual",
    "vol_plan",
    "vol_real",
    "vol_sugerido",
]
detalle = filtro[detalle_cols].copy()
detalle = detalle.sort_values(["periodo_mes", "nom_cliente_alicorp_actual"])

detalle_show = detalle.rename(
    columns={
        "periodo_label": "Periodo",
        "cod_cliente_alicorp_actual": "Código Cliente",
        "nom_cliente_alicorp_actual": "Nombre Cliente",
        "vol_plan": "Vol Plan",
        "vol_real": "Vol Real",
        "vol_sugerido": "Vol Sugerido",
    }
)

detalle_show["Dif Abs (Sug-Plan)"] = (detalle_show["Vol Sugerido"] - detalle_show["Vol Plan"]).abs()

st.dataframe(
    detalle_show[[
        "Periodo",
        "Código Cliente",
        "Nombre Cliente",
        "Vol Plan",
        "Vol Real",
        "Vol Sugerido",
        "Dif Abs (Sug-Plan)",
    ]]
    .style.format(
        {
            "Vol Plan": "{:,.0f}",
            "Vol Real": "{:,.0f}",
            "Vol Sugerido": "{:,.0f}",
            "Dif Abs (Sug-Plan)": "{:,.0f}",
        }
    )
    .set_properties(**{"text-align": "center"}),
    width='stretch',
    hide_index=True,
)

