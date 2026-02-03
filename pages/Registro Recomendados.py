from pathlib import Path

import streamlit as st
import pandas as pd

from shared import load_df_cus

st.set_page_config(page_title="Registro Recomendados", layout="wide")
st.title("Registro de Clientes Recomendados")

CSV_PATH = "inputs/recomendados.csv"
PERIODO_ACTUAL = "202601"


def load_recomendados():
    path = Path(CSV_PATH)
    if not path.exists():
        return pd.DataFrame(columns=["periodo", "cod_cliente", "nombre_cliente", "recomendado"])
    df = pd.read_csv(path, dtype={"cod_cliente": str, "periodo": str})
    expected = ["periodo", "cod_cliente", "nombre_cliente", "recomendado"]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df[expected]


def save_recomendados(df):
    Path(CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)


# Data de clientes
base = load_df_cus()
clientes = base[["cod_cliente_alicorp_actual", "nom_cliente_alicorp_actual"]].drop_duplicates()
clientes = clientes.sort_values("nom_cliente_alicorp_actual")

cliente_dict = dict(zip(clientes["nom_cliente_alicorp_actual"], clientes["cod_cliente_alicorp_actual"].astype(str)))

st.caption(f"Periodo actual: {PERIODO_ACTUAL}")

col_sel, col_btn = st.columns([3, 1])
with col_sel:
    cliente_sel = st.selectbox(
        "Selecciona cliente",
        options=[""] + list(cliente_dict.keys()),
        index=0
    )

with col_btn:
    st.write("")
    if st.button("Marcar recomendado", disabled=(cliente_sel == "")):
        df_rec = load_recomendados()
        cod = cliente_dict[cliente_sel]

        mask = (df_rec["periodo"] == PERIODO_ACTUAL) & (df_rec["cod_cliente"] == cod)
        if mask.any():
            df_rec.loc[mask, "nombre_cliente"] = cliente_sel
            df_rec.loc[mask, "recomendado"] = True
        else:
            df_rec = pd.concat([
                df_rec,
                pd.DataFrame([
                    {
                        "periodo": PERIODO_ACTUAL,
                        "cod_cliente": cod,
                        "nombre_cliente": cliente_sel,
                        "recomendado": True,
                    }
                ])
            ], ignore_index=True)

        save_recomendados(df_rec)
        st.success("Cliente marcado como recomendado.")

st.markdown("---")

st.subheader("Registro de recomendados")
df_rec = load_recomendados()

if df_rec.empty:
    st.info("Aun no hay clientes recomendados.")
else:
    df_view = df_rec[df_rec["periodo"] == PERIODO_ACTUAL].copy()
    if df_view.empty:
        st.info("No hay clientes recomendados para el periodo actual.")
    else:
        st.dataframe(df_view, use_container_width=True, hide_index=True)
