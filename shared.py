import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import unicodedata


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
        value_col = "vol_ton_real"

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


def multiselect_con_nulos(label, serie, opciones_override=None):
    if opciones_override is None:
        opciones = sorted(serie.dropna().unique().tolist())
    else:
        if hasattr(opciones_override, "dropna"):
            opciones = sorted(opciones_override.dropna().unique().tolist())
        else:
            opciones = sorted([val for val in opciones_override if pd.notna(val)])
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




def calcular_abc_dinamico(df_base):
    if df_base.empty:
        return pd.DataFrame(columns=["cod_cliente_alicorp_actual", "ABC"])

    hoy = pd.Timestamp.today().normalize()
    ultimo_mes_cerrado = (hoy.replace(day=1) - pd.Timedelta(days=1)).to_period("M").to_timestamp()
    inicio_ventana = (ultimo_mes_cerrado - pd.DateOffset(months=5)).to_period("M").to_timestamp()

    df_abc_base = df_base[
        (df_base["periodo_mes"] >= inicio_ventana)
        & (df_base["periodo_mes"] <= ultimo_mes_cerrado)
    ].copy()

    if df_abc_base.empty:
        meses_disp = sorted(df_base["periodo_mes"].dropna().unique().tolist())
        if not meses_disp:
            return pd.DataFrame(columns=["cod_cliente_alicorp_actual", "ABC"])
        ultimo_disp = pd.Timestamp(meses_disp[-1])
        inicio_disp = (ultimo_disp - pd.DateOffset(months=5)).to_period("M").to_timestamp()
        df_abc_base = df_base[
            (df_base["periodo_mes"] >= inicio_disp)
            & (df_base["periodo_mes"] <= ultimo_disp)
        ].copy()

    abc = (
        df_abc_base.groupby("cod_cliente_alicorp_actual", as_index=False)
        .agg({"venta_real": "sum"})
        .sort_values("venta_real", ascending=False)
    )

    total_venta = abc["venta_real"].sum()
    if total_venta <= 0:
        abc["ABC"] = "C"
        return abc[["cod_cliente_alicorp_actual", "ABC"]]

    abc["pct_individual"] = abc["venta_real"] / total_venta
    abc["pct_acum"] = abc["pct_individual"].cumsum()
    abc["ABC"] = pd.cut(
        abc["pct_acum"],
        bins=[0, 0.80, 0.95, float("inf")],
        labels=["A", "B", "C"],
        include_lowest=True,
    )
    return abc[["cod_cliente_alicorp_actual", "ABC"]]

BQ_SOURCE_TABLE = "acpe-dev-uc-ml.dev.vcc_split_streamlit"


def _get_bq_client():
    try:
        from google.oauth2 import service_account
        from google.cloud import bigquery
    except Exception as exc:
        raise RuntimeError(
            "Faltan dependencias para BigQuery. Instala: pip install google-cloud-bigquery google-auth"
        ) from exc

    gcp_sa = st.secrets.get("gcp_service_account", None)
    if gcp_sa:
        credentials = service_account.Credentials.from_service_account_info(gcp_sa)
        return bigquery.Client(credentials=credentials, project=credentials.project_id)

    # Fallback a credenciales por defecto (ADC)
    return bigquery.Client()


@st.cache_data(ttl=600, show_spinner=False)
def load_df_base_raw():
    client = _get_bq_client()
    query = f"SELECT * FROM `{BQ_SOURCE_TABLE}`"
    rows = [dict(r) for r in client.query(query).result()]
    return pd.DataFrame(rows)


def _prepare_base_df(df_raw):
    df = df_raw.copy()

    required_cols = [
        "periodo",
        "cod_cliente_alicorp_actual",
        "nom_cliente_alicorp_actual",
        "des_grupo_precio_alicorp",
        "JCC",
        "des_oficina_venta_alicorp",
        "des_grupo_vendedor_alicorp",
        "peso_real",
        "peso_plan",
        "venta_real",
        "venta_plan",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en BigQuery ({BQ_SOURCE_TABLE}): {missing}")

    num_cols = ["peso_real", "peso_plan", "venta_real", "venta_plan"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "vol_ton_real" not in df.columns:
        df["vol_ton_real"] = df["peso_real"] / 1000
    else:
        df["vol_ton_real"] = pd.to_numeric(df["vol_ton_real"], errors="coerce").fillna(df["peso_real"] / 1000)

    if "vol_ton_plan" not in df.columns:
        df["vol_ton_plan"] = df["peso_plan"] / 1000
    else:
        df["vol_ton_plan"] = pd.to_numeric(df["vol_ton_plan"], errors="coerce").fillna(df["peso_plan"] / 1000)

    df["periodo"] = pd.to_datetime(df["periodo"], errors="coerce")
    df = df.dropna(subset=["periodo"]).copy()

    # Normalizar tipos clave
    df["cod_cliente_alicorp_actual"] = df["cod_cliente_alicorp_actual"].astype(str).str.strip()
    df["nom_cliente_alicorp_actual"] = df["nom_cliente_alicorp_actual"].astype(str)

    return df


def load_df_cus():
    # Carga base desde BigQuery y aplica transformaciones
    df = _prepare_base_df(load_df_base_raw())

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
            "vol_ton_real": "sum",
            "vol_ton_plan": "sum",
        },
    ).reset_index()

    df_cus["a?o"] = df_cus["periodo"].dt.year
    df_cus["anio"] = df_cus["a?o"]
    df_cus["a\u00f1o"] = df_cus["a?o"]
    df_cus["mes"] = df_cus["periodo"].dt.month
    df_cus["mes_nombre"] = df_cus["mes"].map(MES_MAP)
    df_cus["periodo_mes"] = df_cus["periodo"].dt.to_period("M").dt.to_timestamp()

    abc = calcular_abc_dinamico(df_cus)
    df_cus = df_cus.merge(abc[["cod_cliente_alicorp_actual", "ABC"]], on="cod_cliente_alicorp_actual", how="left")

    df_cus = calcular_campos_cumplimiento(
        df_cus, real_col="venta_real", plan_col="venta_plan", prefix="sol", escala_real=1_000_000
    )
    df_cus = calcular_campos_cumplimiento(
        df_cus, real_col="peso_real", plan_col="peso_plan", prefix="vol", escala_real=1_000
    )

    return df_cus


def load_df_with_categoria():
    """
    Carga el dataframe con la columna des_categoria para an?lisis de mix por categor?a.
    Similar a load_df_cus() pero mantiene des_categoria en el pivot.
    """
    df = _prepare_base_df(load_df_base_raw())

    if "des_categoria" not in df.columns:
        df["des_categoria"] = "(Sin categoria)"

    df_cat = df.pivot_table(
        index=[
            "periodo",
            "cod_cliente_alicorp_actual",
            "nom_cliente_alicorp_actual",
            "des_grupo_precio_alicorp",
            "JCC",
            "des_oficina_venta_alicorp",
            "des_grupo_vendedor_alicorp",
            "des_categoria",
        ],
        aggfunc={
            "peso_real": "sum",
            "peso_plan": "sum",
            "venta_real": "sum",
            "venta_plan": "sum",
            "vol_ton_real": "sum",
            "vol_ton_plan": "sum",
        },
    ).reset_index()

    df_cat["a?o"] = df_cat["periodo"].dt.year
    df_cat["anio"] = df_cat["a?o"]
    df_cat["a\u00f1o"] = df_cat["a?o"]
    df_cat["mes"] = df_cat["periodo"].dt.month
    df_cat["mes_nombre"] = df_cat["mes"].map(MES_MAP)
    df_cat["periodo_mes"] = df_cat["periodo"].dt.to_period("M").dt.to_timestamp()

    abc = calcular_abc_dinamico(df_cat)
    df_cat = df_cat.merge(abc[["cod_cliente_alicorp_actual", "ABC"]], on="cod_cliente_alicorp_actual", how="left")

    return df_cat


def _xlsx_sheet_to_dataframe_no_openpyxl(path, sheet_name="Consolidado"):
    ns_main = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

    with zipfile.ZipFile(path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {r.attrib.get("Id"): r.attrib.get("Target") for r in rels}

        target = None
        for s in workbook.find("a:sheets", ns_main):
            if s.attrib.get("name") == sheet_name:
                rid = s.attrib.get(rel_ns)
                target = rel_map.get(rid)
                break
        if target is None:
            return pd.DataFrame()

        sst = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in shared.findall("a:si", ns_main):
                txt = "".join([(t.text or "") for t in si.findall(".//a:t", ns_main)])
                sst.append(txt)

        sheet_xml = ET.fromstring(zf.read(f"xl/{target}"))
        rows = sheet_xml.find("a:sheetData", ns_main).findall("a:row", ns_main)

        records = []
        headers = None
        for row in rows:
            values = {}
            for c in row.findall("a:c", ns_main):
                ref = c.attrib.get("r", "")
                col = "".join([ch for ch in ref if ch.isalpha()])
                cell_type = c.attrib.get("t")
                v = c.find("a:v", ns_main)
                if v is None:
                    val = ""
                elif cell_type == "s":
                    idx = int(v.text)
                    val = sst[idx] if 0 <= idx < len(sst) else v.text
                else:
                    val = v.text
                values[col] = val

            if headers is None:
                headers = values
                continue

            rec = {}
            for col_letter, header in headers.items():
                rec[str(header).strip()] = values.get(col_letter, "")
            records.append(rec)

        return pd.DataFrame(records)


def load_df_sugeridos():
    path = Path("inputs") / "Consolidado Sugeridos.xlsx"
    if not path.exists():
        return pd.DataFrame(columns=["cod_cliente_alicorp_actual", "periodo_mes", "vol_sugerido"])

    try:
        df = pd.read_excel(path, sheet_name="Consolidado")
    except Exception:
        df = _xlsx_sheet_to_dataframe_no_openpyxl(path, sheet_name="Consolidado")

    if df.empty:
        return pd.DataFrame(columns=["cod_cliente_alicorp_actual", "periodo_mes", "vol_sugerido"])

    # Normalizar nombres de columnas esperadas
    def _norm_txt(val):
        s = str(val).strip().lower()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = s.replace("?", "n")
        return s

    cols_map = {_norm_txt(c): c for c in df.columns}

    def _pick(candidates):
        for cand in candidates:
            key = _norm_txt(cand)
            if key in cols_map:
                return cols_map[key]
        return None

    col_cod = _pick(["cod_cliente_alicorp_actual", "cod_cliente", "codigo cliente", "cliente"])
    col_mes = _pick(["mes", "mes_nombre"])
    col_anio = _pick(["ano", "anio", "a?o"])
    col_vol = _pick(["vol", "volumen", "vol_sugerido"])

    if not col_cod or not col_mes or not col_anio:
        return pd.DataFrame(columns=["cod_cliente_alicorp_actual", "periodo_mes", "vol_sugerido"])

    out = df[[col_cod, col_mes, col_anio] + ([col_vol] if col_vol else [])].copy()
    rename_map = {
        col_cod: "cod_cliente_alicorp_actual",
        col_mes: "mes_raw",
        col_anio: "anio_raw",
    }
    if col_vol:
        rename_map[col_vol] = "vol_sugerido"
    out = out.rename(columns=rename_map)
    if "vol_sugerido" not in out.columns:
        out["vol_sugerido"] = np.nan

    mes_map_text = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "setiembre": 9, "septiembre": 9, "octubre": 10,
        "noviembre": 11, "diciembre": 12,
        "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
        "jul": 7, "ago": 8, "set": 9, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
    }

    out["cod_cliente_alicorp_actual"] = out["cod_cliente_alicorp_actual"].astype(str).str.strip()
    out["anio"] = pd.to_numeric(out["anio_raw"], errors="coerce")
    out["mes_num"] = (
        out["mes_raw"].astype(str).str.strip().str.lower().map(mes_map_text)
        .fillna(pd.to_numeric(out["mes_raw"], errors="coerce"))
    )

    out["periodo_mes"] = pd.to_datetime(
        {"year": out["anio"], "month": out["mes_num"], "day": 1}, errors="coerce"
    )
    out["vol_sugerido"] = pd.to_numeric(out["vol_sugerido"], errors="coerce")

    out = out.dropna(subset=["cod_cliente_alicorp_actual", "periodo_mes"]).copy()
    return out[["cod_cliente_alicorp_actual", "periodo_mes", "vol_sugerido"]]
