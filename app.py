import io
import re
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -----------------------------
# Helpers
# -----------------------------

def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def detect_branded(keyword: str, brand_terms: List[str]) -> bool:
    k = keyword.lower()
    return any(bt in k for bt in brand_terms if bt)


def estimate_intent(keyword: str) -> float:
    k = keyword.lower()
    buy_terms = [
        "buy", "for sale", "price", "best", "coupon", "discount",
        "deal", "free shipping", "near me", "vs ", "review", "reviews",
        "top", "cheap", "affordable"
    ]
    score = 0.0
    for t in buy_terms:
        if t in k:
            score += 0.08
    wc = len(k.split())
    score += max(0.0, min(0.2, 0.03 * max(0, wc - 2)))
    return min(1.0, score)


def infer_product(keyword: str) -> str:
    k = keyword.lower()
    words = k.split()
    if len(words) == 1:
        return k

    if "tea" in k:
        if "green" in k:
            return "Green Tea"
        if "ginger" in k or "turmeric" in k:
            return "Ginger/Turmeric Tea"
        return "Tea"

    if "coffee" in k:
        return "Coffee"

    if "shower" in k or "rinsekit" in k:
        return "Portable Shower"

    if "tumbler" in k or "bottle" in k:
        return "Drinkware"

    return words[0].capitalize()


def bucket(score: float) -> str:
    if score >= 0.7:
        return "Easy"
    if score >= 0.5:
        return "Moderate"
    return "Hard"


# -----------------------------
# App
# -----------------------------

st.set_page_config(page_title="Keyword Viability Scorer", page_icon="📈", layout="wide")
st.title("Keyword Viability Scorer")
st.write(
    "Upload Google Ads Keyword Planner data and automatically score keyword viability based on search volume, competition, CPC, and branded dominance."
)

uploaded = st.file_uploader("Upload Google Ads Keyword Planner CSV", type=["csv"])


# -----------------------------
# Sidebar Weights
# -----------------------------
with st.sidebar:
    st.header("Scoring Weights")
    w_vol = st.slider("Search volume weight", 0.0, 1.0, 0.35, 0.05)
    w_comp = st.slider("Competition weight (lower is better)", 0.0, 1.0, 0.30, 0.05)
    w_cpc = st.slider("CPC weight (lower is better)", 0.0, 1.0, 0.20, 0.05)
    w_brand = st.slider("Brand penalty weight", 0.0, 1.0, 0.10, 0.05)
    w_intent = st.slider("Intent/long-tail weight", 0.0, 1.0, 0.05, 0.05)
    st.caption("All weights will be normalized to sum to 1.0.")

    brand_csv = st.text_input(
        "Brand terms (comma-separated)",
        value="nike,apple,amazon,rinsekit,joolca,hydroflask,yeti"
    )
    brand_terms = [b.strip().lower() for b in brand_csv.split(",") if b.strip()]


# -----------------------------
# Detect Google Ads columns
# -----------------------------
def detect_google_ads_format(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    required = ["keyword", "avg. monthly searches", "competition"]
    return all(any(req in c for c in cols.keys()) for req in required)


# -----------------------------
# Main Processing
# -----------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded)

    if detect_google_ads_format(df):
        st.success("Google Ads format detected — columns auto-mapped.")
        col_map = {}
        for col in df.columns:
            c = col.lower()
            if "keyword" in c:
                col_map['keyword'] = col
            elif "avg. monthly searches" in c:
                col_map['search_volume'] = col
            elif c.startswith("competition") and "indexed" not in c:
                col_map['competition'] = col
            elif "top of page bid (low range)" in c:
                col_map['cpc_low'] = col
            elif "top of page bid (high range)" in c:
                col_map['cpc_high'] = col
        kw_col = col_map.get('keyword')
        vol_col = col_map.get('search_volume')
        comp_col = col_map.get('competition')
        cpc_low_col = col_map.get('cpc_low')
        cpc_high_col = col_map.get('cpc_high')
    else:
        st.warning("Could not auto-detect columns — please rename to standard Google Ads format.")
        st.stop()

    # -----------------------------
    # Parsing functions
    # -----------------------------
    out = pd.DataFrame()
    out["keyword"] = df[kw_col].astype(str)

    def parse_sv(x):
        if pd.isna(x):
            return np.nan
        s = str(x).lower().replace(",", "").strip()
        m = re.match(r"(\d+(?:\.\d+)?)[kKmM]?\s*[-–]\s*(\d+(?:\.\d+)?)[kKmM]?", s)
        if m:
            a, b = m.groups()
            a = float(a) * (1000 if "k" in s else (1_000_000 if "m" in s else 1))
            b = float(b) * (1000 if "k" in s else (1_000_000 if "m" in s else 1))
            return (a + b) / 2
        m2 = re.match(r"(\d+(?:\.\d+)?)([kKmM])?", s)
        if m2:
            val = float(m2.group(1))
            suf = m2.group(2)
            if suf and suf.lower() == "k":
                val *= 1000
            elif suf and suf.lower() == "m":
                val *= 1_000_000
            return val
        try:
            return float(s)
        except Exception:
            return np.nan

    out["search_volume_raw"] = df[vol_col].apply(parse_sv)

    def parse_comp(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in {"low", "l"}:
            return 0.2
        if s in {"medium", "med", "m"}:
            return 0.5
        if s in {"high", "h"}:
            return 0.8
        try:
            v = float(s)
            return max(0.0, min(1.0, v))
        except Exception:
            return np.nan

    out["competition_raw"] = df[comp_col].apply(parse_comp)

    def parse_cpc(x):
        if pd.isna(x):
            return np.nan
        s = re.sub(r"[^0-9.]+", "", str(x))
        try:
            return float(s)
        except Exception:
            return np.nan

    cpc_low = df[cpc_low_col].apply(parse_cpc) if cpc_low_col else pd.Series(np.nan, index=df.index)
    cpc_high = df[cpc_high_col].apply(parse_cpc) if cpc_high_col else pd.Series(np.nan, index=df.index)
    out["cpc_raw"] = np.nanmean(np.vstack([cpc_low, cpc_high]), axis=0)

    out["product"] = out["keyword"].apply(infer_product)

    out["volume_norm"] = normalize_series(out["search_volume_raw"].fillna(out["search_volume_raw"].median()))
    out["comp_norm"] = normalize_series(out["competition_raw"].fillna(out["competition_raw"].median()))
    out["cpc_norm"] = normalize_series(out["cpc_raw"].fillna(out["cpc_raw"].median()))

    out["is_branded"] = out["keyword"].apply(lambda k: detect_branded(k, brand_terms))
    out["brand_penalty"] = np.where(out["is_branded"], 1.0, 0.0)

    out["intent_score"] = out["keyword"].apply(estimate_intent)

    weights = np.array([w_vol, w_comp, w_cpc, w_brand, w_intent], dtype=float)
    if weights.sum() == 0:
        weights = np.array([0.35, 0.30, 0.20, 0.10, 0.05])
    weights = weights / weights.sum()

    out["viability_score"] = (
        weights[0] * out["volume_norm"]
        + weights[1] * (1 - out["comp_norm"])
        + weights[2] * (1 - out["cpc_norm"])
        + weights[3] * (1 - out["brand_penalty"])
        + weights[4] * out["intent_score"]
    )

    out["difficulty_bucket"] = out["viability_score"].apply(bucket)

    # -----------------------------
    # Product-level summary
    # -----------------------------
    prod_agg = (
        out.groupby("product")
        .agg(
            keywords=("keyword", "count"),
            branded_share=("is_branded", "mean"),
            avg_volume=("search_volume_raw", "mean"),
            avg_comp=("competition_raw", "mean"),
            avg_cpc=("cpc_raw", "mean"),
            avg_intent=("intent_score", "mean"),
            avg_score=("viability_score", "mean"),
        )
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )

    # -----------------------------
    # Display Results
    # -----------------------------
    st.subheader("Keyword-level results")
    st.dataframe(
        out[[
            "keyword", "product", "search_volume_raw", "competition_raw", "cpc_raw",
            "is_branded", "intent_score", "viability_score", "difficulty_bucket"
        ]].sort_values("viability_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Product-level summary")
    st.dataframe(prod_agg, use_container_width=True, hide_index=True)

    st.subheader("📊 Keyword Viability Visualizations")

    top_products_chart = (
        alt.Chart(prod_agg.head(10))
        .mark_bar()
        .encode(
            x=alt.X("avg_score:Q", title="Average Viability Score"),
            y=alt.Y("product:N", sort="-x", title="Product"),
            tooltip=["avg_score", "avg_volume", "avg_cpc", "avg_comp"],
            color=alt.Color("avg_score:Q", scale=alt.Scale(scheme="greenblue"))
        )
        .properties(title="Top 10 Easiest Products to Launch")
    )

    difficulty_chart = (
        alt.Chart(out)
        .mark_bar()
        .encode(
            x=alt.X("difficulty_bucket:N", title="Difficulty Level"),
            y=alt.Y("count():Q", title="Number of Keywords"),
            color=alt.Color("difficulty_bucket:N", scale=alt.Scale(scheme="set2"))
        )
        .properties(title="Keyword Difficulty Distribution")
    )

    col1, col2 = st.columns(2)
    col1.altair_chart(top_products_chart, use_container_width=True)
    col2.altair_chart(difficulty_chart, use_container_width=True)

    # -----------------------------
    # CSV Export
    # -----------------------------
    csv_keywords = out.to_csv(index=False).encode("utf-8")
    csv_products = prod_agg.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download keyword results (CSV)",
        data=csv_keywords,
        file_name="keyword_viability_scores.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download product summary (CSV)",
        data=csv_products,
        file_name="product_viability_summary.csv",
        mime="text/csv",
    )

    # ============================================================
    # === NEO4J GRAPH DATABASE INTEGRATION (added section) =======
    # ============================================================

    from graph_store import GraphStore
    import os

    st.subheader("📡 Save Results to Neo4j Graph Database")

    with st.expander("Neo4j Connection Settings"):
        neo4j_uri = st.text_input(
            "Neo4j URI",
            value=os.getenv("NEO4J_URI", "bolt://localhost:7687")
        )
        neo4j_user = st.text_input(
            "Neo4j Username",
            value=os.getenv("NEO4J_USER", "neo4j")
        )
        neo4j_password = st.text_input(
            "Neo4j Password",
            type="password",
            value=os.getenv("NEO4J_PASSWORD", "example")
        )

    if st.button("Save to Neo4j"):
        try:
            store = GraphStore(neo4j_uri, neo4j_user, neo4j_password)

            # WARNING: resets DB
            store.reset()

            # Insert product nodes
            store.upsert_products(prod_agg)

            # Insert keyword nodes
            keyword_cols = [
                "product", "keyword", "search_volume_raw", "competition_raw",
                "cpc_raw", "is_branded", "intent_score",
                "viability_score", "difficulty_bucket"
            ]
            store.upsert_keywords(out[keyword_cols])

            store.close()
            st.success("✔ Data successfully saved to Neo4j!")

        except Exception as e:
            st.error(f"❌ Failed to save to Neo4j: {e}")

