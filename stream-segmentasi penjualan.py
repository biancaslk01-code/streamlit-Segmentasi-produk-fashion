import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

st.set_page_config(page_title="Segmentasi Produk Fashion", layout="wide")

# ===================== BACKGROUND ======================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #fce7f3, #e0f2fe, #ffffff);
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
h1, h2, h3 {
    color: #0f172a;
    font-family: 'Arial';
}
.stButton>button {
    background-color: #0f172a;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ===================== TITLE ======================
st.title("📊 Segmentasi Produk Fashion Berdasarkan Pola Penjualan Bulanan")
st.write("Metode: **K-Means & Hierarchical Clustering**")

# ==============================
# UPLOAD DATA
# ==============================
uploaded_file = st.file_uploader("Upload file Excel atau CSV", type=["xlsx", "csv"])

if uploaded_file is not None:

    # ---- Baca file ----
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📌 Preview Data")
    st.dataframe(df.head())

    # ==============================
    # AUTO DETECT COLUMN
    # ==============================
    col_item = None
    col_qty = None
    col_sales = None

    for col in df.columns:
        if "item" in col.lower() or "produk" in col.lower() or "artikel" in col.lower():
            col_item = col
        if "qty" in col.lower() or "jumlah" in col.lower() or "unit" in col.lower():
            col_qty = col
        if "total" in col.lower() or "sales" in col.lower() or "penjualan" in col.lower() or "nett" in col.lower():
            col_sales = col

    if col_item is None or col_qty is None or col_sales is None:
        st.error("❌ Kolom tidak terdeteksi otomatis. Pastikan file punya:")
        st.write("- Item / Produk")
        st.write("- Qty / Jumlah")
        st.write("- Total Sales / Penjualan")
        st.stop()

    df = df[[col_item, col_qty, col_sales]]
    df.columns = ["Item", "Qty", "Sales"]

    # ==============================
    # CLEANING DATA
    # ==============================
    df["Item"] = df["Item"].astype(str).str.upper().str.strip()
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)

    # Gabungkan per produk
    data_produk = df.groupby("Item").agg({
        "Qty": "sum",
        "Sales": "sum"
    }).reset_index()

    st.subheader("📦 Data per Produk")
    st.dataframe(data_produk)

    # ==============================
    # NORMALIZATION
    # ==============================
    X = data_produk[["Qty", "Sales"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==============================
    # K-MEANS CLUSTERING
    # ==============================
    k = st.slider("Jumlah Cluster (K-Means & Hierarchical)", 2, 8, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    data_produk["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

    # ==============================
    # HIERARCHICAL CLUSTERING
    # ==============================
    hc = AgglomerativeClustering(n_clusters=k)
    data_produk["Cluster_Hierarchical"] = hc.fit_predict(X_scaled)

    # ==============================
    # LABEL SEGMENTASI
    # ==============================
    def label_cluster(x):
        if x == 0:
            return "Kurang Laris"
        elif x == 1:
            return "Sedang"
        else:
            return "Paling Laris"

    data_produk["Label_KMeans"] = data_produk["Cluster_KMeans"].apply(label_cluster)
    data_produk["Label_Hierarchical"] = data_produk["Cluster_Hierarchical"].apply(label_cluster)

    # ==============================
    # RESULT
    # ==============================
    st.subheader("✅ Hasil Segmentasi Produk (SEMUA TERBACA)")
    st.dataframe(data_produk)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Ringkasan K-Means")
        st.dataframe(data_produk["Cluster_KMeans"].value_counts().reset_index())

    with col2:
        st.subheader("📊 Ringkasan Hierarchical")
        st.dataframe(data_produk["Cluster_Hierarchical"].value_counts().reset_index())

    # ==============================
    # DOWNLOAD RESULT
    # ==============================
    st.subheader("⬇ Download hasil segmentasi")

    csv = data_produk.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download hasil sebagai CSV",
        data=csv,
        file_name="hasil_segmentasi_produk.csv",
        mime="text/csv"
    )

else:
    st.info("Silakan upload file data penjualan untuk mulai segmentasi.")
    st.markdown("Pastikan file memiliki kolom: **Item / Produk, Qty, Total Sales**")






















































