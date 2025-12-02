import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

# =========================
# CONFIG + BACKGROUND
# =========================

st.set_page_config(layout="wide")

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

# =========================
# TITLE
# =========================

st.title("SEGMENTASI PRODUK FASHION BERDASARKAN POLA PENJUALAN BULANAN")
st.write("Kolom yang wajib ada: **Item | Qty | Sales**")

# =========================
# UPLOAD DATA
# =========================

file = st.file_uploader("Upload file Excel (kolom: Item, Qty, Sales)", type=["xlsx"])

if file is not None:
    df = pd.read_excel(file)

    # =========================
    # VALIDASI KOLOM
    # =========================

    required_cols = ["Item", "Qty", "Sales"]

    if not all(col in df.columns for col in required_cols):
        st.error("❌ FILE HARUS ADA KOLOM: Item, Qty, Sales (sesuai nama ini)")
        st.stop()

    df = df[["Item", "Qty", "Sales"]]

    st.subheader("✅ DATA TERBACA")
    st.dataframe(df.head(10), use_container_width=True)

    # =========================
    # CLUSTERING
    # =========================

    X = df[["Qty","Sales"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

    # Hierarchical
    Z = linkage(X_scaled, method='ward')
    df["Hierarchical_Cluster"] = fcluster(Z, 3, criterion='maxclust')

    # =========================
    # HASIL
    # =========================

    st.subheader("✅ HASIL SEGMENTASI PER ARTIKEL")

    st.dataframe(df, use_container_width=True, height=600)

    # =========================
    # RINGKASAN CLUSTER
    # =========================

    st.subheader("📊 JUMLAH DATA PER CLUSTER")

    col1, col2 = st.columns(2)

    with col1:
        st.write("KMEANS")
        st.write(df["KMeans_Cluster"].value_counts().sort_index())

    with col2:
        st.write("HIERARCHICAL")
        st.write(df["Hierarchical_Cluster"].value_counts().sort_index())

    # =========================
    # DOWNLOAD
    # =========================

    df.to_excel("HASIL_SEGMENTASI.xlsx", index=False)
    st.success("✅ File berhasil disimpan sebagai: HASIL_SEGMENTASI.xlsx")

else:
    st.warning("⏳ Upload file Excel dulu...")


















































