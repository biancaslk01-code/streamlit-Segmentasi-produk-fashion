import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG & BACKGROUND
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


st.title("📊 SEGMENTASI PRODUK FASHION")
st.write("**Metode: K-Means & Hierarchical Clustering**")
st.write("Kolom yang digunakan: `Item`, `Total Qty`, `Total Nett Sales`")

# ===================================
# UPLOAD DATA
# ===================================

uploaded_file = st.file_uploader("Upload File Excel / CSV", type=["xlsx", "csv"])

if uploaded_file is not None:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("✅ Data Asli (15 baris pertama)")
    st.dataframe(df.head())

    # ===================================
    # CLEANING DATA
    # ===================================

    df.columns = df.columns.astype(str).str.strip()

    needed_cols = ["Item", "Total Qty", "Total Nett Sales"]

    if not all(col in df.columns for col in needed_cols):
        st.error("❌ Kolom Wajib tidak ditemukan: Item, Total Qty, Total Nett Sales")
        st.stop()

    df = df[needed_cols].copy()

    df["Item"] = df["Item"].astype(str).str.strip().str.upper()
    df = df[df["Item"] != "NAN"]
    df = df[df["Item"] != ""]
    df = df[~df["Item"].str.contains("TOTAL", na=False)]

    # Bersihkan angka
    df["Total Qty"] = df["Total Qty"].astype(str).str.replace(",", "").str.strip()
    df["Total Nett Sales"] = df["Total Nett Sales"].astype(str).str.replace(",", "").str.strip()

    df["Total Qty"] = pd.to_numeric(df["Total Qty"], errors="coerce").fillna(0)
    df["Total Nett Sales"] = pd.to_numeric(df["Total Nett Sales"], errors="coerce").fillna(0)

    df = df.groupby("Item", as_index=False).agg({
        "Total Qty": "sum",
        "Total Nett Sales": "sum"
    })

    df = df.rename(columns={
        "Total Qty": "Qty",
        "Total Nett Sales": "Sales"
    })

    st.subheader("✅ Data Setelah Dibersihkan")
    st.dataframe(df)

    if len(df) < 3:
        st.error("Data terlalu sedikit untuk clustering (minimal 3 produk diperlukan)")
        st.stop()

    # ===================================
    # SCALING
    # ===================================

    X = df[["Qty", "Sales"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ===================================
    # KMEANS CLUSTERING
    # ===================================

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

    # ===================================
    # HIERARCHICAL CLUSTERING
    # ===================================

    linked = linkage(X_scaled, method="ward")
    df["Hierarchical_Cluster"] = fcluster(linked, t=3, criterion='maxclust')

    # ===================================
    # HASIL AKHIR
    # ===================================

    st.subheader("✅ HASIL SEGMENTASI PRODUK")
    st.dataframe(df)

    # ===================================
    # VISUALISASI KMEANS
    # ===================================

    st.subheader("📈 Visualisasi K-Means Clustering")

    fig1, ax1 = plt.subplots()
    scatter1 = ax1.scatter(df["Qty"], df["Sales"], c=df["KMeans_Cluster"])

    for i, txt in enumerate(df["Item"]):
        ax1.annotate(txt, (df["Qty"][i], df["Sales"][i]), fontsize=7)

    ax1.set_xlabel("Total Quantity")
    ax1.set_ylabel("Total Sales")
    ax1.set_title("K-Means Clustering")

    st.pyplot(fig1)

    # ===================================
    # VISUALISASI HIERARCHICAL
    # ===================================

    st.subheader("📈 Visualisasi Hierarchical Clustering")

    fig2, ax2 = plt.subplots()
    scatter2 = ax2.scatter(df["Qty"], df["Sales"], c=df["Hierarchical_Cluster"])

    for i, txt in enumerate(df["Item"]):
        ax2.annotate(txt, (df["Qty"][i], df["Sales"][i]), fontsize=7)

    ax2.set_xlabel("Total Quantity")
    ax2.set_ylabel("Total Sales")
    ax2.set_title("Hierarchical Clustering")

    st.pyplot(fig2)

    # ===================================
    # DOWNLOAD RESULT
    # ===================================

    st.subheader("⬇ Download Hasil Segmentasi")
    st.download_button(
        "Download Excel",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="hasil_segmentasi_produk.csv",
        mime="text/csv"
    )

else:
    st.warning("Silakan upload file data terlebih dahulu")
