import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

#############################################
# CONFIG & BACKGROUND
#############################################

st.set_page_config(
    page_title="Segmentasi Produk Fashion",
    layout="wide"
)

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

#############################################
# TITLE
#############################################

st.title("📊 Segmentasi Produk Fashion")
st.markdown("""
Metode yang digunakan:
- ✅ K-Means Clustering
- ✅ Hierarchical Clustering

Segmentasi dilakukan berdasarkan:
- Total Quantity
- Total Sales
""")

#############################################
# UPLOAD FILE
#############################################

uploaded_file = st.file_uploader("📤 Upload file Excel / CSV (Penjualan Produk)", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Preview Data Asli")
    st.dataframe(df.head())

    #############################################
    # AUTO FILTER COLUMN (ANTI ERROR)
    #############################################

    kolom = df.columns.str.lower()

    if "item" in kolom:
        col_item = df.columns[kolom.tolist().index("item")]
    elif "produk" in kolom:
        col_item = df.columns[kolom.tolist().index("produk")]
    else:
        st.error("❌ Kolom ITEM / PRODUK tidak ditemukan")
        st.stop()

    if "qty" in kolom:
        col_qty = df.columns[kolom.tolist().index("qty")]
    elif "total qty" in kolom:
        col_qty = df.columns[kolom.tolist().index("total qty")]
    else:
        st.error("❌ Kolom QTY tidak ditemukan")
        st.stop()

    if "sales" in kolom:
        col_sales = df.columns[kolom.tolist().index("sales")]
    elif "total nett sales" in kolom:
        col_sales = df.columns[kolom.tolist().index("total nett sales")]
    elif "total sales" in kolom:
        col_sales = df.columns[kolom.tolist().index("total sales")]
    else:
        st.error("❌ Kolom SALES tidak ditemukan")
        st.stop()

    df = df[[col_item, col_qty, col_sales]]
    df.columns = ["Item", "Qty", "Sales"]

    #############################################
    # GROUPING PER ARTIKEL
    #############################################

    produk = df.groupby("Item", as_index=False).agg({
        "Qty": "sum",
        "Sales": "sum"
    })

    st.subheader("📑 Data Setelah Digabung per Artikel")
    st.dataframe(produk)

    #############################################
    # SCALING
    #############################################

    X = produk[['Qty', 'Sales']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #############################################
    # K-MEANS
    #############################################

    kmeans = KMeans(n_clusters=3, random_state=42)
    produk["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

    #############################################
    # HIERARCHICAL
    #############################################

    hier = AgglomerativeClustering(n_clusters=3, linkage="ward")
    produk["Cluster_Hierarchical"] = hier.fit_predict(X_scaled)

    #############################################
    # RESULT TABLE
    #############################################

    st.subheader("✅ Hasil Segmentasi Produk")
    st.dataframe(produk)

    #############################################
    # VISUALISASI K-MEANS
    #############################################

    st.subheader("📈 Visualisasi K-Means Clustering")

    fig1, ax1 = plt.subplots()
    scatter = ax1.scatter(
        produk["Qty"],
        produk["Sales"],
        c=produk["Cluster_KMeans"]
    )

    for i, txt in enumerate(produk["Item"]):
        ax1.annotate(txt, (produk["Qty"][i], produk["Sales"][i]), fontsize=7)

    ax1.set_xlabel("Total Quantity")
    ax1.set_ylabel("Total Sales")
    ax1.set_title("K-Means Clustering Produk")

    st.pyplot(fig1)

    #############################################
    # VISUALISASI HIERARCHICAL
    #############################################

    st.subheader("📈 Visualisasi Hierarchical Clustering")

    fig2, ax2 = plt.subplots()
    scatter2 = ax2.scatter(
        produk["Qty"],
        produk["Sales"],
        c=produk["Cluster_Hierarchical"]
    )

    for i, txt in enumerate(produk["Item"]):
        ax2.annotate(txt, (produk["Qty"][i], produk["Sales"][i]), fontsize=7)

    ax2.set_xlabel("Total Quantity")
    ax2.set_ylabel("Total Sales")
    ax2.set_title("Hierarchical Clustering Produk")

    st.pyplot(fig2)

    #############################################
    # DOWNLOAD RESULT
    #############################################

    st.subheader("⬇️ Download Hasil Segmentasi")

    csv = produk.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV Hasil Segmentasi",
        data=csv,
        file_name="hasil_segmentasi_fashion.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload dulu file Excel/CSV data penjualanmu")





















































