
# Install Streamlit if not already installed
!pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Segmentasi Produk Fashion", layout="wide")

# =========================
# JUDUL APLIKASI
# =========================

st.title("Segmentasi Produk Fashion")
st.markdown('''
Aplikasi ini digunakan untuk melakukan **segmentasi produk fashion berdasarkan pola penjualan bulanan**
menggunakan metode **K-Means** dan **Hierarchical Clustering**.
''')

# =========================
# LOAD DATA
# =========================

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV / Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data Awal")
    st.dataframe(df)

    # =========================
    # PILIH KOLOM BULAN
    # =========================

    st.sidebar.header("Pilih Kolom Penjualan Bulanan")
    columns = st.sidebar.multiselect("Pilih kolom penjualan (Jan - Des):", df.columns)

    if len(columns) > 1:

        X = df[columns]

        # =========================
        # SCALING
        # =========================

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # =========================
        # INPUT JUMLAH CLUSTER
        # =========================

        st.sidebar.header("Pengaturan Clustering")
        k = st.sidebar.slider("Jumlah cluster (K)", 2, 10, 3)

        # =========================
        # K-MEANS
        # =========================

        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)

        df["Cluster_KMeans"] = kmeans_labels

        # =========================
        # HIERARCHICAL
        # =========================

        hierarchical = AgglomerativeClustering(n_clusters=k)
        hier_labels = hierarchical.fit_predict(X_scaled)

        df["Cluster_Hierarchical"] = hier_labels

        st.success("Clustering berhasil dilakukan!")

        # =========================
        # HASIL DATA
        # =========================

        st.subheader("Hasil Segmentasi")
        st.dataframe(df)

        # =========================
        # VISUALISASI
        # =========================

        st.subheader("Visualisasi Hasil Clustering (2D)")

        fig, ax = plt.subplots(1, 2, figsize=(14,6))

        sns.scatterplot(
            x=X_scaled[:, 0],
            y=X_scaled[:, 1],
            hue=kmeans_labels,
            palette="Set1",
            ax=ax[0]
        )
        ax[0].set_title("K-Means Clustering")

        sns.scatterplot(
            x=X_scaled[:, 0],
            y=X_scaled[:, 1],
            hue=hier_labels,
            palette="Set2",
            ax=ax[1]
        )
        ax[1].set_title("Hierarchical Clustering")

        st.pyplot(fig)

        # =========================
        # DOWNLOAD HASIL
        # =========================

        st.subheader("Download Hasil")
        result_csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Hasil Segmentasi",
            data=result_csv,
            file_name='hasil_segmentasi.csv',
            mime='text/csv'
        )

    else:
        st.warning("Pilih minimal 2 kolom penjualan bulanan!")

else:
    st.info("Silakan upload dataset terlebih dahulu")
