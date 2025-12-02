import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# ======================
# JUDUL APLIKASI
# ======================
st.set_page_config(layout="wide")
st.title("SEGMENTASI PRODUK FASHION BERDASARKAN POLA PENJUALAN BULANAN")
st.markdown("### Metode K-Means & Hierarchical Clustering")

# ======================
# UPLOAD DATA
# ======================
st.subheader("1. Upload Data Penjualan")
uploaded_file = st.file_uploader("Upload file Excel atau CSV", type=["csv", "xlsx"])

if uploaded_file is not None:

    # ======================
    # READ FILE
    # ======================
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, header=1)

    st.success("✅ File berhasil di-upload")

    # ======================
    # DATA CLEANING
    # ======================
    df = df[['Tanggal', 'Item', 'Total Qty', 'Total Nett Sales']]
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    df = df.dropna()

    # Pastikan SEMUA produk (MEN + WM) terbaca
    df['Item'] = df['Item'].astype(str).str.upper().str.strip()

    df['Bulan'] = df['Tanggal'].dt.month

    st.subheader("2. Data Bersih")
    st.dataframe(df.head())

    # ======================
    # AGREGASI BULANAN
    # ======================
    monthly_sales = df.groupby(['Item', 'Bulan']).agg({
        'Total Qty': 'sum',
        'Total Nett Sales': 'sum'
    }).reset_index()

    st.subheader("3. Total Penjualan Bulanan Per Produk")
    st.dataframe(monthly_sales)

    # ======================
    # PIVOT TABLE
    # ======================
    pivot = monthly_sales.pivot_table(
        index='Item',
        columns='Bulan',
        values='Total Qty',
        fill_value=0
    )

    st.subheader("4. Pola Penjualan Bulanan (Pivot)")
    st.dataframe(pivot)

    # ======================
    # NORMALISASI
    # ======================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot)

    # ======================
    # K-MEANS
    # ======================
    st.subheader("5. K-Means Clustering")

    k = st.slider("Pilih jumlah cluster", 2, 8, 4)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    pivot['Segment (KMeans)'] = clusters

    # ======================
    # MEMBERI LABEL SEGMENT
    # ======================
    total_per_produk = df.groupby('Item')['Total Qty'].sum()

    pivot['Total_Penjualan'] = total_per_produk
    pivot = pivot.sort_values(by='Total_Penjualan', ascending=False)

    def beri_label(cluster):
        if cluster == 0:
            return "LOW SALES"
        elif cluster == 1:
            return "MEDIUM SALES"
        elif cluster == 2:
            return "HIGH SALES"
        else:
            return "VERY HIGH SALES"

    pivot['Keterangan Segment'] = pivot['Segment (KMeans)'].apply(beri_label)

    st.subheader("6. HASIL SEGMENTASI PRODUK (Semua Produk)")
    st.dataframe(pivot)

    # ======================
    # VISUALISASI
    # ======================
    st.subheader("7. Visualisasi Segmentasi (K-Means)")

    fig1, ax1 = plt.subplots(figsize=(12,5))
    ax1.scatter(pivot.index, pivot['Total_Penjualan'])
    plt.xticks(rotation=90)
    plt.ylabel("Total Penjualan")
    plt.xlabel("Produk")
    st.pyplot(fig1)

    # ======================
    # HIERARCHICAL CLUSTERING
    # ======================
    st.subheader("8. Hierarchical Clustering")

    linked = linkage(X_scaled, method='ward')

    fig2, ax2 = plt.subplots(figsize=(15, 6))
    dendrogram(
        linked,
        labels=pivot.index.tolist(),
        leaf_rotation=90
    )
    st.pyplot(fig2)

    # ======================
    # DOWNLOAD HASIL
    # ======================
    st.subheader("9. Download Hasil Segmentasi")

    output = pivot.reset_index()
    csv = output.to_csv(index=False).encode('utf-8')

    st.do
