import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# ----- BACKGROUND COLOR -----
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #4b6cb7, #182848);
        color: white;
    }
    .stDataFrame {
        background-color: white;
    }
    h1, h2, h3, h4, h5 {
        color: white !important;
    }
    .css-1d391kg, .css-1v3fvcr {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("SEGMENTASI PRODUK FASHION BERDASARKAN POLA PENJUALAN BULANAN")
st.markdown("### Metode K-Means & Hierarchical Clustering")

st.subheader("1. Upload Data Penjualan")
uploaded_file = st.file_uploader("Upload file Excel atau CSV", type=["csv", "xlsx"])

if uploaded_file is not None:

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("📌 Kolom yang terdeteksi:")
    st.write(df.columns.tolist())

    # NORMALISASI FORMAT NAMA KOLOM
    df.columns = df.columns.str.strip().str.lower()

    # MAPPING NAMA KOLOM
    column_map = {}

    for col in df.columns:
        if 'tanggal' in col or 'date' in col:
            column_map['tanggal'] = col
        elif 'item' in col or 'produk' in col or 'nama' in col:
            column_map['item'] = col
        elif 'total qty' in col or 'qty' in col:
            column_map['total qty'] = col
        elif 'nett' in col or 'sales' in col or 'penjualan' in col:
            column_map['total nett sales'] = col

    try:
        df = df[
            [
                column_map['tanggal'],
                column_map['item'],
                column_map['total qty'],
                column_map['total nett sales']
            ]
        ]

        df.columns = ['Tanggal', 'Item', 'Total Qty', 'Total Nett Sales']

    except:
        st.error("Kolom tidak dikenali. Pastikan ada kolom Tanggal, Item, Qty, dan Total Sales")
        st.stop()

    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    df = df.dropna()
    df['Item'] = df['Item'].astype(str).str.upper().str.strip()
    df['Bulan'] = df['Tanggal'].dt.month

    st.subheader("2. Data setelah dibersihkan")
    st.dataframe(df.head())

    # AGREGASI
    monthly_sales = df.groupby(['Item', 'Bulan']).agg({
        'Total Qty': 'sum',
        'Total Nett Sales': 'sum'
    }).reset_index()

    st.subheader("3. Penjualan Bulanan")
    st.dataframe(monthly_sales)

    pivot = monthly_sales.pivot_table(
        index='Item',
        columns='Bulan',
        values='Total Qty',
        fill_value=0
    )

    st.subheader("4. Pivot Table")
    st.dataframe(pivot)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot)

    st.subheader("5. K-Means Clustering")

    k = st.slider("Jumlah cluster", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    pivot['Cluster'] = kmeans.fit_predict(X_scaled)

    total_jual = df.groupby('Item')['Total Qty'].sum()
    pivot['Total Penjualan'] = total_jual

    def label_cluster(x):
        if x == 0:
            return "LOW SALES"
        elif x == 1:
            return "MEDIUM SALES"
        elif x == 2:
            return "HIGH SALES"
        else:
            return "VERY HIGH SALES"

    pivot['Segment'] = pivot['Cluster'].apply(label_cluster)

    st.subheader("✅ HASIL SEGMENTASI SEMUA PRODUK (MEN + WM + DLL)")
    st.dataframe(pivot)

    st.subheader("6. Visualisasi")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(pivot.index, pivot['Total Penjualan'])
    plt.xticks(rotation=90)
    plt.title("Total Penjualan per Produk")
    st.pyplot(fig)

    st.subheader("7. Hierarchical Clustering")

    linked = linkage(X_scaled, method='ward')

    fig2, ax2 = plt.subplots(figsize=(15, 5))
    dendrogram(linked, labels=pivot.index.tolist(), leaf_rotation=90)
    st.pyplot(fig2)

    st.success("🎉 Segmentasi Produk Berhasil!")

else:
    st.info("Silakan upload file penjualan terlebih dahulu")
