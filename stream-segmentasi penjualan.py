import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

# =========================
# PAGE SETTING + BACKGROUND
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
# JUDUL
# =========================

st.title("SEGMENTASI ARTIKEL PRODUK (MEN & WOMEN)")
st.write("File harus bernama **data.xlsx** dan punya minimal 2 kolom numerik (Qty & Sales / Revenue / Harga).")

# =========================
# LOAD DATA
# =========================

try:
    df = pd.read_excel("data.xlsx")
except:
    st.error("❌ File data.xlsx tidak ditemukan. Simpan file kamu dengan nama **data.xlsx** dulu.")
    st.stop()

st.subheader("✅ KOLOM YANG TERBACA DARI FILE:")
st.write(list(df.columns))

# =========================
# AMBIL KOLOM ANGKA
# =========================

num_df = df.select_dtypes(include=['int64','float64'])

if num_df.shape[1] < 2:
    st.error("❌ HARUS ADA MINIMAL 2 KOLOM ANGKA (contoh: Qty & Sales)")
    st.stop()

# Ambil 2 kolom numerik pertama untuk clustering
X = num_df.iloc[:, :2]

# =========================
# STANDARDISASI
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# K-MEANS
# =========================

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# =========================
# HIERARCHICAL
# =========================

Z = linkage(X_scaled, method="ward")
df["Hierarchical_Cluster"] = fcluster(Z, t=3, criterion="maxclust")

# =========================
# PREVIEW DATA (MIN 10 BARIS)
# =========================

st.subheader("📌 PREVIEW DATA ASLI (MINIMAL 10 BARIS)")
st.dataframe(df.head(10), use_container_width=True)

st.write(f"📊 TOTAL DATA: **{len(df)} BARIS TERBACA**")

# =========================
# HASIL SEGMENTASI
# =========================

st.subheader("✅ HASIL SEGMENTASI (FULL DATA + CLUSTER)")
st.success("Kolom hasil ada di 👉 **KMeans_Cluster** & **Hierarchical_Cluster**")

st.dataframe(df, use_container_width=True, height=600)

# =========================
# RINGKASAN
# =========================

st.subheader("📊 RINGKASAN CLUSTER")

col1, col2 = st.columns(2)

with col1:
    st.write("KMeans")
    st.write(df["KMeans_Cluster"].value_counts().sort_index())

with col2:
    st.write("Hierarchical")
    st.write(df["Hierarchical_Cluster"].value_counts().sort_index())

# =========================
# SIMPAN FILE
# =========================

df.to_excel("HASIL_SEGMENTASI.xlsx", index=False)
st.success("✅ File disimpan sebagai: **HASIL_SEGMENTASI.xlsx**")



















































