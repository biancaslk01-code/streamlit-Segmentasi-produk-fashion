import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import io
import matplotlib.pyplot as plt

# for dendrogram
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

# Optional UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# Page config & title requested
st.set_page_config(page_title="Segmentasi Produk Fashion", layout="wide")
st.title("Segmentasi Produk Fashion Berdasarkan Pola Penjualan Bulanan Menggunakan Metode K-Means dan Hierarchical Clustering")

# Funny background / caption
st.markdown("### ✨ BACKGROUND LUCU: SI KLOPA SI ARTIKEL PUNYA RAHSIA ✨")
st.caption("Katanya tiap produk fashion kalau dikumpulin bareng-bareng jadi klan — ada yang best-seller, ada yang cuma numpang lewat. Kita bantu cari teman seperjuangan supaya toko nggak sepi!")

# Sidebar options
st.sidebar.header("Pengaturan")
default_path = "/mnt/data/Data Penjualan 2024.xlsx"
uploaded_file = st.sidebar.file_uploader("Unggah file Excel (.xlsx / .xls) — atau kosongkan pakai file default", type=["xlsx", "xls"])
use_default_if_exists = False
if uploaded_file is None and os.path.exists(default_path):
    use_default_if_exists = st.sidebar.checkbox("Gunakan file default (Data Penjualan 2024.xlsx) dari server", value=True)
n_clusters = st.sidebar.slider("Jumlah cluster (k)", 2, 10, 4)
agg_method = st.sidebar.selectbox("Agglomerative linkage", ["ward", "complete", "average", "single"])
visual_method = st.sidebar.selectbox("Visualisasi 2D", ["UMAP (jika tersedia)", "PCA (fallback)"])
show_dendrogram = st.sidebar.checkbox("Tampilkan dendrogram (Hierarchical)", value=True)

# Load Excel (all sheets)
if uploaded_file is not None:
    try:
        xls = pd.read_excel(uploaded_file, sheet_name=None)
        st.sidebar.success("File Excel terbaca (multiple sheets). Memakai sheet pertama sebagai default.")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file: {e}")
        st.stop()
elif use_default_if_exists:
    try:
        xls = pd.read_excel(default_path, sheet_name=None)
        st.sidebar.success(f"Membaca {default_path}")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca default file: {e}")
        st.stop()
else:
    st.warning("Tidak ada file. Unggah file Excel atau pilih default di sidebar.")
    st.stop()

# Choose sheet
sheet_names = list(xls.keys())
sheet_choice = st.selectbox("Pilih sheet untuk diproses", sheet_names, index=0)
data = xls[sheet_choice].copy()
st.write("Preview data (5 baris):")
st.dataframe(data.head())

# Detect identifier-like columns for product/article
cols_lower = [c.lower() for c in data.columns.astype(str)]
candidates = ["artikel", "article", "product", "item", "nama", "nama barang", "kode", "sku", "id", "produk", "product_name"]
detected = [c for c in data.columns if c.lower() in candidates]
id_col = None
if detected:
    id_col = st.selectbox("Kolom identifier terdeteksi — pilih kolom identifier (artikel/product)", detected, index=0)
else:
    st.info("Tidak menemukan kolom identifier umum otomatis.")
    id_choice = st.selectbox("Pilih kolom identifier (jika ada) atau pilih 'RowID' untuk treat tiap baris sebagai produk", ["RowID"] + list(data.columns.astype(str)))
    if id_choice == "RowID":
        data = data.reset_index().rename(columns={"index": "RowID"})
        id_col = "RowID"
    else:
        id_col = id_choice

st.write(f"Identifier yang dipakai: **{id_col}**")

# Numeric features detection / selection
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    # try coercing to numeric
    coerced = {}
    for c in data.columns:
        coerced_col = pd.to_numeric(data[c], errors="coerce")
        if coerced_col.notna().sum() > 0:
            coerced[c + "_num"] = coerced_col
    if coerced:
        data = pd.concat([data, pd.DataFrame(coerced)], axis=1)
        numeric_cols = list(coerced.keys())

st.write("Kolom numerik terdeteksi:", numeric_cols)
custom_features = st.multiselect("Pilih kolom numerik untuk fitur (biarkan kosong untuk pakai semua terdeteksi)", numeric_cols)
if custom_features:
    features = custom_features
else:
    features = numeric_cols

if not features:
    st.error("Tidak ada fitur numerik yang cukup untuk segmentasi. Pastikan ada kolom numerik (contoh: penjualan, qty, harga).")
    st.stop()

st.write("Fitur yang digunakan:", features)

# Aggregate per identifier (sum, mean, median, std)
st.info("Mengagregasi fitur per identifier (sum, mean, median, std) agar tiap produk punya ringkasan pola penjualan.")
agg_funcs = {f: ["sum", "mean", "median", "std"] for f in features}
grouped = data.groupby(id_col).agg(agg_funcs)
grouped.columns = ["_".join(map(str, c)).strip() for c in grouped.columns.values]
grouped = grouped.fillna(0)
st.write("Contoh fitur agregat (5 baris):")
st.dataframe(grouped.head())

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(grouped.values)

# KMeans
st.write("Menjalankan KMeans & Agglomerative (Hierarchical)...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# Agglomerative
agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=agg_method)
agg_labels = agg.fit_predict(X)

# Add labels to dataframe
grouped_with_labels = grouped.copy()
grouped_with_labels["KMeans_cluster"] = kmeans_labels
grouped_with_labels["Agg_cluster"] = agg_labels

# Silhouette
try:
    sil_k = silhouette_score(X, kmeans_labels) if 1 < n_clusters < len(X) else None
except Exception:
    sil_k = None
try:
    sil_a = silhouette_score(X, agg_labels) if 1 < n_clusters < len(X) else None
except Exception:
    sil_a = None

c1, c2 = st.columns(2)
with c1:
    st.metric("Silhouette KMeans", f"{sil_k:.4f}" if sil_k is not None else "N/A")
with c2:
    st.metric("Silhouette Agglomerative", f"{sil_a:.4f}" if sil_a is not None else "N/A")

# Show results table
st.subheader("Hasil Segmentasi — Contoh (top 100)")
st.dataframe(grouped_with_labels.reset_index().head(100))

# Cluster summaries (count + mean of features)
def cluster_summary(df_labels, label_col):
    cols = [c for c in df_labels.columns if c not in ["KMeans_cluster", "Agg_cluster"]]
    summary = df_labels.groupby(label_col)[cols].agg(["count", "mean"])
    summary.columns = ["_".join(map(str, c)).strip() for c in summary.columns.values]
    return summary

st.subheader("Ringkasan Cluster KMeans")
st.dataframe(cluster_summary(grouped_with_labels, "KMeans_cluster").reset_index())

st.subheader("Ringkasan Cluster Agglomerative")
st.dataframe(cluster_summary(grouped_with_labels, "Agg_cluster").reset_index())

# Examples per cluster
st.subheader("Contoh anggota cluster (maks 10 anggota tiap cluster)")
examples = []
for method in ["KMeans_cluster", "Agg_cluster"]:
    for cl in sorted(grouped_with_labels[method].unique()):
        members = grouped_with_labels[grouped_with_labels[method] == cl].reset_index().iloc[:10][id_col].astype(str).tolist()
        examples.append({"method": method, "cluster": int(cl), "count": int((grouped_with_labels[method] == cl).sum()), "examples": ", ".join(members)})
examples_df = pd.DataFrame(examples)
st.dataframe(examples_df)

# 2D Visualization (UMAP or PCA)
st.subheader("Visualisasi 2D Cluster")
if visual_method == "UMAP (jika tersedia)" and UMAP_AVAILABLE:
    reducer = umap.UMAP(n_components=2, random_state=42)
    X2 = reducer.fit_transform(X)
    vis_note = "UMAP used"
else:
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    vis_note = "PCA used"

viz_df = pd.DataFrame(X2, columns=["x", "y"], index=grouped_with_labels.index)
viz_df["KMeans_cluster"] = grouped_with_labels["KMeans_cluster"].values
viz_df["Agg_cluster"] = grouped_with_labels["Agg_cluster"].values

# Plot KMeans clusters
fig_k, axk = plt.subplots(figsize=(7, 5))
for cl in sorted(viz_df["KMeans_cluster"].unique()):
    sel = viz_df[viz_df["KMeans_cluster"] == cl]
    axk.scatter(sel["x"], sel["y"], label=f"c{cl}", alpha=0.7, s=30)
axk.set_title(f"KMeans clusters (2D) — {vis_note}")
axk.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig_k)

# Plot Agglomerative clusters
fig_a, axa = plt.subplots(figsize=(7, 5))
for cl in sorted(viz_df["Agg_cluster"].unique()):
    sel = viz_df[viz_df["Agg_cluster"] == cl]
    axa.scatter(sel["x"], sel["y"], label=f"c{cl}", alpha=0.7, s=30)
axa.set_title(f"Agglomerative clusters (2D) — {vis_note}")
axa.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig_a)

# Dendrogram (optional)
if show_dendrogram:
    st.subheader("Dendrogram (Hierarchical)")
    # compute condensed distance matrix
    try:
        # Use the scaled features X
        dist_matrix = pdist(X)  # pairwise distances
        Z = hierarchy.linkage(dist_matrix, method=agg_method)
        fig_d, axd = plt.subplots(figsize=(10, 4))
        hierarchy.dendrogram(Z, truncate_mode='level', p=8, no_labels=True, ax=axd)
        axd.set_title("Dendrogram (truncated)")
        st.pyplot(fig_d)
    except Exception as e:
        st.warning(f"Gagal membuat dendrogram: {e}")

# Save CSV results to disk and provide downloads
out_dir = "/mnt/data/segmentation_results"
os.makedirs(out_dir, exist_ok=True)
labels_path = os.path.join(out_dir, "segmentation_labels.csv")
kmeans_summary_path = os.path.join(out_dir, "kmeans_summary.csv")
agg_summary_path = os.path.join(out_dir, "agg_summary.csv")

grouped_with_labels.reset_index().to_csv(labels_path, index=False)
cluster_summary(grouped_with_labels, "KMeans_cluster").reset_index().to_csv(kmeans_summary_path, index=False)
cluster_summary(grouped_with_labels, "Agg_cluster").reset_index().to_csv(agg_summary_path, index=False)

st.markdown("**Hasil disimpan (server):**")
st.code(labels_path)
st.code(kmeans_summary_path)
st.code(agg_summary_path)

# Download buttons
st.markdown("**Unduh hasil CSV langsung:**")
csv_buf = io.StringIO()
grouped_with_labels.reset_index().to_csv(csv_buf, index=False)
st.download_button("Download segmentation_labels.csv", data=csv_buf.getvalue(), file_name="segmentation_labels.csv", mime="text/csv")

csv_buf2 = io.StringIO()
cluster_summary(grouped_with_labels, "KMeans_cluster").reset_index().to_csv(csv_buf2, index=False)
st.download_button("Download kmeans_summary.csv", data=csv_buf2.getvalue(), file_name="kmeans_summary.csv", mime="text/csv")

csv_buf3 = io.StringIO()
cluster_summary(grouped_with_labels, "Agg_cluster").reset_index().to_csv(csv_buf3, index=False)
st.download_button("Download agg_summary.csv", data=csv_buf3.getvalue(), file_name="agg_summary.csv", mime="text/csv")

st.success("Selesai.")
