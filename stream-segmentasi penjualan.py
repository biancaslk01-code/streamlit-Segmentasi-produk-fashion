import streamlit as st
import pandas as pd

# ===============================
# BACKGROUND & STYLE
# ===============================
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

# ===============================
# TITLE
# ===============================
st.title("📊 Segmentasi Penjualan Produk Fashion")
st.write("Upload file Excel/CSV dengan data penjualan kamu")

# ===============================
# UPLOAD FILE
# ===============================
file = st.file_uploader("Upload file (.xlsx / .csv)", type=["csv", "xlsx"])

if file is not None:

    # Baca file
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("📄 Kolom terdeteksi di file kamu:")
    st.write(list(df.columns))

    # ===============================
    # NORMALISASI NAMA KOLOM
    # ===============================
    column_mapping = {}

    for col in df.columns:
        if "tgl" in col.lower() or "tanggal" in col.lower() or "date" in col.lower():
            column_mapping[col] = "Tanggal"
        if "item" in col.lower() or "produk" in col.lower() or "product" in col.lower():
            column_mapping[col] = "Item"
        if "qty" in col.lower() or "jumlah" in col.lower():
            column_mapping[col] = "Qty"
        if "total" in col.lower() or "sales" in col.lower() or "net" in col.lower():
            column_mapping[col] = "Total Sales"

    df.rename(columns=column_mapping, inplace=True)

    # ===============================
    # CEK KOLOM WAJIB
    # ===============================
    required_cols = ["Tanggal", "Item", "Qty", "Total Sales"]

    if all(col in df.columns for col in required_cols):

        df = df[required_cols]

        # Pastikan numerik
        df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
        df["Total Sales"] = pd.to_numeric(df["Total Sales"], errors="coerce")

        # ===============================
        # SEGMENTASI PRODUK
        # ===============================
        def segmentasi(item):
            item = str(item).lower()

            if "woman" in item or "wanita" in item or "wm" in item or "girls" in item:
                return "WOMEN"
            elif "man" in item or "men" in item or "pria" in item:
                return "MEN"
            elif "kid" in item or "anak" in item:
                return "KIDS"
            elif "shoe" in item or "sepatu" in item:
                return "FOOTWEAR"
            elif "bag" in item or "tas" in item:
                return "BAG"
            else:
                return "LAINNYA"

        df["Segmen"] = df["Item"].apply(segmentasi)

        st.success("✅ Data berhasil dibaca & disegmentasi")

        # ===============================
        # SHOW DATA
        # ===============================
        st.subheader("📌 Data Setelah Segmentasi")
        st.dataframe(df, use_container_width=True)

        # ===============================
        # REKAP SEGMENTASI
        # ===============================
        seg_summary = df.groupby("Segmen").agg({
            "Qty": "sum",
            "Total Sales": "sum"
        }).reset_index()

        st.subheader("📊 Rekap Segmentasi")
        st.dataframe(seg_summary, use_container_width=True)

        # ===============================
        # DOWNLOAD
        # ===============================
        st.download_button(
            "⬇️ Download hasil segmentasi (Excel)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="hasil_segmentasi.csv",
            mime="text/csv"
        )

    else:
        st.error("❌ File kamu belum memiliki semua kolom penting")
        st.warning("Kolom wajib: Tanggal, Item, Qty, Total Sales")
        st.info("Silakan sesuaikan nama kolom di file Excel kamu")

