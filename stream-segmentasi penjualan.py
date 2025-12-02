
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load the trained scaler and KMeans model ---
try:
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    st.success("Scaler and KMeans model loaded successfully.")
except FileNotFoundError:
    st.error("Error: scaler.pkl or kmeans_model.pkl not found. Please ensure they are in the same directory as this script.")
    st.stop()

# --- 2. Define a function to preprocess new data and predict clusters ---
def predict_product_clusters(new_sales_data_df):
    '''Processes new monthly sales data for products and predicts their clusters.

    Args:
        new_sales_data_df (pd.DataFrame): A DataFrame containing new sales data.
                                           Expected columns: 'Tanggal', 'Item', 'Qty (Normal)',
                                           'Nett Sales (Normal)', 'Qty (Discount)',
                                           'Nett Sales (Discount)', 'Total Qty', 'Total Nett Sales'.

    Returns:
        pd.DataFrame: A DataFrame with product items and their predicted K-Means clusters.
    '''
    df_processed = new_sales_data_df.copy()

    # Basic cleaning steps (similar to notebook)
    df_processed.columns = df_processed.columns.astype(str)
    df_processed = df_processed.loc[:, ~df_processed.columns.str.contains('Unnamed')]
    df_processed['Tanggal'] = pd.to_datetime(df_processed['Tanggal'], errors='coerce')
    df_processed = df_processed.dropna(subset=['Tanggal'])
    df_processed['Month'] = df_processed['Tanggal'].dt.month

    numerical_cols_to_clean = [
        'Qty (Normal)', 'Nett Sales (Normal)', 'Qty (Discount)',
        'Nett Sales (Discount)', 'Total Qty', 'Total Nett Sales'
    ]
    for col in numerical_cols_to_clean:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
            df_processed[col] = df_processed[col].apply(lambda x: max(x, 0))

    # Clean 'Item' column
    items_to_remove = ['Total', 'Item', 'Qty', 'MARKETING', 'SHOECARE']
    df_processed = df_processed[~df_processed['Item'].isin(items_to_remove)].copy()
    df_processed['Item'] = df_processed['Item'].astype(str).str.upper().str.strip()
    item_mapping = {
        'WMNFLATS': 'WMNFLAT',
        'WMNFLAT': 'WMNFLAT',
        'WMNFL.AT': 'WMNFLAT',
        'MENSANOAL': 'MENSANDAL',
        'MEMBAGS': 'MENBAGS',
        'MEMBELT': 'MENBELT',
        'WMNBA GS': 'WMNBAGS',
        'WlNHEELS': 'WMNHEELS',
        'WLNHEELS': 'WMNHEELS',
        'MENSFORMAL': 'MENFORMAL'
    }
    df_processed['Item'] = df_processed['Item'].replace(item_mapping)

    # Aggregate the new data to get monthly sales and quantity per item
    df_monthly_sales_new = df_processed.groupby(['Item', 'Month']).agg(
        total_nett_sales=('Total Nett Sales', 'sum'),
        total_qty=('Total Qty', 'sum')
    ).reset_index()

    # Pivot the data to create features similar to the training data
    df_sales_pivot_new = df_monthly_sales_new.pivot_table(
        index='Item', columns='Month', values='total_nett_sales', fill_value=0
    )
    df_sales_pivot_new.columns = [f'Sales_Month_{col}' for col in df_sales_pivot_new.columns]

    df_qty_pivot_new = df_monthly_sales_new.pivot_table(
        index='Item', columns='Month', values='total_qty', fill_value=0
    )
    df_qty_pivot_new.columns = [f'Qty_Month_{col}' for col in df_qty_pivot_new.columns]

    # Merge sales and quantity features
    # Use df_sales_pivot_new.index as the base to ensure all items are included
    all_items = df_sales_pivot_new.index.union(df_qty_pivot_new.index)
    df_features_new = pd.DataFrame(index=all_items)
    df_features_new = df_features_new.merge(df_sales_pivot_new, left_index=True, right_index=True, how='left')
    df_features_new = df_features_new.merge(df_qty_pivot_new, left_index=True, right_index=True, how='left')
    df_features_new = df_features_new.fillna(0) # Fill NaN if a month is missing for Qty/Sales

    # Ensure all original month columns are present, fill with 0 if not
    # These months were observed during training (1, 2, 3, 4, 9, 11)
    original_months = [1, 2, 3, 4, 9, 11]
    training_feature_names = [
        f'Sales_Month_{m}' for m in original_months
    ] + [
        f'Qty_Month_{m}' for m in original_months
    ]

    for col in training_feature_names:
        if col not in df_features_new.columns:
            df_features_new[col] = 0
    
    # Reorder columns to match the order used during training (important for scaler)
    # Filter to only include columns used in training
    df_features_for_scaling = df_features_new[training_feature_names]

    # Scale the features
    df_scaled_features_new = scaler.transform(df_features_for_scaling)

    # Predict clusters
    new_clusters = kmeans.predict(df_scaled_features_new)

    # Assign clusters back to items
    df_results = pd.DataFrame({'Item': df_features_new.index, 'Predicted_Cluster': new_clusters})
    return df_results

# --- Streamlit App --- 
st.set_page_config(page_title="Product Segmentation App", layout="wide")
st.title("🛍️ Product Sales Segmentation")
st.write("Upload your new monthly sales data to get product cluster predictions.")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            new_data_df = pd.read_csv(uploaded_file)
        else:
            new_data_df = pd.read_excel(uploaded_file, header=1) # Assuming same header as training data
        
        st.sidebar.success("File uploaded successfully!")
        st.subheader("Raw Data Preview (First 5 rows)")
        st.dataframe(new_data_df.head())

        st.subheader("Predicting Product Clusters...")
        predictions_df = predict_product_clusters(new_data_df)
        
        st.subheader("Predicted Clusters")
        st.dataframe(predictions_df)

        st.subheader("Cluster Summary")
        cluster_counts = predictions_df['Predicted_Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Number of Products']
        st.dataframe(cluster_counts)

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.write("Please ensure your file has the correct format and columns, similar to the training data.")
        st.write("Expected columns: 'Tanggal', 'Item', 'Qty (Normal)', 'Nett Sales (Normal)', 'Qty (Discount)', 'Nett Sales (Discount)', 'Total Qty', 'Total Nett Sales'")
else:
    st.info("Please upload a sales data file to get predictions.")
    st.markdown("**Expected Data Format:**")
    st.markdown("- Excel (.xlsx) or CSV (.csv) file.")
    st.markdown("- Columns should include: 'Tanggal', 'Item', 'Qty (Normal)', 'Nett Sales (Normal)', 'Qty (Discount)', 'Nett Sales (Discount)', 'Total Qty', 'Total Nett Sales'.")
    st.markdown("- For Excel files, assume header is in the second row (header=1).")
