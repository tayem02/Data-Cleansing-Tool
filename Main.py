import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================
# Function Definitions
# ===============================

def load_and_convert_data(uploaded_file):

    # Read file based on extension
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Auto-convert columns that appear numeric
    converted_columns = []
    for col in df.select_dtypes(include=['object']).columns:
        converted = pd.to_numeric(df[col], errors='coerce')
        ratio = converted.notna().mean()
        if ratio >= 0.8:
            df[col] = converted
            converted_columns.append(col)
    return df, converted_columns

def display_data_preview_and_types(df):

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head(20))
    with col2:
        st.subheader("Data Columns and Types")
        col_info = pd.DataFrame({"Data Type": df.dtypes.astype(str)})
        st.dataframe(col_info)

def compute_correlation_matrix(df, selected_columns):

    df_numeric = df[selected_columns].select_dtypes(include=[np.number])
    corr_matrix = df_numeric.corr()
    return corr_matrix, df_numeric

def plot_correlation_heatmap(corr_matrix, df_numeric):

    fig, ax = plt.subplots(figsize=(len(df_numeric.columns) * 0.8, len(df_numeric.columns) * 0.8))
    
    # Set overall style
    sns.set(style="white")
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        square=True,
        ax=ax,
        cbar_kws={'shrink': 0.7},
        mask=np.triu(np.ones_like(corr_matrix, dtype=bool))
    )
    
    # Customize background colors
    ax.set_facecolor('#f7f7f7')
    fig.patch.set_facecolor('#f7f7f7')
    
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)

def impute_missing_values(df, columns_to_impute, method):

    df_cleaned = df.copy()
    for col in columns_to_impute:
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            if method == "Mean":
                imputed_value = df_cleaned[col].mean()
            else:
                imputed_value = df_cleaned[col].median()
            df_cleaned[col].fillna(imputed_value, inplace=True)
        else:
            # For non-numeric columns, fill missing values with the mode if available
            if not df_cleaned[col].mode().empty:
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
            else:
                df_cleaned[col].fillna("Unknown", inplace=True)
    return df_cleaned

# ===============================
# Streamlit App Code
# ===============================

# Set page configuration (optional)
st.set_page_config(page_title="Correlation & Data Cleaning App", layout="wide")
st.title("Correlation Analysis and Data Cleaning App")

# -------------------------------
# 1. Data Upload
# -------------------------------
st.sidebar.header("1. Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Load data and auto-convert columns as needed
        df, converted_columns = load_and_convert_data(uploaded_file)
        
        if converted_columns:
            st.info(f"The following columns were auto-converted to numeric: {', '.join(converted_columns)}")
        
        # Display the data preview and the data types table
        display_data_preview_and_types(df)
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Awaiting file upload...")

# Proceed only if data is loaded
if uploaded_file is not None:
    # -------------------------------
    # 2. Correlation Matrix Calculation
    # -------------------------------
    st.header("2. Correlation Matrix")
    st.markdown(
        "Select the columns you want to include in the correlation analysis. "
        "Only numeric columns will be used for the calculation."
    )
    selected_columns = st.multiselect("Select columns for correlation analysis", options=df.columns.tolist())

    if selected_columns:
        corr_matrix, df_numeric = compute_correlation_matrix(df, selected_columns)
        if df_numeric.shape[1] == 0:
            st.warning("None of the selected columns are numeric. Please select at least one numeric column.")
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Correlation Matrix (Table)")
                st.dataframe(corr_matrix.style.format(precision=2))
            with col2:
                st.subheader("Correlation Matrix (Heatmap)")
                plot_correlation_heatmap(corr_matrix, df_numeric)
    else:
        st.info("Please select one or more columns for correlation analysis.")

    # -------------------------------
    # 3. Missing Value Imputation
    # -------------------------------
    st.header("3. Missing Value Imputation")
    st.markdown(
        "From the columns chosen above, select which ones you'd like to impute missing values for. "
        "Choose your imputation method: **Mean** or **Median** (for numeric columns)."
    )
    if selected_columns:
        columns_to_impute = st.multiselect("Select columns to impute missing values", options=selected_columns)
        imputation_method = st.selectbox("Choose imputation method", options=["Mean", "Median"])
        
        if columns_to_impute:
            if st.button("Fill Missing Values"):
                df_cleaned = impute_missing_values(df, columns_to_impute, imputation_method)
                st.success("Missing values have been filled.")
                
                # -------------------------------
                # 4. Display and Download Cleansed Data
                # -------------------------------
                st.header("4. Final Cleansed Data")
                st.dataframe(df_cleaned)
                
                # Convert the cleansed dataframe to CSV and add a download button
                csv_data = df_cleaned.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleansed Data as CSV",
                    data=csv_data,
                    file_name='cleansed_data.csv',
                    mime='text/csv'
                )
        else:
            st.info("Select one or more columns to impute missing values.")
    else:
        st.info("Please perform the correlation analysis step first to select columns.")
