import streamlit as st
import pandas as pd
import numpy as np

from data_loader import DataLoader

st.set_page_config(
    page_title="NZ Crime Rate Forcasting",
    layout="wide",
)
# - Dark theme CSS -
st.markdown("""
<style>
    /* Dividers */
    hr { border-color: #30363d; }
 
    /* Select box */
    div[data-baseweb="select"] > div {
        background-color: #1c2230;
        border-color: #30363d;
        color: #FFFFFF;
    }
 
    /* DataFrames */
    .stDataFrame { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

# - App layout -
st.title("NZ Crime Rate Forecasting")
st.caption("Crime Rate Forecasting for New Zealand")
st.divider()

# Load data
with st.spinner("Loading dataset…"):
    try:
        loader = DataLoader()
        rcvs_tables = loader.fetch_rcvs()
        rcos_tables = loader.fetch_rcos()
        activity_tables = loader.fetch_activity()
        data_ok = True
    except Exception as e:
        st.error(f"Could not fetch live data: {e}")
        data_ok = False
 
if data_ok:

    # Clean data - RCVS
    rcvs_tables["Boundary bar AEG.csv"] = loader.rename_column(df=rcvs_tables["Boundary bar AEG.csv"], 
                                                               cols=["Police District/TA", "Unnamed: 1"],
                                                               names=["Police District", "Count"])
    rcvs_tables["ANSOC Bar AEG.csv"] = loader.rename_column(df=rcvs_tables["ANSOC Bar AEG.csv"], 
                                                                   cols=["Unnamed: 1"],
                                                                   names=["Count"])
    rcvs_tables["Trend AEG.csv"] = loader.rename_column(df=rcvs_tables["Trend AEG.csv"], 
                                                                       cols=["Month of Year Month", "Unnamed: 1"],
                                                                       names=["Date", "Count"])
    rcvs_tables["TableA.csv"] = loader.promote_first_row_to_header(df=rcvs_tables["TableA.csv"])
    rcvs_tables["TableB.csv"] = loader.promote_first_row_to_header(df=rcvs_tables["TableB.csv"])
    for df_name, df in rcvs_tables.items():
        rcvs_tables[df_name] = loader.clean_count_columns(df=df)
    
    table_a = rcvs_tables["TableA.csv"]

    st.dataframe(
            table_a.head(5).style.format(precision=2),
            use_container_width=True,
            height=420,
        )