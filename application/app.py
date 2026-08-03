import streamlit as st
import pandas as pd
import numpy as np

from data_loader import DataLoader
from forecasting import run_forecast_pipeline

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

def run_forecasting(df, id_col, 
                    partial_year:int=2026, # current-year column that's not a full year yet
                    months_elapsed:int=7, # current-year column that's not a full year yet
                    n_years_ahead:int=2 # forecast both 2026 (full-year) and 2027
                    ):
    
    results = run_forecast_pipeline(
        df,
        id_col=id_col,
        partial_year=partial_year,      
        months_elapsed=months_elapsed,   
        n_years_ahead=n_years_ahead,        
    )

    return results

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
    rcvs_tables["Trend AEG.csv"] = loader.to_date(rcvs_tables["Trend AEG.csv"], "Date")
    for df_name, df in rcvs_tables.items():
        rcvs_tables[df_name] = loader.clean_count_columns(df=df)

    # Clean data - RCOS
    rcos_tables["Boundary bar AEG.csv"] = loader.rename_column(df=rcos_tables["Boundary bar AEG.csv"], 
                                                                cols=["Police Districts", "Unnamed: 1"],
                                                                names=["Police District", "Count"])
    rcos_tables["ANSOC Bar AEG.csv"] = loader.rename_column(df=rcos_tables["ANSOC Bar AEG.csv"], 
                                                                    cols=["Unnamed: 1"],
                                                                    names=["Count"])
    rcos_tables["Trend AEG.csv"] = loader.rename_column(df=rcos_tables["Trend AEG.csv"], 
                                                                        cols=["Month of Year Month", "Unnamed: 1"],
                                                                        names=["Date", "Count"])
    rcos_tables["Ethnicity AES.csv"] = loader.rename_column(df=rcos_tables["Ethnicity AES.csv"], 
                                                                            cols=[f"% of Total Proceedings along Ethnic Group", "Proceedings"],
                                                                            names=["Percentage of Proceedings", "Count"])
    rcos_tables["TableA.csv"] = loader.promote_first_row_to_header(df=rcos_tables["TableA.csv"])
    rcos_tables["TableB.csv"] = loader.promote_first_row_to_header(df=rcos_tables["TableB.csv"])
    rcos_tables["Trend AEG.csv"] = loader.to_date(rcos_tables["Trend AEG.csv"], "Date", formatting=f"%b%Y")
    rcos_tables["Age and Sex AES.csv"] = loader.add_header_row(rcos_tables["Age and Sex AES.csv"], ["Age", "Count"])
    for df_name, df in rcos_tables.items():
        rcos_tables[df_name] = loader.clean_count_columns(df=df)

    # Clean data - Activity and Report
    activity_tables["Boundary Districts.csv"] = loader.promote_first_row_to_header(df=activity_tables["Boundary Districts.csv"])
    activity_tables["Boundary Districts.csv"] = activity_tables["Boundary Districts.csv"].fillna(0)
    activity_tables["Occ Type.csv"] = loader.promote_first_row_to_header(df=activity_tables["Occ Type.csv"])
    activity_tables["Occ Type.csv"] = activity_tables["Occ Type.csv"].fillna(0) 
    activity_tables["TableA.csv"] = loader.promote_first_row_to_header(df=activity_tables["TableA.csv"])
    activity_tables["TableB.csv"] = loader.promote_first_row_to_header(df=activity_tables["TableB.csv"])
    activity_tables["TableA.csv"] = loader.rename_column(df=activity_tables["TableA.csv"], 
                                                                    cols=["Police District/Region"],
                                                                    names=["Police District"])
    activity_tables["TableA.csv"].columns = list(activity_tables["TableA.csv"].columns[:1]) + list(pd.to_datetime(activity_tables["TableA.csv"].columns[1:]))
    for df_name, df in activity_tables.items():
        activity_tables[df_name] = loader.clean_count_columns(df=df)
    
    
    # Forecast Result
    # rcvs_tableA_results = run_forecasting(rcvs_tables["TableA.csv"], id_col="Police District")
    # rcvs_tableB_results = run_forecasting(rcvs_tables["TableB.csv"], id_col="ANZSOC Division")
    # rcos_tableA_results = run_forecasting(rcos_tables["TableA.csv"], id_col="Police District")
    # rcos_tableB_results = run_forecasting(rcos_tables["TableB.csv"], id_col="Anzsoc Division")
    # activity_tableA_results = run_forecasting(activity_tables["TableA.csv"], id_col="Police District", n_years_ahead=6)
    activity_tableB_results = run_forecasting(activity_tables["TableB.csv"], id_col="Occurrence Type Category", n_years_ahead=6)

    st.dataframe(
            activity_tableB_results["long_df"] .head(5).style.format(precision=2),
            use_container_width=True,
            height=420,
        )