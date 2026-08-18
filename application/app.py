import streamlit as st
import pandas as pd
import numpy as np
import locale
locale.setlocale(locale.LC_ALL, "C")

from data_loader import DataLoader
from boundary_loader import BoundaryDataLoader
from forecasting import run_forecast_pipeline
from model import compare_models, sklearn_forecast, statsmodels_forecast, prophet_forecast

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

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

@st.cache_data
def forecast_dataset(dataset_name):
    dataset = DATA_INFO[dataset_name]["dataset"]
    tableA_results = run_forecasting(dataset, id_col="Police District")
    tableA_results["rate_per_capita"] = boundary_loader.merge_with_crime_data(long_df=tableA_results["long_df"])
    return tableA_results



def model_selection(long_df, model_name, district):
    if model_name == "Linear Regression":
        model = LinearRegression()
        forecast_result = sklearn_forecast(long_df=long_df, district=district, model=model, tune=False)
    elif model_name == "SVR":
        model = Pipeline([
            ("scalar", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=100, epsilon=0.01))
        ])
        forecast_result = sklearn_forecast(long_df=long_df, district=district, model=model, tune=False)
    elif model_name == "KNN (k=10, scaled)":
        model = Pipeline([
            ("scalar", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=10))
        ])
        forecast_result = sklearn_forecast(long_df=long_df, district=district, model=model, tune=False)
    elif model_name == "Random Forest Regressor":
        model = Pipeline([
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        forecast_result = sklearn_forecast(long_df=long_df, district=district, model=model, tune=False)
    elif model_name == "Neural Network Regressor":
        model = Pipeline([
            ("scalar", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(150, 50, 10), activation='relu', 
                                solver='adam', max_iter=300))
        ])
        forecast_result = sklearn_forecast(long_df=long_df, district=district, model=model, tune=False)
    elif model_name == "Histogram Gradient Boosting Regressor":
        model = Pipeline([
            ("scalar", StandardScaler()),
            ("hgbr", HistGradientBoostingRegressor(learning_rate=0.01, max_iter=200, max_depth=50))
        ])
        forecast_result = sklearn_forecast(long_df=long_df, district=district, model=model, tune=False)
    elif model_name == "Prophet":
        forecast_result = prophet_forecast(long_df=long_df, district=district)
    elif model_name == "Statsmodel (ETS)":
        forecast_result = statsmodels_forecast(long_df=long_df, district=district, method="ets")
    elif model_name == "Statsmodel (Sarima)":
        forecast_result = statsmodels_forecast(long_df=long_df, district=district, method="sarima")
    elif model_name == "Gradient Boosting":
        forecast_result = sklearn_forecast(long_df=long_df, district=district)

    return forecast_result

@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_data():
    try:
        loader = DataLoader()
        rcvs_tables = loader.fetch_rcvs()
        rcos_tables = loader.fetch_rcos()
        activity_tables = loader.fetch_activity()

        boundary_loader = BoundaryDataLoader(
            meshblock_shp="../dataset/kx-2023-census-electoral-population-meshblock-2025-version-2-SHP/2023-census-electoral-population-meshblock-2025-version-2.shp",
            district_shp="../dataset/kx-nz-police-district-boundaries-29-april-2021-SHP/nz-police-district-boundaries-29-april-2021.shp",
        )

        data_cleaned, rcvs_tables, rcos_tables, activity_tables = clean_dataset(loader, 
                                                                  rcvs_tables, rcos_tables, 
                                                                  activity_tables)
        data_ok = True
    except Exception as e:
        st.error(f"Could not fetch live data: {e}")
        data_ok = False

    return data_ok, data_cleaned, rcvs_tables, rcos_tables, activity_tables, boundary_loader

def clean_dataset(loader:DataLoader, rcvs_tables, rcos_tables, activity_tables):
    try:
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
        rcvs_tables["TableA.csv"] = rcvs_tables["TableA.csv"].replace("Sept", "Sep")
        rcvs_tables["TableA.csv"].iloc[0] = rcvs_tables["TableA.csv"].iloc[2].astype(str).str.cat(rcvs_tables["TableA.csv"].iloc[0].astype(str), sep='')
        rcvs_tables["TableA.csv"] = loader.promote_first_row_to_header(df=rcvs_tables["TableA.csv"])
        rcvs_tables["TableA.csv"] = loader.rename_column(df=rcvs_tables["TableA.csv"], 
                                                                                cols=[np.nan],
                                                                                names=["Police District"])
        rcvs_tables["TableA.csv"] = rcvs_tables["TableA.csv"].iloc[2:].reset_index(drop=True)
        rcvs_tables["TableA.csv"] = rcvs_tables["TableA.csv"].drop(columns="TotalTotal")
        rcvs_tables["TableA.csv"].columns = list(rcvs_tables["TableA.csv"].columns[:1]) + list(pd.to_datetime(rcvs_tables["TableA.csv"].columns[1:], format=f"%b%Y"))
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
        rcos_tables["TableA.csv"] = rcos_tables["TableA.csv"].replace("Sept", "Sep")
        rcos_tables["TableA.csv"].iloc[0] = rcos_tables["TableA.csv"].iloc[2].astype(str).str.cat(rcos_tables["TableA.csv"].iloc[0].astype(str), sep='')
        rcos_tables["TableA.csv"] = loader.promote_first_row_to_header(df=rcos_tables["TableA.csv"])
        rcos_tables["TableA.csv"] = loader.rename_column(df=rcos_tables["TableA.csv"], 
                                                                                cols=[np.nan],
                                                                                names=["Police District"])
        rcos_tables["TableA.csv"] = rcos_tables["TableA.csv"].iloc[2:].reset_index(drop=True)
        rcos_tables["TableA.csv"] = rcos_tables["TableA.csv"].drop(columns="TotalTotal")
        rcos_tables["TableA.csv"].columns = list(rcos_tables["TableA.csv"].columns[:1]) + list(pd.to_datetime(rcos_tables["TableA.csv"].columns[1:], format=f"%b%Y"))
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

        data_cleaned = True
    except Exception as e:
        st.error(f"Could not clean data: {e}")
        data_cleaned = False

    return data_cleaned, rcvs_tables, rcos_tables, activity_tables



# - App layout -
st.title("NZ Crime Rate Forecasting")
st.caption("Crime Rate Forecasting for New Zealand")
st.divider()

if "data_ok" not in st.session_state:
    st.session_state["data_ok"] = False

# Load data
if st.session_state["data_ok"] == False:
    data_ok, data_cleaned, rcvs_tables, rcos_tables, activity_tables, boundary_loader = load_data()
    if data_ok and data_cleaned:
        st.session_state["data_ok"] = True
        st.session_state["rcvs_tables"] = rcvs_tables
        st.session_state["rcos_tables"] = rcos_tables
        st.session_state["activity_tables"] = activity_tables
        st.session_state["boundary_loader"] = boundary_loader

if st.session_state["data_ok"] == True:
    rcvs_tables = st.session_state["rcvs_tables"]
    rcos_tables = st.session_state["rcos_tables"]
    activity_tables = st.session_state["activity_tables"]
    boundary_loader = st.session_state["boundary_loader"]

    st.header("Data Selection")
    data_sel, data_desc= st.columns([2, 2])
    
    DATA_INFO = {
        "RCVS": {
            "desc": "Recorded Crime Victims Statistics.",
            "dataset": rcvs_tables["TableA.csv"],
        },
        "RCOS": {
            "desc": "Recorded Crime Offenders Statistics.",
            "dataset": rcos_tables["TableA.csv"],
        },
        # "Activity": {
        #     "desc": "Activities.",
        #     "dataset": rcos_tables["TableA.csv"],
        # }
    }

    if not "data_chosen" in st.session_state:
            st.session_state["data_chosen"] = "RCOS"
    with data_sel: 
        data_choice = st.selectbox(
            "Choose a dataset",
            list(DATA_INFO.keys()),
            index=0,
        )
    with data_desc:
        st.markdown(f"**Description:** {DATA_INFO[data_choice]['desc']}")

    if data_choice != st.session_state["data_chosen"]:
        st.session_state["build_features"] = forecast_dataset(
            data_choice
        )
        st.session_state["data_chosen"] = data_choice

    # Forecast Result
    
    # rcvs_tableB_results = run_forecasting(rcvs_tables["TableB.csv"], id_col="ANZSOC Division")
    # rcos_tableA_results = run_forecasting(rcos_tables["TableA.csv"], id_col="Police District")
    # rcos_tableB_results = run_forecasting(rcos_tables["TableB.csv"], id_col="Anzsoc Division")
    # activity_tableA_results = run_forecasting(activity_tables["TableA.csv"], id_col="Police District", n_years_ahead=6)
    # activity_tableB_results = run_forecasting(activity_tables["TableB.csv"], id_col="Occurrence Type Category", n_years_ahead=6)

    # rcos_tableA_results["rate_per_capita"] = boundary_loader.merge_with_crime_data(long_df=rcos_tableA_results["long_df"])
    # activity_tableA_results["rate_per_capita"] = boundary_loader.merge_with_crime_data(long_df=activity_tableA_results["long_df"])

    st.header("Model and District Selection")
    col_sel, col_desc, district_sel, select_button = st.columns([2, 2, 2, 1])
    MODEL_INFO = {
        "Gradient Boosting": {
            "desc": "Prediction using Gradient Boosting.",
        },
        "Linear Regression": {
            "desc": "Prediction using Linear Regression.",
        },
        "SVR": {
            "desc": "Prediction using SVR.",
        },
        "KNN (k=10, scaled)": {
            "desc": "Prediction using K-Nearest Neighbours Regression.",
        },
        "Random Forest Regressor": {
            "desc": "Prediction using Random Forest Regression.",
        },
        "Neural Network Regressor": {
            "desc": "Prediction using Neural Network Regression Feed Foward.",
        },
        "Histogram Gradient Boosting Regressor": {
            "desc": "Prediction using Neural Network Regression Feed Foward.",
        },
        "Prophet": {
            "desc": "Prediction using Prophet.",
        },
        "Statsmodel (ETS)": {
            "desc": "Prediction using Statsmodel (ETS).",
        },
        "Statsmodel (Sarima)": {
            "desc": "Prediction using Statsmodel (Sarima).",
        }
    }

    if "build_features" in st.session_state:
        built_table =  st.session_state["build_features"]

    DISTRICTS_SEL = built_table["rate_per_capita"]["district"].unique()
 
    with col_sel:
        chosen = st.selectbox(
            "Choose a prediction model",
            list(MODEL_INFO.keys()),
            index=0,
        )
    with col_desc:
        st.markdown(f"**Description:** {MODEL_INFO[chosen]['desc']}")
    with district_sel:
        chosen2 = st.selectbox(
            "Choose a district",
            DISTRICTS_SEL,
            index=0,
        )
    with select_button:
        if st.button("Select"):
            st.session_state["forecast_result"] = model_selection(
                built_table["long_df"],
                chosen,
                chosen2
            )

    if "forecast_result" in st.session_state:
        forecast_result = st.session_state["forecast_result"]

        st.dataframe(
            forecast_result["forecast"][["date", "forecast_count"]],
            use_container_width=True,
        )