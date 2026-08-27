import streamlit as st
import pandas as pd
import numpy as np
import re
import locale
locale.setlocale(locale.LC_ALL, "C")

from data_loader import DataLoader
from boundary_loader import BoundaryDataLoader
from forecasting import run_forecast_pipeline
from model import sklearn_forecast, statsmodels_forecast, prophet_forecast
from visualize import (plot_rate_choropleth, plot_rate_matrix, 
                       plot_anzsoc_bar, plot_stacked_bar, 
                       plot_anzsoc_area, plot_anzsoc_treemap)

import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
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

def run_forecasting(df, id_col):
    
    results = run_forecast_pipeline(df, id_col=id_col)

    return results


@st.cache_data(show_spinner="Building features")
def forecast_dataset(dataset_name, id_col="Police District"):
    dataset = DATA_INFO[dataset_name]["dataset"]
    tableA_results = run_forecasting(dataset, id_col=id_col)
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
        for tbl_name in ["TableA.csv", "TableB.csv"]:
            rcvs_tables[tbl_name] = rcvs_tables[tbl_name].replace("Sept", "Sep")
            rcvs_tables[tbl_name].iloc[0] = rcvs_tables[tbl_name].iloc[2].astype(str).str.cat(rcvs_tables[tbl_name].iloc[0].astype(str), sep='')
            rcvs_tables[tbl_name] = loader.promote_first_row_to_header(df=rcvs_tables[tbl_name])
            rcvs_tables[tbl_name] = loader.rename_column(df=rcvs_tables[tbl_name], 
                                                                cols=[np.nan],
                                                                names=["Police District"])
            rcvs_tables[tbl_name] = rcvs_tables[tbl_name].iloc[2:].reset_index(drop=True)
            rcvs_tables[tbl_name] = rcvs_tables[tbl_name].drop(columns="TotalTotal")
            rcvs_tables[tbl_name].columns = list(rcvs_tables[tbl_name].columns[:1]) + list(pd.to_datetime(rcvs_tables[tbl_name].columns[1:], format=f"%b%Y"))
            
        for df_name, df in rcvs_tables.items():
            rcvs_tables[df_name] = loader.clean_count_columns(df=df)

        # Clean data - RCOS
        rcos_tables["Ethnicity AES.csv"] = loader.rename_column(df=rcos_tables["Ethnicity AES.csv"], 
                                                                                cols=[f"% of Total Proceedings along Ethnic Group", "Proceedings"],
                                                                                names=["Percentage of Proceedings", "Count"])
        for tbl_name in ["TableA.csv", "TableB.csv"]:
            rcos_tables[tbl_name] = rcos_tables[tbl_name].replace("Sept", "Sep")
            rcos_tables[tbl_name].iloc[0] = rcos_tables[tbl_name].iloc[2].astype(str).str.cat(rcos_tables[tbl_name].iloc[0].astype(str), sep='')
            rcos_tables[tbl_name] = loader.promote_first_row_to_header(df=rcos_tables[tbl_name])
            rcos_tables[tbl_name] = loader.rename_column(df=rcos_tables[tbl_name], 
                                                                cols=[np.nan],
                                                                names=["Police District"])
            rcos_tables[tbl_name] = rcos_tables[tbl_name].iloc[2:].reset_index(drop=True)
            rcos_tables[tbl_name] = rcos_tables[tbl_name].drop(columns="TotalTotal")
            rcos_tables[tbl_name].columns = list(rcos_tables[tbl_name].columns[:1]) + list(pd.to_datetime(rcos_tables[tbl_name].columns[1:], format=f"%b%Y"))

        for df_name, df in rcos_tables.items():
            rcos_tables[df_name] = loader.clean_count_columns(df=df)

        # Clean data - Activity and Report
        activity_tables["Occ Type.csv"] = loader.promote_first_row_to_header(df=activity_tables["Occ Type.csv"])
        activity_tables["Occ Type.csv"] = activity_tables["Occ Type.csv"].fillna(0) 
        activity_tables["TableB.csv"] = loader.promote_first_row_to_header(df=activity_tables["TableB.csv"])
        activity_tables["TableB.csv"].columns = activity_tables["TableB.csv"].columns.map(lambda x: re.sub(r'\bSept(\d{4})\b', r'Sep\1', x))
        activity_tables["TableB.csv"].columns = list(activity_tables["TableB.csv"].columns[:2]) + list(pd.to_datetime(activity_tables["TableB.csv"].columns[2:], format=f"%b%Y"))
        activity_tables["TableB.csv"] = activity_tables["TableB.csv"].fillna(0)
        
        for df_name, df in activity_tables.items():
            activity_tables[df_name] = loader.clean_count_columns(df=df)

        data_cleaned = True
    except Exception as e:
        st.error(f"Could not clean data: {e}")
        data_cleaned = False

    return data_cleaned, rcvs_tables, rcos_tables, activity_tables


# - App layout -
st.title("NZ Crime Rate Forecasting")
st.caption("""Crime Rate Forecasting for New Zealand 
    (dataset taken from NZ Police Data-https://www.police.govt.nz/about-us/publications-statistics/data-and-statistics/policedatanz)""")
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


    #TODO: Selection of Dataset -> Build Lag Features
    st.header("Data Selection")
    data_sel, data_desc= st.columns([2, 2])
    
    DATA_INFO = {
        "RCVS - District": {
            "desc": "Recorded Crime Victims Statistics - by Police Boundary",
            "dataset": rcvs_tables["TableA.csv"],
        },
        "RCVS - Anzsoc": {
                    "desc": "Recorded Crime Victims Statistics - by Crime Type",
                    "dataset": rcvs_tables["TableB.csv"],
                },
        "RCOS - District": {
            "desc": "Recorded Crime Offenders Statistics - by Police Boundary",
            "dataset": rcos_tables["TableA.csv"],
        },
        "RCOS - Anzsoc": {
                    "desc": "Recorded Crime Offenders Statistics - by Crime Type",
                    "dataset": rcos_tables["TableB.csv"],
                },
    }

    if not "data_chosen" in st.session_state:
        st.session_state["data_chosen"] = "RCVS - Anzsoc"

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
            data_choice, id_col=DATA_INFO[data_choice]['dataset'].columns[0]
        )


    #TODO: Selection of Model and Division -> Forecast
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
        "Histogram Gradient Boosting Regressor": {
            "desc": "Prediction using Histogram Gradient Boosting Regressor.",
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
            "Choose a division",
            DISTRICTS_SEL,
            format_func=lambda x: "New Zealand (Nation)" if x == "Total" else x,
            index=0,
        )
    with select_button:
        if st.button("Select"):
            st.session_state["forecast_result"] = model_selection(
                built_table["long_df"],
                chosen,
                chosen2
            )


    #TODO: Output of Results and Visualization
    if "forecast_result" in st.session_state:
        forecast_result = st.session_state["forecast_result"]

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", chosen)
        m2.metric("RMSE", f"{forecast_result['metrics_holdout']['rmse']:,.2f}")
        m3.metric("Mean Absolute Error", f"{forecast_result['metrics_holdout']['mae']:.2f}")
        m4.metric("Mean Absolute Percentage Error", f"{forecast_result['metrics_holdout']['mape']:,.2f}")

        st.divider()
        
        st.header("Prediction Output")
        tab1, = st.tabs(["Predicted and Actual Over Time"])
    
        with tab1:
            y_actual = built_table["rate_per_capita"].loc[built_table["rate_per_capita"]["district"] == chosen2].copy() 
            y_pred = forecast_result["forecast"].copy()

            # Convert Period -> Timestamp and Count -> Forecast_count for seamless concat
            y_actual["date"] = y_actual["period"].dt.to_timestamp()
            y_actual["forecast_count"] = y_actual["count"]
            last_actual = y_actual.iloc[-1]

            forecast_plot = pd.concat([
                y_actual[["date", "forecast_count"]],
                y_pred[["date", "forecast_count"]]
            ], join="outer", ignore_index=True)

            fig = go.Figure()
            # Actual and predicted
            fig.add_trace(
                go.Scatter(
                    x=forecast_plot["date"],
                    y=forecast_plot["forecast_count"],
                    mode="lines",
                    name="Predicted",
                    line=dict(color="#F83003", width=2),
                )
            )

            # Add vertical line to differentiate actual and predicted
            fig.add_vline(
                x=last_actual["date"],
                line_dash="dash",
                line_color="white",
            )

            fig.add_vrect(
                x0=last_actual["date"],
                x1=forecast_plot["date"].max(),
                opacity=0.2,
                fillcolor="lightsalmon",
                line_width=0,
                layer="below",
            )

            fig.update_layout(
                title=f"Predicted vs Actual Demand Over Time — {chosen}",
                xaxis_title="Date",
                yaxis_title="Count",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="#e0e0e0",
                legend=dict(bgcolor="#161b22", bordercolor="#30363d")
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                forecast_plot,
                use_container_width=True,
            )


    #TODO: Exploratory Data Analysis
    boundary_district = boundary_loader.load_districts()
    boundary_district = boundary_district.rename(columns={"DISTRICT_N": "district"})
    boundary_district = boundary_district.replace("Bay of Plenty", "Bay Of Plenty") 
    if "act_tableB" not in st.session_state:
        x = run_forecasting(activity_tables["TableB.csv"], id_col=("Occurrence Type Category", "Occurrence Division"))
        st.session_state["act_tableB"] = x["long_df"]
    if data_choice != st.session_state["data_chosen"]:
        st.session_state["data_chosen"] = data_choice
        st.session_state.pop("fig1", None)
        st.session_state.pop("fig2", None)
        st.session_state.pop("fig3", None)
        st.session_state.pop("fig4", None)
        if data_choice == "RCVS - District":
            tabA, tabB = st.tabs(["Rate Matrix", "Rate Choropleth"])
            st.session_state["fig1"] = plot_rate_matrix(built_table["rate_per_capita"])
            st.session_state["fig2"] = plot_rate_choropleth(boundary_district, built_table["rate_per_capita"])
        elif data_choice == "RCOS - District":
            tabA, tabB = st.tabs(["Rate Matrix", "Rate Choropleth"])
            st.session_state["fig1"] = plot_rate_matrix(built_table["rate_per_capita"], title="Offender rate per 10,000 people — by district")
            st.session_state["fig2"] = plot_rate_choropleth(boundary_district, built_table["rate_per_capita"], title="Offender rate per 10,000 people — by district")
        elif data_choice == "RCVS - Anzsoc":
            tabA, tabB, tabC = st.tabs(["Bar Chart", "Rate Matrix", "Area Chart"])
            st.session_state["fig1"] = plot_anzsoc_bar(built_table["rate_per_capita"])
            st.session_state["fig3"] = plot_rate_matrix(built_table["rate_per_capita"], value_col="count", title="Crime classification - victim", ytitle="Type of Crime")        
            st.session_state["fig4"] = plot_anzsoc_area(built_table["rate_per_capita"])    
        elif data_choice == "RCOS - Anzsoc":
            tabA, tabB, tabC = st.tabs(["Bar Chart", "Rate Matrix", "Area Chart"])
            st.session_state["fig1"] = plot_anzsoc_bar(built_table["rate_per_capita"], title="Crime classification - offender")
            st.session_state["fig3"] = plot_rate_matrix(built_table["rate_per_capita"], value_col="count", title="Crime classification - offender", ytitle="Type of Crime")        
            st.session_state["fig4"] = plot_anzsoc_area(built_table["rate_per_capita"], title="Crime classification - offender")     
    
    if "fig1" in st.session_state:
        with tabA:
            st.plotly_chart(st.session_state["fig1"], use_container_width=True)
    if "fig2" in st.session_state:
        with tabB:
            st.pyplot(st.session_state["fig2"], use_container_width=True, clear_figure=True)
    if "fig3" in st.session_state:
        with tabB:
            st.plotly_chart(st.session_state["fig3"], use_container_width=True)
    if "fig4" in st.session_state:
        with tabC:
            st.plotly_chart(st.session_state["fig4"], use_container_width=True)

    # Data and statistics toggle
    columns_to_keep = ['district', 'count', 'period', 'year', 'population', 'rate_per_capita']
    built_table["short_df"] = built_table["rate_per_capita"][columns_to_keep]
    with st.expander("Data Viewer"):
        st.dataframe(rcvs_tables["TableB.csv"], use_container_width=True, height=420)
    with st.expander("Summary Statistics"):
        st.dataframe(built_table["short_df"].describe().T, use_container_width=True)

    st.divider()

    st.header("Exploratory Data")
    fig1 = plot_stacked_bar(activity_tables["Occ Type.csv"])
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = plot_anzsoc_treemap(st.session_state["act_tableB"])
    st.plotly_chart(fig2, use_container_width=True)

