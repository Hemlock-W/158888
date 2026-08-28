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
# --------------------------------------------------------------------------
# Dark Theme CSS
# --------------------------------------------------------------------------
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
            ("svr", SVR(kernel="rbf", C=100, epsilon=0.1))
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
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42))
        ])
        forecast_result = sklearn_forecast(long_df=long_df, district=district, model=model, tune=False)
    elif model_name == "Histogram Gradient Boosting Regressor":
        model = Pipeline([
            ("scalar", StandardScaler()),
            ("hgbr", HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200, max_depth=30))
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
        # --------------------------------------------------------------------------
        # Clean Data RCVS
        # --------------------------------------------------------------------------
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

        # --------------------------------------------------------------------------
        # Clean Data RCOS 
        # --------------------------------------------------------------------------
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

        # --------------------------------------------------------------------------
        # Clean Data Activity and Report
        # --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# App Layout
# --------------------------------------------------------------------------
st.title("NZ Crime Rate Forecasting")
st.caption("""Crime Rate Forecasting for New Zealand 
    (dataset taken from NZ Police Data-https://www.police.govt.nz/about-us/publications-statistics/data-and-statistics/policedatanz)""")
st.divider()

if "data_ok" not in st.session_state:
    st.session_state["data_ok"] = False

# --------------------------------------------------------------------------
# Load Data
# --------------------------------------------------------------------------
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


    # --------------------------------------------------------------------------
    # 1. Selection of Dataset -> Build Lag Features
    # --------------------------------------------------------------------------
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
        st.session_state["data_chosen"] = data_choice
        st.session_state["build_features"] = forecast_dataset(
            data_choice, id_col=DATA_INFO[data_choice]['dataset'].columns[0]
        )


    # --------------------------------------------------------------------------
    # 2. Selection of Model and Division -> Model Forecast
    # --------------------------------------------------------------------------
    st.header("Model and District Selection")
    col_sel, col_desc, district_sel, select_button = st.columns([2, 2, 2, 1])
    MODEL_INFO = {
        "Gradient Boosting": {
            "desc": "Builds an ensemble of decision trees sequentially, where each new tree corrects the errors of previous trees.",
        },

        "Linear Regression": {
            "desc": "Models a linear relationship between features and the target using a best-fit straight line.",
        },

        "SVR": {
            "desc": "Uses support vectors and kernel functions to find a hyperplane that best fits the data within a specified error margin.",
        },

        "KNN (k=10, scaled)": {
            "desc": "Predicts values based on the average of the 10 nearest neighbouring observations in the feature space.",
        },

        "Random Forest Regressor": {
            "desc": "Combines predictions from many decision trees built on random subsets of data to improve accuracy and reduce overfitting.",
        },

        "Histogram Gradient Boosting Regressor": {
            "desc": "A faster gradient boosting method that groups feature values into histograms before building trees, making it efficient for large datasets.",
        },

        "Prophet": {
            "desc": "A decomposable time-series model that captures trend, seasonality, and holiday effects automatically.",
        },

        "Statsmodel (ETS)": {
            "desc": "Uses Error, Trend, and Seasonal components to model and forecast time-series patterns through exponential smoothing.",
        },

        "Statsmodel (Sarima)": {
            "desc": "Models autoregressive, differencing, moving average, and seasonal patterns to forecast time-series data.",
        },
    }

    if "build_features" in st.session_state:
        built_table =  st.session_state["build_features"]

    DISTRICTS_SEL = built_table["rate_per_capita"]["district"].unique()

    if not "chosen2" in st.session_state:
        st.session_state["chosen2"] = DISTRICTS_SEL[0]

    with col_sel:
        chosen = st.selectbox(
            "Choose a prediction model",
            list(MODEL_INFO.keys()),
            index=8,
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
            st.session_state["chosen2"] = chosen2
            st.session_state["data_chosen"] = data_choice
            st.session_state["forecast_result"] = model_selection(
                built_table["long_df"],
                chosen,
                chosen2
            )


    # --------------------------------------------------------------------------
    # 3. Output of Results and Visualization 
    # --------------------------------------------------------------------------
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
        tab1, tab2 = st.tabs(["Predicted and Actual Over Time", "Actual vs Predicted"])
    
        with tab1:
            y_actual = built_table["rate_per_capita"].loc[built_table["rate_per_capita"]["district"] == st.session_state["chosen2"]].copy() 
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
        with tab2:
            hist = forecast_result["holdout"]
            lo, hi = min(hist["actual"].min(), hist["predicted"].min()), max(hist["actual"].max(), hist["predicted"].max())

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=hist["actual"], y=hist["predicted"],
                mode="markers", name="Prediction",
                marker=dict(color="#166b4d", opacity=0.65, size=6),
            ))
            fig1.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi],
                mode="lines", name="Perfect Prediction",
                line=dict(color="#f78166", width=2, dash="dash"),
            ))
        
            fig1.update_layout(
                title=f"{ 'New Zealand (Nation)' if st.session_state['chosen2'] == 'Total' else st.session_state['chosen2']}: Actual vs Predicted (no future forecast)",
                xaxis_title="Actual",
                yaxis_title="Predicted",
                plot_bgcolor="#dee7f9",
                paper_bgcolor="#0e1117",
                font_color="#ebe9e9",
                legend=dict(bgcolor="#566b88", bordercolor="#30363d"),
            )
            st.plotly_chart(fig1, use_container_width=True)

    # --------------------------------------------------------------------------
    # 4. Exploratory Data Analysis 
    # --------------------------------------------------------------------------
    boundary_district = boundary_loader.load_districts()
    boundary_district = boundary_district.rename(columns={"DISTRICT_N": "district"})
    boundary_district = boundary_district.replace("Bay of Plenty", "Bay Of Plenty") 
    if "act_tableB" not in st.session_state:
        x = run_forecasting(activity_tables["TableB.csv"], id_col=("Occurrence Type Category", "Occurrence Division"))
        st.session_state["act_tableB"] = x["long_df"]

    if st.session_state["data_chosen"] == "RCVS - District":
        tabA, tabB = st.tabs(["Rate Matrix", "Rate Choropleth"])
        with tabA:
            st.plotly_chart(plot_rate_matrix(built_table["rate_per_capita"]), use_container_width=True)
        with tabB:
            st.pyplot(plot_rate_choropleth(boundary_district, built_table["rate_per_capita"]), use_container_width=True)
    elif st.session_state["data_chosen"] == "RCOS - District":
        tabA, tabB = st.tabs(["Rate Matrix", "Rate Choropleth"])
        with tabA:
            st.plotly_chart(plot_rate_matrix(built_table["rate_per_capita"], title="Offender rate per 10,000 people — by district"), use_container_width=True)
        with tabB:
            st.pyplot(plot_rate_choropleth(boundary_district, built_table["rate_per_capita"], title="Offender rate per 10,000 people — by district"), use_container_width=True)
    elif st.session_state["data_chosen"] == "RCVS - Anzsoc":
        tabA, tabB, tabC = st.tabs(["Bar Chart", "Rate Matrix", "Area Chart"])
        with tabA:
            st.plotly_chart(plot_anzsoc_bar(built_table["rate_per_capita"]), use_container_width=True)
        with tabB:
            st.plotly_chart(plot_rate_matrix(built_table["rate_per_capita"], value_col="count", title="Crime classification - victim", ytitle="Type of Crime"), use_container_width=True)       
        with tabC:
            st.plotly_chart(plot_anzsoc_area(built_table["rate_per_capita"]), use_container_width=True)
    elif st.session_state["data_chosen"] == "RCOS - Anzsoc":
        tabA, tabB, tabC = st.tabs(["Bar Chart", "Rate Matrix", "Area Chart"])
        with tabA:
            st.plotly_chart(plot_anzsoc_bar(built_table["rate_per_capita"], title="Crime classification - offender"), use_container_width=True)
        with tabB:
            st.plotly_chart(plot_rate_matrix(built_table["rate_per_capita"], value_col="count", title="Crime classification - offender", ytitle="Type of Crime"), use_container_width=True)   
        with tabC:
            st.plotly_chart(plot_anzsoc_area(built_table["rate_per_capita"], title="Crime classification - offender"), use_container_width=True)   
    

    # Data and statistics toggle
    columns_to_keep = ['district', 'count', 'period', 'year', 'population', 'rate_per_capita']
    built_table["short_df"] = built_table["rate_per_capita"][columns_to_keep]
    with st.expander(f"Data Viewer ({st.session_state['data_chosen']})"):
        st.dataframe(rcvs_tables["TableB.csv"], use_container_width=True, height=420)
    with st.expander("Summary Statistics"):
        st.dataframe(built_table["short_df"].describe().T, use_container_width=True)

    st.divider()

    st.header("Exploratory Data")
    fig_1 = plot_stacked_bar(activity_tables["Occ Type.csv"])
    st.plotly_chart(fig_1, use_container_width=True)

    fig_2 = plot_anzsoc_treemap(st.session_state["act_tableB"])
    st.plotly_chart(fig_2, use_container_width=True)


