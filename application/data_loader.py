import pandas as pd
import streamlit as st
import re

class DataLoader:
    def __init__(
        self,
        base_dir: str = "../dataset",
        rcvs_subdir: str = "RCVS",
        rcos_subdir: str = "RCOS",
        activity_subdir: str = "Activity",
        rcvs_filenames=None,
        rcos_filenames=None,
        activity_filenames=None,
    ):
        self.base_dir = base_dir
        self.rcvs_subdir = f"{base_dir}/{rcvs_subdir}"
        self.rcos_subdir = f"{base_dir}/{rcos_subdir}"
        self.activity_subdir = f"{base_dir}/{activity_subdir}"

        self.rcvs_filenames = rcvs_filenames or [
            "ANSOC Bar AEG.csv", "Boundary bar AEG.csv", "Map sheet.csv",
            "TableA.csv", "TableB.csv", "Trend AEG.csv",
        ]
        self.rcos_filenames = rcos_filenames or [
            "ANSOC Bar AEG.csv", "Boundary bar AEG.csv", "Age and Sex AES.csv",
            "Ethnicity AES.csv", "TableA.csv", "TableB.csv", "Trend AEG.csv",
        ]
        self.activity_filenames = activity_filenames or [
            "Boundary Districts.csv", "Occ Type.csv", "TableA.csv", "TableB.csv",
        ]

    @st.cache_data(show_spinner="Fetching CSV files…")
    def _fetch_csvs(_self, directory: str, filenames: list[str]) -> dict[str, pd.DataFrame]:
        """Read a list of CSVs from self.dir, keyed by filename."""
        return {
            fname: pd.read_csv(f"{directory}/{fname}", sep="\t", encoding="utf-16")
            for fname in filenames
        }

    def fetch_rcvs(self) -> dict[str, pd.DataFrame]:
        return self._fetch_csvs(self.rcvs_subdir, self.rcvs_filenames)

    def fetch_rcos(self) -> dict[str, pd.DataFrame]:
        return self._fetch_csvs(self.rcos_subdir, self.rcos_filenames)

    def fetch_activity(self) -> dict[str, pd.DataFrame]:
        return self._fetch_csvs(self.activity_subdir, self.activity_filenames)


    def rename_column(_self, df:pd.DataFrame, cols:list[str], names:list[str]):
        if len(cols) != len(names):
            raise ValueError(
                f"cols and names must be the same length (got {len(cols)} and {len(names)})"
            )
        rename_map = dict(zip(cols, names))
        return df.rename(columns=rename_map)

    def clean_count_column(_self, series: pd.Series) -> pd.Series:
        """Convert a 'Count' column of strings like '10,000' into integers.
        Uses pandas' nullable Int64 dtype so missing/blank values become <NA>
        instead of raising or forcing a fallback to float."""
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"": None, "nan": None, "None": None})
        )
        return pd.to_numeric(cleaned, errors="coerce").astype("Int64")
    def clean_count_columns(_self, df:pd.DataFrame):
        for col in df.columns:
            if col.strip().lower() == "count":
                df[col] = _self.clean_count_column(df[col])
        return df

    def promote_first_row_to_header(_self, df:pd.DataFrame) -> pd.DataFrame:
        """Use the values in row 0 as the new column names, then drop that row."""
        df = df.copy()
        df.columns = df.iloc[0].astype(str).str.strip()
        df = df.iloc[1:].reset_index(drop=True)
        df.columns.name = None  # cosmetic: drops the leftover index name pandas sometimes carries over
        return df

    def print_statement(_self, tables):
        for df_name, df in tables.items():
            st.markdown(df.describe())