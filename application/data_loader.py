"""
Load all datasets (RCVS, RCOS and Activity from specific path)
Dataset cleaning or reformatting
"""

import pandas as pd
import streamlit as st
#  import dateparser
import re

_COMMA_NUMBER_RE = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$")

def _comma_number(value) -> bool:
    if pd.isna(value):
        return True  # missing values don't disqualify the column
    return bool(_COMMA_NUMBER_RE.match(str(value).strip()))


def _clean_count_column(series:pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": None, "nan": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("Int64")

class DataLoader:
    def __init__(
        self,
        base_dir:str        = "../dataset",
        rcvs_subdir:str     = "RCVS",
        rcos_subdir:str     = "RCOS",
        activity_subdir:str = "Activity",
        rcvs_filenames      = None,
        rcos_filenames      = None,
        activity_filenames  = None,
    ):
        self.base_dir = base_dir
        self.rcvs_subdir = f"{base_dir}/{rcvs_subdir}"
        self.rcos_subdir = f"{base_dir}/{rcos_subdir}"
        self.activity_subdir = f"{base_dir}/{activity_subdir}"

        self.rcvs_filenames = rcvs_filenames or [
            "TableA.csv", "TableB.csv",
        ]
        self.rcos_filenames = rcos_filenames or [
            "Ethnicity AES.csv", "TableA.csv", "TableB.csv",
        ]
        self.activity_filenames = activity_filenames or [
            "Occ Type.csv", "TableB.csv",
        ]

    @st.cache_data(show_spinner="Fetching CSV files…")
    def _fetch_csvs(_self, directory:str, filenames:list[str]) -> dict[str, pd.DataFrame]:
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


    def rename_column(_self, df:pd.DataFrame, cols:list[str], names:list[str]) -> pd.DataFrame:
        if len(cols) != len(names):
            raise ValueError(
                f"cols and names must be the same length (got {len(cols)} and {len(names)})"
            )
        rename_map = dict(zip(cols, names))
        return df.rename(columns=rename_map)

    def clean_count_columns(_self, df:pd.DataFrame) -> pd.DataFrame:
        """Convert data like 10,000 into int"""
        for col in df.columns:
            # if df[col].dtype != object:
            #     continue
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            match_ratio = non_null.apply(_comma_number).mean()
            if match_ratio >= 0.9:
                df[col] = _clean_count_column(df[col]) 
        return df

    def promote_first_row_to_header(_self, df:pd.DataFrame) -> pd.DataFrame:
        """Use the values in row 0 as the new column names, then drop that row."""
        df = df.copy()
        df.columns = df.iloc[0].astype(str).str.strip()
        df = df.iloc[1:].reset_index(drop=True)
        df.columns.name = None
        return df

    def to_date(_self, df:pd.DataFrame, col, formatting = "%B %Y") -> pd.DataFrame:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = pd.to_datetime(df[col], format=formatting)
        return df

    def add_header_row(_self, df:pd.DataFrame, col) -> pd.DataFrame:
        """Move values from header to row 0, then add header row"""
        df.loc[-1] = df.columns
        df.index = df.index + 1
        df = df.sort_index()
        df.columns = col
        return df