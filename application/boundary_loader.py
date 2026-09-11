"""
Aggregates meshblock population  to NZ Police district
level to be converted into rates per capita.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import numpy as np

def _norm(s: str) -> str:
    return str(s).strip().casefold()

class BoundaryDataLoader:
    def __init__(
        self,
        meshblock_shp:str,
        district_shp:str,
        population_col              = ("GENERAL_EL", "MAORI_ELEC"),
        district_name_col:str       = "DISTRICT_N",
        missing_value_sentinel:int  = -999,
    ):
        self.meshblock_shp          = meshblock_shp
        self.district_shp           = district_shp
        self.population_col         = list(population_col)
        self.district_name_col      = district_name_col
        self.missing_value_sentinel = missing_value_sentinel

    # ------------------------------------------------------------------
    # Loading + inspection
    # ------------------------------------------------------------------
    def load_meshblocks(self) -> gpd.GeoDataFrame:
        return gpd.read_file(self.meshblock_shp)

    def load_districts(self) -> gpd.GeoDataFrame:
        return gpd.read_file(self.district_shp)

    def drop_geometry(self, gdf:gpd.GeoDataFrame) -> pd.DataFrame:
        gdf_df = pd.DataFrame(gdf)
        gdf_df.drop(columns=["geometry"], inplace=True)
        return gdf_df

    def inspect_columns(self) -> dict:
        mb = self.load_meshblocks()
        dist = self.load_districts()

        return {
            "meshblock_columns": list(mb.columns),
            "meshblock_crs": str(mb.crs),
            "meshblock_sample": self.drop_geometry(mb),
            "district_columns": list(dist.columns),
            "district_crs": str(dist.crs),
            "district_sample": self.drop_geometry(dist),
        }


    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def population_by_district(self) -> pd.DataFrame:
        """
        Returns DataFrame: one row per district, with total
        population summed from meshblocks assigned to it.
        """
        meshblocks = self.load_meshblocks()
        districts = self.load_districts()

        if meshblocks.crs != districts.crs:
            meshblocks = meshblocks.to_crs(districts.crs)

        for col in self.population_col:
            meshblocks[col] = meshblocks[col].replace(self.missing_value_sentinel, 0)

        mb_points = meshblocks.copy()
        # Use centroid from geometry
        # Avoid duplication of area for intersect places
        mb_points["geometry"] = mb_points.geometry.centroid
        joined = gpd.sjoin(
            mb_points,
            districts[[self.district_name_col, "geometry"]],
            how="left",
            predicate="within",
        )
        pop_by_district = (
            joined.groupby(self.district_name_col)[self.population_col]
            .sum()
            .reset_index()
        )

        pop_by_district["population"] = pop_by_district[self.population_col].sum(axis=1)
        pop_by_district.rename(columns={self.district_name_col: "district"}, inplace=True)
        return pop_by_district

        
    def missing_population(
        self, long_df:pd.DataFrame, pop_df:pd.DataFrame,
        drop_labels = ("Not Specified",), total_label:str = "Total",
    ) -> pd.DataFrame:
        long_df = long_df.copy()
        pop_df = pop_df.copy()

        # Drop not specified districts
        drop_keys = {_norm(d) for d in drop_labels}
        drop_mask = long_df["district"].apply(_norm).isin(drop_keys)
        n_dropped = int(drop_mask.sum())
        if n_dropped:
            print(f"Dropping {n_dropped} row(s) with district in {list(drop_labels)}")
        long_df = long_df.loc[~drop_mask].copy()

        # Match case
        long_df["_key"] = long_df["district"].apply(_norm)
        pop_df["_key"] = pop_df["district"].apply(_norm)

        # Sum population for total
        total_key = _norm(total_label)
        total_population = pop_df["population"].mean()
    
        pop_df = pd.concat(
            [pop_df, pd.DataFrame([{"district": total_label, "population": total_population, "_key": total_key}])],
            ignore_index=True,
        )
    
        # Merge on the normalized key, keep the original display name
        merged = long_df.merge(pop_df[["_key", "population"]], on="_key", how="left")
        merged = merged.drop(columns=["_key"])
 
        still_missing = merged.loc[merged["population"].isna(), "district"].unique()
        if len(still_missing):
            print(f"Warning: still no population match for: {list(still_missing)}")
 
        return merged

    
    # ------------------------------------------------------------------
    # Merge with long_df
    # ------------------------------------------------------------------
    def merge_with_crime_data(
        self, long_df:pd.DataFrame, per:int = 10000
    ) -> pd.DataFrame:
        pop_df = self.population_by_district()
        merged = long_df.merge(pop_df, on="district", how="left")

        missing = merged.loc[merged["population"].isna(), "district"].unique()
        if len(missing):
            print(f"Warning: no population match for districts: {list(missing)}")

        merged = self.missing_population(long_df=long_df, pop_df=pop_df)
        merged["rate_per_capita"] = np.round((merged["count"] / merged["population"]) * per, 3)

        return merged
