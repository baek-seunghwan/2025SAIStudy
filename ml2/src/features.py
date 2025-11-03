from __future__ import annotations
import pandas as pd
import numpy as np

MONTH_MAP = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
DOW_MAP = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}

class FeatureBuilder:
    """Collect all feature engineering here so train/infer are consistent."""

    def __init__(self, fit_stats: dict | None = None):
        self.fit_stats = fit_stats or {}

    def _base(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # safe numeric conversions
        if 'month' in out.columns:
            out['month_num'] = out['month'].map(MONTH_MAP).fillna(0).astype(int)
        if 'claim_day_of_week' in out.columns:
            out['claim_day_of_week_num'] = out['claim_day_of_week'].map(DOW_MAP).fillna(0).astype(int)

        # common ratios/diffs if columns exist
        def safe_ratio(a, b):
            return np.where((b==0) | (b.isna()), 0, a/b)

        col_pairs = [
            ('claim_est_payout','annual_income','payout_income_ratio'),
            ('driver_age','vehicle_age','driver_vehicle_age_ratio'),
        ]
        for a,b,newc in col_pairs:
            if a in out.columns and b in out.columns:
                out[newc] = safe_ratio(out[a], out[b])

        if 'driver_age' in out.columns and 'vehicle_age' in out.columns:
            out['driver_vehicle_age_diff'] = out['driver_age'] - out['vehicle_age']

        if 'liab_prct' in out.columns and 'claim_est_payout' in out.columns:
            out['liab_payout'] = out['liab_prct'] * out['claim_est_payout']

        # fill remaining NA
        num_cols = out.select_dtypes(include=['number']).columns
        out[num_cols] = out[num_cols].fillna(0)
        obj_cols = out.select_dtypes(include=['object','category']).columns
        out[obj_cols] = out[obj_cols].fillna('Unknown')

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._base(df)
        # collect simple stats for future (optional)
        self.fit_stats['num_cols'] = out.select_dtypes(include=['number']).columns.tolist()
        self.fit_stats['cat_cols'] = out.select_dtypes(include=['object','category']).columns.tolist()
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._base(df)
