import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StellarFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering for the SDSS17 stellar classification dataset."""

    ID_COLS = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID',
               'spec_obj_ID', 'plate', 'MJD', 'fiber_ID']
    BANDS = ['u', 'g', 'r', 'i', 'z']

    RS_BINS = [-np.inf, 0.005, 0.1, 0.5, 1.5, np.inf]
    RS_LABELS = ['Stellar', 'Local', 'Mid', 'High', 'VeryHigh']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_eng = X.copy()

        X_eng['u_g'] = X_eng['u'] - X_eng['g']
        X_eng['g_r'] = X_eng['g'] - X_eng['r']
        X_eng['r_i'] = X_eng['r'] - X_eng['i']
        X_eng['i_z'] = X_eng['i'] - X_eng['z']
        X_eng['u_r'] = X_eng['u'] - X_eng['r']

        X_eng['mean_mag'] = X_eng[self.BANDS].mean(axis=1)
        X_eng['mag_range'] = X_eng[self.BANDS].max(axis=1) - X_eng[self.BANDS].min(axis=1)

        X_eng['redshift_abs'] = X_eng['redshift'].abs()
        X_eng['log_redshift'] = np.log1p(X_eng['redshift_abs'])
        X_eng['IsNearZeroRedshift'] = (X_eng['redshift_abs'] < 0.005).astype(int)
        X_eng['RedshiftGroup'] = pd.cut(
            X_eng['redshift'], bins=self.RS_BINS, labels=self.RS_LABELS
        ).astype(str)

        X_eng = X_eng.drop(columns=[c for c in self.ID_COLS if c in X_eng.columns])
        return X_eng
