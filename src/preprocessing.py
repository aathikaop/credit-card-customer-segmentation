# Import Libraries

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


# Drop unwanted columns

drop_cols = [
    "CUST_ID"
]

class ColumnDropper(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        return X.drop(
            columns=self.columns,
            errors="ignore"
        )


# Impute missing values with median

impute_cols = [
    "MINIMUM_PAYMENTS",
    "CREDIT_LIMIT"
]

class MedianImputer(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):

        self.medians_ = {
            col: X[col].median()
            for col in self.columns
        }

        return self

    def transform(self, X):

        X = X.copy()

        for col, median in self.medians_.items():
            X[col] = X[col].fillna(median)

        return X


# Bounded columns

skip_outlier_cols = [
    "BALANCE_FREQUENCY",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "PRC_FULL_PAYMENT",
    "TENURE"
]

class OutlierCapper(BaseEstimator, TransformerMixin):

    def __init__(self, skip_columns):
        self.skip_columns = skip_columns

    def fit(self, X, y=None):

        self.bounds_ = {}

        for col in X.select_dtypes(include=np.number).columns:

            if col in self.skip_columns:
                continue

            Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            IQR = Q3 - Q1

            self.bounds_[col] = (
                Q1 - 1.5 * IQR,
                Q3 + 1.5 * IQR
            )

        return self

    def transform(self, X):

        X = X.copy()

        for col, (lower, upper) in self.bounds_.items():
            X[col] = X[col].clip(lower, upper)

        return X


# Heavily right-skewed columns

skewed_cols = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "CREDIT_LIMIT"
]

class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        for col in self.columns:
            X[col] = np.log1p(X[col].clip(lower=0))

        return X


# Numerical scaling Pipeline

numeric_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        )
    ])


# Create Preprocessing Pipeline

def create_preprocessing_pipeline():

    preprocessing_pipeline = Pipeline([
        (
            "drop_columns",
            ColumnDropper(drop_cols)
        ),
        (
            "imputer",
            MedianImputer(impute_cols)
        ),
        (
            "outlier_capper",
            OutlierCapper(skip_outlier_cols)
        ),
        (
            "log_transform",
            LogTransformer(skewed_cols)
        ),
        (
            "scaler",
            numeric_pipeline
        )
    ])

    return preprocessing_pipeline