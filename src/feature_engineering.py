import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ONEOFF_RATIO"] = df["ONEOFF_PURCHASES"] / (df["PURCHASES"] + 1)
    df["LIMIT_USAGE"] = df["BALANCE"] / (df["CREDIT_LIMIT"] + 1)
    df["PAYMENT_RATIO"] = df["PAYMENTS"] / (df["MINIMUM_PAYMENTS"] + 1)
    df["MONTHLY_PURCHASE"] = df["PURCHASES"] / df["TENURE"]
    df["MONTHLY_CASH_ADVANCE"] = df["CASH_ADVANCE"] / df["TENURE"]

    return df