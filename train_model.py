import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


@dataclass
class ModelResults:
    name: str
    mae: float
    rmse: float
    r2: float


def download_data(start_date: str = "2017-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    df = yf.download("BTC-USD", start=start_date, end=end_date, auto_adjust=False)
    if df.empty:
        raise ValueError("No data downloaded. Please check your internet connection.")
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Return_1d"] = data["Close"].pct_change()
    data["MA_7"] = data["Close"].rolling(7).mean()
    data["MA_30"] = data["Close"].rolling(30).mean()
    data["Volatility_7"] = data["Return_1d"].rolling(7).std()
    data["Lag_1"] = data["Close"].shift(1)
    data["Lag_2"] = data["Close"].shift(2)
    data["Lag_3"] = data["Close"].shift(3)
    data["Target"] = data["Close"].shift(-1)  # next-day close
    data = data.dropna().reset_index(drop=True)
    return data


def evaluate(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> ModelResults:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return ModelResults(model_name, mae, rmse, r2)


def save_visualizations(df: pd.DataFrame, out_dir: str = "outputs") -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)
    sns.set_style("darkgrid")

    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"], color="royalblue")
    plt.title("Bitcoin Close Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/price_trend.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(df["Close"], bins=50, kde=True, color="purple")
    plt.title("Distribution of Bitcoin Close Price")
    plt.xlabel("Close Price (USD)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/price_distribution.png")
    plt.close()


def main() -> None:
    raw_df = download_data()
    save_visualizations(raw_df)
    data = create_features(raw_df)

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Return_1d",
        "MA_7",
        "MA_30",
        "Volatility_7",
        "Lag_1",
        "Lag_2",
        "Lag_3",
    ]
    X = data[feature_cols]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    lr_results = evaluate(y_test, lr_pred, "Linear Regression")
    xgb_results = evaluate(y_test, xgb_pred, "XGBoost Regressor")

    results_df = pd.DataFrame(
        [
            {"Model": lr_results.name, "MAE": lr_results.mae, "RMSE": lr_results.rmse, "R2": lr_results.r2},
            {"Model": xgb_results.name, "MAE": xgb_results.mae, "RMSE": xgb_results.rmse, "R2": xgb_results.r2},
        ]
    )

    print("\nModel Performance:")
    print(results_df.to_string(index=False))

    predictions_df = pd.DataFrame(
        {
            "Date": data["Date"].iloc[X_test.index],
            "Actual_Next_Close": y_test.values,
            "LR_Prediction": lr_pred,
            "XGB_Prediction": xgb_pred,
        }
    ).reset_index(drop=True)

    predictions_df.to_csv("outputs/predictions.csv", index=False)
    results_df.to_csv("outputs/model_metrics.csv", index=False)
    print("\nSaved outputs to outputs/ folder.")


if __name__ == "__main__":
    main()