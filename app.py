import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf
from scipy import stats
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


@st.cache_data
def load_data(start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download("BTC-USD", start=start_date, end=end_date, auto_adjust=False).reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.drop_duplicates(subset=["Date"]).dropna().sort_values("Date").reset_index(drop=True)
    return data


def wrangle_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Month_Name"] = data["Date"].dt.strftime("%b")
    data["Return_1d"] = data["Close"].pct_change()
    data["Price_Range"] = data["High"] - data["Low"]
    return data


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Return_1d"] = data["Close"].pct_change()
    data["MA_7"] = data["Close"].rolling(7).mean()
    data["MA_30"] = data["Close"].rolling(30).mean()
    data["Volatility_7"] = data["Return_1d"].rolling(7).std()
    data["Lag_1"] = data["Close"].shift(1)
    data["Lag_2"] = data["Close"].shift(2)
    data["Lag_3"] = data["Close"].shift(3)
    data["Target"] = data["Close"].shift(-1)
    return data.dropna().reset_index(drop=True)


def run_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    returns = df["Return_1d"].dropna()
    median_return = returns.median()
    high_group = returns[returns >= median_return]
    low_group = returns[returns < median_return]

    if len(high_group) > 1 and len(low_group) > 1:
        t_stat, t_pvalue = stats.ttest_ind(high_group, low_group, equal_var=False)
    else:
        t_stat, t_pvalue = float("nan"), float("nan")

    if len(returns) > 3:
        shapiro_stat, shapiro_pvalue = stats.shapiro(returns.sample(min(5000, len(returns)), random_state=42))
    else:
        shapiro_stat, shapiro_pvalue = float("nan"), float("nan")

    corr_coef, corr_pvalue = stats.pearsonr(df["Volume"], df["Close"])

    return pd.DataFrame(
        [
            {
                "Hypothesis": "Mean returns differ between higher-return and lower-return groups",
                "Test": "Welch t-test",
                "Statistic": t_stat,
                "P_Value": t_pvalue,
            },
            {
                "Hypothesis": "Daily returns are normally distributed",
                "Test": "Shapiro-Wilk",
                "Statistic": shapiro_stat,
                "P_Value": shapiro_pvalue,
            },
            {
                "Hypothesis": "Volume and close price have linear correlation",
                "Test": "Pearson correlation",
                "Statistic": corr_coef,
                "P_Value": corr_pvalue,
            },
        ]
    )


st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")
st.title("Bitcoin Price Prediction (Simple Project)")
st.caption("Dataset Source: Yahoo Finance (`BTC-USD`) via yfinance.")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", pd.to_datetime("2019-01-01"))
with col2:
    end_date = st.date_input("End Date", pd.to_datetime("2025-12-31"))

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

df = load_data(str(start_date), str(end_date))
if df.empty:
    st.error("No data available for the selected date range.")
    st.stop()

st.subheader("Raw Data Preview")
st.dataframe(df.tail(10), use_container_width=True)

clean_df = clean_data(df)
wrangled_df = wrangle_data(clean_df)

st.subheader("Data Cleaning and Wrangling")
clean_col1, clean_col2, clean_col3 = st.columns(3)
clean_col1.metric("Rows (Raw)", len(df))
clean_col2.metric("Rows (Cleaned)", len(clean_df))
clean_col3.metric("Missing Values (Cleaned)", int(clean_df.isna().sum().sum()))
st.dataframe(wrangled_df.tail(10), use_container_width=True)

st.subheader("Bitcoin Close Price Trend")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(wrangled_df["Date"], wrangled_df["Close"], color="royalblue")
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price (USD)")
st.pyplot(fig1)

st.subheader("Close Price Distribution")
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.histplot(wrangled_df["Close"], bins=40, kde=True, ax=ax2, color="purple")
ax2.set_xlabel("Close Price (USD)")
st.pyplot(fig2)

st.subheader("Correlation Heatmap")
fig_heat, ax_heat = plt.subplots(figsize=(7, 4))
corr_df = wrangled_df[["Open", "High", "Low", "Close", "Volume"]].corr()
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_heat)
st.pyplot(fig_heat)

st.subheader("Monthly Average Close Price")
monthly_avg = (
    wrangled_df.assign(MonthDate=wrangled_df["Date"].dt.to_period("M").dt.to_timestamp())
    .groupby("MonthDate", as_index=False)["Close"]
    .mean()
)
fig_month, ax_month = plt.subplots(figsize=(10, 4))
ax_month.plot(monthly_avg["MonthDate"], monthly_avg["Close"], color="darkorange")
ax_month.set_xlabel("Month")
ax_month.set_ylabel("Avg Close Price (USD)")
st.pyplot(fig_month)

st.subheader("Hypothesis Testing")
hypothesis_df = run_hypothesis_tests(wrangled_df)
st.dataframe(hypothesis_df, use_container_width=True)

data = create_features(wrangled_df)
features = [
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

X = data[features]
y = data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

result_df = pd.DataFrame(
    {
        "Date": data["Date"].iloc[X_test.index].values,
        "Actual_Next_Close": y_test.values,
        "Predicted_Next_Close": pred,
    }
)

st.subheader("Prediction Results (Last 20 Rows)")
st.dataframe(result_df.tail(20), use_container_width=True)

st.subheader("Actual vs Predicted Next-Day Close")
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(result_df["Date"], result_df["Actual_Next_Close"], label="Actual", color="green")
ax3.plot(result_df["Date"], result_df["Predicted_Next_Close"], label="Predicted", color="orange")
ax3.legend()
ax3.set_xlabel("Date")
ax3.set_ylabel("Price (USD)")
st.pyplot(fig3)

next_day_prediction = model.predict(X.tail(1))[0]
st.metric("Predicted Next-Day Close (USD)", f"{next_day_prediction:,.2f}")