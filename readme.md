# Bitcoin Price Prediction (Simple Python Project)

This is a beginner-friendly project for **Bitcoin next-day price prediction** using:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- Frontend: `streamlit`

## Dataset Source

This project uses historical Bitcoin data for `BTC-USD` from **Yahoo Finance** (downloaded using `yfinance`).

- Yahoo Finance BTC-USD page: https://finance.yahoo.com/quote/BTC-USD/history/
- Programmatic access via `yfinance`: https://github.com/ranaroussi/yfinance

## Project Files

- `train_model.py` - downloads data, creates features, trains models, and saves outputs.
- `app.py` - simple Streamlit frontend to visualize data and predictions.
- `requirements.txt` - dependencies.

## How to Run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train and generate output files

```bash
python train_model.py
```

This creates:

- `outputs/price_trend.png`
- `outputs/price_distribution.png`
- `outputs/model_metrics.csv`
- `outputs/predictions.csv`

### 3) Run the frontend

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Output
![Bitcoin Trend](outputs/image.png)

![Price Distribution](outputs/image-1.png)

## Notes

- This is a simple educational model, not financial advice.
- Crypto prices are volatile, so real-world prediction quality can vary significantly.