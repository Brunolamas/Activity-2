# app.py — Multi-Ticker Anomaly Dashboard (CSV/Excel + yfinance)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Market Anomaly Dashboard", layout="wide")

# ============================================================
#                       TITLE & DESCRIPTION
# ============================================================
st.title("📈 Multi-Ticker Market Anomaly Dashboard — Momentum (12–1 Approx.)")

st.markdown("""
### Team Members
Bruno Lombardo Lamas Alvarado A01644369  
Damián Urbieta Ramírez A01644801  
Armando Allende Sedano A0169476  
Darío Vázquez Romero A01644735  
José Alberto Alcaraz Baños A01067875
""")

st.write("""
**This dashboard implements a Momentum / Anomaly-based strategy** with:
- Data cleaning + feature construction (returns, indicators, anomaly signal)
- Anomaly-driven **BUY / SELL** signals
- MACD (line, signal, histogram)
- Choice of **chart type** and **indicators**
- A **basic backtest** versus buy-and-hold
""")

# ============================================================
#                         SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Settings")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="AAPL,MSFT"
)

uploaded_file = st.sidebar.file_uploader("Or upload a CSV file", type=["csv", "xlsx"])

start_date = st.sidebar.date_input("Start date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))

interval = st.sidebar.selectbox(
    "Interval",
    options=["1d", "1wk", "1mo"],
    index=0
)

chart_type = st.sidebar.selectbox(
    "Chart type",
    options=["Line", "Candlestick"],
    index=0
)

indicators_selected = st.sidebar.multiselect(
    "Indicators to show",
    options=["SMA (50)", "SMA (200)", "MACD", "RSI (14)", "Anomaly (Momentum Z-score)"],
    default=["SMA (50)", "MACD", "Anomaly (Momentum Z-score)"]
)

st.sidebar.markdown("---")
st.sidebar.write("The **first ticker** will be used for backtesting and detailed charts.")

# ============================================================
#                    DATA DOWNLOAD FUNCTION
# ============================================================
@st.cache_data
def load_data(ticker: str, start, end, interval="1d") -> pd.DataFrame:
    # Added progress=False to keep the terminal clean
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    
    if df.empty:
        return df

    # --- FIX STARTS HERE ---
    # If yfinance returns a MultiIndex (e.g., ('Close', 'AAPL')), flatten it to just 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # --- FIX ENDS HERE ---

    df = df.reset_index()

    # Normalize all naming cases
    cols = {c.lower(): c for c in df.columns}

    # Priority: adjclose > close > any OHLC available
    if "adj close" in df.columns:
        df["Price"] = df["Adj Close"]
    elif "close" in df.columns:
        df["Price"] = df["Close"]
    elif "adjclose" in cols:
        df["Price"] = df[cols["adjclose"]]
    elif "close" in cols:
        df["Price"] = df[cols["close"]]
    else:
        # Fail-safe: default to OHLC mean
        # Ensure we only select numeric columns for the mean calculation to avoid errors
        df["Price"] = df[["Open", "High", "Low", "Close"]].mean(axis=1)

    return df


# ============================================================
#     DATA CLEANING + FEATURE CONSTRUCTION (returns, indicators,
#                         anomaly signal)
# ============================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Basic cleaning
    df = df.sort_values("Date")
    df = df.dropna(subset=["Close"])
    df = df.reset_index(drop=True)

    # Returns
    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log1p(df["return"])

    # Momentum (12–1 approx) using trading days (21 ~ 1 month, 252 ~ 12 months)
    # Ret_12m ~ 252 days, Ret_1m ~ 21 days
    lookback_12m = 252
    lookback_1m = 21
    df["ret_12m"] = df["Close"].pct_change(lookback_12m)
    df["ret_1m"] = df["Close"].pct_change(lookback_1m)
    df["mom_12_1"] = df["ret_12m"] - df["ret_1m"]

    # Anomaly: Z-score of momentum
    mom_mean = df["mom_12_1"].rolling(252, min_periods=60).mean()
    mom_std = df["mom_12_1"].rolling(252, min_periods=60).std()
    df["mom_zscore"] = (df["mom_12_1"] - mom_mean) / mom_std

    # Define anomaly signal:
    # mom_zscore > 1.5  -> anomaly_up (potential BUY)
    # mom_zscore < -1.5 -> anomaly_down (potential SELL)
    df["anomaly"] = 0
    df.loc[df["mom_zscore"] > 1.5, "anomaly"] = 1
    df.loc[df["mom_zscore"] < -1.5, "anomaly"] = -1

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # RSI (14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Simple moving averages
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # Trading signals based on anomaly:
    # position = 1 when anomaly > 0, else 0 (flat)
    df["position"] = np.where(df["anomaly"] > 0, 1, 0)

    # Strategy returns: use previous day's position
    df["strategy_return"] = df["position"].shift(1).fillna(0) * df["return"]

    # Cumulative performance
    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_buy_hold"] = (1 + df["return"]).cumprod()

    return df

# ============================================================
#                     LOAD DATA
# ============================================================
data_dict = {}

# 1. CHECK FOR UPLOADED FILE
if 'uploaded_file' in locals() and uploaded_file is not None:
    try:
        # Load file based on extension
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

        # --- FIX 1: Normalize "Date" column name ---
        # Look for any column that looks like "date" (case-insensitive)
        date_col = None
        for col in df_raw.columns:
            if col.lower() == 'date':
                date_col = col
                break
        
        if date_col:
            df_raw.rename(columns={date_col: 'Date'}, inplace=True)
            df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        else:
            st.error("CSV must contain a 'Date' column.")
            st.stop()

        # --- FIX 2: Handle "Wide" Format (Date, TSLA.Close, TSM.Close...) ---
        # We loop through every column that isn't "Date" and treat it as a Ticker
        found_tickers = []
        
        for col in df_raw.columns:
            if col == 'Date':
                continue
            
            # Extract clean ticker name (e.g., "TSLA.Close" -> "TSLA")
            ticker_name = col.replace(".Close", "").replace(".close", "").strip()
            
            # Create a temporary dataframe for just this ticker
            # The 'add_features' function expects: Date, Close (and optionally Open/High/Low)
            temp_df = df_raw[['Date', col]].copy()
            temp_df.rename(columns={col: "Close"}, inplace=True)
            
            # Add required OHLC columns (fill with Close price as fallback)
            temp_df["Open"] = temp_df["Close"]
            temp_df["High"] = temp_df["Close"]
            temp_df["Low"] = temp_df["Close"]
            
            # Run calculations
            df_feat = add_features(temp_df)
            
            data_dict[ticker_name] = df_feat
            found_tickers.append(ticker_name)

        if not found_tickers:
            st.error("No ticker columns found in CSV.")
            st.stop()

        # Update the global tickers list so the sidebar works
        tickers = found_tickers
        
        st.success(f"Successfully loaded {len(tickers)} tickers from file.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

# 2. FALLBACK TO YFINANCE (if no file)
else:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    for t in tickers:
        df_raw = load_data(t, start_date, end_date, interval)
        if df_raw.empty:
            st.warning(f"No data for ticker: {t}")
            continue
        df_feat = add_features(df_raw)
        data_dict[t] = df_feat

if not data_dict:
    st.stop()

# Set the primary ticker for the view
# Use the first ticker from the file OR the text input
primary_ticker = tickers[0]

# If the user selects a different ticker in the UI, we can add a selector here later,
# but for now, we rely on the fact that 'primary_ticker' drives the charts.
# Since you have 1000+ tickers, let's add a specific selector for the uploaded data:
if 'uploaded_file' in locals() and uploaded_file is not None:
    primary_ticker = st.selectbox("Select Ticker to Analyze", options=tickers)

df_main = data_dict[primary_ticker]

# ============================================================
#          💹 PRICE CHART WITH TRADING SIGNALS (ANOMALY-BASED)
# ============================================================
st.subheader(f"💹 Price Chart with Trading Signals — {primary_ticker}")

fig_price = go.Figure()

if chart_type == "Candlestick":
    fig_price.add_trace(go.Candlestick(
        x=df_main["Date"],
        open=df_main["Open"],
        high=df_main["High"],
        low=df_main["Low"],
        close=df_main["Close"],
        name="Price"
    ))
else:
    fig_price.add_trace(go.Scatter(
        x=df_main["Date"],
        y=df_main["Close"],
        mode="lines",
        name="Close"
    ))

# Overlays: SMAs
if "SMA (50)" in indicators_selected:
    fig_price.add_trace(go.Scatter(
        x=df_main["Date"],
        y=df_main["SMA_50"],
        mode="lines",
        name="SMA 50"
    ))

if "SMA (200)" in indicators_selected:
    fig_price.add_trace(go.Scatter(
        x=df_main["Date"],
        y=df_main["SMA_200"],
        mode="lines",
        name="SMA 200"
    ))

# BUY / SELL markers based on anomaly
buy_points = df_main[df_main["anomaly"] == 1]
sell_points = df_main[df_main["anomaly"] == -1]

fig_price.add_trace(go.Scatter(
    x=buy_points["Date"],
    y=buy_points["Close"],
    mode="markers",
    name="BUY (Anomaly Up)",
    marker=dict(symbol="triangle-up", size=10),
))

fig_price.add_trace(go.Scatter(
    x=sell_points["Date"],
    y=sell_points["Close"],
    mode="markers",
    name="SELL (Anomaly Down)",
    marker=dict(symbol="triangle-down", size=10),
))

fig_price.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig_price, use_container_width=True)

# ============================================================
#            ANOMALY VISUALIZATION (MOMENTUM Z-SCORE)
# ============================================================
if "Anomaly (Momentum Z-score)" in indicators_selected:
    st.subheader(f"📉 Anomaly View — Momentum Z-score ({primary_ticker})")

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=df_main["Date"],
        y=df_main["mom_zscore"],
        mode="lines",
        name="Momentum Z-score"
    ))
    # Highlight anomaly zones
    fig_anom.add_hrect(y0=1.5, y1=10, opacity=0.1, line_width=0, fillcolor="green")
    fig_anom.add_hrect(y0=-10, y1=-1.5, opacity=0.1, line_width=0, fillcolor="red")
    fig_anom.add_hline(y=0, line=dict(dash="dash"))

    fig_anom.update_layout(
        xaxis_title="Date",
        yaxis_title="Z-score",
    )

    st.plotly_chart(fig_anom, use_container_width=True)

# ============================================================
#                        MACD INDICATOR
# ============================================================
if "MACD" in indicators_selected:
    st.subheader(f"📊 MACD Indicator — {primary_ticker}")

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(
        x=df_main["Date"],
        y=df_main["MACD"],
        mode="lines",
        name="MACD"
    ))
    fig_macd.add_trace(go.Scatter(
        x=df_main["Date"],
        y=df_main["MACD_signal"],
        mode="lines",
        name="Signal"
    ))
    fig_macd.add_trace(go.Bar(
        x=df_main["Date"],
        y=df_main["MACD_hist"],
        name="Histogram",
        opacity=0.4
    ))

    fig_macd.update_layout(
        xaxis_title="Date",
        yaxis_title="MACD",
        barmode="relative"
    )

    st.plotly_chart(fig_macd, use_container_width=True)

# ============================================================
#                         RSI INDICATOR
# ============================================================
if "RSI (14)" in indicators_selected:
    st.subheader(f"📏 RSI (14) — {primary_ticker}")

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df_main["Date"],
        y=df_main["RSI_14"],
        mode="lines",
        name="RSI (14)"
    ))
    fig_rsi.add_hline(y=70, line=dict(dash="dash"))
    fig_rsi.add_hline(y=30, line=dict(dash="dash"))
    fig_rsi.update_layout(
        xaxis_title="Date",
        yaxis_title="RSI"
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

# ============================================================
#          SIGNAL SUMMARY: COUNTS & DATA POINTS
# ============================================================
buy_count = len(buy_points)
sell_count = len(sell_points)
total_trades = buy_count + sell_count
data_points = len(df_main)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 BUY Signals", buy_count)
col2.metric("🔴 SELL Signals", sell_count)
col3.metric("📈 Total Signals (BUY+SELL)", total_trades)
col4.metric("📊 Data Points", data_points)

# ============================================================
#               BASIC BACKTEST & PERFORMANCE CHECK
# ============================================================
st.subheader("📈 Strategy Backtest vs Buy-and-Hold")

perf_df = df_main.dropna(subset=["cum_strategy", "cum_buy_hold"])

if len(perf_df) > 0:
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=perf_df["Date"],
        y=perf_df["cum_strategy"],
        mode="lines",
        name="Strategy (Anomaly-based)"
    ))
    fig_bt.add_trace(go.Scatter(
        x=perf_df["Date"],
        y=perf_df["cum_buy_hold"],
        mode="lines",
        name="Buy & Hold"
    ))
    fig_bt.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Growth (× initial capital)"
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Basic metrics
    final_strat = perf_df["cum_strategy"].iloc[-1]
    final_bh = perf_df["cum_buy_hold"].iloc[-1]
    total_ret_strat = (final_strat - 1) * 100
    total_ret_bh = (final_bh - 1) * 100

    # Approximate annualized return (assuming 252 trading days)
    n_days = (perf_df["Date"].iloc[-1] - perf_df["Date"].iloc[0]).days
    if n_days > 0:
        years = n_days / 252
        ann_strat = (final_strat ** (1 / years) - 1) * 100 if years > 0 else np.nan
        ann_bh = (final_bh ** (1 / years) - 1) * 100 if years > 0 else np.nan
    else:
        ann_strat, ann_bh = np.nan, np.nan

    # Simple volatility & Sharpe (no RF)
    strat_ret = perf_df["strategy_return"].dropna()
    if len(strat_ret) > 1:
        vol_strat = strat_ret.std() * np.sqrt(252) * 100
        sharpe_strat = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
    else:
        vol_strat, sharpe_strat = np.nan, np.nan

    bh_ret = perf_df["return"].dropna()
    if len(bh_ret) > 1:
        vol_bh = bh_ret.std() * np.sqrt(252) * 100
        sharpe_bh = (bh_ret.mean() / bh_ret.std()) * np.sqrt(252)
    else:
        vol_bh, sharpe_bh = np.nan, np.nan

    metrics_df = pd.DataFrame({
        "Metric": [
            "Total Return (%)",
            "Annualized Return (%)",
            "Volatility (%)",
            "Sharpe (no RF)"
        ],
        "Strategy": [
            round(total_ret_strat, 2),
            round(ann_strat, 2) if pd.notna(ann_strat) else np.nan,
            round(vol_strat, 2) if pd.notna(vol_strat) else np.nan,
            round(sharpe_strat, 2) if pd.notna(sharpe_strat) else np.nan,
        ],
        "Buy & Hold": [
            round(total_ret_bh, 2),
            round(ann_bh, 2) if pd.notna(ann_bh) else np.nan,
            round(vol_bh, 2) if pd.notna(vol_bh) else np.nan,
            round(sharpe_bh, 2) if pd.notna(sharpe_bh) else np.nan,
        ]
    })

    st.dataframe(metrics_df, use_container_width=True)
else:
    st.info("Not enough data to run backtest.")

st.success("Dashboard loaded successfully!")
