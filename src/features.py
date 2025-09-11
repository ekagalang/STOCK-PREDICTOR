import pandas as pd
import numpy as np
import ta

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Fallback kolom + paksa numeric
    close = _to_num(d["Adj Close"] if "Adj Close" in d.columns else d["Close"])
    high  = _to_num(d["High"])
    low   = _to_num(d["Low"])
    vol   = _to_num(d["Volume"]).fillna(0)

    # Jika setelah casting mayoritas NaN -> stop dini (biar UI kasih pesan)
    if close.dropna().empty:
        raise ValueError("Data 'Close/Adj Close' kosong/non-numeric setelah casting.")

    d["ret"] = close.pct_change()

    # Indikator inti
    d["SMA20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    d["SMA50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    d["EMA12"] = ta.trend.EMAIndicator(close, window=12).ema_indicator()
    d["RSI14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    d["ATR14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    d["Turnover"] = close * vol

    vol_mean20 = vol.rolling(20).mean()
    vol_std20  = vol.rolling(20).std().replace(0, np.nan)
    d["VolZ"] = (vol - vol_mean20) / vol_std20

    to_mean20 = d["Turnover"].rolling(20).mean()
    to_std20  = d["Turnover"].rolling(20).std().replace(0, np.nan)
    d["TurnoverZ"] = (d["Turnover"] - to_mean20) / to_std20

    # Proxy akumulasi/distribusi
    d["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    d["MFI"] = ta.volume.MFIIndicator(high, low, close, vol, window=14).money_flow_index()

    # A/D manual (handle div0)
    rng = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / rng
    clv = clv.replace([np.inf, -np.inf], np.nan).fillna(0)
    d["AD"] = (clv * vol).cumsum()

    # VWAP distance
    pv_cum = (close * vol).cumsum()
    v_cum  = vol.cumsum().replace(0, np.nan)
    vwap   = pv_cum / v_cum
    d["VWAP_Dist"] = (close - vwap) / vwap

    # Momentum/volatilitas
    d["ROC5"] = ta.momentum.ROCIndicator(close, window=5).roc()
    d["RetStd20"] = d["ret"].rolling(20).std()

    # --- Indikator lanjutan (baru) ---
    macd = ta.trend.MACD(close)
    d["MACD"] = macd.macd()
    d["MACD_Signal"] = macd.macd_signal()
    d["MACD_Hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    d["BB_Middle"] = bb.bollinger_mavg()
    d["BB_Upper"]  = bb.bollinger_hband()
    d["BB_Lower"]  = bb.bollinger_lband()
    d["BB_Width"]  = (d["BB_Upper"] - d["BB_Lower"]) / d["BB_Middle"]

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    d["STOCH_K"] = stoch.stoch()
    d["STOCH_D"] = stoch.stoch_signal()

    # Ichimoku (optional: beberapa nilai akan NaN diawal)
    try:
        d["ICH_TENKAN"] = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52).ichimoku_conversion_line()
        d["ICH_KIJUN"]  = ta.trend.IchimokuIndicator(high, low).ichimoku_base_line()
    except Exception:
        d["ICH_TENKAN"] = np.nan
        d["ICH_KIJUN"]  = np.nan

    return d
