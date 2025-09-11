import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib
from typing import Tuple

# Fitur yang harus sinkron dengan src/features.py
FEATURES = [
    "SMA20","SMA50","EMA12","RSI14","ATR14",
    "VolZ","TurnoverZ","AD","OBV","MFI","VWAP_Dist",
    "ROC5","RetStd20"
]

def _safe_xy(df_feat: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Ambil X dari df_feat[FEATURES] dan selaraskan y.
    Drop baris yang punya NaN di fitur.
    """
    X = df_feat[FEATURES].copy()
    mask = ~X.isna().any(axis=1)
    X = X[mask]
    y = y.loc[X.index]
    return X, y

def train_walkforward(df_feat: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42):
    """
    Latih model GradientBoosting secara walk-forward (TimeSeriesSplit).
    Return dict: {"auc": float, "preds": pd.Series(prob_up)}
    """
    X, y = _safe_xy(df_feat, y)

    # Jaga-jaga kalau data pendek: minimal 3 split, maksimal proporsional ukuran data
    n_splits = max(3, min(n_splits, max(2, len(X) // 50)))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    preds = pd.Series(index=X.index, dtype=float)
    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr = y.iloc[tr_idx]
        model = GradientBoostingClassifier(random_state=random_state)
        model.fit(X_tr, y_tr)
        preds.iloc[te_idx] = model.predict_proba(X_te)[:, 1]

    valid_mask = ~preds.isna()
    auc = roc_auc_score(y[valid_mask], preds[valid_mask]) if valid_mask.any() else float("nan")
    return {"auc": auc, "preds": preds}

def backtest_bars(df: pd.DataFrame, preds: pd.Series, threshold: float = 0.55, n_bars: int = 3, fee_bp: float = 10):
    """
    Backtest sederhana berbasis bar:
    - Entry di close bar t jika prob(t) > threshold
    - Exit di close bar t+n_bars
    - fee_bp = biaya roundtrip (basis poin), contoh 10 -> 0.10%
    Return: equity (pd.Series), signal (pd.Series[int])
    """
    preds = preds.dropna()
    sig = (preds > threshold).astype(int)

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    price = df.loc[preds.index, price_col].copy()
    fwd_ret = (price.shift(-n_bars) / price) - 1.0

    gross = fwd_ret * sig
    # Perkiraan jumlah transaksi pakai perubahan sinyal
    turns = sig.diff().abs().fillna(0)
    fees = (turns * (fee_bp / 10000.0)).clip(lower=0)

    net = (gross - fees).fillna(0)
    equity = (1 + net).cumprod()
    return equity, sig

def fit_full_model(df_feat: pd.DataFrame, y: pd.Series, save_path: str = "models/model.joblib"):
    """
    Latih model pada seluruh data (setelah fitur siap) dan simpan ke disk.
    """
    X, y = _safe_xy(df_feat, y)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    joblib.dump(model, save_path)
    return model

def predict_with_model(model, df_feat: pd.DataFrame) -> pd.Series:
    """
    Hitung probabilitas naik untuk bar yang fitur-fiturnya lengkap.
    """
    X = df_feat[FEATURES].copy()
    mask = ~X.isna().any(axis=1)
    X = X[mask]
    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="proba_up")
    return proba

def backtest_sl_tp(
    df: pd.DataFrame,
    preds: pd.Series,
    threshold: float = 0.55,
    n_bars_exit: int = 3,
    sl_pct: float = 0.01,   # 1% stop loss
    tp_pct: float = 0.02,   # 2% take profit
    fee_bp: float = 10,     # 0.10% roundtrip
    position_frac: float = 1.0
):
    """
    Backtest dengan aturan:
    - Entry di close bar t jika prob(t) > threshold
    - Exit lebih awal jika:
        * drawdown dari entry mencapai -sl_pct
        * gain dari entry mencapai +tp_pct
      jika tidak kena SL/TP, exit paksa di t + n_bars_exit
    - Biaya transaksi fee_bp (basis poin) sekali per roundtrip
    - position_frac: ukuran posisi relatif (0..1)
    Return:
      equity: pd.Series
      sig:    pd.Series[int] (0/1)
      exits:  pd.DataFrame dengan info exit (index sama dengan preds.index)
    """
    preds = preds.dropna()
    sig = (preds > threshold).astype(int)

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    px = df.loc[preds.index, price_col].astype(float)

    # container hasil
    ret_list = []
    exit_reason = []  # 'tp'/'sl'/'time'
    exit_index = []

    fee = fee_bp / 10000.0

    i = 0
    idx = preds.index.to_list()
    n = len(idx)
    while i < n:
        t = idx[i]
        if sig.loc[t] != 1:
            ret_list.append(0.0)
            exit_reason.append(None)
            exit_index.append(t)
            i += 1
            continue

        # Entry
        entry_px = px.loc[t]
        exit_px = None
        reason = "time"
        j = 1
        # scanning ke depan hingga n_bars_exit
        while j <= n_bars_exit and (i + j) < n:
            tt = idx[i + j]
            curr_px = px.loc[tt]
            rr = (curr_px / entry_px) - 1.0
            if rr >= tp_pct:
                exit_px = curr_px; reason = "tp"; break
            if rr <= -sl_pct:
                exit_px = curr_px; reason = "sl"; break
            j += 1
        if exit_px is None:
            # tidak kena TP/SL -> exit time based (bar terakhir yang tersedia)
            tt = idx[min(i + n_bars_exit, n - 1)]
            exit_px = px.loc[tt]
            reason = "time"

        gross = (exit_px / entry_px) - 1.0
        net = (gross - fee) * position_frac

        # Simpan satu return untuk bar entry (supaya equity cumprod sinkron dengan preds index)
        ret_list.append(net)
        exit_reason.append(reason)
        exit_index.append(t)

        # Lompati ke bar setelah exit
        # jika exit sebelum n_bars_exit, set i ke posisi j yang sudah dipakai
        i = min(i + j, n - 1)
        i += 1

    equity = (1 + pd.Series(ret_list, index=exit_index).reindex(preds.index).fillna(0)).cumprod()
    exits = pd.DataFrame({"exit_reason": pd.Series(exit_reason, index=exit_index)}).reindex(preds.index)
    return equity, sig, exits
