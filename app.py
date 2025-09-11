import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import re
import traceback

from src.data import load_ohlcv
from src.features import build_features
from src.label import make_label_next_up_nbars
from src.model import train_walkforward, backtest_bars, fit_full_model, predict_with_model
from src.model import backtest_sl_tp

# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Predictor (MA, RSI, Volume, Bandar-Proxy)")

# =========================
# Helpers UI
# =========================
def signal_label(prob, thr):
    if pd.isna(prob):
        return "UNKNOWN", "gray"
    if prob >= max(0.75, thr + 0.10):
        return "STRONG BUY", "green"
    if prob >= thr:
        return "BUY", "green"
    if prob >= max(0.50, thr - 0.05):
        return "NEUTRAL", "orange"
    return "HOLD/SELL", "gray"

def badge(text, color):
    colors = {
        "green": "#16a34a", "orange": "#f59e0b", "gray": "#6b7280",
        "red": "#dc2626", "blue": "#2563eb"
    }
    c = colors.get(color, "#6b7280")
    return f"""
    <span style="
      background:{c};color:white;padding:6px 10px;border-radius:999px;
      font-weight:600;font-size:0.85rem;display:inline-block;">
      {text}
    </span>
    """

def fmt_pct(x):
    return f"{x*100:,.1f}%" if pd.notna(x) else "—"

# =========================
# Sidebar: Input pengguna
# =========================
ticker_input = st.sidebar.text_input("Ticker (IDX pakai .JK)", value="BBCA.JK")
tick_first = re.split(r"[,\s;]+", ticker_input.strip())[0]
if tick_first != ticker_input:
    st.sidebar.warning(f"Mendeteksi multi-ticker. Dipakai: **{tick_first}**")
ticker = tick_first

interval = st.sidebar.selectbox(
    "Interval",
    ["1m","5m","15m","30m","1h","1d","1wk"],
    index=5  # default 1d
)

period_options_map = {
    "1m":  ["7d","14d"],
    "5m":  ["14d","30d","60d"],
    "15m": ["30d","60d","90d"],
    "30m": ["30d","60d","90d"],
    "1h":  ["30d","60d","180d","1y"],
    "1d":  ["2y","5y","10y"],
    "1wk": ["5y","10y"],
}
period = st.sidebar.selectbox("Period", period_options_map.get(interval, ["2y","5y","10y"]))

horizon_n = st.sidebar.slider("Horizon (jumlah bar ke depan)", 1, 20, 3, 1)
threshold = st.sidebar.slider("Ambang sinyal (prob naik)", 0.50, 0.80, 0.55, 0.01)
sl_pct = st.sidebar.slider("Stop Loss (%)", 0.2, 5.0, 1.0, 0.1) / 100.0
tp_pct = st.sidebar.slider("Take Profit (%)", 0.2, 10.0, 2.0, 0.1) / 100.0
train_button = st.sidebar.button("Latih Ulang & Backtest")
debug = st.sidebar.checkbox("Mode debug", value=False)

# =========================
# Load data (sekali saja)
# =========================
try:
    df = load_ohlcv(ticker, period=period, interval=interval)
except Exception as e:
    st.error("Gagal memuat data (load_ohlcv). Detail:")
    st.code("".join(traceback.format_exception_only(type(e), e)))
    st.stop()

# =========================
# Debug aman (hindari duplikat kolom)
# =========================
if debug:
    st.caption("🔎 Debug: shape setelah load_ohlcv")
    st.write(df.shape)
    st.caption("🔎 Debug: daftar kolom")
    st.write(list(df.columns))
    if len(df) > 0 and not df.columns.duplicated().any():
        st.caption("🔎 Debug: 10 bar terakhir")
        st.dataframe(df.tail(10))
    elif len(df) > 0:
        st.warning("Kolom duplikat terdeteksi — tabel mentah tidak ditampilkan agar PyArrow tidak error.")

# =========================
# Validasi awal
# =========================
if df is None or len(df) == 0:
    st.error("Data ticker kosong/invalid. Coba ticker lain (AAPL/MSFT) atau period lain. Bisa juga Yahoo lagi blank.")
    st.stop()

core_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
missing_core = [c for c in core_cols if c not in df.columns]
if len(missing_core) >= 4:
    st.error(f"Kolom inti hilang terlalu banyak: {missing_core}. Coba ticker/period lain.")
    st.stop()

non_numeric_cols = [c for c in core_cols if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
if non_numeric_cols:
    st.error(f"Kolom non-numeric terdeteksi: {non_numeric_cols}. Coba reload/ubah ticker.")
    st.stop()

if len(df) < 60 and interval in ["1d","1wk"]:
    st.error("Data terlalu sedikit untuk menghitung indikator. Coba period lebih panjang.")
    st.stop()

# =========================
# Bangun fitur + label
# =========================
try:
    feat = build_features(df)
    y = make_label_next_up_nbars(df, n=horizon_n)
except Exception as e:
    st.error("Gagal membangun fitur/label. Detail:")
    st.code("".join(traceback.format_exception_only(type(e), e)))
    st.stop()

# =========================
# Chart harga + MA + RSI + Volume (+ markers)
# =========================
with st.expander("Chart Harga & MA/RSI", expanded=True):
    try:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="OHLC"
        ))
        if "SMA20" in feat.columns:
            fig.add_trace(go.Scatter(x=feat.index, y=feat["SMA20"], name="SMA20"))
        if "SMA50" in feat.columns:
            fig.add_trace(go.Scatter(x=feat.index, y=feat["SMA50"], name="SMA50"))

        # ==== MARKERS dari hasil training terakhir (jika ada) ====
        try:
            preds = st.session_state.get("last_preds", None)
            sig = st.session_state.get("last_sig", None)
            exits = st.session_state.get("last_exits", None)
            base_close = "Adj Close" if "Adj Close" in df.columns else "Close"

            if preds is not None and sig is not None:
                buy_idx = preds.index[sig == 1]
                if len(buy_idx) > 0:
                    fig.add_trace(go.Scatter(
                        x=buy_idx, y=df.loc[buy_idx, base_close],
                        mode="markers", name="BUY",
                        marker=dict(symbol="triangle-up", size=10, color="green",
                                    line=dict(width=1, color="white"))
                    ))

            if exits is not None and "exit_reason" in exits.columns:
                ex = exits.dropna(subset=["exit_reason"])
                if not ex.empty:
                    ex_tp = ex[ex["exit_reason"] == "tp"].index
                    ex_sl = ex[ex["exit_reason"] == "sl"].index
                    ex_tm = ex[ex["exit_reason"] == "time"].index

                    if len(ex_tp) > 0:
                        fig.add_trace(go.Scatter(
                            x=ex_tp, y=df.loc[ex_tp, base_close],
                            mode="markers", name="EXIT TP",
                            marker=dict(symbol="circle", size=9, color="#16a34a")
                        ))
                    if len(ex_sl) > 0:
                        fig.add_trace(go.Scatter(
                            x=ex_sl, y=df.loc[ex_sl, base_close],
                            mode="markers", name="EXIT SL",
                            marker=dict(symbol="x", size=9, color="#dc2626")
                        ))
                    if len(ex_tm) > 0:
                        fig.add_trace(go.Scatter(
                            x=ex_tm, y=df.loc[ex_tm, base_close],
                            mode="markers", name="EXIT TIME",
                            marker=dict(symbol="square", size=8, color="#f59e0b")
                        ))
        except Exception:
            pass
        # =========================================================

        st.plotly_chart(fig, use_container_width=True)

        if "RSI14" in feat.columns:
            st.line_chart(feat["RSI14"], use_container_width=True)
        if "Volume" in df.columns:
            st.bar_chart(df["Volume"], use_container_width=True)

        try:
            st.caption(f"Last bar: {df.index[-1]}  |  Close: {df['Close'].iloc[-1]:,.2f}")
        except Exception:
            pass
    except Exception as e:
        st.warning("Gagal menampilkan chart. Detail:")
        st.code("".join(traceback.format_exception_only(type(e), e)))

# =========================
# Training & Backtest (walk-forward + SL/TP)
# =========================
if train_button:
    st.write("Melatih model walk-forward…")
    try:
        out = train_walkforward(feat, y)
        st.metric("AUC (validasi)", f"{out['auc']:.3f}")

        # Backtest SL/TP
        equity, sig, exits = backtest_sl_tp(
            df, out["preds"], threshold=threshold,
            n_bars_exit=horizon_n, sl_pct=sl_pct, tp_pct=tp_pct, fee_bp=10
        )
        st.subheader(f"Equity Curve (SL/TP, exit≤{horizon_n} bar)")
        st.line_chart(equity)

        # Ringkasan performa sinyal
        preds = out["preds"].dropna()
        y_aligned = y.loc[preds.index]
        signal_mask = preds > threshold
        wins = (y_aligned[signal_mask] == 1).sum()
        total = int(signal_mask.sum())
        winrate = (wins / total) if total > 0 else np.nan

        base_close = "Adj Close" if "Adj Close" in df.columns else "Close"
        px = df.loc[preds.index, base_close]
        fwd = (px.shift(-horizon_n) / px - 1.0)
        exp_ret = (fwd[signal_mask]).mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Sinyal (count)", f"{total:,}")
        c2.metric("Win-rate @signal", fmt_pct(winrate))
        c3.metric("Avg fwd return", fmt_pct(exp_ret if pd.notna(exp_ret) else 0.0))

        # Simpan ke session_state untuk plotting marker
        st.session_state["last_preds"] = preds
        st.session_state["last_sig"] = (preds > threshold).astype(int)
        st.session_state["last_threshold"] = threshold
        st.session_state["last_horizon"] = horizon_n
        st.session_state["last_exits"] = exits
    except Exception as e:
        st.error("Gagal melatih/backtest. Detail:")
        st.code("".join(traceback.format_exception_only(type(e), e)))

# =========================
# Prediksi terkini (model full) + Signal Card + Gauge
# =========================
st.divider()
st.subheader("Prediksi Terkini (Model Full)")

col1, col2 = st.columns([1, 1])

prob_now = None
with col1:
    try:
        model_path = Path("models/model.joblib")
        if model_path.exists():
            model = joblib.load(model_path)
            proba_series = predict_with_model(model, feat).rename("proba_up")
            if not proba_series.empty and pd.notna(proba_series.iloc[-1]):
                prob_now = float(proba_series.iloc[-1])
                label, color = signal_label(prob_now, threshold)

                st.markdown("**Status Sinyal Saat Ini**")
                st.markdown(badge(label, color), unsafe_allow_html=True)
                st.write(f"**Prob. Naik:** {prob_now:.3f}  |  **Ambang:** {threshold:.2f}")
            else:
                st.info("Belum ada probabilitas valid untuk bar terbaru.")
        else:
            st.info("Belum ada model tersimpan. Klik tombol di samping untuk melatih & simpan.")
    except Exception as e:
        st.error("Gagal menghitung prediksi terkini. Detail:")
        st.code("".join(traceback.format_exception_only(type(e), e)))

with col2:
    if st.button("Latih Full & Simpan"):
        try:
            model = fit_full_model(feat, y, save_path="models/model.joblib")
            st.success("Model tersimpan: models/model.joblib")
        except Exception as e:
            st.error("Gagal melatih & menyimpan model. Detail:")
            st.code("".join(traceback.format_exception_only(type(e), e)))

# Gauge probabilitas
try:
    if prob_now is not None:
        g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_now * 100,
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'thickness': 0.3},
                'steps': [
                    {'range': [0, threshold*100], 'color': '#9ca3af'},
                    {'range': [threshold*100, max(80, threshold*100 + 5)], 'color': '#f59e0b'},
                    {'range': [max(80, threshold*100 + 5), 100], 'color': '#16a34a'},
                ],
                'threshold': {'line': {'color': 'black', 'width': 2}, 'thickness': 0.75, 'value': threshold*100},
            },
            title={'text': "Probabilitas Naik (Gauge)"}
        ))
        st.plotly_chart(g, use_container_width=True)
except Exception:
    pass

# =========================
# Feature Importance (model full)
# =========================
st.divider()
with st.expander("🔬 Feature Importance (model full)"):
    try:
        model_path = Path("models/model.joblib")
        if model_path.exists():
            model = joblib.load(model_path)
            if hasattr(model, "feature_importances_"):
                feats = [
                    "SMA20","SMA50","EMA12","RSI14","ATR14",
                    "VolZ","TurnoverZ","AD","OBV","MFI","VWAP_Dist",
                    "ROC5","RetStd20"
                ]
                imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=True)
                fig_imp = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h'))
                fig_imp.update_layout(margin=dict(l=80, r=20, t=30, b=20), height=420)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Model tidak menyediakan feature_importances_.")
        else:
            st.info("Latih & simpan model terlebih dahulu.")
    except Exception as e:
        st.warning("Gagal menampilkan feature importance.")
        st.code("".join(traceback.format_exception_only(type(e), e)))
