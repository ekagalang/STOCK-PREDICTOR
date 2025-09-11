# src/data.py
import yfinance as yf
import pandas as pd
from pathlib import Path
import re
from datetime import datetime, timedelta

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def _normalize_single_ticker(ticker_input: str) -> str:
    parts = re.split(r"[,\s;]+", ticker_input.strip())
    return parts[0]

def _flatten_multiindex_concat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Jika kolom MultiIndex -> gabungkan semua level dengan underscore.
    Contoh: ('BBCA.JK','Open') atau ('Open','BBCA.JK') -> 'BBCA.JK_Open' / 'Open_BBCA.JK'
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join(str(x) for x in tup if x is not None).strip() for tup in df.columns.to_list()]
    return df

def _set_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Jika punya kolom Date/Datetime -> jadikan index
    for name in ["Date", "date", "Datetime", "datetime"]:
        if name in df.columns:
            df[name] = pd.to_datetime(df[name], errors="coerce")
            df = df.set_index(name)
            break
    # Pastikan index datetime & naik
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
    except Exception:
        pass
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df

def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Tambahkan suffix .1, .2, ... untuk kolom duplikat (sementara, sebelum mapping)
    df = df.copy()
    if df.columns.duplicated().any():
        counts = {}
        newcols = []
        for c in df.columns:
            if c not in counts:
                counts[c] = 0
                newcols.append(c)
            else:
                counts[c] += 1
                newcols.append(f"{c}.{counts[c]}")
        df.columns = newcols
    return df

def _map_to_core_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Cari kolom yang mengandung token open/high/low/close/adj/volume
    meskipun ada ticker di depan/belakang (e.g., 'BBCA.JK_Open', 'Open_BBCA.JK').
    """
    df = df.copy()
    cols_map = {}
    # pola token (abaikan non-huruf)
    def tok(c):
        return re.sub(r"[^a-z]", "", c.lower())

    # kandidat mapping
    core_targets = {
        "Open":   {"open"},
        "High":   {"high"},
        "Low":    {"low"},
        "Close":  {"close"},
        "Adj Close": {"adjclose","adjustedclose","adj"},
        "Volume": {"volume"},
    }

    used = set()
    for col in df.columns:
        t = tok(col)
        # buang nama ticker dari string agar tokenisasi lebih bersih (opsional)
        t = t.replace(tok(ticker), "")
        for target, keys in core_targets.items():
            if any(k in t for k in keys):
                # Ambil hanya pertama yang ketemu untuk tiap target
                if target not in cols_map:
                    cols_map[target] = col
                    used.add(col)
                break

    # Buat frame baru hanya dari core yang berhasil dipetakan
    core_cols = {}
    for target in ["Open","High","Low","Close","Adj Close","Volume"]:
        if target in cols_map:
            core_cols[target] = df[cols_map[target]]

    out = pd.DataFrame(index=df.index)
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        if c in core_cols:
            out[c] = pd.to_numeric(core_cols[c], errors="coerce")

    # Fallback Close <-> Adj Close
    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]
    if "Adj Close" not in out.columns and "Close" in out.columns:
        out["Adj Close"] = out["Close"]

    # Buang baris tanpa Close bila ada
    if "Close" in out.columns:
        out = out.dropna(subset=["Close"], how="any")

    return out

def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    # urutkan inti dulu
    core = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    return df[core + [c for c in df.columns if c not in core]]

def _attempts(ticker: str, period: str, interval: str):
    # 5 jalur pengambilan data
    yield dict(fn="download", kw=dict(group_by="column", auto_adjust=False))
    yield dict(fn="history_period", kw=dict(actions=False, auto_adjust=False))
    yield dict(fn="history_explicit", kw=dict(actions=False, auto_adjust=False))
    yield dict(fn="download_auto", kw=dict(group_by="column", auto_adjust=True))
    yield dict(fn="download_weekly", kw=dict(group_by="column", auto_adjust=True, interval="1wk"))

def _call_yf(ticker: str, period: str, interval: str, mode: str, kw: dict) -> pd.DataFrame:
    if mode == "download":
        return yf.download(ticker, period=period, interval=interval, progress=False, **kw)
    if mode == "download_auto":
        return yf.download(ticker, period=period, interval=interval, progress=False, **kw)
    if mode == "download_weekly":
        return yf.download(ticker, period=period, interval=kw.pop("interval"), progress=False, **kw)
    if mode == "history_period":
        return yf.Ticker(ticker).history(period=period, interval=interval, **kw)
    if mode == "history_explicit":
        today = datetime.utcnow().date()
        if period == "2y":
            start = today - timedelta(days=365*3)
        elif period == "5y":
            start = today - timedelta(days=365*7)
        else:
            start = today - timedelta(days=365*12)
        return yf.Ticker(ticker).history(start=start.isoformat(), end=today.isoformat(), interval=interval, **kw)
    return pd.DataFrame()

def load_ohlcv(ticker_input: str, period="5y", interval="1d") -> pd.DataFrame:
    ticker = _normalize_single_ticker(ticker_input)

    for att in _attempts(ticker, period, interval):
        try:
            raw = _call_yf(ticker, period, interval, att["fn"], att["kw"])
            if raw is None or len(raw) == 0:
                continue

            # Step 1: flatten kolom apapun bentuknya
            raw = _flatten_multiindex_concat(raw)

            # Step 2: dedup sebelum di-render / diproses
            raw = _dedup_columns(raw)

            # Step 3: pastikan index datetime
            raw = _set_datetime_index(raw)

            # Step 4: mapping ke core cols
            core = _map_to_core_columns(raw, ticker)

            # Step 5: kalau ini jalur mingguan, resample ke harian (ffill)
            if att["fn"] == "download_weekly" and len(core) > 0:
                core = core.resample("1D").ffill()

            # Step 6: finalize
            core = _finalize(core)

            if len(core) > 0 and ("Close" in core.columns or "Adj Close" in core.columns):
                return core
        except Exception:
            continue

    # gagal semua
    return pd.DataFrame()
