import pandas as pd

def make_label_next_up(df: pd.DataFrame) -> pd.Series:
    """
    Label biner untuk 1 bar ke depan.
    1 jika Adj Close_{t+1} > Adj Close_t, else 0.
    """
    base = "Adj Close" if "Adj Close" in df.columns else "Close"
    return (df[base].shift(-1) > df[base]).astype(int)

def make_label_next_up_nbars(df: pd.DataFrame, n: int = 3) -> pd.Series:
    """
    Label biner untuk n bar ke depan.
    1 jika Adj Close_{t+n} > Adj Close_t, else 0.
    """
    base = "Adj Close" if "Adj Close" in df.columns else "Close"
    return (df[base].shift(-n) > df[base]).astype(int)