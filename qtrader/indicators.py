"""技术指标(纯 pandas 实现,拒绝 TA-Lib 依赖)。全部输入输出均含当日。"""
from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=1).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    avg_up = up.ewm(alpha=1 / window, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_up / avg_down
    out = 100 - 100 / (1 + rs)
    out = out.mask(avg_down <= 0, 100.0)          # 无下跌 -> 超买
    out = out.mask((avg_up <= 0) & (avg_down <= 0), 50.0)  # 无波动
    return out.fillna(50.0).astype(float)


def vol_ma(volume: pd.Series, window: int = 5) -> pd.Series:
    return volume.rolling(window, min_periods=1).mean()


def roll_high(high: pd.Series, window: int) -> pd.Series:
    return high.rolling(window, min_periods=1).max()


def drawdown_from_high(close: pd.Series, window: int) -> pd.Series:
    """当前价距 N 日最高收盘的回撤比例(正值,0.15 表示回撤 15%)"""
    hh = close.rolling(window, min_periods=1).max()
    return (hh - close) / hh


def range_pos(df_row: pd.Series) -> float:
    """收盘价位于当日振幅位置:0=最低,1=最高"""
    hi, lo, c = float(df_row["high"]), float(df_row["low"]), float(df_row["close"])
    if hi <= lo:
        return 1.0
    return (c - lo) / (hi - lo)
