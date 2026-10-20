"""DataFeed 全链路集成验证(真实网络)。"""
import sys
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")

from qtrader.config import load_config
from qtrader.datafeed import DataFeed

cfg = load_config()
feed = DataFeed(cfg, "data")

print("== 1) 全A快照(新浪兜底)==")
spot = feed.spot_all(force=True)
print(spot.shape)
print(spot[["symbol", "name", "price", "pct_chg", "amount"]].head(3).to_string())
print("pct_chg 范围:", round(spot["pct_chg"].min(), 4), "~", round(spot["pct_chg"].max(), 4))

print("\n== 2) 个股日K(腾讯兜底)==")
h = feed.hist("SH600519", "2026-06-01", "2026-10-19")
print(h.tail(3).to_string())

print("\n== 3) ETF日K(腾讯兜底)==")
h2 = feed.hist("SH510300", "2026-06-01", "2026-10-19")
print("ETF rows:", len(h2), "| tail close:", None if h2.empty else h2["close"].iloc[-1])

print("\n== 4) 港股日K(新浪)==")
h3 = feed.hist("HK00700", "2026-06-01", "2026-10-19")
print("HK rows:", len(h3), "| tail close:", None if h3.empty else h3["close"].iloc[-1])

print("\n== 5) 腾讯实时报价(A+HK 混合)==")
q = feed.quote(["SH600519", "HK00700", "SH510300", "SZ000001"])
print(q[["symbol", "name", "price", "pct_chg", "amount"]].to_string())

print("\n== 6) 交易日历/大盘指数 ==")
print("今天(2026-10-20 周二)是交易日:", feed.is_trading_day())
r = feed.regime_hist("SH000001", "2026-05-01", "2026-10-19")
print("指数 bars:", len(r), "| 最新:", r["close"].iloc[-1])
