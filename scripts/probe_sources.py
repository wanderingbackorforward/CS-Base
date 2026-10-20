"""数据源探测:逐个测试本环境可用性(探测用,不入主流程)。"""
import sys, time
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")

import akshare as ak

def probe(name, fn, timeout_desc=""):
    t0 = time.time()
    try:
        df = fn()
        cost = time.time() - t0
        if df is None or len(df) == 0:
            print(f"[FAIL] {name}: empty ({cost:.1f}s)")
        else:
            print(f"[ OK ] {name}: {len(df)} 行, {cost:.1f}s, 列: {list(df.columns)[:8]}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:90]} ({time.time()-t0:.1f}s)")

probe("em.spot 全A快照", lambda: ak.stock_zh_a_spot_em())
probe("sina.spot 全A快照", lambda: ak.stock_zh_a_spot())
probe("em.hist 个股日K", lambda: ak.stock_zh_a_hist(symbol="600519", period="daily",
      start_date="20260601", end_date="20261019", adjust="qfq"))
probe("tx.hist 腾讯日K", lambda: ak.stock_zh_a_hist_tx(symbol="sh600519",
      start_date="20260601", end_date="20261019"))
probe("sina.hk_daily 港股日K", lambda: ak.stock_hk_daily(symbol="00700"))
probe("sina.calendar 交易日历", lambda: ak.tool_trade_date_hist_sina())
probe("sina.index_daily 指数", lambda: ak.stock_zh_index_daily(symbol="sh000001"))
probe("em.concepts 概念板块", lambda: ak.stock_board_concept_name_em())
probe("em.etf_hist ETF日K", lambda: ak.fund_etf_hist_em(symbol="510300", period="daily",
      start_date="20260601", end_date="20261019", adjust="qfq"))
