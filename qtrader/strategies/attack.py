"""策略:进攻仓(投资系统准则 v5 H1-H10)。重仓赢家,早走输家。

狠体现在两处:单只仓位大、涨了不卖(H9 不止盈);输家由论点卡和均线
尽早请出去(H7/H8)。个股收益极度偏斜,所以规则只围绕两件事:
拿到少数大赢家时拿得住,拿到输家时走得早。

- H1 候选至少一个来源:投资人13F前10大 / 掌舵人公司(名单 P1-P5)
- H2 硬条件:美股 市值≥100亿+近4季累计净利润/经营现金流为正;
      A股沿用 v3 U2/U3 硬筛(不在池里=不满足)
- H3 论点卡:买入前置;机器规则系统判,人工规则季度提醒
- H4 3~5只,单只目标=进攻仓÷只数(≤20%/10%),放不满的钱留在核心
- H5 两步建仓:第一步月末(美股>10月均线/A股估值≤50%分位)买一半;
      第二步下个季度调仓日价格高于第一步买入价才买满;不满足继续拿半仓等
- H6 不摊平:价格低于上次买入价不加仓(H8 恢复除外)
- H7 满足任一条次日全卖:论点卡卖出条件/H2不再满足/来源消失满2季/A股估值>90%
- H8 趋势(美股):月末收盘<10月均线,浮盈<30%全卖;≥30%先卖一半,
      连续3个月末在均线下再卖另一半;重新站上均线次日直接恢复到目标仓位
- H9 不止盈:单只>25%(A股15%)减回20%(A股12%);进攻仓整体>70%减回60%
- H10 隔离:进攻仓市值<累计投入的一半 -> 12个月不进新钱,核心永不补亏

R1 穿透30%/R2 行业40%(A股20%)/R3 最小下单额:买入口径上拦截,不自动卖。
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd

from ..config import Section
from ..models import PendingSignal, Position, Side
from .base import Context, Strategy
from .persons import Persons, ThesisCards

log = logging.getLogger("qtrader.attack")


def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    for day in (d.day, 28, 27, 26):
        try:
            return date(y, m, day)
        except ValueError:
            continue
    return d


def _quarter_boundaries_between(d1: date, d2: date) -> int:
    """d1~d2 之间跨过几个季度界(近似:季首月1日;美股实际是13F次一交易日)。"""
    cnt = 0
    for year in range(d1.year, d2.year + 1):
        for mm in (1, 4, 7, 10):
            b = date(year, mm, 1)
            if d1 < b <= d2:
                cnt += 1
    return cnt


def _d(s: str) -> date | None:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _monthly_closes(hist) -> pd.Series | None:
    """月度收盘序列(复权日线聚合到月末);无法聚合返回 None,绝不猜。"""
    if hist is None or len(hist) == 0:
        return None
    try:
        if "dt" in getattr(hist, "columns", []):
            idx = pd.to_datetime(hist["dt"])
        elif isinstance(hist.index, pd.DatetimeIndex):
            idx = hist.index
        else:
            return None
        s = pd.Series(hist["close"].astype(float).values,
                      index=idx).sort_index()
        for freq in ("ME", "M"):
            try:
                return s.resample(freq).last().dropna()
            except ValueError:
                continue
    except Exception as e:  # noqa: BLE001
        log.debug("月线聚合失败: %s", e)
    return None


class Attack(Strategy):
    name = "attack"

    def __init__(self, section: Section, store, *, market: str = "us",
                 pool_manager=None) -> None:
        self.store = store
        self.market = market               # "cn" | "us"
        self.enabled = True
        a = section["attack"]
        alloc = section["allocation"]
        self.pool_manager = pool_manager
        self.min_names = int(a.get("min_names", 3))
        self.max_names = int(a.get("max_names", 5))
        self.attack_pct = float(alloc.get("attack_pct",
                                           a.get("target_pct", 0.30)))
        self.per_name_max = float(a.get("per_name_max", 0.20))
        self.per_name_trim_at = float(a.get("per_name_trim_at", 0.25))
        self.per_name_trim_to = float(a.get("per_name_trim_to", 0.20))
        self.total_trim_at = float(a.get("total_trim_at", 0.70))
        self.total_trim_to = float(a.get("total_trim_to", 0.60))
        self.entry_val_max = float(a.get("entry_val_max", 0.50))
        self.exit_val_max = float(a.get("exit_val_max", 0.90))
        self.use_ma = bool(a.get("use_ma", market == "us"))
        self.ma_months = int(a.get("ma_months", 10))
        self.h8_profit_kept = float(a.get("h8_profit_kept", 0.30))
        self.h8_streak_months = int(a.get("h8_streak_months", 3))
        self.source_gone_quarters = int(a.get("source_gone_quarters", 2))
        self.isolation_line = float(a.get("isolation_line", 0.50))
        self.isolation_months = int(a.get("isolation_months", 12))
        self.min_order_amount = float(a.get("min_order_amount", 3600.0))
        self.industry_cap = float(a.get("industry_cap", 0.40))
        self.lookthrough_cap = float(a.get("lookthrough_cap", 0.30))
        self.hard = dict(a.get("hard") or {})
        # 人物名单 + 论点卡 + 穿透权重
        self.persons = Persons(str(a.get("persons_file", "persons.yaml")))
        self.cards = ThesisCards(str(a.get("thesis_file", "thesis_cards.yaml")))
        self.index_weights: dict = self._load_index_weights(
            str(a.get("index_weights_file", "")))

    def _load_index_weights(self, path: str) -> dict:
        if not path:
            return {}
        try:
            from pathlib import Path
            import yaml
            p = Path(path)
            if not p.exists():
                return {}
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:  # noqa: BLE001
            log.warning("R1 穿透权重加载失败: %s", e)
            return {}

    # ---------------- 状态 ----------------
    def _st(self, sym: str) -> dict:
        d = self.store.pool_state(sym, default=None) or {}
        d.setdefault("step", 0)
        d.setdefault("step1_price", 0.0)
        d.setdefault("step1_date", "")
        d.setdefault("h8_streak", 0)
        d.setdefault("h8_partial", False)
        d.setdefault("h8_exited", False)
        d.setdefault("gone_since", None)
        return d

    def _save(self, sym: str, d: dict) -> None:
        self.store.save_pool_state(sym, d)

    def _acc(self) -> dict:
        acc = self.store.get_meta("attack_acc", default=None) or {}
        acc.setdefault("invested", 0.0)
        acc.setdefault("isolation_until", "")
        acc.setdefault("isolation_since", "")
        acc.setdefault("review", "")
        return acc

    def _save_acc(self, acc: dict) -> None:
        self.store.set_meta("attack_acc", acc)

    def _held(self, ctx: Context) -> dict[str, Position]:
        return {s: p for s, p in ctx.positions.items()
                if p.strategy == self.name}

    def _attack_mv(self, ctx: Context) -> float:
        return sum(p.qty * ctx.price(s) for s, p in self._held(ctx).items())

    # ---------------- H10 隔离 ----------------
    def _is_isolated(self, today: str) -> bool:
        acc = self._acc()
        until = _d(acc.get("isolation_until") or "")
        return bool(until and _d(today) and _d(today) <= until)

    def _h10_update(self, ctx: Context) -> None:
        acc = self._acc()
        today_d = _d(ctx.today)
        until = _d(acc.get("isolation_until") or "")
        if until and today_d and today_d > until:
            # 隔离期满:复盘后从当前市值重新计量(否则刚解禁立刻再触发)
            acc["invested"] = self._attack_mv(ctx)
            acc["isolation_until"] = ""
            acc["isolation_since"] = ""
            acc["review"] = f"H10 隔离至{until}期满,已复盘;invested 重置为当前市值"
            self._save_acc(acc)
            log.warning("H10 隔离期满( %s ),进攻仓重新计量:invested=%.0f",
                        ctx.today, acc["invested"])
            return
        if until:
            return
        mv = self._attack_mv(ctx)
        invested = float(acc.get("invested") or 0.0)
        if invested > 0 and mv < invested * self.isolation_line:
            until_d = _add_months(today_d, self.isolation_months) \
                if today_d else today_d
            acc["isolation_until"] = until_d.isoformat() if today_d else ""
            acc["isolation_since"] = ctx.today
            acc["review"] = (f"H10 隔离:进攻市值{mv:.0f} < 累计投入"
                             f"{invested:.0f}的一半,12个月内新钱不进进攻仓,"
                             "并做一次完整复盘")
            self._save_acc(acc)
            log.critical("H10 隔离触发:%s 进攻市值%.0f<投入%.0f的%.0f%%,"
                         "隔离至%s(核心永不补亏)", ctx.today, mv, invested,
                         self.isolation_line * 100, acc["isolation_until"])

    # ---------------- H2 / 来源 / 卡 ----------------
    def _pool_entry(self, ctx: Context, sym: str) -> dict:
        return next((e for e in (ctx.pool or []) if e.get("symbol") == sym), {})

    def _h2_check(self, ctx: Context, sym: str) -> tuple[bool, str]:
        """H2 硬条件;返回(通过,说明)。美股读候选财务字段,A股看池内标记。"""
        e = self._pool_entry(ctx, sym)
        if not e:
            return False, "H2 不满足:不在候选池(U4 被移出/未入池)"
        if self.market == "cn":
            if not e.get("hard_pass", False):
                return False, f"H2 不满足:U2/U3 硬筛未过({e.get('hard_fail', '')})"
            return True, ""
        # 美股:市值 + 近4季累计净利润/经营现金流(不看自由现金流)
        cap = float(e.get("market_cap_usd") or 0)
        npf = float(e.get("net_profit_4q") or 0)
        ocf = float(e.get("ocf_4q") or 0)
        upd = e.get("financial_updated") or ""
        min_cap = float(self.hard.get("min_mcap_usd", 10.0e9))
        max_age = int(self.hard.get("max_financial_age_days", 200))
        if cap < min_cap:
            return False, f"H2 不满足:市值{cap / 1e9:.1f}B < {min_cap / 1e9:.0f}B"
        if npf <= 0:
            return False, "H2 不满足:近4季累计净利润≤0"
        if ocf <= 0:
            return False, "H2 不满足:近4季累计经营现金流≤0"
        upd_d = _d(upd)
        today_d = _d(ctx.today)
        if not upd_d or (today_d and (today_d - upd_d).days > max_age):
            return False, (f"H2 数据过期:财务字段更新于{upd or '未填'},"
                          f"超{max_age}天,先更新 us_pool.yaml 再买")
        return True, ""

    def _source_check(self, ctx: Context, sym: str) -> tuple[bool, str]:
        """H1 来源 + P1/P2/P4;P3 只挡新买。"""
        p = self.persons.source_of(sym)
        if p is None:
            return False, f"H1 无来源:{sym} 既不是掌舵人的公司,也不在13F前10大"
        ok, why = self.persons.eligible(p, ctx.today)
        if not ok:
            return False, why
        if self.persons.net_sell_blocked(sym):
            return False, f"P3 移出候选:{sym} 掌舵人近12个月净卖出超20%"
        return True, f"来源:{p.get('name')}({p.get('type')})"

    def _h7_reasons(self, ctx: Context, sym: str) -> list[str]:
        """H7 卖出条件核查(论点卡机器规则 + H2 + 来源消失满2季 + A股估值)。"""
        reasons: list[str] = []
        st = self._st(sym)
        # 论点卡机器规则
        for rule in self.cards.auto_rules(sym):
            e = self._pool_entry(ctx, sym)
            if rule == "ocf_4q_negative" and self.market == "us" \
                    and float(e.get("ocf_4q") or 0) < 0:
                reasons.append("论点卡:近4季累计经营现金流转负")
            if rule == "profit_4q_negative" and self.market == "us" \
                    and float(e.get("net_profit_4q") or 0) < 0:
                reasons.append("论点卡:近4季累计净利润转负")
            if rule == "out_of_pool" and self.market == "cn" and not e:
                reasons.append("论点卡:被移出准入名单(U4)")
        # H2 不再满足
        h2ok, h2why = self._h2_check(ctx, sym)
        if not h2ok:
            reasons.append(h2why)
        # P4 来源消失满2个季度
        p = self.persons.source_of(sym)
        if p is not None and p.get("gone"):
            if not st.get("gone_since"):
                st["gone_since"] = ctx.today
                self._save(sym, st)
            since = _d(st.get("gone_since") or ctx.today)
            today_d = _d(ctx.today)
            if since and today_d and _quarter_boundaries_between(
                    since, today_d) >= self.source_gone_quarters:
                reasons.append(f"P4 来源消失满{self.source_gone_quarters}个季度"
                               f"({p.get('name')}离任/清仓)")
        elif p is not None and st.get("gone_since"):
            st["gone_since"] = None              # 来源恢复(名单改回)
            self._save(sym, st)
        # A股估值 > 90%
        if self.market == "cn":
            q = (ctx.pool_val_q or {}).get(sym)
            if q is not None and q > self.exit_val_max:
                reasons.append(f"估值分位{q:.0%}>{self.exit_val_max:.0%}")
        return reasons

    # ---------------- 买入口径检查 ----------------
    def _r2_industry_ok(self, ctx: Context, sym: str, add_pct: float) -> bool:
        ind = ctx.industry_of(sym)
        if not ind:
            return True
        equity = ctx.equity()
        held = 0.0
        for s, p in self._held(ctx).items():
            if ctx.industry_of(s) == ind:
                held += p.qty * ctx.price(s)
        if equity > 0 and (held / equity + add_pct) > self.industry_cap:
            log.warning("R2 行业上限:%s(%s)现%.1f%%+新买%.1f%%>%.0f%%,拦截",
                        sym, ind, held / equity * 100, add_pct * 100,
                        self.industry_cap * 100)
            return False
        return True

    def _r1_lookthrough_pct(self, ctx: Context, sym: str) -> float:
        """直接持仓 + 通过指数ETF间接持有的比例。"""
        equity = ctx.equity() or 1.0
        pos = ctx.positions.get(sym)
        direct = pos.qty * ctx.price(sym) / equity if pos else 0.0
        indirect = 0.0
        for etf, weights in (self.index_weights or {}).items():
            w = float((weights or {}).get(sym, 0.0))
            if w <= 0:
                continue
            ep = ctx.positions.get(etf)
            if ep is not None:
                indirect += ep.qty * ctx.price(etf) / equity * w
        return direct + indirect

    def _r1_ok(self, ctx: Context, sym: str, add_pct: float) -> bool:
        total = self._r1_lookthrough_pct(ctx, sym) + add_pct
        if total > self.lookthrough_cap:
            log.warning("R1 穿透上限:%s 直接+穿透=%.1f%%>%.0f%%,拦截",
                        sym, total * 100, self.lookthrough_cap * 100)
            return False
        return True

    def _ma_state(self, ctx: Context, sym: str) -> tuple[float, float] | None:
        """(月末收盘, 10个月均线);数据不足返回 None。"""
        mc = _monthly_closes((ctx.hist or {}).get(sym))
        if mc is None or len(mc) < self.ma_months:
            return None
        ma = float(mc.iloc[-self.ma_months:].mean())
        return float(mc.iloc[-1]), ma

    def _entry_ok(self, ctx: Context, sym: str) -> tuple[bool, str]:
        """H5 第一步门槛:美股月末>10月均线;A股估值分位≤50%。"""
        if self.market == "us":
            m = self._ma_state(ctx, sym)
            if m is None:
                return False, "H5 月线数据不足10个月,不买(绝不猜)"
            close, ma = m
            if close <= ma:
                return False, f"H5 月末收盘{close:.2f}≤{self.ma_months}月均线{ma:.2f}"
            return True, f"月末{close:.2f}>均线{ma:.2f}"
        q = (ctx.pool_val_q or {}).get(sym)
        if q is None:
            return False, "H5 估值分位不可知,不买(绝不猜)"
        if q > self.entry_val_max:
            return False, f"H5 估值分位{q:.0%}>{self.entry_val_max:.0%}"
        return True, f"估值分位{q:.0%}≤{self.entry_val_max:.0%}"

    # ---------------- 收盘后买入信号 ----------------
    def entry_signals(self, ctx: Context) -> list[PendingSignal]:
        out: list[PendingSignal] = []
        equity = ctx.equity()
        if equity <= 0:
            return out
        today = ctx.today
        pending = {(s.symbol, s.rule) for s in ctx.pending}
        self._h10_update(ctx)                     # H10 每天查
        isolated = self._is_isolated(today)
        if isolated:
            acc = self._acc()
            log.warning("H10 隔离中(至%s):进攻仓今日只卖不买",
                        acc.get("isolation_until"))
        held = self._held(ctx)
        pool_syms = [e.get("symbol") for e in (ctx.pool or [])]
        # H4:只数与单只目标(放不满的钱留在核心)
        n_new = sum(1 for e in (ctx.pool or [])
                    if e.get("symbol") not in held)
        n = max(len(held), min(len(held) + n_new, self.max_names))
        n = max(n, 1)
        per_target = min(self.attack_pct / n, self.per_name_max)

        for e in ctx.pool or []:
            sym = e.get("symbol")
            if not sym or sym in held:
                continue
            st = self._st(sym)
            # H3 论点卡(买入前置)
            card_ok, card_why = self.cards.card_ok_and_risks_written(
                sym, self.persons)
            src_ok, src_why = self._source_check(ctx, sym)   # H1/P1-P3
            h2_ok, h2_why = self._h2_check(ctx, sym)
            if not (card_ok and src_ok and h2_ok):
                log.info("%s 暂不买:%s / %s / %s", sym, card_why, src_why, h2_why)
                continue
            if isolated:
                continue

            buy_pct = 0.0
            rule = ""
            why = ""
            if st.get("h8_exited") and self.use_ma:
                # H8 恢复:重新站上均线 -> 次日直接恢复到目标仓位(不走两步)
                if not ctx.month_end:
                    continue
                m = self._ma_state(ctx, sym)
                if m is None or m[0] <= m[1]:
                    continue
                buy_pct = per_target
                rule, why = "H8r", (f"H8 恢复:月末收盘{m[0]:.2f}重新站上"
                                    f"{self.ma_months}月均线{m[1]:.2f},"
                                    f"次日直接恢复到目标{per_target:.1%}")
            elif ctx.month_end and st.get("step", 0) == 0:
                # H5 第一步:月末检查,买目标的一半
                ok, w = self._entry_ok(ctx, sym)
                if not ok:
                    log.info("%s %s", sym, w)
                    continue
                buy_pct = round(per_target / 2, 5)
                rule, why = "H5a", (f"H5 第一步:{w},买目标的一半"
                                   f"({buy_pct:.2%};第二步下个季度调仓日"
                                   "价格高于买入价才买满)")
            if not rule:
                continue
            amount = buy_pct * equity
            if amount < self.min_order_amount:    # R3 最小下单额
                log.warning("R3 拦截 %s %s:金额%.0f<%.0f(每笔佣金≤0.5%%;"
                            "放不下就减少只数,钱留在核心)",
                            rule, sym, amount, self.min_order_amount)
                continue
            if not self._r2_industry_ok(ctx, sym, buy_pct):
                continue
            if not self._r1_ok(ctx, sym, buy_pct):
                continue
            if (sym, rule) in pending:
                continue
            out.append(PendingSignal(
                symbol=sym, side=Side.BUY, signal_date=today,
                strategy=self.name, rule=rule, size_hint_pct=buy_pct,
                reason=f"{rule}({src_why}){why}"))
            log.warning("买入信号 %s %s %s", rule, sym, why)

        # H5 第二步:季度调仓日,价格高于第一步买入价才买满(H6 不摊平)
        if ctx.quarter_day and not isolated:
            for sym, pos in held.items():
                st = self._st(sym)
                if st.get("step", 0) != 1:
                    continue
                if (sym, "H5b") in pending:
                    continue
                h7 = self._h7_reasons(ctx, sym)
                if h7:
                    continue                    # 卖出条件已触发,等 manage 处理
                px = ctx.price(sym)
                s1 = float(st.get("step1_price") or 0)
                if px <= 0 or s1 <= 0 or px <= s1:
                    log.info("H5b %s 继续拿半仓等:现价%.2f ≤ 第一步买入价%.2f"
                             "(H6 不摊平)", sym, px, s1)
                    continue
                cur_pct = pos.qty * px / equity
                rest = round(min(per_target - cur_pct, per_target / 2), 5)
                if rest <= 0:
                    st["step"] = 2
                    self._save(sym, st)
                    continue
                if rest * equity < self.min_order_amount:
                    log.warning("R3 拦截 H5b %s:补仓金额%.0f<%.0f", sym,
                                rest * equity, self.min_order_amount)
                    continue
                if not self._r2_industry_ok(ctx, sym, rest):
                    continue
                out.append(PendingSignal(
                    symbol=sym, side=Side.BUY, signal_date=today,
                    strategy=self.name, rule="H5b", size_hint_pct=rest,
                    reason=(f"H5 第二步:现价{px:.2f}>第一步买入价{s1:.2f}"
                            f"且卖出条件未触发,买满目标{per_target:.1%}"
                            f"(现{cur_pct:.1%},补{rest:.2%})")))
                log.warning("买入信号 H5b %s 现价%.2f>首步价%.2f,买满", sym, px, s1)

        # H8 恢复(半仓持有者):月末站上均线 -> 次日恢复到目标仓位
        if self.use_ma and ctx.month_end and not isolated:
            for sym, pos in held.items():
                st = self._st(sym)
                if not st.get("h8_partial"):
                    continue
                if (sym, "H8r") in pending:
                    continue
                m = self._ma_state(ctx, sym)
                if m is None or m[0] <= m[1]:
                    continue
                cur_pct = pos.qty * ctx.price(sym) / equity
                rest = round(per_target - cur_pct, 5)
                if rest <= 0 or rest * equity < self.min_order_amount:
                    continue
                if not self._r2_industry_ok(ctx, sym, rest):
                    continue
                if not self._r1_ok(ctx, sym, rest):
                    continue
                out.append(PendingSignal(
                    symbol=sym, side=Side.BUY, signal_date=today,
                    strategy=self.name, rule="H8r", size_hint_pct=rest,
                    reason=(f"H8 恢复:月末{m[0]:.2f}重新站上均线{m[1]:.2f},"
                            f"从{cur_pct:.1%}恢复到目标{per_target:.1%}")))
                log.warning("买入信号 H8r %s 恢复到目标仓位", sym)
        return out

    # ---------------- 收盘后卖出信号 ----------------
    def manage(self, ctx: Context) -> list[PendingSignal]:
        out: list[PendingSignal] = []
        equity = ctx.equity()
        if equity <= 0:
            return out
        today = ctx.today
        pending = {(s.symbol, s.rule) for s in ctx.pending}
        held = self._held(ctx)

        # ---- H7:任一条件触发,次日全部卖出(每天查,财报/公告后即推) ----
        for sym, pos in held.items():
            if (sym, "H7") in pending:
                continue
            reasons = self._h7_reasons(ctx, sym)
            if not reasons:
                continue
            qty = pos.available_qty if self.market == "us" else pos.available_qty
            if self.market == "cn":
                qty -= qty % 100
            if qty <= 0:
                log.warning("H7 %s 触发但无可卖(T+1),次日再试:%s",
                            sym, ";".join(reasons))
                continue
            out.append(PendingSignal(
                symbol=sym, side=Side.SELL, signal_date=today,
                strategy=self.name, rule="H7", qty=qty, sell_ratio=1.0,
                reason=f"H7 全部卖出:{';'.join(reasons)}"))
            log.warning("卖出信号 H7 %s:%s", sym, ";".join(reasons))

        # ---- H8:趋势退出(美股,月末检查) ----
        if self.use_ma and ctx.month_end:
            for sym, pos in held.items():
                st = self._st(sym)
                m = self._ma_state(ctx, sym)
                if m is None:
                    continue
                close, ma = m
                if close >= ma:
                    if st.get("h8_streak", 0):
                        st["h8_streak"] = 0
                        self._save(sym, st)
                    continue
                # 月末收盘在均线下方
                if st.get("h8_partial") and \
                        st.get("h8_streak", 0) >= self.h8_streak_months:
                    qty = pos.available_qty
                    if self.market == "cn":
                        qty -= qty % 100
                    if qty > 0 and (sym, "H8c") not in pending:
                        out.append(PendingSignal(
                            symbol=sym, side=Side.SELL, signal_date=today,
                            strategy=self.name, rule="H8c", qty=qty,
                            sell_ratio=1.0,
                            reason=(f"H8 连续{st['h8_streak']}个月末在"
                                    f"{self.ma_months}月均线下方,卖出剩余一半")))
                        log.warning("卖出信号 H8c %s 连续3个月末在均线下", sym)
                    continue
                profit = close / pos.avg_cost - 1 if pos.avg_cost > 0 else 0.0
                if profit < self.h8_profit_kept:
                    qty = pos.available_qty
                    if self.market == "cn":
                        qty -= qty % 100
                    if qty > 0 and (sym, "H8a") not in pending:
                        out.append(PendingSignal(
                            symbol=sym, side=Side.SELL, signal_date=today,
                            strategy=self.name, rule="H8a", qty=qty,
                            sell_ratio=1.0,
                            reason=(f"H8 月末收盘{close:.2f}<{self.ma_months}"
                                    f"月均线{ma:.2f},浮盈{profit:.0%}<"
                                    f"{self.h8_profit_kept:.0%},全部卖出"
                                    "(论点卡保留,站上均线直接恢复)")))
                        log.warning("卖出信号 H8a %s 跌破均线且浮盈%.0f%%,全卖",
                                    sym, profit * 100)
                    continue
                # 浮盈≥30%:先卖一半(税),连续3个月末在均线下再卖另一半
                qty = pos.available_qty // 2
                if self.market == "cn":
                    qty -= qty % 100
                if qty > 0 and (sym, "H8b") not in pending:
                    out.append(PendingSignal(
                        symbol=sym, side=Side.SELL, signal_date=today,
                        strategy=self.name, rule="H8b", qty=qty,
                        sell_ratio=round(qty / max(pos.qty, 1), 5),
                        reason=(f"H8 月末收盘{close:.2f}<{self.ma_months}"
                                f"月均线{ma:.2f},浮盈{profit:.0%}≥"
                                f"{self.h8_profit_kept:.0%},先卖一半省税;"
                                f"连续{self.h8_streak_months}个月末在均线下"
                                "再卖另一半")))
                    st["h8_partial"] = True
                    st["h8_streak"] = st.get("h8_streak", 0) + 1
                    self._save(sym, st)
                    log.warning("卖出信号 H8b %s 跌破均线,浮盈%.0f%%先卖一半",
                                sym, profit * 100)
                elif (sym, "H8b") in pending:
                    pass
                else:
                    st["h8_streak"] = st.get("h8_streak", 0) + 1
                    self._save(sym, st)

        # ---- H9:不止盈,只在季度调仓日把超限部分减回 ----
        if ctx.quarter_day:
            attack_pct_total = self._attack_mv(ctx) / equity
            scale = None
            if attack_pct_total > self.total_trim_at:
                scale = self.total_trim_to / attack_pct_total
                log.warning("H9 进攻仓整体%.0f%%>%.0f%%,按比例减回%.0f%%",
                            attack_pct_total * 100, self.total_trim_at * 100,
                            self.total_trim_to * 100)
            for sym, pos in held.items():
                if (sym, "H9") in pending:
                    continue
                px = ctx.price(sym)
                pct = pos.qty * px / equity
                target_pct = pct * scale if scale is not None else pct
                target_pct = min(target_pct, self.per_name_trim_to)
                if pct <= self.per_name_trim_at and scale is None:
                    continue
                if pct <= target_pct:
                    continue
                qty = int((pct - target_pct) * equity / px)
                if self.market == "cn":
                    qty -= qty % 100
                if qty <= 0:
                    continue
                reasons = []
                if pct > self.per_name_trim_at:
                    reasons.append(f"单只{pct:.1%}>{self.per_name_trim_at:.0%}"
                                   f"减回{self.per_name_trim_to:.0%}")
                if scale is not None:
                    reasons.append(f"进攻仓整体{attack_pct_total:.0%}>"
                                    f"{self.total_trim_at:.0%}减回"
                                    f"{self.total_trim_to:.0%}")
                out.append(PendingSignal(
                    symbol=sym, side=Side.SELL, signal_date=today,
                    strategy=self.name, rule="H9", qty=qty,
                    sell_ratio=round(qty / max(pos.qty, 1), 5),
                    reason=("H9 减回(不是止盈,赢家继续拿着):"
                            + ";".join(reasons))))
                log.warning("卖出信号 H9 %s:%s", sym, ";".join(reasons))
        return out

    # ---------------- 成交回调 ----------------
    def on_fill(self, sig: PendingSignal, filled_qty: int, ok: bool,
                price: float = 0.0) -> None:
        if sig.strategy != self.name:
            return
        sym = sig.symbol
        st = self._st(sym)
        acc = self._acc()
        amount = filled_qty * price if price > 0 else 0.0
        if ok and sig.side == Side.BUY and amount > 0:
            st["invested"] = float(st.get("invested", 0.0)) + amount
            acc["invested"] = float(acc.get("invested", 0.0)) + amount
        if sig.side == Side.BUY:
            if not ok:
                pass                      # 档位不变,信号次日重试或作废
            elif sig.rule == "H5a":
                st.update({"step": 1, "step1_price": price,
                           "step1_date": sig.signal_date})
            elif sig.rule in ("H5b", "H8r"):
                st.update({"step": 2, "h8_partial": False, "h8_exited": False})
        else:
            if not ok:
                if sig.rule == "H8b":
                    st["h8_partial"] = False   # 卖单失败回退,次日重试
            elif sig.rule in ("H8a", "H8c"):
                st.update({"step": 0, "h8_partial": False, "h8_exited": True,
                           "h8_streak": 0})
            elif sig.rule == "H7":
                st.update({"step": 0, "h8_partial": False, "h8_exited": False,
                           "h8_streak": 0, "gone_since": None})
            elif sig.rule == "H8b":
                st["h8_partial"] = True
        self._save(sym, st)
        self._save_acc(acc)

    # ---------------- 展示 ----------------
    def snapshot(self, ctx: Context | None = None) -> dict:
        acc = self._acc()
        return {"invested": acc.get("invested", 0.0),
                "isolation_until": acc.get("isolation_until", ""),
                "isolation_review": acc.get("review", ""),
                "persons": self.persons.snapshot(),
                "cards": {s: {"manual": self.cards.manual_conditions(s),
                             "risks": self.cards.risks(s)}
                          for s in self.cards.cards}}
