"""人物名单 P1-P5 与论点卡 H3 的加载/校验(v5)。

名人不再是照抄对象,只是找好公司的线索:
- 投资人:13F 前10大持仓(SEC EDGAR,按实际公布日生效);
- 掌舵人:正在经营的上市公司(创始人任CEO/董事长,或连续任CEO满5年);
- A股没有13F,只用掌舵人这一类。

P1 不选以加密货币为主业/用自己发行资产拉盘的人;
P2 欺诈定罪/认罪/禁任高管 -> 排除;和解未认罪 -> 不排除,风险必须写进论点卡;
P3 掌舵人近12个月净卖出 > 本人持股20% -> 该公司移出候选(已持仓的按 H7/P4 逻辑走);
P4 掌舵人离任/投资人清仓 = 来源消失,满2个季度按 H7 卖出;
P5 名单只在季度调仓日改,下一季度生效。

论点卡(H3)是买入前置条件:没写卡的候选,进攻仓绝不买。
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

log = logging.getLogger("qtrader.persons")


def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        log.error("读取 %s 失败: %s", path, e)
        return {}


class Persons:
    """人物名单:只提供候选来源,买不买由 H2/H3/H5 决定。"""

    def __init__(self, path: str) -> None:
        self.path = path
        data = _load_yaml(path)
        self.persons: list[dict] = list(data.get("persons") or [])
        self._by_company: dict[str, dict] = {}
        for p in self.persons:
            for comp in (p.get("company") or "").split(","):
                comp = comp.strip()
                if comp:
                    self._by_company[comp] = p
            for h in (p.get("top_holdings") or []):
                self._by_company.setdefault(h, p)

    # ---------------- 查询 ----------------
    def by_id(self, pid: str) -> dict | None:
        return next((p for p in self.persons if p.get("id") == pid), None)

    def source_of(self, symbol: str) -> dict | None:
        """给候选找来源(H1):掌舵人的公司,或投资人13F前10大。"""
        return self._by_company.get(symbol)

    # ---------------- P1/P2/P3/P4 判定 ----------------
    def eligible(self, person: dict, today: str) -> tuple[bool, str]:
        """这个人现在还能不能当来源;返回 (可用, 原因)。"""
        pid = person.get("id", "?")
        if not person.get("verified", False):
            log.info("P 待核实:%s(%s)——仍可作为来源,但请核实名单条件", pid,
                     person.get("name"))
        if person.get("crypto_native"):            # P1
            return False, f"P1 排除:{pid} 以加密货币为主业或拉盘"
        if person.get("banned"):                  # P2 定罪/认罪/禁任
            return False, f"P2 排除:{pid} 有欺诈定罪/认罪或被禁任高管"
        if person.get("gone"):                    # P4 来源消失
            return False, f"P4 来源消失:{pid} 离任/清仓(满2个季度按H7卖出)"
        return True, ""

    def net_sell_blocked(self, symbol: str, ratio: float = 0.20) -> bool:
        """P3:掌舵人近12个月净卖出超过本人持股的20% -> 移出候选。"""
        p = self.source_of(symbol)
        if not p or p.get("type") != "helmsman":
            return False
        try:
            return float(p.get("net_sell_12m") or 0.0) > ratio
        except (TypeError, ValueError):
            return False

    def settlements_of(self, symbol: str) -> list[str]:
        """P2:和解未认罪的记录 -> 必须出现在论点卡的风险里。"""
        p = self.source_of(symbol)
        return list(p.get("settlements") or []) if p else []

    def candidates_of(self, today: str) -> list[str]:
        """当前所有可用来源对应的标的(供候选池标注)。"""
        out = []
        for comp, p in self._by_company.items():
            ok, _ = self.eligible(p, today)
            if ok:
                out.append(comp)
        return out

    def snapshot(self) -> list[dict]:
        return self.persons


class ThesisCards:
    """论点卡(H3):买入前置;机器规则系统判,人工规则季度提醒。"""

    AUTO_RULES = ("ocf_4q_negative", "profit_4q_negative", "out_of_pool")

    def __init__(self, path: str) -> None:
        self.path = path
        self.cards: dict[str, dict] = _load_yaml(path)

    def get(self, symbol: str) -> dict | None:
        return self.cards.get(symbol)

    def has_card(self, symbol: str) -> bool:
        c = self.cards.get(symbol)
        return bool(c and c.get("thesis"))

    def auto_rules(self, symbol: str) -> list[str]:
        c = self.cards.get(symbol) or {}
        return [r for r in (c.get("sell_conditions") or {}).get("auto") or []
                if r in self.AUTO_RULES]

    def manual_conditions(self, symbol: str) -> list[str]:
        c = self.cards.get(symbol) or {}
        return list((c.get("sell_conditions") or {}).get("manual") or [])

    def risks(self, symbol: str) -> str:
        return str((self.cards.get(symbol) or {}).get("risks") or "")

    def card_ok_and_risks_written(self, symbol: str,
                                  persons: Persons | None = None) -> tuple[bool, str]:
        """H3:卡存在且 P2 和解风险已写进卡;返回 (通过, 说明)。"""
        c = self.cards.get(symbol)
        if not c or not c.get("thesis"):
            return False, f"H3 未写论点卡:{symbol} 在写卡前不买"
        risks = str(c.get("risks") or "").strip()
        if persons is not None:
            st = persons.settlements_of(symbol)
            if st and not risks:
                return False, (f"H3 {symbol} 的来源有和解记录({st[0][:40]}…),"
                               "必须写进论点卡 risks 后才能买")
        return True, ""
