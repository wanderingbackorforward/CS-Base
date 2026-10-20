"""看板:FastAPI + 单页暗色前端(净值/持仓/信号流/热度榜,30 秒自刷新)。

页面纯静态,所有数据走 /api/*,前端 fetch 渲染——服务端零模板、零构建。
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .engine import Engine

log = logging.getLogger("qtrader.web")

_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QTrader 量化看板</title>
<style>
:root{--bg:#0b0f14;--card:#131a22;--line:#1e2833;--fg:#d7e2ee;--dim:#7c8a9a;
--acc:#4f9cf9;--up:#e14b4b;--dn:#2fbf71}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",monospace}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:20px;letter-spacing:2px}h1 small{color:var(--dim);font-weight:400;margin-left:10px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card .k{color:var(--dim);font-size:12px}.card .v{font-size:22px;margin-top:6px;font-weight:600}
.pos,.up{color:var(--up)}.neg,.dn{color:var(--dn)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px;margin:14px 0}
.panel h2{font-size:14px;color:var(--acc);margin:0 0 10px;letter-spacing:1px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 8px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:400}
.badge{display:inline-block;padding:1px 7px;border-radius:4px;font-size:11px;
background:#1d2a3a;color:var(--acc);margin-right:6px}
.watch{background:#3a2a12;color:#f5b64c}
#chart{width:100%;height:300px}
.empty{color:var(--dim);padding:20px;text-align:center}
.toast{position:fixed;right:16px;bottom:16px;background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:10px 14px;font-size:12px;
color:var(--dim);display:none}
</style></head><body><div class="wrap">
<h1>QTRADER<small>纸面交易看板 · A股/港股</small></h1>
<div class="cards">
<div class="card"><div class="k">净值</div><div class="v" id="equity">--</div></div>
<div class="card"><div class="k">现金</div><div class="v" id="cash">--</div></div>
<div class="card"><div class="k">持仓市值</div><div class="v" id="mv">--</div></div>
<div class="card"><div class="k">今日盈亏</div><div class="v" id="daypnl">--</div></div>
<div class="card"><div class="k">总盈亏</div><div class="v" id="pnl">--</div></div>
<div class="card"><div class="k">状态</div><div class="v" style="font-size:14px" id="status">--</div></div>
</div>
<div class="panel"><h2>净值曲线</h2><div id="chart"><div class="empty">加载中…</div></div></div>
<div class="panel"><h2>待执行信号(收盘后生成,次日开盘撮合 X1)</h2><div id="signals"></div></div>
<div class="panel"><h2>当前持仓</h2><div id="positions"></div></div>
<div class="panel"><h2>档位状态机(G3-G8 各份生命周期)</h2><div id="tranches"></div></div>
<div class="panel"><h2>成交记录(每笔带规则编号,无编号=违规)</h2><div id="orders"></div></div>
<div class="toast" id="toast"></div></div>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"
 onerror="window.__noecharts=1"></script>
<script>
const $=id=>document.getElementById(id);
const fmt=(x,d=0)=>x==null||isNaN(x)?'--':Number(x).toLocaleString('zh-CN',
 {minimumFractionDigits:d,maximumFractionDigits:d});
const cls=x=>x>0?'up':(x<0?'dn':'');
function row(cells){return '<tr>'+cells.map(c=>`<td class="${c[1]||''}">${c[0]}</td>`).join('')+'</tr>'}
let chart=null;
function drawEquity(pts){
 if(window.__noecharts||typeof echarts==='undefined'){$('chart').innerHTML=
  '<div class="empty">图表库(CDN)不可达,仅表格模式。共 '+pts.length+' 个净值点</div>';return}
 chart=echarts.init($('chart'));
 chart.setOption({backgroundColor:'transparent',
  tooltip:{trigger:'axis'},grid:{left:60,right:20,top:10,bottom:26},
  xAxis:{type:'category',data:pts.map(p=>p.ts),axisLine:{lineStyle:{color:'#2a3644'}},
   axisLabel:{color:'#7c8a9a'}},
  yAxis:{type:'value',scale:true,axisLabel:{color:'#7c8a9a'},
   splitLine:{lineStyle:{color:'#1a232d'}}},
  series:[{type:'line',data:pts.map(p=>p.equity),symbol:'none',
   lineStyle:{color:'#4f9cf9',width:2},
   areaStyle:{color:'rgba(79,156,249,.12)'}}]});
}
async function refresh(){
 try{
  const s=await (await fetch('/api/state')).json();
  $('equity').textContent=fmt(s.equity,0);
  $('cash').textContent=fmt(s.cash,0);
  $('mv').textContent=fmt(s.market_value,0);
  const dp=s.day_pnl; $('daypnl').textContent=(dp>0?'+':'')+fmt(dp,0);
  $('daypnl').className='v '+cls(dp);
  const p=s.total_pnl; $('pnl').textContent=(p>0?'+':'')+fmt(p,0)+' ('+fmt(p/s.equity*100,1)+'%)';
  $('pnl').className='v '+cls(p);
  $('status').textContent=s.killed?'已熔断(人工复位)':(s.cooldown?'冷却中':'运行中');
  $('status').style.color=s.killed?'var(--up)':(s.cooldown?'#f5b64c':'var(--dn)');
  const eq=await (await fetch('/api/equity')).json();
  drawEquity(eq);
  const pos=s.positions.map(p=>row([[p.symbol,''],['badge',p.strategy],
   [p.qty+' × '+fmt(p.avg_cost,2),''],
   [(p.unrealized>0?'+':'')+fmt(p.unrealized,0),cls(p.unrealized)],
   [(p.unrealized_pct>0?'+':'')+fmt(p.unrealized_pct*100,1)+'%',cls(p.unrealized_pct)],
   [fmt(p.stop_price,2),''],[p.days_held,'']]));
  $('positions').innerHTML=s.positions.length?'<table><tr><th>标的</th><th>策略</th>'+
   '<th>数量×成本</th><th>浮盈</th><th>盈亏%</th><th>止损线</th><th>持有日</th></tr>'+
   pos.join('')+'</table>':'<div class="empty">空仓(等信号)</div>';
  const sg=(s.signals||[]).map(g=>row([[g.signal_date.slice(5),''],
   ['<span class="badge">'+g.rule+'</span>',''],
   [g.side==='BUY'?'买入':'卖出',g.side==='BUY'?'up':'dn'],
   [g.symbol,''],[g.note||'挂单待撮合',g.note?'dn':'']]));
  $('signals').innerHTML=sg.length?'<table><tr><th>信号日</th><th>规则</th><th>方向</th>'+
   '<th>标的</th><th>备注</th></tr>'+sg.join('')+'</table>':
   '<div class="empty">无待执行信号(等收盘后15:10生成)</div>';
  const tr=(s.tranches||[]).map(t=>row([[t.symbol,''],
   ['<span class="badge">'+t.tranche+'</span>',''],[t.state,''],
   [fmt(t.trigger_high,2),''],[t.qty,''],
   [t.needs_review?'⚠ '+t.review_note:(t.rule_hint||''),'',t.needs_review?'dn':'']]));
  $('tranches').innerHTML=tr.length?'<table><tr><th>标的</th><th>档位</th><th>状态</th>'+
   '<th>触发前高</th><th>数量</th><th>备注</th></tr>'+tr.join('')+'</table>':
   '<div class="empty">全部档位空闲(等G3触发)</div>';
  const od=s.orders.map(o=>row([[o.ts.slice(5),''],
   [o.side==='BUY'?'买入':'卖出',o.side==='BUY'?'up':'dn'],
   [o.symbol,''],[o.status==='FILLED'?fmt(o.filled_qty):'-',''],
   [o.status==='FILLED'?fmt(o.filled_price,3):'-',''],[fmt(o.fee,2),''],
   [o.reason,o.status==='REJECTED'?'dn':'']]));
  $('orders').innerHTML=s.orders.length?'<table><tr><th>时间</th><th>方向</th><th>标的</th>'+
   '<th>数量</th><th>价格</th><th>费用</th><th>理由</th></tr>'+od.join('')+'</table>':
   '<div class="empty">还没有成交</div>';
 }catch(e){
  $('toast').style.display='block';$('toast').textContent='刷新失败: '+e;
 }
}
refresh();setInterval(refresh,30000);setInterval(()=>{if(chart)chart.resize()},1000);
</script></body></html>"""


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="QTrader", docs_url=None, redoc_url=None)
    store = engine.store

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    def _state() -> dict[str, Any]:
        acc = store.load_account()
        positions = store.all_positions()
        quote = engine._quotes_for(list(positions))
        mv = engine._market_value(positions, quote=quote)
        equity = acc.equity(mv) if acc else 0.0
        day_pnl = (equity - acc.day_start_equity) if acc else 0.0
        rows = []
        for sym, p in positions.items():
            price = 0.0
            if len(quote):
                r = quote[quote["symbol"] == sym]
                if len(r):
                    price = float(r.iloc[0]["price"])
            price = price or p.avg_cost
            unreal = (price - p.avg_cost) * p.qty
            rows.append({
                "symbol": sym, "strategy": p.strategy, "qty": p.qty,
                "avg_cost": round(p.avg_cost, 3), "price": round(price, 3),
                "unrealized": round(unreal, 0),
                "unrealized_pct": round(price / p.avg_cost - 1, 4) if p.avg_cost else 0,
                "stop_price": round(p.stop_price, 2), "days_held": p.days_held,
            })
        orders = []
        for o in store.recent_orders(60):
            d = o.to_dict()
            d["reason"] = (o.reason or "")[:120]
            orders.append(d)
        signals = []
        for g in store.pending_signals():
            d = g.to_dict()
            d["reason"] = (g.reason or "")[:120]
            signals.append(d)
        tranches = []
        for sym, st in store.all_pool_states().items():
            if st.get("needs_review"):
                tranches.append({"symbol": sym, "tranche": "-", "state": "复核",
                                 "trigger_high": 0, "qty": 0,
                                 "needs_review": True,
                                 "review_note": str(st.get("review_note", ""))[:80]})
                continue
            for key, tr in (st.get("tranches") or {}).items():
                if tr.get("state") not in ("available",):
                    tranches.append({"symbol": sym, "tranche": key,
                                     "state": tr.get("state", ""),
                                     "trigger_high": tr.get("trigger_high", 0),
                                     "qty": tr.get("qty", 0),
                                     "needs_review": False, "rule_hint": ""})
        return {
            "equity": round(equity, 0), "cash": round(acc.cash, 0) if acc else 0,
            "market_value": round(mv, 0), "day_pnl": round(day_pnl, 0),
            "total_pnl": round(equity - engine.capital, 0),
            "killed": bool(acc.killed) if acc else False,
            "cooldown": store.get_meta("cooldown_until"),
            "positions": rows, "orders": orders,
            "signals": signals, "tranches": tranches,
        }

    @app.get("/api/state")
    def api_state() -> JSONResponse:
        return JSONResponse(_state())

    @app.get("/api/equity")
    def api_equity() -> JSONResponse:
        return JSONResponse(store.equity_curve())

    @app.post("/api/reset")
    def api_reset() -> dict:
        engine.reset()
        return {"ok": True}

    return app
