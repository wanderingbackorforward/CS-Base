"""看板:密码登录 + A股/美股双账户单页暗色前端(30秒自刷新)。

安全模型(美股准则:系统只算信号/发提醒/记日志,下单人工确认):
- 登录:POST /api/login 校验 config.yaml server.password(常数时间比较),
  成功颁发随机 token(内存保存,重启失效);此后所有 /api/* 走
  Authorization: Bearer <token>,401 时前端弹回登录页;
- 暴力破解缓解:同一 IP 连错 5 次锁 60 秒;
- 页面纯静态,数据全走 /api/*,前端 fetch 渲染——服务端零模板、零构建。

双账户:A股(CNY,./data)与美股(USD,./data_us)状态完全隔离,
各自展示净值/信号/持仓/档位状态机/成交记录;美股额外展示
三仓占比(核心70/卫星20/现金10)与 USD/CNY 报税参考。
"""
from __future__ import annotations

import hmac
import logging
import secrets
import time
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .engine import Engine
from .models import Side
from .us_engine import UsEngine

log = logging.getLogger("qtrader.web")

_TOKENS: set[str] = set()
_LOGIN_FAILS: dict[str, tuple[int, float]] = {}

_STATE_LABEL = {"available": "空闲", "pending": "待成交", "held": "持有",
                "selling": "卖出中", "selling_23": "卖2/3中",
                "partial": "留1/3止盈中", "selling_last": "卖尾仓中",
                "deployed": "已部署"}


class LoginBody(BaseModel):
    password: str


class FillBody(BaseModel):
    signal_id: str
    price: float
    qty: int
    fee: float = 0.0
    mood: int = 0


class SkipBody(BaseModel):
    signal_id: str
    note: str = ""


class CashBody(BaseModel):
    market: str = "cn"
    delta: float = 0.0
    set_to: Optional[float] = None
    note: str = ""


_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QTrader 双账户看板</title>
<style>
:root{--bg:#0b0f14;--card:#131a22;--line:#1e2833;--fg:#d7e2ee;--dim:#7c8a9a;
--acc:#4f9cf9;--up:#e14b4b;--dn:#2fbf71;--warn:#f5b64c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",monospace}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 60px}
.head{display:flex;align-items:center;justify-content:space-between}
h1{font-size:20px;letter-spacing:2px;margin:0}h1 small{color:var(--dim);
font-weight:400;margin-left:10px;font-size:13px}
.logout{color:var(--dim);cursor:pointer;font-size:12px;border:1px solid var(--line);
padding:3px 10px;border-radius:6px}
.logout:hover{color:var(--fg)}
.tabs{display:flex;gap:8px;margin:16px 0}
.tab{flex:0 0 auto;padding:8px 24px;border:1px solid var(--line);border-radius:8px;
background:var(--card);color:var(--dim);cursor:pointer;font-size:14px}
.tab.active{color:#fff;border-color:var(--acc);background:#16222f}
.tab .cur{font-size:11px;margin-left:6px;opacity:.8}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:12px;margin:16px 0}
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
.badge.core{background:#1a2c46;color:#7db4f7}
.badge.sat{background:#3a2a12;color:#f5b64c}
.badge.park{background:#12331f;color:#4fd08e}
.chart{width:100%;height:300px}
.empty{color:var(--dim);padding:20px;text-align:center}
.hint{color:var(--dim);font-size:12px;margin-top:8px}
.alloc-bar{position:relative;display:flex;height:28px;border-radius:6px;overflow:hidden;
background:#1a232d;border:1px solid var(--line);margin-bottom:12px}
.seg{height:100%;min-width:0;transition:width .4s}
.seg-core{background:linear-gradient(180deg,#5aa8ff,#3d84d6)}
.seg-sat{background:linear-gradient(180deg,#f7c55a,#d99a2b)}
.seg-cash{background:linear-gradient(180deg,#3fd28a,#27995f)}
.mark{position:absolute;top:-3px;bottom:-3px;width:0;
border-left:1px dashed rgba(255,255,255,.55)}
.toast{position:fixed;right:16px;bottom:16px;background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:10px 14px;font-size:12px;
color:var(--dim);display:none;z-index:100}
.overlay{position:fixed;inset:0;background:rgba(5,8,12,.94);display:none;
align-items:center;justify-content:center;z-index:99}
.login-card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:28px 32px;width:330px;text-align:center}
.login-card h2{margin:0 0 6px;font-size:18px;letter-spacing:1px}
.login-card p{color:var(--dim);font-size:12px;margin:0 0 16px}
.login-card input{width:100%;padding:10px 12px;border-radius:8px;
border:1px solid var(--line);background:#0d141c;color:var(--fg);
font-size:15px;letter-spacing:2px;outline:none}
.login-card input:focus{border-color:var(--acc)}
.login-card button{width:100%;margin-top:12px;padding:10px;border:none;
border-radius:8px;background:var(--acc);color:#fff;font-size:15px;cursor:pointer}
.login-card button:hover{filter:brightness(1.1)}
#login-err{color:var(--up);font-size:12px;min-height:18px;margin-top:10px}
.mini{padding:2px 8px;border:1px solid var(--acc);background:transparent;
color:var(--acc);border-radius:4px;font-size:11px;cursor:pointer;margin-right:4px}
.mini:hover{background:#16222f}
.mini.warn{border-color:var(--warn);color:var(--warn)}
.mfield{margin-top:10px;text-align:left}
.mfield label{color:var(--dim);font-size:12px;display:block;margin-bottom:4px}
.mfield input{width:100%;padding:8px 10px;border-radius:6px;border:1px solid var(--line);
background:#0d141c;color:var(--fg);font-size:14px;outline:none;letter-spacing:0}
.mrow{display:flex;gap:10px;margin-top:14px}
.mrow button{flex:1;margin:0}
</style></head><body>
<div id="login" class="overlay"><div class="login-card">
<h2>QTRADER 看板</h2><p>A股 + 美股双账户 · 密码登录</p>
<input id="pwd" type="password" placeholder="访问密码" autocomplete="current-password">
<button id="btn-login">进入看板</button>
<div id="login-err"></div></div></div>
<div class="wrap">
<div class="head"><h1>QTRADER<small>A股 / 美股双账户 · 信号 · 下单 · 记账</small></h1>
<div><span class="logout" id="btn-cash" style="margin-right:8px" onclick="openCash()">入金/出金</span>
<span class="logout" id="btn-logout">退出登录</span></div></div>
<div class="tabs">
<button class="tab active" id="tab-cn" onclick="switchTab('cn')">A股<span class="cur">CNY</span></button>
<button class="tab" id="tab-us" onclick="switchTab('us')">美股<span class="cur">USD</span></button>
</div>

<div id="view-cn">
<div class="cards">
<div class="card"><div class="k">净值(CNY)</div><div class="v" id="equity-cn">--</div></div>
<div class="card"><div class="k">现金</div><div class="v" id="cash-cn">--</div></div>
<div class="card"><div class="k">持仓市值</div><div class="v" id="mv-cn">--</div></div>
<div class="card"><div class="k">今日盈亏</div><div class="v" id="daypnl-cn">--</div></div>
<div class="card"><div class="k">总盈亏</div><div class="v" id="pnl-cn">--</div></div>
<div class="card"><div class="k">状态</div><div class="v" style="font-size:14px" id="status-cn">--</div></div>
</div>
<div class="panel"><h2>净值曲线(CNY)</h2><div id="chart-cn" class="chart"></div></div>
<div class="panel"><h2>待执行信号(收盘后生成,次日开盘撮合 X1)</h2><div id="signals-cn"></div></div>
<div class="panel"><h2>当前持仓</h2><div id="positions-cn"></div></div>
<div class="panel"><h2>档位状态机(G3-G8 各份生命周期)</h2><div id="tranches-cn"></div></div>
<div class="panel"><h2>成交记录(每笔带规则编号,无编号=违规 D3)</h2><div id="orders-cn"></div></div>
</div>

<div id="view-us" style="display:none">
<div class="cards">
<div class="card"><div class="k">净值(USD)</div><div class="v" id="equity-us">--</div></div>
<div class="card"><div class="k">现金</div><div class="v" id="cash-us">--</div></div>
<div class="card"><div class="k">持仓市值</div><div class="v" id="mv-us">--</div></div>
<div class="card"><div class="k">总盈亏</div><div class="v" id="pnl-us">--</div></div>
<div class="card"><div class="k">今日盈亏</div><div class="v" id="daypnl-us">--</div></div>
<div class="card"><div class="k">信号基准日(美股收盘)</div><div class="v" style="font-size:15px" id="session-us">--</div></div>
</div>
<div class="panel"><h2>三仓占比(目标:核心 70% / 卫星 20% / 现金池 10%)</h2>
<div class="alloc-bar"><div class="seg seg-core" id="seg-core"></div>
<div class="seg seg-sat" id="seg-sat"></div>
<div class="seg seg-cash" id="seg-cash"></div>
<div class="mark" style="left:70%"></div><div class="mark" style="left:90%"></div></div>
<div id="alloc-legend"></div></div>
<div class="panel"><h2>报税参考(USD/CNY,每笔交易按人民币记账)</h2><div id="fx-us"></div></div>
<div class="panel"><h2>净值曲线(USD)</h2><div id="chart-us" class="chart"></div></div>
<div class="panel"><h2>待执行信号(美股收盘后生成,次日开盘撮合)</h2><div id="signals-us"></div></div>
<div class="panel"><h2>当前持仓</h2><div id="positions-us"></div></div>
<div class="panel"><h2>档位状态机(C4-C8 底仓与档位 / S6-S13 个股档位)</h2><div id="tranches-us"></div></div>
<div class="panel"><h2>成交记录(每笔带规则编号,无编号=违规 D3)</h2><div id="orders-us"></div></div>
</div>

<div class="toast" id="toast"></div></div>
<div id="modal" class="overlay"><div class="login-card" style="text-align:left;width:360px">
<h2 id="modal-title" style="font-size:15px;text-align:center"></h2>
<div id="modal-body"></div>
<div class="mrow"><button id="modal-ok">确定</button>
<button id="modal-cancel" style="background:#1d2a3a">取消</button></div>
</div></div>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"
 onerror="window.__noecharts=1"></script>
<script>
const $=id=>document.getElementById(id);
let TOKEN=localStorage.getItem('qt_token')||'';
let mkt='cn';
const charts={};
const MODE={cn:'paper',us:'paper'};
const fmt=(x,d=0)=>x==null||isNaN(x)?'--':Number(x).toLocaleString('zh-CN',
 {minimumFractionDigits:d,maximumFractionDigits:d});
const cls=x=>x>0?'up':(x<0?'dn':'');
function row(cells){return '<tr>'+cells.map(c=>`<td class="${c[1]||''}">${c[0]}</td>`).join('')+'</tr>'}
function showLogin(){$('login').style.display='flex'}
function hideLogin(){$('login').style.display='none'}
function toast(msg){$('toast').style.display='block';$('toast').textContent=msg;
 setTimeout(()=>{$('toast').style.display='none'},4000)}
async function api(path){
 const r=await fetch(path,{headers:{'Authorization':'Bearer '+TOKEN}});
 if(r.status===401){showLogin();throw new Error('unauthorized')}
 if(!r.ok)throw new Error('http '+r.status);
 return r.json();
}
async function postApi(path,body){
 const r=await fetch(path,{method:'POST',
  headers:{'Authorization':'Bearer '+TOKEN,'Content-Type':'application/json'},
  body:JSON.stringify(body)});
 if(r.status===401){showLogin();throw new Error('unauthorized')}
 if(!r.ok){const d=await r.json().catch(()=>({}));throw new Error(d.msg||d.detail||('http '+r.status))}
 return r.json();
}
let _modalSubmit=null;
function showModal(title,fields,onOk){
 $('modal-title').textContent=title;
 const b=$('modal-body');b.innerHTML='';
 fields.forEach(f=>{
  const d=document.createElement('div');d.className='mfield';
  const l=document.createElement('label');l.textContent=f.label;
  const i=document.createElement('input');i.type=f.type||'text';
  if(f.value!=null)i.value=f.value;i.placeholder=f.ph||'';
  d.appendChild(l);d.appendChild(i);b.appendChild(d);f._el=i;});
 $('modal').style.display='flex';
 _modalSubmit=async()=>{
  const v={};fields.forEach(f=>{v[f.key]=f._el.value.trim()});
  let ok=true;try{ok=await onOk(v)}catch(e){toast('失败: '+e.message);ok=false}
  if(ok!==false){$('modal').style.display='none';refresh()}};
}
function openFill(sid,sym,side){
 showModal('回填真实成交 · '+sym+' '+side,[
  {key:'price',label:'实际成交价(以券商App成交回报为准)',type:'number'},
  {key:'qty',label:'成交数量(股)',type:'number'},
  {key:'fee',label:'实际费用(佣金等,可填0)',type:'number',value:'0'},
  {key:'mood',label:'下单时情绪1-5(D2纪律,可空)',type:'number',ph:'如 3'}],
  async v=>{
   if(!v.price||!v.qty){toast('成交价与数量必填');return false}
   const r=await postApi('/api/fill',{signal_id:sid,price:+v.price,
    qty:+v.qty,fee:+(v.fee||0),mood:+(v.mood||0)});
   toast(r.ok?'已入账: '+r.msg:'回填失败: '+r.msg);return r.ok});
}
async function doSkip(sid){
 try{const r=await postApi('/api/skip',{signal_id:sid,note:'看板人工放弃'});
  toast(r.msg);refresh()}catch(e){toast('失败: '+e.message)}
}
function openCash(){
 const m=mkt;
 showModal((m==='us'?'美股账户(USD)':'A股账户(CNY)')+' 资金校准',[
  {key:'delta',label:'入金为正 / 出金为负',type:'number',ph:'如 5000 或 -2000'},
  {key:'set',label:'或直接校准到该现金绝对值(优先)',type:'number',ph:'券商App可用资金'},
  {key:'note',label:'备注',type:'text',ph:'如 入金'}],
  async v=>{
   if(v.delta===''&&v.set===''){toast('至少填一项');return false}
   const r=await postApi('/api/cash',{market:m,
    delta:v.set!==''?0:+(v.delta||0),set_to:v.set!==''?+v.set:null,note:v.note});
   toast(r.ok?r.msg:'失败: '+r.msg);return r.ok});
}
async function doLogin(){
 const pwd=$('pwd').value;$('login-err').textContent='';
 try{
  const r=await fetch('/api/login',{method:'POST',
   headers:{'Content-Type':'application/json'},
   body:JSON.stringify({password:pwd})});
  if(r.ok){TOKEN=(await r.json()).token;
   localStorage.setItem('qt_token',TOKEN);hideLogin();boot();}
  else{const d=await r.json().catch(()=>({}));$('login-err').textContent=d.detail||'密码错误'}
 }catch(e){$('login-err').textContent='网络错误: '+e}
}
function doLogout(){TOKEN='';localStorage.removeItem('qt_token');
 charts.cn=null;charts.us=null;showLogin()}
function switchTab(m){mkt=m;
 $('tab-cn').classList.toggle('active',m==='cn');
 $('tab-us').classList.toggle('active',m==='us');
 $('view-cn').style.display=m==='cn'?'':'none';
 $('view-us').style.display=m==='us'?'':'none';
 const c=charts[m];if(c)c.resize();}
function drawEquity(m,pts){
 const el=$('chart-'+m);
 if(window.__noecharts||typeof echarts==='undefined'){el.innerHTML=
  '<div class="empty">图表库(CDN)不可达,仅表格模式。共 '+pts.length+' 个净值点</div>';return}
 charts[m]=charts[m]||echarts.init(el);
 charts[m].setOption({backgroundColor:'transparent',
  tooltip:{trigger:'axis'},grid:{left:64,right:20,top:10,bottom:26},
  xAxis:{type:'category',data:pts.map(p=>p.ts),axisLine:{lineStyle:{color:'#2a3644'}},
   axisLabel:{color:'#7c8a9a'}},
  yAxis:{type:'value',scale:true,axisLabel:{color:'#7c8a9a',
   formatter:v=>m==='us'?'$'+fmt(v,0):fmt(v,0)},
   splitLine:{lineStyle:{color:'#1a232d'}}},
  series:[{type:'line',data:pts.map(p=>p.equity),symbol:'none',
   lineStyle:{color:'#4f9cf9',width:2},
   areaStyle:{color:'rgba(79,156,249,.12)'}}]});
}
function renderSignals(s,m,cur){
 const live=MODE[m]==='live';
 const sg=(s.signals||[]).map(g=>{
  const ops=live?('<button class="mini" onclick="openFill(\''+g.signal_id+
   '\',\''+g.symbol+'\',\''+(g.side==='BUY'?'买入':'卖出')+'\')">回填成交</button>'+
   '<button class="mini warn" onclick="doSkip(\''+g.signal_id+'\')">放弃</button>'):'';
  return row([[g.signal_date.slice(5),''],
  ['<span class="badge">'+(g.rule||'-')+'</span>',''],
  [g.side==='BUY'?'买入':'卖出',g.side==='BUY'?'up':'dn'],
  [g.symbol,''],
  [g.est_amount?cur+fmt(g.est_amount,0):'--',''],
  [g.note||(g.reason||'').slice(0,50)||(live?'等你在券商App下单后回填':'挂单待撮合'),
   g.note?'warn':''],
  [ops,'']])});
 const banner=live?'<div class="hint" style="color:var(--warn)">实盘模式:去券商App按信号下单,成交后点「回填成交」录入实际价/量/费,系统按真实资金流记账</div>':'';
 return banner+(sg.length?'<table><tr><th>信号日</th><th>规则</th><th>方向</th>'+
  '<th>标的</th><th>预估金额</th><th>备注</th>'+(live?'<th>操作</th>':'')+'</tr>'+
  sg.join('')+'</table>':
  '<div class="empty">无待执行信号(收盘后按规则生成,没有信号就不打扰)</div>');
}
function renderTranches(s){
 const tr=(s.tranches||[]).map(t=>row([[t.symbol,''],
  ['<span class="badge">'+(t.tranche||'-')+'</span>',''],
  [t.state_label||t.state||'',''],
  [t.trigger_high>0?fmt(t.trigger_high,2):'--',''],
  [t.qty>0?fmt(t.qty):'--',''],
  [t.needs_review?'⚠ '+(t.review_note||''):(t.extra||t.rule_hint||''),
   t.needs_review?'dn':'']]));
 return tr.length?'<table><tr><th>标的</th><th>档位</th><th>状态</th>'+
  '<th>触发前高</th><th>数量</th><th>备注</th></tr>'+tr.join('')+'</table>':
  '<div class="empty">全部档位空闲(等回调触发)</div>';
}
function renderOrders(s,cur){
 const od=(s.orders||[]).map(o=>row([[o.ts.slice(5),''],
  [o.side==='BUY'?'买入':'卖出',o.side==='BUY'?'up':'dn'],
  [o.symbol,''],[o.status==='FILLED'?fmt(o.filled_qty):'-',''],
  [o.status==='FILLED'?fmt(o.filled_price,3):'-',''],
  [cur+fmt(o.fee,2),''],
  [(o.reason||'').slice(0,80),o.status==='REJECTED'?'dn':'']]));
 return od.length?'<table><tr><th>时间</th><th>方向</th><th>标的</th>'+
  '<th>数量</th><th>价格</th><th>费用</th><th>理由</th></tr>'+od.join('')+'</table>':
  '<div class="empty">还没有成交</div>';
}
function renderCN(s){
 $('equity-cn').textContent='¥'+fmt(s.equity,0);
 $('cash-cn').textContent='¥'+fmt(s.cash,0);
 $('mv-cn').textContent='¥'+fmt(s.market_value,0);
 const dp=s.day_pnl;$('daypnl-cn').textContent=(dp>0?'+':'')+'¥'+fmt(dp,0);
 $('daypnl-cn').className='v '+cls(dp);
 const p=s.total_pnl;
 $('pnl-cn').textContent=(p>0?'+':'')+'¥'+fmt(p,0)+' ('+fmt(p/s.equity*100,1)+'%)';
 $('pnl-cn').className='v '+cls(p);
 $('status-cn').textContent=(MODE.cn==='live'?'实盘·人工下单 · ':'')+
  (s.killed?'已熔断(人工复位)':(s.cooldown?'冷却中':'运行中'));
 $('status-cn').style.color=s.killed?'var(--up)':(s.cooldown?'var(--warn)':'var(--dn)');
 const pos=(s.positions||[]).map(p=>row([[p.symbol,''],
  ['<span class="badge">'+p.strategy+'</span>',''],
  [p.qty+' × '+fmt(p.avg_cost,2)+' @'+fmt(p.price,2),''],
  [(p.unrealized>0?'+':'')+'¥'+fmt(p.unrealized,0),cls(p.unrealized)],
  [(p.unrealized_pct>0?'+':'')+fmt(p.unrealized_pct*100,1)+'%',cls(p.unrealized_pct)],
  [fmt(p.stop_price,2),''],[p.days_held,'']]));
 $('positions-cn').innerHTML=pos.length?'<table><tr><th>标的</th><th>策略</th>'+
  '<th>数量×成本@现价</th><th>浮盈</th><th>盈亏%</th><th>止损线</th><th>持有日</th></tr>'+
  pos.join('')+'</table>':'<div class="empty">空仓(等信号)</div>';
 $('signals-cn').innerHTML=renderSignals(s,'cn','¥');
 $('tranches-cn').innerHTML=renderTranches(s);
 $('orders-cn').innerHTML=renderOrders(s,'¥');
}
function renderUS(s){
 if(s.disabled){
  ['signals-us','positions-us','tranches-us','orders-us','fx-us'].forEach(
   id=>{$(id).innerHTML='<div class="empty">美股板块未启用(config.yaml us.enabled)</div>'});
  return}
 $('equity-us').textContent='$'+fmt(s.equity,0);
 $('cash-us').textContent='$'+fmt(s.cash,0);
 $('mv-us').textContent='$'+fmt(s.market_value,0);
 const dp=s.day_pnl;$('daypnl-us').textContent=(dp>0?'+':'')+'$'+fmt(dp,0);
 $('daypnl-us').className='v '+cls(dp);
 const p=s.total_pnl;
 $('pnl-us').textContent=(p>0?'+':'')+'$'+fmt(p,0)+' ('+fmt(p/s.equity*100,1)+'%)';
 $('pnl-us').className='v '+cls(p);
 $('session-us').textContent=(MODE.us==='live'?'实盘·人工下单 · ':'')+
  (s.session||'未生成(等06:10信号周期)');
 const a=s.alloc||{};
 $('seg-core').style.width=((a.core_pct||0)*100).toFixed(1)+'%';
 $('seg-sat').style.width=((a.sat_pct||0)*100).toFixed(1)+'%';
 $('seg-cash').style.width=((a.cash_pct||0)*100).toFixed(1)+'%';
 const eq=s.equity||0;
 const mk=(nm,cur,tgt,cls_,note)=>row([[nm,''],
  [fmt(cur*100,1)+'%',Math.abs(cur-tgt)>0.05?'warn':''],
  [fmt(cur*eq,0)+' USD',''],[fmt(tgt*100,0)+'%',''],
  [(cur-tgt>0?'+':'')+fmt((cur-tgt)*100,1)+' pp',cls(cur-tgt)],
  [note||'','']]);
 $('alloc-legend').innerHTML='<table><tr><th>仓位</th><th>当前占比</th>'+
  '<th>约市值</th><th>目标</th><th>偏离</th><th>明细</th></tr>'+
  mk('核心仓(纳指100 C)',a.core_pct||0,0.70,cls(a.core_pct-0.70),
   '已部署 '+fmt((a.core_deployed_pct||0)*100,1)+'%,其余待命停泊SGOV')+
  mk('卫星仓(个股 S)',a.sat_pct||0,0.20,cls(a.sat_pct-0.20),
   '已部署 '+fmt((a.sat_deployed_pct||0)*100,1)+'%,其余待命停泊SGOV')+
  mk('现金池(SGOV+现金)',a.cash_pct||0,0.10,cls(a.cash_pct-0.10),
   'R5:低于5%时卖出回款优先补足')+'</table>'+
  '<div class="hint">白色虚线=目标边界(70%/90%);R1:每年1月首个交易日检查,偏离&gt;5pp才调回;待命份额按归属记账(C9)</div>';
 if(s.fx){$('fx-us').innerHTML=
  'USD/CNY ≈ <b>'+s.fx.toFixed(3)+'</b> &nbsp;|&nbsp; 净值折人民币 ≈ <b>¥'+
  fmt(s.equity*s.fx,0)+'</b>'+
  '<div class="hint">境外所得次年3月1日-6月30日自行申报(20%);同一纳税年度境外股票盈亏可相抵,不可跨年</div>'}
 else{$('fx-us').innerHTML='<div class="empty">汇率暂不可用(不影响交易)</div>'}
 const pos=(s.positions||[]).map(p=>row([[p.symbol,''],
  ['<span class="badge '+p.role_cls+'">'+p.role+'</span>',''],
  [p.qty+' × '+fmt(p.avg_cost,2)+' @'+fmt(p.price,2),''],
  [(p.unrealized>0?'+':'')+'$'+fmt(p.unrealized,0),cls(p.unrealized)],
  [(p.unrealized_pct>0?'+':'')+fmt(p.unrealized_pct*100,1)+'%',cls(p.unrealized_pct)],
  [p.days_held,'']]));
 $('positions-us').innerHTML=pos.length?'<table><tr><th>标的</th><th>角色</th>'+
  '<th>数量×成本@现价</th><th>浮盈</th><th>盈亏%</th><th>持有日</th></tr>'+
  pos.join('')+'</table>':
  '<div class="empty">空仓(启动后 C4 买QQQM底仓,C9 现金停泊SGOV)</div>';
 $('signals-us').innerHTML=renderSignals(s,'us','$');
 $('tranches-us').innerHTML=renderTranches(s);
 $('orders-us').innerHTML=renderOrders(s,'$');
}
async function refresh(){
 try{
  const [cn,us,cnEq,usEq]=await Promise.all([
   api('/api/state?market=cn'),api('/api/state?market=us'),
   api('/api/equity?market=cn'),api('/api/equity?market=us')]);
  MODE.cn=cn.mode||'paper';MODE.us=us.mode||'paper';
  renderCN(cn);drawEquity('cn',cnEq||[]);
  renderUS(us);drawEquity('us',usEq||[]);
 }catch(e){if(e.message!=='unauthorized')toast('刷新失败: '+e.message)}
}
function boot(){refresh();setInterval(refresh,30000)}
$('btn-login').onclick=doLogin;
$('btn-logout').onclick=doLogout;
$('modal-ok').onclick=()=>{if(_modalSubmit)_modalSubmit()};
$('modal-cancel').onclick=()=>{$('modal').style.display='none'};
$('pwd').addEventListener('keydown',e=>{if(e.key==='Enter')doLogin()});
window.addEventListener('resize',()=>Object.values(charts).forEach(c=>c&&c.resize()));
if(TOKEN){boot()}else{showLogin()}
</script></body></html>"""


def _signal_rows(signals, positions, quote, equity: float) -> list[dict]:
    """信号行 + 预估金额(准则:提醒写明规则编号/标的/股数/预估金额)。"""
    px: dict[str, float] = {}
    if quote is not None and len(quote):
        for _, r in quote.iterrows():
            px[str(r["symbol"])] = float(r["price"])
    out = []
    for g in signals:
        d = g.to_dict()
        est = 0.0
        if g.side == Side.BUY and g.size_hint_pct:
            est = g.size_hint_pct * equity
        else:
            p = positions.get(g.symbol)
            price = px.get(g.symbol) or (p.avg_cost if p else 0.0)
            if g.qty and price:
                est = g.qty * price
            elif p and g.sell_ratio and price:
                est = p.qty * g.sell_ratio * price
        d["est_amount"] = round(est, 0)
        out.append(d)
    return out


def _order_rows(store, limit: int = 60) -> list[dict]:
    out = []
    for o in store.recent_orders(limit):
        d = o.to_dict()
        d["reason"] = (o.reason or "")[:120]
        out.append(d)
    return out


def _tranche_rows_cn(store) -> list[dict]:
    rows = []
    for sym, st in store.all_pool_states().items():
        if st.get("needs_review"):
            rows.append({"symbol": sym, "tranche": "-", "state": "复核",
                         "state_label": "⚠ 待复核", "trigger_high": 0, "qty": 0,
                         "needs_review": True,
                         "review_note": str(st.get("review_note", ""))[:80],
                         "extra": "", "rule_hint": ""})
            continue
        for key, tr in (st.get("tranches") or {}).items():
            if tr.get("state") in ("available", None):
                continue
            rows.append({"symbol": sym, "tranche": key,
                         "state": tr.get("state", ""),
                         "state_label": _STATE_LABEL.get(tr.get("state", ""),
                                                         tr.get("state", "")),
                         "trigger_high": tr.get("trigger_high", 0),
                         "qty": tr.get("qty", 0), "needs_review": False,
                         "review_note": "", "extra": "", "rule_hint": ""})
    return rows


def _tranche_rows_us(store, us_engine: UsEngine) -> list[dict]:
    rows = []
    core = us_engine._us_core
    core_sym = core.symbol if core else "USQQQM"
    park_sym = core.park_symbol if core else "USSGOV"
    for sym, st in store.all_pool_states().items():
        if sym == park_sym:
            continue
        if sym == core_sym:
            base = st.get("base") or {}
            if base.get("deployed") or base.get("qty"):
                rows.append({"symbol": sym, "tranche": "底仓",
                             "state": "deployed", "state_label": "底仓已部署",
                             "trigger_high": 0, "qty": base.get("qty", 0),
                             "needs_review": False, "review_note": "",
                             "extra": "C4/C7:底仓不卖,仅R1年度再平衡调整", "rule_hint": ""})
            for key, tr in (st.get("tranches") or {}).items():
                if tr.get("state") == "available":
                    continue
                rows.append({"symbol": sym, "tranche": f"C5-{key}",
                             "state": tr.get("state", ""),
                             "state_label": _STATE_LABEL.get(tr.get("state", ""),
                                                             tr.get("state", "")),
                             "trigger_high": tr.get("trigger_high", 0),
                             "qty": tr.get("qty", 0), "needs_review": False,
                             "review_note": "",
                             "extra": "C8:收盘涨回触发日前高即整份卖回", "rule_hint": ""})
            continue
        # 卫星仓个股状态
        if st.get("needs_review"):
            rows.append({"symbol": sym, "tranche": "-", "state": "复核",
                         "state_label": "⚠ 待复核", "trigger_high": 0, "qty": 0,
                         "needs_review": True,
                         "review_note": str(st.get("review_note", ""))[:80],
                         "extra": "", "rule_hint": ""})
        vol = "高波动(止盈30%)" if st.get("vol_class") == "high" \
            else "普通(止盈25%)"
        for key, tr in (st.get("tranches") or {}).items():
            if tr.get("state") == "available":
                continue
            extra = f"{vol};S10移动止盈线 " \
                    f"{fmt2(tr.get('trailing_high', 0) * (0.70 if st.get('vol_class') == 'high' else 0.75))}" \
                    if tr.get("state") == "partial" else vol
            rows.append({"symbol": sym, "tranche": f"S6-{key}",
                         "state": tr.get("state", ""),
                         "state_label": _STATE_LABEL.get(tr.get("state", ""),
                                                         tr.get("state", "")),
                         "trigger_high": tr.get("trigger_high", 0),
                         "qty": tr.get("qty", 0),
                         "needs_review": False, "review_note": "",
                         "extra": extra, "rule_hint": ""})
    return rows


def fmt2(x: float) -> str:
    return f"{x:.2f}" if x else "--"


def create_app(engine: Engine, us_engine: Optional[UsEngine] = None) -> FastAPI:
    app = FastAPI(title="QTrader", docs_url=None, redoc_url=None)
    store = engine.store
    password = str(engine.cfg["server"].get("password", "") or "")

    def _authed(request: Request) -> bool:
        if not password:
            return True  # 未配置密码:开放访问(不建议)
        auth = request.headers.get("authorization", "")
        tok = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        return bool(tok) and tok in _TOKENS

    def _guard(request: Request) -> Optional[JSONResponse]:
        if not _authed(request):
            return JSONResponse({"detail": "未登录或会话已过期"}, status_code=401)
        return None

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/api/login")
    def api_login(body: LoginBody, request: Request):
        ip = request.client.host if request.client else "?"
        now = time.time()
        fails, last = _LOGIN_FAILS.get(ip, (0, 0.0))
        if now - last > 60:
            fails = 0
        if fails >= 5:
            return JSONResponse({"detail": "尝试次数过多,请1分钟后再试"},
                                status_code=429)
        if password and hmac.compare_digest(str(body.password), password):
            _LOGIN_FAILS[ip] = (0, 0.0)
            tok = secrets.token_hex(16)
            _TOKENS.add(tok)
            return {"token": tok}
        _LOGIN_FAILS[ip] = (fails + 1, now)
        time.sleep(0.5)
        return JSONResponse({"detail": "密码错误"}, status_code=401)

    # ---------------- A股状态 ----------------
    def _state_cn() -> dict[str, Any]:
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
        return {
            "currency": "CNY",
            "mode": engine.mode,
            "equity": round(equity, 0), "cash": round(acc.cash, 0) if acc else 0,
            "market_value": round(mv, 0), "day_pnl": round(day_pnl, 0),
            "total_pnl": round(equity - engine.capital, 0),
            "killed": bool(acc.killed) if acc else False,
            "cooldown": store.get_meta("cooldown_until"),
            "positions": rows,
            "orders": _order_rows(store),
            "signals": _signal_rows(store.pending_signals(), positions, quote, equity),
            "tranches": _tranche_rows_cn(store),
        }

    # ---------------- 美股状态 ----------------
    def _state_us() -> dict[str, Any]:
        if us_engine is None:
            return {"disabled": True, "currency": "USD"}
        us = us_engine
        acc = us.account
        ustore = us.store
        positions = ustore.all_positions()
        quote = us._quotes_for(list(positions))
        mv = us._market_value(positions, quote=quote)
        equity = acc.equity(mv)
        day_pnl = equity - acc.day_start_equity
        core = us._us_core
        park_sym = core.park_symbol if core else "USSGOV"
        core_sym = core.symbol if core else "USQQQM"
        rows = []
        for sym, p in positions.items():
            price = 0.0
            if len(quote):
                r = quote[quote["symbol"] == sym]
                if len(r):
                    price = float(r.iloc[0]["price"])
            price = price or p.avg_cost
            unreal = (price - p.avg_cost) * p.qty
            if p.strategy == "us_satellite":
                role, role_cls = "卫星", "sat"
            elif sym == park_sym:
                role, role_cls = "停泊", "park"
            else:
                role, role_cls = "核心", "core"
            rows.append({
                "symbol": sym, "role": role, "role_cls": role_cls,
                "qty": p.qty, "avg_cost": round(p.avg_cost, 3),
                "price": round(price, 3),
                "unrealized": round(unreal, 0),
                "unrealized_pct": round(price / p.avg_cost - 1, 4) if p.avg_cost else 0,
                "days_held": p.days_held,
            })
        try:
            alloc = us.allocation()
        except Exception:  # noqa: BLE001
            alloc = {"core_pct": 0, "sat_pct": 0, "cash_pct": 0}
        try:
            fx = us.usd_cny()
        except Exception:  # noqa: BLE001
            fx = None
        return {
            "currency": "USD",
            "mode": us.mode,
            "equity": round(equity, 0), "cash": round(acc.cash, 0),
            "market_value": round(mv, 0), "day_pnl": round(day_pnl, 0),
            "total_pnl": round(equity - us.capital, 0),
            "killed": bool(acc.killed),
            "alloc": {"core_pct": round(alloc["core_pct"], 4),
                      "sat_pct": round(alloc["sat_pct"], 4),
                      "cash_pct": round(alloc["cash_pct"], 4),
                      "core_deployed_pct": round(alloc.get("core_deployed_pct", 0), 4),
                      "sat_deployed_pct": round(alloc.get("sat_deployed_pct", 0), 4),
                      "park_waiting_pct": round(alloc.get("park_waiting_pct", 0), 4)},
            "fx": round(fx, 4) if fx else None,
            "session": ustore.get_meta("us:session"),
            "positions": rows,
            "orders": _order_rows(ustore),
            "signals": _signal_rows(ustore.pending_signals(), positions, quote, equity),
            "tranches": _tranche_rows_us(ustore, us),
        }

    @app.get("/api/state")
    def api_state(request: Request, market: str = "cn"):
        denied = _guard(request)
        if denied:
            return denied
        return JSONResponse(_state_us() if market == "us" else _state_cn())

    @app.get("/api/equity")
    def api_equity(request: Request, market: str = "cn"):
        denied = _guard(request)
        if denied:
            return denied
        if market == "us":
            data = us_engine.store.equity_curve() if us_engine else []
        else:
            data = store.equity_curve()
        return JSONResponse(data)

    @app.post("/api/reset")
    def api_reset(request: Request, market: str = "cn"):
        denied = _guard(request)
        if denied:
            return denied
        if market == "us":
            if us_engine:
                us_engine.reset()
        else:
            engine.reset()
        return {"ok": True}

    # ---------------- live 人工通道:回填/放弃/资金校准 ----------------
    def _engine_of_signal(signal_id: str):
        from qtrader import fill as F
        engines = [e for e in (engine, us_engine) if e is not None]
        return F.resolve_engine(engines, signal_id)

    @app.post("/api/fill")
    def api_fill(body: FillBody, request: Request):
        denied = _guard(request)
        if denied:
            return denied
        from qtrader import fill as F
        eng = _engine_of_signal(body.signal_id)
        if eng is None:
            return JSONResponse({"ok": False, "msg": "找不到待执行信号"},
                                status_code=404)
        if eng.mode != "live":
            return JSONResponse(
                {"ok": False, "msg": f"{eng.mode} 模式为系统自动撮合,"
                                     f"无需人工回填(config.yaml 切 live 后可用)"},
                status_code=400)
        r = F.fill_signal(eng, body.signal_id, body.price, body.qty,
                          body.fee, body.mood)
        return JSONResponse(r)

    @app.post("/api/skip")
    def api_skip(body: SkipBody, request: Request):
        denied = _guard(request)
        if denied:
            return denied
        from qtrader import fill as F
        eng = _engine_of_signal(body.signal_id)
        if eng is None:
            return JSONResponse({"ok": False, "msg": "找不到待执行信号"},
                                status_code=404)
        if eng.mode != "live":
            return JSONResponse(
                {"ok": False, "msg": f"{eng.mode} 模式不支持人工放弃,信号按规则自动处理"},
                status_code=400)
        return JSONResponse(F.skip_signal(eng, body.signal_id, body.note))

    @app.post("/api/cash")
    def api_cash(body: CashBody, request: Request):
        denied = _guard(request)
        if denied:
            return denied
        from qtrader import fill as F
        eng = us_engine if body.market == "us" else engine
        if eng is None:
            return JSONResponse({"ok": False, "msg": "该市场引擎未启用"},
                                status_code=404)
        return JSONResponse(F.cash_adjust(eng, delta=body.delta,
                                          set_to=body.set_to, note=body.note))

    return app
