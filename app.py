import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ASX Valuation Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1rem; }

    .title-card {
        background: linear-gradient(135deg, #1B2A4A 0%, #2E4172 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .title-card h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .title-card p {
        color: #a0b0d0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: #1e2535;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #2e3d5e;
    }
    .metric-label {
        color: #8899bb;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        color: white;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-sub {
        color: #8899bb;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    .signal-buy  { background:#1a4a2e; color:#4ade80;
                   padding:0.4rem 1rem; border-radius:20px;
                   font-weight:700; font-size:1rem; }
    .signal-hold { background:#4a3a00; color:#fbbf24;
                   padding:0.4rem 1rem; border-radius:20px;
                   font-weight:700; font-size:1rem; }
    .signal-sell { background:#4a1a1a; color:#f87171;
                   padding:0.4rem 1rem; border-radius:20px;
                   font-weight:700; font-size:1rem; }

    .method-tag {
        background:#2e3d5e; color:#a0b0d0;
        padding:0.2rem 0.6rem; border-radius:6px;
        font-size:0.8rem; margin-right:0.3rem;
    }
    .warning-tag {
        background:#4a3000; color:#fbbf24;
        padding:0.2rem 0.6rem; border-radius:6px;
        font-size:0.8rem;
    }

    .section-header {
        color: #a0b0d0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #2e3d5e;
    }

    .disclaimer {
        background: #1a1f2e;
        border-left: 3px solid #2e4172;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #6b7a99;
        font-size: 0.8rem;
        margin-top: 1rem;
    }

    div[data-testid="stDataFrame"] { border-radius: 10px; }
    .stSelectbox > div { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MACRO INPUTS
# ══════════════════════════════════════════════════════════════════════════════
RISK_FREE_RATE   = 0.044
EQUITY_RISK_PREM = 0.055
TAX_RATE         = 0.30
TERMINAL_GROWTH  = 0.025

BIG_4_BANKS = {"CBA.AX","NAB.AX","WBC.AX","ANZ.AX"}
OTHER_BANKS  = {"MQG.AX"}
BANKS        = BIG_4_BANKS | OTHER_BANKS
INSURERS     = {"SUN.AX","QBE.AX","IAG.AX","AMP.AX"}
FINANCIALS   = BANKS | INSURERS
BIG_4_FRANCHISE_PREMIUM = 0.15

SECTOR_EV_EBITDA = {
    "utilities":  8.0, "energy": 6.0,
    "healthcare": 14.0,"technology":16.0,
    "materials":  7.0, "default":10.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# ALL VALUATION FUNCTIONS (same as asx_screener.py)
# ══════════════════════════════════════════════════════════════════════════════
def cost_of_equity_capm(beta):
    return RISK_FREE_RATE + (beta or 1.0) * EQUITY_RISK_PREM

def calculate_wacc(info):
    beta         = info.get("beta",1.0) or 1.0
    ke           = cost_of_equity_capm(beta)
    total_debt   = info.get("totalDebt",0) or 0
    interest_exp = abs(info.get("interestExpense",0) or 0)
    kd           = (interest_exp/total_debt) if total_debt>0 and interest_exp>0 else 0.05
    kd_at        = kd*(1-TAX_RATE)
    market_cap   = info.get("marketCap",0) or 0
    total_cap    = market_cap+total_debt
    if total_cap==0: return None
    return (market_cap/total_cap*ke)+(total_debt/total_cap*kd_at)

def get_fcf(stock):
    try:
        cf = stock.cashflow
        if cf.empty: return None
        op_labels  = ["Operating Cash Flow",
                      "Cash Flow From Continuing Operating Activities",
                      "Total Cash From Operating Activities"]
        cap_labels = ["Capital Expenditure","Purchase Of Ppe",
                      "Capital Expenditures",
                      "Purchases Of Property Plant And Equipment"]
        op_cf,capex = None,0
        for l in op_labels:
            if l in cf.index: op_cf=cf.loc[l].iloc[0]; break
        for l in cap_labels:
            if l in cf.index: capex=cf.loc[l].iloc[0]; break
        if op_cf is not None: return op_cf+capex
        if "Free Cash Flow" in cf.index:
            return cf.loc["Free Cash Flow"].iloc[0]
        return None
    except: return None

def get_3yr_cagr(stock, info):
    try:
        fin = stock.financials
        if fin.empty: raise ValueError()
        for label in ["Total Revenue","Revenue"]:
            if label in fin.index:
                rev = fin.loc[label].dropna()
                n   = min(len(rev)-1, 3)
                if n<1: raise ValueError()
                cagr = (rev.iloc[0]/rev.iloc[n])**(1/n)-1
                return max(0.02,min(abs(cagr),0.25)),n
        raise ValueError()
    except:
        fb = info.get("revenueGrowth") or info.get("earningsGrowth") or 0.05
        try:
            if np.isnan(fb): fb=0.05
        except: fb=0.05
        return max(0.02,min(abs(fb),0.20)),1

def valuation_dcf(stock, info):
    fcf = get_fcf(stock)
    if fcf is None: return None,"No FCF data"
    if fcf<=0:      return None,"Negative FCF"
    growth,n_yrs = get_3yr_cagr(stock,info)
    gr = [growth,growth,growth*0.9,growth*0.75,growth*0.60]
    wacc = calculate_wacc(info)
    if wacc is None or wacc<=TERMINAL_GROWTH: wacc=TERMINAL_GROWTH+0.03
    proj,disc,cf = [],[],fcf
    for yr,g in enumerate(gr,1):
        cf=cf*(1+g); proj.append(cf)
        disc.append(cf/(1+wacc)**yr)
    tv    = proj[-1]*(1+TERMINAL_GROWTH)/(wacc-TERMINAL_GROWTH)
    pv_tv = tv/(1+wacc)**5
    ev    = sum(disc)+pv_tv
    net_d = (info.get("totalDebt",0) or 0)-(info.get("totalCash",0) or 0)
    fv    = (ev-net_d)/(info.get("sharesOutstanding",1) or 1)
    return round(fv,2),{
        "method":"DCF","wacc":round(wacc*100,1),
        "growth":round(growth*100,1),"growth_yrs":n_yrs,
        "pv_terminal":round(pv_tv/ev*100,1),
        "proj_fcfs":proj,"disc_fcfs":disc,"pv_tv":pv_tv
    }

def valuation_pb(ticker, stock, info):
    try:
        bvps = info.get("bookValue")
        if not bvps or bvps<=0: return None,"No book value"
        try:
            ni  = stock.financials.loc["Net Income"]
            eq  = stock.balance_sheet.loc["Stockholders Equity"]
            roe = np.mean([ni.iloc[i]/eq.iloc[i]
                           for i in range(min(3,len(ni),len(eq)))
                           if eq.iloc[i]>0])
        except:
            roe = info.get("returnOnEquity")
        if roe is None: return None,"No ROE data"
        try:
            if np.isnan(roe): return None,"Invalid ROE"
        except: pass
        ke  = cost_of_equity_capm(info.get("beta",1.0) or 1.0)
        if ke<=TERMINAL_GROWTH: return None,"Ke too low"
        pb  = max(0.3,min((roe-TERMINAL_GROWTH)/(ke-TERMINAL_GROWTH),4.0))
        fv  = pb*bvps
        is_big4 = ticker in BIG_4_BANKS
        if is_big4: fv*=(1+BIG_4_FRANCHISE_PREMIUM)
        return round(fv,2),{
            "method":"P/B","justified_pb":round(pb,2),
            "roe":round(roe*100,1),"ke":round(ke*100,1),
            "big4_premium":is_big4
        }
    except Exception as e:
        return None,f"P/B error: {e}"

def valuation_ev_ebitda(stock, info):
    try:
        ebitda = info.get("ebitda")
        if not ebitda or ebitda<=0: return None,"No EBITDA"
        sec = (info.get("sector") or "").lower()
        if "util"   in sec: m=SECTOR_EV_EBITDA["utilities"]
        elif "energy" in sec: m=SECTOR_EV_EBITDA["energy"]
        elif "health" in sec: m=SECTOR_EV_EBITDA["healthcare"]
        elif "tech"  in sec: m=SECTOR_EV_EBITDA["technology"]
        elif "mater" in sec: m=SECTOR_EV_EBITDA["materials"]
        else: m=SECTOR_EV_EBITDA["default"]
        net_d = (info.get("totalDebt",0) or 0)-(info.get("totalCash",0) or 0)
        fv    = (m*ebitda-net_d)/(info.get("sharesOutstanding",1) or 1)
        return round(fv,2),{
            "method":"EV/EBITDA","multiple":m,
            "sector":info.get("sector","Unknown")
        }
    except Exception as e:
        return None,f"EV/EBITDA error: {e}"

def sensitivity_table(stock, info, base_wacc_pct, base_growth_pct):
    try:
        fcf = get_fcf(stock)
        if not fcf or fcf<=0: return None,None,None
        wacc_r   = [round(base_wacc_pct+i,1) for i in [-2,-1,0,1,2]]
        growth_r = [round(base_growth_pct+i,1) for i in [-2,-1,0,1,2]]
        net_d    = (info.get("totalDebt",0) or 0)-(info.get("totalCash",0) or 0)
        shares   = info.get("sharesOutstanding",1) or 1
        table    = {}
        for w_p in wacc_r:
            w = max(w_p/100, TERMINAL_GROWTH+0.01)
            row = {}
            for g_p in growth_r:
                g  = max(g_p/100,0.01)
                gr = [g,g,g*0.9,g*0.75,g*0.60]
                proj,disc,cf2 = [],[],fcf
                for yr,rate in enumerate(gr,1):
                    cf2=cf2*(1+rate); proj.append(cf2)
                    disc.append(cf2/(1+w)**yr)
                tv    = proj[-1]*(1+TERMINAL_GROWTH)/(w-TERMINAL_GROWTH)
                pv_tv = tv/(1+w)**5
                ev    = sum(disc)+pv_tv
                row[g_p] = round((ev-net_d)/shares,2)
            table[w_p] = row
        return table,wacc_r,growth_r
    except: return None,None,None

@st.cache_data(ttl=900)  # cache for 15 minutes
def value_stock_cached(ticker_symbol, rf_rate, erp, terminal_g):
    global RISK_FREE_RATE, EQUITY_RISK_PREM, TERMINAL_GROWTH
    RISK_FREE_RATE   = rf_rate
    EQUITY_RISK_PREM = erp
    TERMINAL_GROWTH  = terminal_g
    try:
        stock = yf.Ticker(ticker_symbol)
        info  = stock.info
        if not info: return None,"No data"

        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        if price<=0: return None,"No price data"

        valuations,method_details = [],[]

        if ticker_symbol in FINANCIALS:
            fv,det = valuation_pb(ticker_symbol,stock,info)
            if fv: valuations.append((fv,1.0)); method_details.append(("P/B",fv,det))
        else:
            dcf_fv, dcf_det   = valuation_dcf(stock,info)
            eveb_fv,eveb_det  = valuation_ev_ebitda(stock,info)
            if dcf_fv and eveb_fv:
                valuations += [(dcf_fv,0.60),(eveb_fv,0.40)]
                method_details += [("DCF",dcf_fv,dcf_det),
                                   ("EV/EBITDA",eveb_fv,eveb_det)]
            elif dcf_fv:
                valuations.append((dcf_fv,1.0))
                method_details.append(("DCF",dcf_fv,dcf_det))
            elif eveb_fv:
                valuations.append((eveb_fv,1.0))
                method_details.append(("EV/EBITDA",eveb_fv,eveb_det))

        if not valuations: return None,"All methods failed"

        tw  = sum(w for _,w in valuations)
        bfv = round(sum(fv*w for fv,w in valuations)/tw,2)
        fvs = [fv for fv,_ in valuations]

        upside = (bfv-price)/price*100
        signal = "🟢 BUY" if upside>15 else ("🔴 SELL" if upside<-15 else "🟡 HOLD")

        sens,wr,gr = None,None,None
        for name,fv,det in method_details:
            if name=="DCF" and isinstance(det,dict):
                sens,wr,gr = sensitivity_table(
                    stock,info,det["wacc"],det["growth"])
                break

        return {
            "ticker":         ticker_symbol,
            "name":           info.get("shortName",ticker_symbol),
            "sector":         info.get("sector","Unknown"),
            "industry":       info.get("industry",""),
            "current_price":  round(price,2),
            "blended_fv":     bfv,
            "fv_low":         min(fvs),
            "fv_high":        max(fvs),
            "upside_pct":     round(upside,1),
            "signal":         signal,
            "method_details": method_details,
            "analyst_target": info.get("targetMeanPrice"),
            "n_analysts":     info.get("numberOfAnalystOpinions"),
            "market_cap":     info.get("marketCap"),
            "pe_ratio":       info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high":       info.get("fiftyTwoWeekHigh"),
            "52w_low":        info.get("fiftyTwoWeekLow"),
            "beta":           info.get("beta"),
            "sens_table":     sens,
            "wacc_range":     wr,
            "growth_range":   gr,
        }, None
    except Exception as e:
        return None,f"Error: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# ASX 200 TICKER LIST (sample — we'll expand later)
# ══════════════════════════════════════════════════════════════════════════════
ASX_TICKERS = [
    "CBA.AX","NAB.AX","WBC.AX","ANZ.AX","MQG.AX",
    "SUN.AX","QBE.AX","IAG.AX",
    "BHP.AX","RIO.AX","FMG.AX","S32.AX",
    "WES.AX","WOW.AX","COL.AX","JBH.AX",
    "QAN.AX","CSL.AX","REA.AX","SEK.AX",
    "TLS.AX","ORG.AX","AMC.AX","AGL.AX","ALL.AX",
    "WDS.AX","STO.AX","NST.AX","NCM.AX","EVN.AX"
]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔍 Search")

    mode = st.radio(
        "Mode",
        ["Single Stock", "Run Full Screener"],
        help="Single Stock: deep dive one company. Full Screener: scan all stocks."
    )

    if mode == "Single Stock":
        ticker_input = st.text_input(
            "Enter ASX Ticker",
            value="CSL.AX",
            placeholder="e.g. BHP.AX",
            help="Add .AX suffix for ASX stocks"
        ).upper().strip()

        if not ticker_input.endswith(".AX"):
            ticker_input += ".AX"

    st.markdown("---")
    st.markdown("### ⚙️ Model Assumptions")
    rf   = st.number_input("Risk-Free Rate (%)",
                            value=4.4, step=0.1, format="%.1f") / 100
    erp  = st.number_input("Equity Risk Premium (%)",
                            value=5.5, step=0.1, format="%.1f") / 100
    tg   = st.number_input("Terminal Growth Rate (%)",
                            value=2.5, step=0.1, format="%.1f") / 100

    RISK_FREE_RATE  = rf
    EQUITY_RISK_PREM = erp
    TERMINAL_GROWTH  = tg

    st.markdown("---")
    st.markdown("""
    <div style='color:#6b7a99; font-size:0.75rem'>
    <b>Data:</b> Yahoo Finance (live)<br>
    <b>Methods:</b> DCF · P/B · EV/EBITDA<br>
    <b>Built by:</b> Aneeq Mohamed<br>
    <b>UNSW Econometrics & Finance</b>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='title-card'>
    <h1>📊 ASX Valuation Screener</h1>
    <p>Multi-method equity valuation · DCF · Justified P/B · EV/EBITDA ·
       Live data via Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE STOCK MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single Stock":

    with st.spinner(f"Valuing {ticker_input}..."):
        result, error = value_stock_cached(ticker_input, rf, erp, tg)
    if error:
        st.error(f"Could not value {ticker_input}: {error}")
        st.stop()

    r = result

    # ── Company header ────────────────────────────────────────────────────────
    col_name, col_sig = st.columns([3,1])
    with col_name:
        st.markdown(f"## {r['name']}")
        st.markdown(
            f"<span class='method-tag'>{r['sector']}</span>"
            f"<span class='method-tag'>{r['industry']}</span>",
            unsafe_allow_html=True
        )
    with col_sig:
        sig = r["signal"]
        cls = ("signal-buy" if "BUY" in sig
               else "signal-sell" if "SELL" in sig
               else "signal-hold")
        st.markdown(
            f"<div style='text-align:right; padding-top:1rem'>"
            f"<span class='{cls}'>{sig}</span></div>",
            unsafe_allow_html=True
        )

    # ── Key metrics ───────────────────────────────────────────────────────────
    st.markdown("<p class='section-header'>Valuation</p>",
                unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)

    def metric_card(col, label, value, sub=""):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div>"
            f"<div class='metric-sub'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    metric_card(c1, "Current Price",
                f"${r['current_price']:.2f}", "Live")
    metric_card(c2, "Blended Fair Value",
                f"${r['blended_fv']:.2f}",
                f"${r['fv_low']:.2f} – ${r['fv_high']:.2f}")
    upside_color = ("#4ade80" if r["upside_pct"]>15
                    else "#f87171" if r["upside_pct"]<-15
                    else "#fbbf24")
    c3.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>Upside / Downside</div>"
        f"<div class='metric-value' style='color:{upside_color}'>"
        f"{r['upside_pct']:+.1f}%</div>"
        f"<div class='metric-sub'>vs current price</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    analyst_str = (f"${r['analyst_target']:.2f} "
                   f"({r['n_analysts']} analysts)"
                   if r["analyst_target"] else "N/A")
    metric_card(c4, "Analyst Consensus", analyst_str, "Mean target")

    mkt_cap = r.get("market_cap")
    if mkt_cap:
        if mkt_cap >= 1e9:
            cap_str = f"${mkt_cap/1e9:.1f}B"
        else:
            cap_str = f"${mkt_cap/1e6:.0f}M"
    else:
        cap_str = "N/A"
    metric_card(c5, "Market Cap", cap_str, "")

    # ── Method breakdown ──────────────────────────────────────────────────────
    st.markdown("<p class='section-header'>Valuation Methods</p>",
                unsafe_allow_html=True)

    method_cols = st.columns(len(r["method_details"]))
    for col, (name, fv, detail) in zip(method_cols, r["method_details"]):
        with col:
            if isinstance(detail, dict):
                if name == "DCF":
                    col.metric(
                        f"DCF Fair Value",
                        f"${fv:.2f}",
                        f"WACC {detail['wacc']}% | "
                        f"Growth {detail['growth']}% "
                        f"({detail['growth_yrs']}yr CAGR)"
                    )
                elif name == "P/B":
                    premium = " + Big4 ✓" if detail.get("big4_premium") else ""
                    col.metric(
                        "P/B Fair Value",
                        f"${fv:.2f}",
                        f"P/B {detail['justified_pb']}x | "
                        f"ROE {detail['roe']}%{premium}"
                    )
                elif name == "EV/EBITDA":
                    col.metric(
                        "EV/EBITDA Fair Value",
                        f"${fv:.2f}",
                        f"{detail['multiple']}x multiple | "
                        f"{detail['sector']}"
                    )

    # ── Stock stats ───────────────────────────────────────────────────────────
    st.markdown("<p class='section-header'>Stock Statistics</p>",
                unsafe_allow_html=True)

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("P/E Ratio",
              f"{r['pe_ratio']:.1f}x" if r["pe_ratio"] else "N/A")
    s2.metric("Dividend Yield",
              f"{r['dividend_yield']*100:.2f}%"
              if r["dividend_yield"] and r["dividend_yield"] < 1
              else f"{r['dividend_yield']:.2f}%"
              if r["dividend_yield"] else "N/A")
    s3.metric("Beta", f"{r['beta']:.2f}" if r["beta"] else "N/A")
    s4.metric("52W Range",
              f"${r['52w_low']:.2f} – ${r['52w_high']:.2f}"
              if r["52w_low"] else "N/A")

    # ── Sensitivity table ─────────────────────────────────────────────────────
    if r["sens_table"] and r["wacc_range"] and r["growth_range"]:
        st.markdown("<p class='section-header'>DCF Sensitivity Analysis</p>",
                    unsafe_allow_html=True)
        st.caption(
            "Fair value ($) across WACC and growth rate assumptions. "
            "Base case highlighted. Green = above current price. "
            "Red = below current price."
        )

        rows = []
        for w in r["wacc_range"]:
            row = {"WACC \\ Growth": f"{w:.1f}%"}
            for g in r["growth_range"]:
                val = r["sens_table"].get(w,{}).get(g,"N/A")
                row[f"{g:.1f}%"] = val
            rows.append(row)

        df_sens = pd.DataFrame(rows).set_index("WACC \\ Growth")

        def colour_cell(val):
            try:
                v = float(val)
                price = r["current_price"]
                if v > price*1.15:
                    return "background-color:#1a4a2e; color:#4ade80"
                elif v < price*0.85:
                    return "background-color:#4a1a1a; color:#f87171"
                else:
                    return "background-color:#4a3a00; color:#fbbf24"
            except:
                return ""

        styled = df_sens.style.map(colour_cell)
        st.dataframe(styled, use_container_width=True)

    # ── Price chart ───────────────────────────────────────────────────────────
    st.markdown("<p class='section-header'>12-Month Price History</p>",
                unsafe_allow_html=True)
    try:
        hist = yf.Ticker(ticker_input).history(period="1y")
        if not hist.empty:
            chart_df = hist[["Close"]].copy()
            chart_df.columns = ["Price ($)"]
            st.line_chart(chart_df, use_container_width=True)
    except:
        st.info("Price history unavailable")

# ══════════════════════════════════════════════════════════════════════════════
# FULL SCREENER MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("### 📋 Full ASX Screener")
    st.info("This will value all stocks in the list. Takes 2–3 minutes.")

    if st.button("🚀 Run Full Screener", type="primary"):
        results = []
        progress = st.progress(0)
        status   = st.empty()

        for i, ticker in enumerate(ASX_TICKERS):
            status.text(f"Valuing {ticker}... ({i+1}/{len(ASX_TICKERS)})")
            result, error = value_stock_cached(ticker, rf, erp, tg)
            if result:
                results.append(result)
            progress.progress((i+1)/len(ASX_TICKERS))

        status.text("✅ Done!")
        progress.empty()

        if results:
            buy  = [r for r in results if "BUY"  in r["signal"]]
            hold = [r for r in results if "HOLD" in r["signal"]]
            sell = [r for r in results if "SELL" in r["signal"]]

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Stocks Valued", len(results))
            m2.metric("🟢 BUY",  len(buy))
            m3.metric("🟡 HOLD", len(hold))
            m4.metric("🔴 SELL", len(sell))

            # Results table
            table_data = []
            for r in results:
                table_data.append({
                    "Ticker":       r["ticker"],
                    "Company":      r["name"],
                    "Sector":       r["sector"],
                    "Price ($)":    r["current_price"],
                    "Fair Value ($)":r["blended_fv"],
                    "Upside (%)":   r["upside_pct"],
                    "Signal":       r["signal"],
                    "Analyst ($)":  r["analyst_target"] or "N/A",
                    "Method":       " + ".join(
                        [n for n,_,_ in r["method_details"]]
                    ),
                })

            df = pd.DataFrame(table_data)

            def colour_signal(val):
                if "BUY"  in str(val):
                    return "background-color:#1a4a2e; color:#4ade80"
                elif "SELL" in str(val):
                    return "background-color:#4a1a1a; color:#f87171"
                else:
                    return "background-color:#4a3a00; color:#fbbf24"

            def colour_upside(val):
                try:
                    v = float(val)
                    if v>15:  return "color:#4ade80"
                    if v<-15: return "color:#f87171"
                    return "color:#fbbf24"
                except: return ""

            styled_df = (df.style
                         .applymap(colour_signal,  subset=["Signal"])
                         .applymap(colour_upside,  subset=["Upside (%)"]))

            st.dataframe(styled_df, height=600)

# ══════════════════════════════════════════════════════════════════════════════
# DISCLAIMER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='disclaimer'>
⚠️ <b>Disclaimer:</b> This tool is for educational and research purposes only.
It is not financial advice. All valuations are quantitative first-pass estimates
based on publicly available data from Yahoo Finance. Always conduct thorough
due diligence before making any investment decisions. Past performance does not
guarantee future results.
</div>
""", unsafe_allow_html=True)