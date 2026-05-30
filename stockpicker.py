"""
STOCK PICKER S&P 500 - Versione potenziata
===========================================
Pipeline a 8 stadi:
1) Universo S&P 500 da Wikipedia
2) Regime di mercato (trend/volatilità/breadth/risk-on)
3) Contesto macro (TNX, DXY, yield curve, FRED CPI/Fed funds)
4) Screening tecnico su 500 ticker con scoring REGIME-AWARE
5) Selezione top-10 per score
6) Arricchimento fondamentale FMP + valutazione relativa per settore
7) Correlazioni (vs SPY, vs sector ETF, cross tra candidati)
8) News + sintesi AI Gemini con probabilità empiriche da calibration.json

Invia segnale su Telegram solo se conviction >= soglia.
"""

import os
import sys
import time
import html
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURAZIONE
# ============================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN_PICKER")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID_PICKER")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FMP_API_KEY = os.environ.get("FMP_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
FRED_API_KEY = os.environ.get("FRED_API_KEY")  # NUOVO: per dati macro

GEMINI_MODELS_FALLBACK = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
]

CONVICTION_THRESHOLD = 70
TOP_N_CANDIDATES = 10
TOP_N_FOR_NEWS = 5

# Risk management
ATR_STOP_MULTIPLIER = 1.8
ATR_TARGET_MULTIPLIER = 3.5
MIN_STOP_PCT = 5.0
MAX_STOP_PCT = 12.0
MIN_TARGET_PCT = 8.0
MAX_TARGET_PCT = 25.0

# File calibrazione (output del backtest)
CALIBRATION_FILE = Path(__file__).parent / "calibration.json"

# File posizioni (storico trade aperti/chiusi)
POSITIONS_FILE = Path(__file__).parent / "positions.json"

# Mappa settore -> ETF di settore (per relative valuation e correlazioni)
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
}

DEFENSIVE_ETFS = ["XLU", "XLP", "XLV"]
CYCLICAL_ETFS = ["XLK", "XLY", "XLF", "XLI"]


# ============================================================
# UTILITY GENERICHE
# ============================================================
def safe_pct(v):
    return None if v is None or pd.isna(v) else round(float(v), 2)


def fmt(v, suff="%"):
    return f"{v:+.2f}{suff}" if v is not None else "N/D"


# ============================================================
# STEP 1 - UNIVERSO S&P 500
# ============================================================
def get_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(url)[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"✅ Universo S&P500: {len(tickers)} ticker")
        return tickers
    except Exception as e:
        print(f"⚠️ Errore Wikipedia: {e}, uso fallback")
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
                "BRK-B", "JPM", "LLY", "V", "XOM", "UNH", "MA", "PG", "JNJ", "HD",
                "COST", "MRK", "ABBV", "CVX", "WMT", "BAC", "KO", "ADBE", "CRM"]


# ============================================================
# STEP 2 - REGIME DI MERCATO
# ============================================================
def rileva_regime_mercato():
    """
    Classifica il regime corrente su 4 dimensioni:
    - trend: bull / bear / range
    - volatilita: low / medium / high (da VIX)
    - breadth: strong / mixed / weak (% S&P500 sopra MA50 - proxy via SPY internals)
    - risk_mode: on / off (defensive vs cyclical performance YTD)
    """
    regime = {
        "trend": "unknown",
        "volatilita": "unknown",
        "breadth": "unknown",
        "risk_mode": "unknown",
        "vix_level": None,
        "spy_vs_ma200_pct": None,
        "etichetta": "neutral",
    }

    try:
        # SPY: trend di mercato
        spy = yf.Ticker("SPY").history(period="1y", auto_adjust=True)
        if not spy.empty and len(spy) >= 200:
            prezzo_spy = float(spy["Close"].iloc[-1])
            ma200 = float(spy["Close"].rolling(200).mean().iloc[-1])
            ma50 = float(spy["Close"].rolling(50).mean().iloc[-1])
            diff_ma200 = (prezzo_spy - ma200) / ma200 * 100
            regime["spy_vs_ma200_pct"] = round(diff_ma200, 2)

            if diff_ma200 > 3 and prezzo_spy > ma50:
                regime["trend"] = "bull"
            elif diff_ma200 < -3:
                regime["trend"] = "bear"
            else:
                regime["trend"] = "range"

        # VIX: volatilità
        vix = yf.Ticker("^VIX").history(period="1mo", auto_adjust=True)
        if not vix.empty:
            vix_lvl = float(vix["Close"].iloc[-1])
            regime["vix_level"] = round(vix_lvl, 2)
            if vix_lvl < 15:
                regime["volatilita"] = "low"
            elif vix_lvl < 25:
                regime["volatilita"] = "medium"
            else:
                regime["volatilita"] = "high"

        # Risk-on vs risk-off: confronta defensive vs cyclical (YTD ~3m)
        perf_def = []
        perf_cyc = []
        for tk in DEFENSIVE_ETFS:
            try:
                h = yf.Ticker(tk).history(period="3mo", auto_adjust=True)["Close"]
                perf_def.append((h.iloc[-1] / h.iloc[0] - 1) * 100)
            except Exception:
                pass
        for tk in CYCLICAL_ETFS:
            try:
                h = yf.Ticker(tk).history(period="3mo", auto_adjust=True)["Close"]
                perf_cyc.append((h.iloc[-1] / h.iloc[0] - 1) * 100)
            except Exception:
                pass
        if perf_def and perf_cyc:
            spread = float(np.mean(perf_cyc) - np.mean(perf_def))
            regime["risk_mode"] = "on" if spread > 1 else ("off" if spread < -1 else "neutral")

        # Breadth: proxy con SPY equal-weight (RSP) vs SPY cap-weight
        try:
            rsp = yf.Ticker("RSP").history(period="3mo", auto_adjust=True)["Close"]
            spy_3m = yf.Ticker("SPY").history(period="3mo", auto_adjust=True)["Close"]
            rsp_perf = (rsp.iloc[-1] / rsp.iloc[0] - 1) * 100
            spy_perf = (spy_3m.iloc[-1] / spy_3m.iloc[0] - 1) * 100
            # Se RSP batte SPY, partecipazione ampia (breadth strong)
            if rsp_perf - spy_perf > 1:
                regime["breadth"] = "strong"
            elif rsp_perf - spy_perf < -2:
                regime["breadth"] = "weak"
            else:
                regime["breadth"] = "mixed"
        except Exception:
            pass

        # Etichetta sintetica
        regime["etichetta"] = sintetizza_regime(regime)
        print(f"✅ Regime: {regime['etichetta']} | VIX={regime['vix_level']} | "
              f"SPY vs MA200: {regime['spy_vs_ma200_pct']}%")
    except Exception as e:
        print(f"⚠️ Errore regime detection: {e}")

    return regime


def sintetizza_regime(r):
    """Etichetta breve usata per scoring e calibration lookup."""
    t = r.get("trend", "unknown")
    v = r.get("volatilita", "unknown")
    if t == "bull" and v in ("low", "medium"):
        return "bull-quiet"
    if t == "bull" and v == "high":
        return "bull-volatile"
    if t == "bear" and v == "high":
        return "bear-volatile"
    if t == "bear":
        return "bear-quiet"
    if t == "range":
        return "range"
    return "neutral"


# ============================================================
# STEP 3 - CONTESTO MACRO
# ============================================================
def fred_get(series_id):
    """Recupera ultimo valore di una series FRED."""
    if not FRED_API_KEY:
        return None
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            obs = r.json().get("observations", [])
            if obs and obs[0].get("value") not in (".", None):
                return float(obs[0]["value"])
    except Exception as e:
        print(f"  FRED {series_id} errore: {e}")
    return None


def raccogli_macro():
    """Costruisce dizionario macro: tassi, DXY, yield curve, CPI."""
    macro = {}
    try:
        # Treasury 10Y e 2Y -> yield curve
        tnx = yf.Ticker("^TNX").history(period="3mo", auto_adjust=True)["Close"]
        if not tnx.empty:
            macro["yield_10y"] = round(float(tnx.iloc[-1]) / 10, 3)  # ^TNX è in decimi
            macro["yield_10y_30g_change_bps"] = round(
                (float(tnx.iloc[-1]) - float(tnx.iloc[-21])) * 10, 1
            ) if len(tnx) > 21 else None

        try:
            tyx = yf.Ticker("^FVX").history(period="3mo", auto_adjust=True)["Close"]
            macro["yield_5y"] = round(float(tyx.iloc[-1]) / 10, 3) if not tyx.empty else None
        except Exception:
            pass

        # DXY (Dollar Index)
        try:
            dxy = yf.Ticker("DX-Y.NYB").history(period="3mo", auto_adjust=True)["Close"]
            if not dxy.empty:
                macro["dxy_level"] = round(float(dxy.iloc[-1]), 2)
                macro["dxy_30g_change_pct"] = round(
                    (float(dxy.iloc[-1]) / float(dxy.iloc[-21]) - 1) * 100, 2
                ) if len(dxy) > 21 else None
        except Exception:
            pass

        # FRED: Fed funds e CPI YoY
        ff = fred_get("FEDFUNDS")
        cpi = fred_get("CPIAUCSL")
        cpi_lag = None
        if FRED_API_KEY:
            try:
                # CPI YoY change: prendiamo 13 osservazioni e calcoliamo variazione 12 mesi
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {"series_id": "CPIAUCSL", "api_key": FRED_API_KEY,
                          "file_type": "json", "sort_order": "desc", "limit": 13}
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 200:
                    obs = r.json().get("observations", [])
                    vals = [float(o["value"]) for o in obs if o.get("value") not in (".", None)]
                    if len(vals) >= 13:
                        cpi_lag = round((vals[0] / vals[12] - 1) * 100, 2)
            except Exception:
                pass

        macro["fed_funds_rate"] = ff
        macro["cpi_yoy_pct"] = cpi_lag
        print(f"✅ Macro: 10Y={macro.get('yield_10y')}% | DXY={macro.get('dxy_level')} | "
              f"Fed={macro.get('fed_funds_rate')}% | CPI YoY={macro.get('cpi_yoy_pct')}%")
    except Exception as e:
        print(f"⚠️ Errore macro: {e}")

    return macro


# ============================================================
# STEP 4 - SCREENING + SCORING REGIME-AWARE
# ============================================================
def calcola_atr(storico, periodo=14):
    if len(storico) < periodo + 1:
        return None
    hl = storico["High"] - storico["Low"]
    hc = (storico["High"] - storico["Close"].shift()).abs()
    lc = (storico["Low"] - storico["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(periodo).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else None


def calcola_score(m, regime_label="neutral"):
    """
    Score 0-100 con PESI REGIME-AWARE:
    - bull-quiet → favorisce momentum
    - bull-volatile → favorisce qualità (sopra MA200) e penalizza alta vol
    - range → favorisce mean-reversion / pullback
    - bear-* → ammette poco rischio, premia low-vol e defensive
    """
    score = 50.0

    r6m = m.get("ret_6m")
    r3m = m.get("ret_3m")
    r1m = m.get("ret_1m")
    vol = m.get("vol_30g")
    dist = m.get("dist_52w_high")

    # Pesi base
    w_momentum_6m = 1.0
    w_momentum_3m = 1.0
    w_pullback = 1.0
    w_above_ma = 1.0
    w_dist_high = 1.0
    w_volatility = 1.0
    w_volume = 1.0

    # Aggiustamento pesi per regime
    if regime_label == "bull-quiet":
        w_momentum_6m = 1.5
        w_momentum_3m = 1.3
        w_pullback = 0.7
    elif regime_label == "bull-volatile":
        w_above_ma = 1.5
        w_volatility = 1.8  # penalizza di più alta vol
        w_momentum_6m = 1.0
    elif regime_label == "range":
        w_pullback = 1.8
        w_momentum_6m = 0.7
    elif regime_label in ("bear-quiet", "bear-volatile"):
        w_volatility = 2.0
        w_above_ma = 1.4
        w_momentum_6m = 0.5

    # Momentum 6M
    if r6m is not None:
        if r6m > 20: score += 15 * w_momentum_6m
        elif r6m > 10: score += 10 * w_momentum_6m
        elif r6m > 0: score += 5 * w_momentum_6m
        elif r6m < -20: score -= 15 * w_momentum_6m
        elif r6m < -10: score -= 8 * w_momentum_6m

    # Momentum 3M
    if r3m is not None:
        if r3m > 10: score += 8 * w_momentum_3m
        elif r3m > 0: score += 4 * w_momentum_3m
        elif r3m < -15: score -= 10 * w_momentum_3m

    # Pullback bonus
    if r1m is not None and r6m is not None:
        if -10 < r1m < -2 and r6m > 10:
            score += 15 * w_pullback
        elif r1m > 15:
            score -= 5

    if m.get("sopra_ma200"):
        score += 8 * w_above_ma
    if m.get("sopra_ma50"):
        score += 5

    if dist is not None:
        if -15 < dist < -3:
            score += 8 * w_dist_high
        elif dist < -35:
            score -= 12

    if vol is not None:
        if vol > 80: score -= 12 * w_volatility
        elif vol > 55: score -= 5 * w_volatility
        elif 20 < vol < 40: score += 3

    if m.get("volume_trend_positivo"):
        score += 5 * w_volume

    return max(0, min(100, round(score, 2)))


def screening_ticker(ticker, regime_label="neutral"):
    try:
        azione = yf.Ticker(ticker)
        h = azione.history(period="1y", auto_adjust=True)
        if h.empty or len(h) < 60:
            return None
        if h.index.tz is not None:
            h.index = h.index.tz_localize(None)

        prezzo = float(h["Close"].iloc[-1])

        def ret(d):
            if len(h) <= d:
                return None
            return ((prezzo - h["Close"].iloc[-d]) / h["Close"].iloc[-d]) * 100

        ma50 = h["Close"].rolling(50).mean().iloc[-1]
        ma200 = h["Close"].rolling(200).mean().iloc[-1] if len(h) >= 200 else None
        high_52w = h["High"].tail(252).max() if len(h) >= 100 else h["High"].max()
        rg = h["Close"].pct_change().dropna().tail(30)
        vol30 = float(rg.std() * np.sqrt(252) * 100) if len(rg) > 0 else None
        vol_10 = h["Volume"].tail(10).mean()
        vol_30 = h["Volume"].tail(30).mean()
        atr = calcola_atr(h)

        # Settore (per relative valuation e correlazioni)
        try:
            info = azione.info
            sector = info.get("sector", "Unknown")
        except Exception:
            sector = "Unknown"

        m = {
            "ticker": ticker,
            "sector": sector,
            "prezzo": round(prezzo, 2),
            "ret_1s": safe_pct(ret(5)),
            "ret_1m": safe_pct(ret(21)),
            "ret_3m": safe_pct(ret(63)),
            "ret_6m": safe_pct(ret(126)),
            "sopra_ma50": bool(prezzo > ma50) if not pd.isna(ma50) else False,
            "sopra_ma200": bool(prezzo > ma200) if ma200 is not None and not pd.isna(ma200) else False,
            "dist_52w_high": round((prezzo - high_52w) / high_52w * 100, 2),
            "vol_30g": safe_pct(vol30),
            "volume_trend_positivo": bool(vol_10 > vol_30 * 1.1),
            "atr": round(atr, 2) if atr is not None else None,
        }
        m["score"] = calcola_score(m, regime_label)
        return m
    except Exception:
        return None


def screening_universo(tickers, regime_label):
    print(f"📊 Screening regime-aware ({regime_label}) su {len(tickers)} ticker...")
    risultati = []
    for i, t in enumerate(tickers):
        r = screening_ticker(t, regime_label)
        if r:
            risultati.append(r)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(tickers)}")
    df = pd.DataFrame(risultati)
    if df.empty:
        return df
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    print(f"✅ Screening: {len(df)} ticker validi")
    return df


# ============================================================
# STEP 5/6 - ARRICCHIMENTO FONDAMENTALE + RELATIVE VALUATION
# ============================================================
def fmp_get(endpoint, params=None):
    if not FMP_API_KEY:
        return None
    base = "https://financialmodelingprep.com/api/v3"
    p = params or {}
    p["apikey"] = FMP_API_KEY
    try:
        r = requests.get(f"{base}/{endpoint}", params=p, timeout=20)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def arricchisci_fondamentale(ticker):
    d = {"ticker": ticker}
    km = fmp_get(f"key-metrics-ttm/{ticker}", {"limit": 1})
    if km and len(km) > 0:
        kv = km[0]
        d["pe_ratio"] = kv.get("peRatioTTM")
        d["ps_ratio"] = kv.get("priceToSalesRatioTTM")
        d["pb_ratio"] = kv.get("pbRatioTTM")
        d["roe"] = kv.get("roeTTM")
        d["debt_equity"] = kv.get("debtToEquityTTM")
        d["fcf_yield"] = kv.get("freeCashFlowYieldTTM")

    g = fmp_get(f"income-statement-growth/{ticker}", {"limit": 1})
    if g and len(g) > 0:
        d["revenue_growth"] = g[0].get("growthRevenue")
        d["eps_growth"] = g[0].get("growthEPS")

    dcf = fmp_get(f"discounted-cash-flow/{ticker}")
    if dcf and len(dcf) > 0:
        dv = dcf[0].get("dcf")
        pr = dcf[0].get("Stock Price") or dcf[0].get("price")
        try:
            if dv and pr:
                d["dcf_value"] = round(float(dv), 2)
                d["dcf_upside_pct"] = round((float(dv) - float(pr)) / float(pr) * 100, 1)
        except (ValueError, TypeError):
            pass

    rt = fmp_get(f"rating/{ticker}")
    if rt and len(rt) > 0:
        d["rating"] = rt[0].get("rating")

    es = fmp_get(f"earnings-surprises/{ticker}", {"limit": 1})
    if es and len(es) > 0:
        est = es[0].get("estimatedEarning")
        act = es[0].get("actualEarningResult")
        try:
            if est and act:
                d["earnings_surprise_pct"] = round((float(act) - float(est)) / abs(float(est)) * 100, 1)
                d["earnings_date"] = es[0].get("date")
        except (ValueError, TypeError, ZeroDivisionError):
            pass

    pr = fmp_get("stock_peers", {"symbol": ticker})
    if pr and isinstance(pr, list) and len(pr) > 0:
        d["peers"] = pr[0].get("peersList", [])[:5]

    tr = fmp_get(f"earning_call_transcript/{ticker}", {"limit": 1})
    if tr and len(tr) > 0:
        full = tr[0].get("content", "")
        d["earnings_transcript_excerpt"] = full[:3500] if full else None

    return d


def valutazione_relativa_settore(candidato, settore):
    """
    Calcola percentile rank di P/E, P/S, ROE del candidato rispetto al settore.
    Strategia: usa l'ETF di settore, prende i suoi top holdings, ne calcola le mediane via FMP.
    Cache settore per non ripetere.
    """
    if not FMP_API_KEY:
        return {}
    etf = SECTOR_ETF_MAP.get(settore)
    if not etf:
        return {}

    # Prendi un paio di peers via FMP come proxy del settore (semplice e leggero)
    peers_data = candidato.get("peers", [])
    if not peers_data:
        return {}

    metriche_settore = {"pe": [], "ps": [], "roe": []}
    for peer in peers_data[:4]:  # limite chiamate API
        km = fmp_get(f"key-metrics-ttm/{peer}", {"limit": 1})
        if km and len(km) > 0:
            v = km[0]
            if v.get("peRatioTTM"): metriche_settore["pe"].append(v["peRatioTTM"])
            if v.get("priceToSalesRatioTTM"): metriche_settore["ps"].append(v["priceToSalesRatioTTM"])
            if v.get("roeTTM"): metriche_settore["roe"].append(v["roeTTM"])

    risultato = {}
    for k, vals in metriche_settore.items():
        if len(vals) >= 2:
            risultato[f"sector_median_{k}"] = round(float(np.median(vals)), 2)

    # Confronto candidato vs mediana settore
    candidato_pe = candidato.get("pe_ratio")
    candidato_ps = candidato.get("ps_ratio")
    candidato_roe = candidato.get("roe")

    if candidato_pe and risultato.get("sector_median_pe"):
        risultato["pe_vs_sector_pct"] = round(
            (candidato_pe / risultato["sector_median_pe"] - 1) * 100, 1
        )
    if candidato_ps and risultato.get("sector_median_ps"):
        risultato["ps_vs_sector_pct"] = round(
            (candidato_ps / risultato["sector_median_ps"] - 1) * 100, 1
        )
    if candidato_roe and risultato.get("sector_median_roe"):
        risultato["roe_vs_sector_diff"] = round(
            candidato_roe - risultato["sector_median_roe"], 2
        )

    return risultato


# ============================================================
# STEP 7 - CORRELAZIONI
# ============================================================
def calcola_correlazioni(top_candidati):
    """
    Per ogni candidato: corr 90gg vs SPY, vs settore ETF.
    + matrice di correlazione tra i top candidati per flaggare ridondanze.
    """
    tickers_cand = [c["screening"]["ticker"] for c in top_candidati]
    settori = list(set(c["screening"].get("sector", "Unknown") for c in top_candidati))
    etf_set = [SECTOR_ETF_MAP[s] for s in settori if s in SECTOR_ETF_MAP]

    tutti = list(set(tickers_cand + ["SPY"] + etf_set))
    print(f"  Scarico storici per {len(tutti)} ticker (correlazioni)...")
    try:
        dati = yf.download(tutti, period="6mo", auto_adjust=True, progress=False)["Close"]
        if isinstance(dati, pd.Series):
            dati = dati.to_frame()
    except Exception as e:
        print(f"  ⚠️ Errore download correlazioni: {e}")
        return {}, pd.DataFrame()

    rendimenti = dati.pct_change().dropna().tail(90)
    if rendimenti.empty:
        return {}, pd.DataFrame()

    corr_dict = {}
    for c in top_candidati:
        tk = c["screening"]["ticker"]
        if tk not in rendimenti.columns:
            continue
        info = {}
        if "SPY" in rendimenti.columns:
            try:
                info["corr_spy_90g"] = round(float(rendimenti[tk].corr(rendimenti["SPY"])), 3)
            except Exception:
                pass
        settore = c["screening"].get("sector", "Unknown")
        etf = SECTOR_ETF_MAP.get(settore)
        if etf and etf in rendimenti.columns:
            try:
                info["corr_settore_90g"] = round(float(rendimenti[tk].corr(rendimenti[etf])), 3)
            except Exception:
                pass
        corr_dict[tk] = info

    # Matrice di correlazione tra i top candidati
    cand_corr = rendimenti[[t for t in tickers_cand if t in rendimenti.columns]].corr().round(2)
    return corr_dict, cand_corr


def trova_pair_ridondanti(matrice_corr, soglia=0.75):
    """Coppie di candidati con corr > soglia: esposizione ridondante."""
    pairs = []
    if matrice_corr.empty:
        return pairs
    for i, t1 in enumerate(matrice_corr.columns):
        for t2 in matrice_corr.columns[i + 1:]:
            v = matrice_corr.loc[t1, t2]
            if pd.notna(v) and v > soglia:
                pairs.append((t1, t2, round(float(v), 2)))
    return sorted(pairs, key=lambda x: -x[2])


# ============================================================
# STEP 8 - NEWS
# ============================================================
def get_news(ticker):
    if not NEWSAPI_KEY:
        return []
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": f'"{ticker}"', "language": "en", "sortBy": "publishedAt",
                    "pageSize": 5,
                    "from": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                    "apiKey": NEWSAPI_KEY},
            timeout=15,
        )
        if r.status_code == 200:
            return [{"titolo": a.get("title", ""),
                     "fonte": a.get("source", {}).get("name", ""),
                     "data": a.get("publishedAt", "")[:10],
                     "descrizione": (a.get("description") or "")[:300]}
                    for a in r.json().get("articles", [])]
    except Exception:
        pass
    return []


# ============================================================
# CALIBRAZIONE / PROBABILITÀ EMPIRICHE
# ============================================================
def carica_calibration():
    """Carica calibration.json se esiste e ha status OK."""
    if not CALIBRATION_FILE.exists():
        print("ℹ️ calibration.json assente: uso euristica")
        return None
    try:
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        if cal.get("status") == "FAILED" or not cal.get("buckets"):
            print(f"⚠️ Calibration con status={cal.get('status')}: uso euristica")
            return None
        print(f"✅ Calibration caricata: {cal.get('n_samples', '?')} samples, "
              f"aggiornata {cal.get('last_updated', '?')}")
        return cal
    except Exception as e:
        print(f"⚠️ Errore lettura calibration: {e}")
        return None


def stima_probabilita_empirica(score, regime_label, calibration):
    """
    Restituisce probabilità empiriche per il bucket (score_bin × regime).
    Output: {p_pos_45d, p_target_10pct, p_stop_8pct, avg_ret_45d, n_samples}
    Se calibration non disponibile o bucket vuoto: None.
    """
    if not calibration:
        return None
    buckets = calibration.get("buckets", {})

    # Identifica bin di score
    if score >= 90: bin_lo, bin_hi = 90, 100
    elif score >= 80: bin_lo, bin_hi = 80, 90
    elif score >= 70: bin_lo, bin_hi = 70, 80
    elif score >= 60: bin_lo, bin_hi = 60, 70
    else: bin_lo, bin_hi = 50, 60

    key_specifico = f"score_{bin_lo}_{bin_hi}_regime_{regime_label}"
    key_generico = f"score_{bin_lo}_{bin_hi}_regime_all"

    bucket = buckets.get(key_specifico) or buckets.get(key_generico)
    if not bucket or bucket.get("n_samples", 0) < 20:
        # Fallback: usa qualunque bucket dello stesso score bin
        bucket = buckets.get(key_generico)
    if not bucket:
        return None
    return bucket


# ============================================================
# AI SYNTHESIS - GEMINI
# ============================================================
def chiamata_gemini(prompt):
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY non configurata"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8192,
            "topP": 0.9,
            "thinkingConfig": {"thinkingBudget": 0},
        }
    }
    ultimo_errore = "nessun modello disponibile"
    for modello in GEMINI_MODELS_FALLBACK:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{modello}:generateContent?key={GEMINI_API_KEY}")
        for tent in range(3):
            try:
                r = requests.post(url, json=payload, timeout=60)
                if r.status_code == 429:
                    attesa = [10, 30, 60][tent]
                    print(f"⏳ 429 su {modello}, retry tra {attesa}s")
                    time.sleep(attesa)
                    continue
                r.raise_for_status()
                d = r.json()
                txt = d["candidates"][0]["content"]["parts"][0]["text"].strip()
                fr = d["candidates"][0].get("finishReason", "?")
                print(f"✅ Gemini OK con {modello} (fr={fr}, len={len(txt)})")
                return txt, None
            except Exception as e:
                ultimo_errore = f"{modello}: {str(e)[:120]}"
                print(f"❌ {ultimo_errore}")
                if tent < 2:
                    time.sleep(5)
    return None, ultimo_errore


def parse_json_robusto(testo):
    if not testo:
        return None
    clean = testo.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        if len(parts) >= 2:
            clean = parts[1]
            if clean.lower().startswith("json"):
                clean = clean[4:]
    clean = clean.strip()
    start = clean.find("{")
    if start < 0:
        return None
    clean = clean[start:]
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    end = clean.rfind("}")
    if end > 0:
        try:
            return json.loads(clean[:end + 1])
        except json.JSONDecodeError:
            pass
    return _ripara_json_troncato(clean)


def _ripara_json_troncato(s):
    res = []
    in_str = False
    esc = False
    stack = []
    for ch in s:
        res.append(ch)
        if esc:
            esc = False; continue
        if ch == "\\":
            esc = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{": stack.append("}")
        elif ch == "[": stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch: stack.pop()
    if in_str: res.append('"')
    testo = "".join(res).rstrip().rstrip(",")
    while stack: testo += stack.pop()
    try:
        return json.loads(testo)
    except Exception:
        return None


def costruisci_dossier(regime, macro, candidati, correlazioni, pair_ridondanti, calibration):
    """Costruisce il prompt enriched con tutto il contesto."""
    out = []

    # Regime
    out.append(f"\n=== REGIME DI MERCATO CORRENTE ===")
    out.append(f"Etichetta: {regime['etichetta']}")
    out.append(f"Trend SPY: {regime['trend']} (SPY vs MA200: {regime['spy_vs_ma200_pct']}%)")
    out.append(f"Volatilità: {regime['volatilita']} (VIX={regime['vix_level']})")
    out.append(f"Breadth: {regime['breadth']} | Risk mode: {regime['risk_mode']}")

    # Macro
    out.append(f"\n=== CONTESTO MACRO ===")
    for k, v in macro.items():
        out.append(f"{k}: {v}")

    # Candidati
    for i, c in enumerate(candidati):
        scr = c["screening"]
        fnd = c.get("fundamentals", {})
        rel = c.get("relative_val", {})
        news = c.get("news", [])
        corr = correlazioni.get(scr["ticker"], {})
        emp = c.get("empirical_prob")

        out.append(f"\n{'=' * 50}")
        out.append(f"CANDIDATO #{i + 1}: {scr['ticker']} | Settore: {scr.get('sector')} | Prezzo: ${scr['prezzo']}")
        out.append(f"Score tecnico (regime-aware): {scr['score']}/100")
        out.append(f"Rendimenti: 1S={scr.get('ret_1s')}% | 1M={scr.get('ret_1m')}% | "
                   f"3M={scr.get('ret_3m')}% | 6M={scr.get('ret_6m')}%")
        out.append(f"Sopra MA50: {scr.get('sopra_ma50')} | Sopra MA200: {scr.get('sopra_ma200')} | "
                   f"Dist 52w high: {scr.get('dist_52w_high')}%")
        out.append(f"Volatilità 30g: {scr.get('vol_30g')}% | Volume trend up: {scr.get('volume_trend_positivo')}")

        if corr:
            out.append(f"Correlazione 90g vs SPY: {corr.get('corr_spy_90g')} | "
                       f"vs settore ETF: {corr.get('corr_settore_90g')}")

        if emp:
            out.append(f"\nPROBABILITÀ EMPIRICA (da backtest storico):")
            out.append(f"  P(rendimento positivo 45gg) = {emp.get('p_pos_45d', 'N/D')}")
            out.append(f"  P(target +10% raggiunto) = {emp.get('p_target_10pct', 'N/D')}")
            out.append(f"  P(stop -8% colpito) = {emp.get('p_stop_8pct', 'N/D')}")
            out.append(f"  Rendimento medio 45gg = {emp.get('avg_ret_45d', 'N/D')}%")
            out.append(f"  N samples nel bucket: {emp.get('n_samples', 0)}")

        if fnd:
            out.append("\nFONDAMENTALI:")
            for k in ["pe_ratio", "ps_ratio", "pb_ratio", "roe", "debt_equity", "fcf_yield",
                      "revenue_growth", "eps_growth", "dcf_value", "dcf_upside_pct",
                      "rating", "earnings_surprise_pct", "earnings_date"]:
                if fnd.get(k) is not None:
                    out.append(f"  {k}: {fnd[k]}")
            if fnd.get("peers"):
                out.append(f"  peers: {', '.join(fnd['peers'])}")

        if rel:
            out.append("\nRELATIVE VALUATION vs SETTORE:")
            for k, v in rel.items():
                out.append(f"  {k}: {v}")

        if fnd.get("earnings_transcript_excerpt"):
            out.append(f"\nULTIMO EARNINGS CALL (estratto):\n{fnd['earnings_transcript_excerpt'][:2200]}")

        if news:
            out.append("\nNEWS RECENTI:")
            for n in news[:3]:
                out.append(f"  - [{n['data']}] {n['titolo']} ({n['fonte']})")

    if pair_ridondanti:
        out.append("\n=== ESPOSIZIONI RIDONDANTI (alta correlazione) ===")
        for t1, t2, c in pair_ridondanti[:5]:
            out.append(f"  {t1} ↔ {t2}: corr={c}")

    if calibration:
        out.append(f"\n=== CONTESTO STATISTICO ===")
        out.append(f"Calibration aggiornata: {calibration.get('last_updated')}")
        out.append(f"Periodo dati: {calibration.get('data_period')}")
        out.append(f"N samples totali: {calibration.get('n_samples')}")

    return "\n".join(out)


def analisi_finale_ai(regime, macro, candidati, correlazioni, pair_ridondanti, calibration):
    dossier = costruisci_dossier(regime, macro, candidati, correlazioni, pair_ridondanti, calibration)

    has_calib = calibration is not None
    nota_prob = ("Hai probabilità empiriche reali dal backtest storico nel dossier — usale come ancora "
                 "principale per il conviction_score. Il conviction non deve discostarsi più di 15 punti "
                 "dalla probabilità empirica p_target_10pct moltiplicata per 100.") if has_calib else \
                ("Non hai probabilità empiriche disponibili: usa euristica basata su score tecnico, "
                 "qualità fondamentali e contesto macro/regime per stimare il conviction.")

    prompt = f"""Sei un analista quantitativo senior specializzato in selezione azionaria a breve termine (1-2 mesi) per portafogli concentrati.

Hai a disposizione: regime di mercato corrente, contesto macroeconomico, {len(candidati)} candidati pre-filtrati con screening regime-aware, fondamentali, relative valuation per settore, correlazioni, news, e {'probabilità empiriche da backtest storico' if has_calib else 'solo dati senza calibration storica'}.

Scegli UNA SOLA azione con la più alta aspettativa risk-adjusted nei prossimi 30-60 giorni dato il regime corrente. {nota_prob}

Rispondi ESATTAMENTE in questo JSON valido (niente markdown, niente backtick, solo JSON):

{{
  "ticker_scelto": "TICKER",
  "company_name": "Nome",
  "conviction_score": <0-100>,
  "tesi_acquisto": "5-7 righe complete che integrano: regime corrente, contesto macro, momentum, valutazione vs settore, qualità fondamentali, correlazione (low beta = difensivo, high beta = aggressivo), e probabilità empirica se disponibile",
  "perche_in_questo_regime": "2 righe sul perché questa scelta è adatta al regime {regime['etichetta']}",
  "catalisti_brevi": ["catalisti con timing", "...", "..."],
  "rischi_principali": ["rischio macro o specifico", "..."],
  "target_orizzonte_giorni": <30, 45 o 60>,
  "beta_categoria": "low (corr SPY <0.5) | medium (0.5-0.75) | high (>0.75)"
}}

Regole:
- Scegli SOLO tra i ticker presenti nel dossier
- Cita SEMPRE numeri concreti (rendimenti, valutazioni, probabilità)
- Se nessun candidato è davvero forte → conviction basso (<70), il bot tace
- Considera correlazioni: evita scelte ridondanti con il regime (es. in bear-volatile niente high beta)
- Non inventare dati assenti

DOSSIER:
{dossier}
"""

    risposta, errore = chiamata_gemini(prompt)
    if not risposta:
        return None, errore
    parsed = parse_json_robusto(risposta)
    if parsed is None:
        return None, f"Parse JSON fallito: {risposta[:300]}"
    return parsed, None


# ============================================================
# RISK MANAGEMENT
# ============================================================
def salva_posizione(scelta, screening, risk, regime, emp_prob):
    """
    Appende il nuovo trade in positions.json come posizione OPEN.
    Il tracker settimanale ne aggiornerà lo stato.
    """
    try:
        posizioni = []
        if POSITIONS_FILE.exists():
            with open(POSITIONS_FILE) as f:
                posizioni = json.load(f)

        ticker = scelta["ticker_scelto"]
        data_entry = datetime.now().strftime("%Y-%m-%d")

        # Evita duplicati: se ho già aperta una posizione su questo ticker, non riapro
        already_open = any(p["ticker"] == ticker and p["status"] == "OPEN" for p in posizioni)
        if already_open:
            print(f"ℹ️ {ticker} già aperto in portfolio, salto save")
            return

        record = {
            "id": f"{ticker}_{data_entry}",
            "ticker": ticker,
            "company_name": scelta.get("company_name", ticker),
            "sector": screening.get("sector", "Unknown"),
            "entry_date": data_entry,
            "entry_price": float(screening["prezzo"]),
            "stop_loss_price": risk["stop_loss_price"],
            "take_profit_price": risk["take_profit_price"],
            "stop_loss_pct": -risk["stop_loss_pct"],
            "take_profit_pct": risk["take_profit_pct"],
            "risk_reward": risk["risk_reward"],
            "conviction": scelta.get("conviction_score", 0),
            "regime_at_entry": regime.get("etichetta", "neutral"),
            "vix_at_entry": regime.get("vix_level"),
            "orizzonte_giorni": scelta.get("target_orizzonte_giorni", 45),
            "beta_categoria": scelta.get("beta_categoria", "?"),
            "emp_p_target_10pct": emp_prob.get("p_target_10pct") if emp_prob else None,
            "emp_p_stop_8pct": emp_prob.get("p_stop_8pct") if emp_prob else None,
            "tesi_breve": scelta.get("tesi_acquisto", "")[:300],
            "status": "OPEN",
            "close_date": None,
            "close_price": None,
            "close_reason": None,
            "pnl_pct": None,
            "days_held": None,
        }
        posizioni.append(record)
        with open(POSITIONS_FILE, "w") as f:
            json.dump(posizioni, f, indent=2)
        print(f"✅ Posizione salvata: {ticker} entry ${screening['prezzo']}")
    except Exception as e:
        print(f"⚠️ Errore salvataggio posizione: {e}")


def calcola_stop_target(prezzo, atr):
    if atr is None or atr <= 0:
        stop_pct, target_pct = 7.0, 14.0
    else:
        stop_pct = (atr * ATR_STOP_MULTIPLIER / prezzo) * 100
        target_pct = (atr * ATR_TARGET_MULTIPLIER / prezzo) * 100
        stop_pct = max(MIN_STOP_PCT, min(MAX_STOP_PCT, stop_pct))
        target_pct = max(MIN_TARGET_PCT, min(MAX_TARGET_PCT, target_pct))
    return {
        "stop_loss_price": round(prezzo * (1 - stop_pct / 100), 2),
        "stop_loss_pct": round(stop_pct, 2),
        "take_profit_price": round(prezzo * (1 + target_pct / 100), 2),
        "take_profit_pct": round(target_pct, 2),
        "risk_reward": round(target_pct / stop_pct, 2),
    }


# ============================================================
# TELEGRAM
# ============================================================
def invia_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram non configurato. Messaggio:\n" + msg)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    MAX = 4000
    parts = [msg[i:i + MAX] for i in range(0, len(msg), MAX)]
    for p in parts:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": p,
                                      "parse_mode": "HTML", "disable_web_page_preview": True})
        print(f"Telegram: {len(p)} char, status {r.status_code}")


def formatta_messaggio(scelta, screening, risk, regime, macro, emp_prob):
    ticker = scelta["ticker_scelto"]
    nome = scelta.get("company_name", ticker)
    conviction = scelta.get("conviction_score", 0)
    tesi = html.escape(scelta.get("tesi_acquisto", ""))
    motivo_regime = html.escape(scelta.get("perche_in_questo_regime", ""))
    catalisti = scelta.get("catalisti_brevi", [])
    rischi = scelta.get("rischi_principali", [])
    orizzonte = scelta.get("target_orizzonte_giorni", 45)
    beta_cat = scelta.get("beta_categoria", "?")

    oggi = datetime.now().strftime("%d/%m/%Y")

    m = f"<b>🎯 SEGNALE DI ACQUISTO - {oggi}</b>\n"
    m += "═══════════════════════\n\n"
    m += f"<b>📈 {html.escape(nome)} ({ticker})</b>\n"
    m += f"Settore: {html.escape(screening.get('sector', '?'))}\n"
    m += f"<b>Entry:</b> ${screening['prezzo']}\n"
    m += f"<b>Conviction:</b> {conviction}/100 ⭐\n"
    m += f"<b>Beta:</b> {beta_cat} | <b>Orizzonte:</b> {orizzonte} giorni\n\n"

    m += "<b>🌐 REGIME &amp; MACRO</b>\n"
    m += f"Regime: <b>{regime['etichetta']}</b> | VIX: {regime['vix_level']}\n"
    m += f"10Y: {macro.get('yield_10y')}% | DXY: {macro.get('dxy_level')} | "
    m += f"CPI YoY: {macro.get('cpi_yoy_pct')}%\n\n"

    m += "<b>🛡️ RISK MANAGEMENT</b>\n"
    m += f"🔴 Stop loss: ${risk['stop_loss_price']} (-{risk['stop_loss_pct']}%)\n"
    m += f"🟢 Take profit: ${risk['take_profit_price']} (+{risk['take_profit_pct']}%)\n"
    m += f"⚖️ R/R: {risk['risk_reward']}:1\n\n"

    if emp_prob:
        m += "<b>📊 PROBABILITÀ EMPIRICHE</b> <i>(backtest storico)</i>\n"
        m += f"P(positivo 45g): {emp_prob.get('p_pos_45d', 'N/D')}\n"
        m += f"P(target +10%): {emp_prob.get('p_target_10pct', 'N/D')}\n"
        m += f"P(stop -8%): {emp_prob.get('p_stop_8pct', 'N/D')}\n"
        m += f"N samples: {emp_prob.get('n_samples', 0)}\n\n"

    m += "<b>📝 TESI</b>\n" + tesi + "\n\n"

    if motivo_regime:
        m += f"<b>🌍 ADATTO AL REGIME</b>\n{motivo_regime}\n\n"

    if catalisti:
        m += "<b>🚀 CATALISTI</b>\n"
        for c in catalisti[:4]:
            m += f"• {html.escape(str(c))}\n"
        m += "\n"

    if rischi:
        m += "<b>⚠️ RISCHI</b>\n"
        for r in rischi[:3]:
            m += f"• {html.escape(str(r))}\n"
        m += "\n"

    m += ("<i>⚠️ Analisi automatizzata, non consulenza finanziaria. "
          "Investi solo capitale che puoi permetterti di perdere.</i>")
    return m


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"🚀 Stock Picker avviato - {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    calibration = carica_calibration()

    print("\n[1/8] Carico universo")
    tickers = get_sp500_tickers()
    if not tickers:
        return

    print("\n[2/8] Rilevo regime di mercato")
    regime = rileva_regime_mercato()

    print("\n[3/8] Raccolgo contesto macro")
    macro = raccogli_macro()

    print(f"\n[4/8] Screening regime-aware ({regime['etichetta']})")
    df = screening_universo(tickers, regime["etichetta"])
    if df.empty:
        return

    print(f"\n[5/8] Top {TOP_N_CANDIDATES} candidati:")
    top_df = df.head(TOP_N_CANDIDATES)
    print(top_df[["ticker", "sector", "prezzo", "score", "ret_1m", "ret_3m"]].to_string())

    print(f"\n[6/8] Arricchimento fondamentale + relative valuation")
    candidati = []
    for i, row in top_df.iterrows():
        tk = row["ticker"]
        print(f"  [{i + 1}/{len(top_df)}] {tk}")
        fnd = arricchisci_fondamentale(tk) if FMP_API_KEY else {}
        rel = valutazione_relativa_settore(fnd, row.get("sector")) if FMP_API_KEY else {}
        emp = stima_probabilita_empirica(row["score"], regime["etichetta"], calibration)
        candidati.append({
            "screening": row.to_dict(),
            "fundamentals": fnd,
            "relative_val": rel,
            "empirical_prob": emp,
            "news": [],
        })

    print(f"\n[7/8] Correlazioni")
    correlazioni, matrice = calcola_correlazioni(candidati)
    pair_ridondanti = trova_pair_ridondanti(matrice)
    if pair_ridondanti:
        print(f"  Coppie ridondanti (corr > 0.75): {len(pair_ridondanti)}")

    print(f"\n[7.5] News per top-{TOP_N_FOR_NEWS}")
    for i, c in enumerate(candidati[:TOP_N_FOR_NEWS]):
        c["news"] = get_news(c["screening"]["ticker"]) if NEWSAPI_KEY else []
        print(f"  [{i + 1}] {c['screening']['ticker']}: {len(c['news'])} news")

    print(f"\n[8/8] Analisi finale Gemini")
    scelta, errore = analisi_finale_ai(regime, macro, candidati, correlazioni, pair_ridondanti, calibration)
    if not scelta:
        print(f"❌ AI: {errore}")
        return

    ticker_scelto = scelta.get("ticker_scelto")
    conviction = scelta.get("conviction_score", 0)
    print(f"\n📊 Scelta: {ticker_scelto} | Conviction: {conviction}/100")

    screening_sel = next((c["screening"] for c in candidati
                          if c["screening"]["ticker"] == ticker_scelto), None)
    emp_sel = next((c.get("empirical_prob") for c in candidati
                    if c["screening"]["ticker"] == ticker_scelto), None)

    if not screening_sel:
        print(f"❌ Ticker {ticker_scelto} non trovato")
        return

    if conviction < CONVICTION_THRESHOLD:
        print(f"⏸️ Conviction {conviction} < soglia {CONVICTION_THRESHOLD}: nessun invio")
        return

    risk = calcola_stop_target(screening_sel["prezzo"], screening_sel.get("atr"))
    msg = formatta_messaggio(scelta, screening_sel, risk, regime, macro, emp_sel)
    invia_telegram(msg)
    print("✅ Segnale inviato")

    # Persisti la posizione per il tracker
    salva_posizione(scelta, screening_sel, risk, regime, emp_sel)


if __name__ == "__main__":
    main()
    top_df = df.head(TOP_N_CANDIDATES)
    print(top_df[["ticker", "sector", "prezzo", "score", "ret_1m", "ret_3m"]].to_string())

    print(f"\n[6/8] Arricchimento fondamentale + relative valuation")
    candidati = []
    for i, row in top_df.iterrows():
        tk = row["ticker"]
        print(f"  [{i + 1}/{len(top_df)}] {tk}")
        fnd = arricchisci_fondamentale(tk) if FMP_API_KEY else {}
        rel = valutazione_relativa_settore(fnd, row.get("sector")) if FMP_API_KEY else {}
        emp = stima_probabilita_empirica(row["score"], regime["etichetta"], calibration)
        candidati.append({
            "screening": row.to_dict(),
            "fundamentals": fnd,
            "relative_val": rel,
            "empirical_prob": emp,
            "news": [],
        })

    print(f"\n[7/8] Correlazioni")
    correlazioni, matrice = calcola_correlazioni(candidati)
    pair_ridondanti = trova_pair_ridondanti(matrice)
    if pair_ridondanti:
        print(f"  Coppie ridondanti (corr > 0.75): {len(pair_ridondanti)}")

    print(f"\n[7.5] News per top-{TOP_N_FOR_NEWS}")
    for i, c in enumerate(candidati[:TOP_N_FOR_NEWS]):
        c["news"] = get_news(c["screening"]["ticker"]) if NEWSAPI_KEY else []
        print(f"  [{i + 1}] {c['screening']['ticker']}: {len(c['news'])} news")

    print(f"\n[8/8] Analisi finale Gemini")
    scelta, errore = analisi_finale_ai(regime, macro, candidati, correlazioni, pair_ridondanti, calibration)
    if not scelta:
        print(f"❌ AI: {errore}")
        return

    ticker_scelto = scelta.get("ticker_scelto")
    conviction = scelta.get("conviction_score", 0)
    print(f"\n📊 Scelta: {ticker_scelto} | Conviction: {conviction}/100")

    screening_sel = next((c["screening"] for c in candidati
                          if c["screening"]["ticker"] == ticker_scelto), None)
    emp_sel = next((c.get("empirical_prob") for c in candidati
                    if c["screening"]["ticker"] == ticker_scelto), None)

    if not screening_sel:
        print(f"❌ Ticker {ticker_scelto} non trovato")
        return

    if conviction < CONVICTION_THRESHOLD:
        print(f"⏸️ Conviction {conviction} < soglia {CONVICTION_THRESHOLD}: nessun invio")
        return

    risk = calcola_stop_target(screening_sel["prezzo"], screening_sel.get("atr"))
    msg = formatta_messaggio(scelta, screening_sel, risk, regime, macro, emp_sel)
    invia_telegram(msg)
    print("✅ Segnale inviato")

    # Persisti la posizione per il tracker
    salva_posizione(scelta, screening_sel, risk, regime, emp_sel)


if __name__ == "__main__":
    main()
)
    top_df = df.head(TOP_N_CANDIDATES)
    print(top_df[["ticker", "sector", "prezzo", "score", "ret_1m", "ret_3m"]].to_string())

    print(f"\n[6/8] Arricchimento fondamentale + relative valuation")
    candidati = []
    for i, row in top_df.iterrows():
        tk = row["ticker"]
        print(f"  [{i + 1}/{len(top_df)}] {tk}")
        fnd = arricchisci_fondamentale(tk) if FMP_API_KEY else {}
        rel = valutazione_relativa_settore(fnd, row.get("sector")) if FMP_API_KEY else {}
        emp = stima_probabilita_empirica(row["score"], regime["etichetta"], calibration)
        candidati.append({
            "screening": row.to_dict(),
            "fundamentals": fnd,
            "relative_val": rel,
            "empirical_prob": emp,
            "news": [],
        })

    print(f"\n[7/8] Correlazioni")
    correlazioni, matrice = calcola_correlazioni(candidati)
    pair_ridondanti = trova_pair_ridondanti(matrice)
    if pair_ridondanti:
        print(f"  Coppie ridondanti (corr > 0.75): {len(pair_ridondanti)}")

    print(f"\n[7.5] News per top-{TOP_N_FOR_NEWS}")
    for i, c in enumerate(candidati[:TOP_N_FOR_NEWS]):
        c["news"] = get_news(c["screening"]["ticker"]) if NEWSAPI_KEY else []
        print(f"  [{i + 1}] {c['screening']['ticker']}: {len(c['news'])} news")

    print(f"\n[8/8] Analisi finale Gemini")
    scelta, errore = analisi_finale_ai(regime, macro, candidati, correlazioni, pair_ridondanti, calibration)
    if not scelta:
        print(f"❌ AI: {errore}")
        return

    ticker_scelto = scelta.get("ticker_scelto")
    conviction = scelta.get("conviction_score", 0)
    print(f"\n📊 Scelta: {ticker_scelto} | Conviction: {conviction}/100")

    screening_sel = next((c["screening"] for c in candidati
                          if c["screening"]["ticker"] == ticker_scelto), None)
    emp_sel = next((c.get("empirical_prob") for c in candidati
                    if c["screening"]["ticker"] == ticker_scelto), None)

    if not screening_sel:
        print(f"❌ Ticker {ticker_scelto} non trovato")
        return

    if conviction < CONVICTION_THRESHOLD:
        print(f"⏸️ Conviction {conviction} < soglia {CONVICTION_THRESHOLD}: nessun invio")
        return

    risk = calcola_stop_target(screening_sel["prezzo"], screening_sel.get("atr"))
    msg = formatta_messaggio(scelta, screening_sel, risk, regime, macro, emp_sel)
    invia_telegram(msg)
    print("✅ Segnale inviato")

    # Persisti la posizione per il tracker
    salva_posizione(scelta, screening_sel, risk, regime, emp_sel)


if __name__ == "__main__":
    main()
