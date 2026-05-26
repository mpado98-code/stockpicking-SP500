"""
STOCK PICKER - Telegram Bot per S&P 500
========================================
Pipeline a 6 stadi:
1) Carica universo S&P 500 da Wikipedia
2) Screening tecnico su tutti i 500 ticker con yfinance (gratis, illimitato)
3) Selezione top-10 candidati per score combinato (momentum + setup pullback)
4) Arricchimento fondamentale con FMP API (free tier: 250 chiamate/giorno)
5) News & sentiment con NewsAPI (free tier: 100 chiamate/giorno)
6) Sintesi con Gemini AI + calcolo stop loss/take profit basato su ATR

Invia il segnale su Telegram SOLO se conviction score >= soglia.
"""

import os
import sys
import time
import html
import json
import warnings
from datetime import datetime, timedelta

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

# Modelli Gemini in ordine di preferenza (fallback automatico se uno fallisce)
GEMINI_MODELS_FALLBACK = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]

# Soglia di conviction sotto la quale NON si invia il segnale
CONVICTION_THRESHOLD = 70

# Numero di candidati da analizzare in profondità (limita le chiamate API FMP)
TOP_N_CANDIDATES = 10
TOP_N_FOR_NEWS = 5

# Parametri risk management
ATR_STOP_MULTIPLIER = 1.8     # Stop loss = entry - 1.8 * ATR
ATR_TARGET_MULTIPLIER = 3.5   # Take profit = entry + 3.5 * ATR (R:R ~ 2:1)
MIN_STOP_PCT = 5.0            # Stop loss minimo: 5% sotto entry
MAX_STOP_PCT = 12.0           # Stop loss massimo: 12% sotto entry
MIN_TARGET_PCT = 8.0          # Target minimo: 8% sopra entry
MAX_TARGET_PCT = 25.0         # Target massimo: 25% sopra entry

# ============================================================
# STEP 1 - UNIVERSO S&P 500
# ============================================================
def get_sp500_tickers():
    """Scarica la lista dei ticker S&P500 da Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tabelle = pd.read_html(url)
        df = tabelle[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"✅ Universo S&P500 caricato: {len(tickers)} ticker")
        return tickers
    except Exception as e:
        print(f"⚠️ Errore caricamento S&P500 da Wikipedia: {e}")
        # Fallback hardcoded (parziale) – almeno permette al bot di funzionare
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
            "BRK-B", "JPM", "LLY", "V", "XOM", "UNH", "MA", "PG", "JNJ", "HD",
            "COST", "MRK", "ABBV", "CVX", "WMT", "BAC", "KO", "ADBE", "CRM",
            "ORCL", "ACN", "AMD", "MCD", "PEP", "TMO", "NFLX", "CSCO", "WFC",
        ]


# ============================================================
# STEP 2 - SCREENING TECNICO (yfinance)
# ============================================================
def calcola_atr(storico, periodo=14):
    """Average True Range, base per stop loss dinamici."""
    if len(storico) < periodo + 1:
        return None
    high_low = storico["High"] - storico["Low"]
    high_close = (storico["High"] - storico["Close"].shift()).abs()
    low_close = (storico["Low"] - storico["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(periodo).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else None


def calcola_score(metriche):
    """
    Score 0-100 che premia: uptrend di medio termine + pullback recente
    + volume sano + volatilità moderata. È il filtro statistico di base.
    """
    score = 50.0

    # Momentum medio termine (6M) - peso alto
    r6m = metriche.get("ret_6m")
    if r6m is not None:
        if r6m > 20: score += 15
        elif r6m > 10: score += 10
        elif r6m > 0: score += 5
        elif r6m < -20: score -= 15
        elif r6m < -10: score -= 8

    # Momentum 3M - conferma
    r3m = metriche.get("ret_3m")
    if r3m is not None:
        if r3m > 10: score += 8
        elif r3m > 0: score += 4
        elif r3m < -15: score -= 10

    # Setup pullback: 1M negativo dentro un trend 6M positivo = opportunità
    r1m = metriche.get("ret_1m")
    if r1m is not None and r6m is not None:
        if -10 < r1m < -2 and r6m > 10:
            score += 15  # bonus "buy the dip"
        elif r1m > 15:
            score -= 5   # già troppo caldo nel breve

    # Posizione vs medie mobili
    if metriche.get("sopra_ma200"):
        score += 8
    if metriche.get("sopra_ma50"):
        score += 5

    # Distanza dal massimo 52 settimane: non vogliamo né top assoluto né crollo
    dist_high = metriche.get("dist_52w_high")
    if dist_high is not None:
        if -15 < dist_high < -3:
            score += 8  # leggermente sotto i massimi: sweet spot
        elif dist_high < -35:
            score -= 12  # crollo serio

    # Volatilità: penalizzo estremi
    vol = metriche.get("vol_30g")
    if vol is not None:
        if vol > 80: score -= 12
        elif vol > 55: score -= 5
        elif 20 < vol < 40: score += 3

    # Volume in aumento = interesse istituzionale
    if metriche.get("volume_trend_positivo"):
        score += 5

    return max(0, min(100, round(score, 2)))


def screening_ticker(ticker):
    """Calcola metriche tecniche per un singolo ticker. Ritorna None se dati mancano."""
    try:
        azione = yf.Ticker(ticker)
        storico = azione.history(period="1y", auto_adjust=True)
        if storico.empty or len(storico) < 60:
            return None
        if storico.index.tz is not None:
            storico.index = storico.index.tz_localize(None)

        prezzo = float(storico["Close"].iloc[-1])

        def ret(giorni):
            if len(storico) <= giorni:
                return None
            past = storico["Close"].iloc[-giorni]
            return ((prezzo - past) / past) * 100

        ret_1s = ret(5)
        ret_1m = ret(21)
        ret_3m = ret(63)
        ret_6m = ret(126) if len(storico) >= 126 else None

        ma50 = storico["Close"].rolling(50).mean().iloc[-1]
        ma200 = storico["Close"].rolling(200).mean().iloc[-1] if len(storico) >= 200 else None

        high_52w = storico["High"].tail(252).max() if len(storico) >= 100 else storico["High"].max()
        dist_52w_high = ((prezzo - high_52w) / high_52w) * 100

        rend_giorn = storico["Close"].pct_change().dropna().tail(30)
        vol_30g = float(rend_giorn.std() * np.sqrt(252) * 100) if len(rend_giorn) > 0 else None

        # Volume trend: media 10gg vs media 30gg
        vol_10 = storico["Volume"].tail(10).mean()
        vol_30 = storico["Volume"].tail(30).mean()
        volume_trend = vol_10 > vol_30 * 1.1

        atr = calcola_atr(storico)

        metriche = {
            "ticker": ticker,
            "prezzo": round(prezzo, 2),
            "ret_1s": round(ret_1s, 2) if ret_1s is not None else None,
            "ret_1m": round(ret_1m, 2) if ret_1m is not None else None,
            "ret_3m": round(ret_3m, 2) if ret_3m is not None else None,
            "ret_6m": round(ret_6m, 2) if ret_6m is not None else None,
            "sopra_ma50": prezzo > ma50 if not pd.isna(ma50) else False,
            "sopra_ma200": prezzo > ma200 if ma200 is not None and not pd.isna(ma200) else False,
            "dist_52w_high": round(dist_52w_high, 2),
            "vol_30g": round(vol_30g, 2) if vol_30g is not None else None,
            "volume_trend_positivo": bool(volume_trend),
            "atr": round(atr, 2) if atr is not None else None,
        }
        metriche["score"] = calcola_score(metriche)
        return metriche

    except Exception as e:
        return None


def screening_universo(tickers):
    """Esegue lo screening su tutto l'universo e ritorna DataFrame ordinato per score."""
    print(f"📊 Screening su {len(tickers)} ticker (può richiedere 5-10 min)...")
    risultati = []
    for i, t in enumerate(tickers):
        m = screening_ticker(t)
        if m:
            risultati.append(m)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(tickers)} processati")
    df = pd.DataFrame(risultati)
    if df.empty:
        return df
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    print(f"✅ Screening completato: {len(df)} ticker validi")
    return df


# ============================================================
# STEP 3-4 - ARRICCHIMENTO FONDAMENTALE (FMP)
# ============================================================
def fmp_get(endpoint, params=None):
    """Wrapper per chiamate FMP con gestione errori."""
    if not FMP_API_KEY:
        return None
    base = "https://financialmodelingprep.com/api/v3"
    params = params or {}
    params["apikey"] = FMP_API_KEY
    try:
        r = requests.get(f"{base}/{endpoint}", params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"  FMP {endpoint}: status {r.status_code}")
            return None
    except Exception as e:
        print(f"  FMP errore {endpoint}: {str(e)[:80]}")
        return None


def arricchisci_fondamentale(ticker):
    """Recupera dati fondamentali, DCF, peers, ultimo earnings call da FMP."""
    dati = {"ticker": ticker}

    # Key metrics: P/E, ROE, debt/equity, ecc.
    km = fmp_get(f"key-metrics-ttm/{ticker}", {"limit": 1})
    if km and len(km) > 0:
        m = km[0]
        dati["pe_ratio"] = m.get("peRatioTTM")
        dati["ps_ratio"] = m.get("priceToSalesRatioTTM")
        dati["pb_ratio"] = m.get("pbRatioTTM")
        dati["roe"] = m.get("roeTTM")
        dati["debt_equity"] = m.get("debtToEquityTTM")
        dati["fcf_yield"] = m.get("freeCashFlowYieldTTM")

    # Crescita
    growth = fmp_get(f"income-statement-growth/{ticker}", {"limit": 1})
    if growth and len(growth) > 0:
        g = growth[0]
        dati["revenue_growth"] = g.get("growthRevenue")
        dati["eps_growth"] = g.get("growthEPS")

    # DCF stimato vs prezzo corrente
    dcf = fmp_get(f"discounted-cash-flow/{ticker}")
    if dcf and len(dcf) > 0:
        d = dcf[0]
        dcf_val = d.get("dcf")
        price = d.get("Stock Price") or d.get("price")
        if dcf_val and price:
            try:
                upside = ((float(dcf_val) - float(price)) / float(price)) * 100
                dati["dcf_value"] = round(float(dcf_val), 2)
                dati["dcf_upside_pct"] = round(upside, 1)
            except (ValueError, TypeError):
                pass

    # Analyst ratings (consensus)
    rating = fmp_get(f"rating/{ticker}")
    if rating and len(rating) > 0:
        dati["rating"] = rating[0].get("rating")
        dati["rating_score"] = rating[0].get("ratingScore")

    # Ultimo earnings: beat o miss
    earnings = fmp_get(f"earnings-surprises/{ticker}", {"limit": 1})
    if earnings and len(earnings) > 0:
        e = earnings[0]
        est = e.get("estimatedEarning")
        act = e.get("actualEarningResult")
        if est and act:
            try:
                surprise_pct = ((float(act) - float(est)) / abs(float(est))) * 100
                dati["earnings_surprise_pct"] = round(surprise_pct, 1)
                dati["earnings_date"] = e.get("date")
            except (ValueError, TypeError, ZeroDivisionError):
                pass

    # Peers
    peers = fmp_get(f"stock_peers", {"symbol": ticker})
    if peers and len(peers) > 0 and isinstance(peers, list):
        peer_list = peers[0].get("peersList", [])[:5]
        dati["peers"] = peer_list

    # Earnings call transcript (più recente)
    transcript = fmp_get(f"earning_call_transcript/{ticker}", {"limit": 1})
    if transcript and len(transcript) > 0:
        full_text = transcript[0].get("content", "")
        # Tronca a 4000 caratteri per non saturare il context Gemini
        dati["earnings_transcript_excerpt"] = full_text[:4000] if full_text else None

    return dati


# ============================================================
# STEP 5 - NEWS (NewsAPI)
# ============================================================
def get_news(ticker, company_name=None):
    """Recupera ultime news per il ticker."""
    if not NEWSAPI_KEY:
        return []
    query = company_name if company_name else ticker
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f'"{query}"',
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "from": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
        "apiKey": NEWSAPI_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            articles = r.json().get("articles", [])
            return [
                {"titolo": a.get("title", ""), "fonte": a.get("source", {}).get("name", ""),
                 "data": a.get("publishedAt", "")[:10], "descrizione": (a.get("description") or "")[:300]}
                for a in articles
            ]
    except Exception as e:
        print(f"  NewsAPI errore {ticker}: {str(e)[:80]}")
    return []


# ============================================================
# STEP 6 - SINTESI AI (Gemini) E SCELTA FINALE
# ============================================================
def costruisci_dossier_per_ai(candidati_arricchiti):
    """Costruisce il payload testuale per Gemini con tutti i dati raccolti."""
    sezioni = []
    for c in candidati_arricchiti:
        scr = c["screening"]
        fnd = c.get("fundamentals", {})
        news = c.get("news", [])

        riga = f"\n{'=' * 50}\nTICKER: {scr['ticker']} | Prezzo: ${scr['prezzo']}\n"
        riga += f"Score tecnico: {scr['score']}/100\n"
        riga += (f"Rendimenti: 1S={scr.get('ret_1s')}% | 1M={scr.get('ret_1m')}% | "
                 f"3M={scr.get('ret_3m')}% | 6M={scr.get('ret_6m')}%\n")
        riga += (f"Sopra MA50: {scr.get('sopra_ma50')} | Sopra MA200: {scr.get('sopra_ma200')} | "
                 f"Dist 52w high: {scr.get('dist_52w_high')}%\n")
        riga += f"Volatilità 30g: {scr.get('vol_30g')}% | Volume trend up: {scr.get('volume_trend_positivo')}\n"

        if fnd:
            riga += "\nFONDAMENTALI:\n"
            for k in ["pe_ratio", "ps_ratio", "pb_ratio", "roe", "debt_equity", "fcf_yield",
                      "revenue_growth", "eps_growth", "dcf_value", "dcf_upside_pct",
                      "rating", "earnings_surprise_pct", "earnings_date"]:
                if fnd.get(k) is not None:
                    riga += f"  {k}: {fnd[k]}\n"
            if fnd.get("peers"):
                riga += f"  peers: {', '.join(fnd['peers'])}\n"
            if fnd.get("earnings_transcript_excerpt"):
                riga += f"\n  ULTIMO EARNINGS CALL (estratto):\n  {fnd['earnings_transcript_excerpt'][:2000]}\n"

        if news:
            riga += "\nNEWS RECENTI:\n"
            for n in news[:3]:
                riga += f"  - [{n['data']}] {n['titolo']} ({n['fonte']})\n"

        sezioni.append(riga)

    return "\n".join(sezioni)


def chiamata_gemini(prompt):
    """Chiama Gemini con fallback su più modelli e retry su 429."""
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY non configurata"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 3500,
            "topP": 0.9,
        }
    }

    ultimo_errore = "Nessun modello disponibile"
    for modello in GEMINI_MODELS_FALLBACK:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{modello}:generateContent?key={GEMINI_API_KEY}")
        for tentativo in range(3):
            try:
                r = requests.post(url, json=payload, timeout=60)
                if r.status_code == 429:
                    attesa = [10, 30, 60][tentativo]
                    print(f"⏳ 429 su {modello}, retry tra {attesa}s...")
                    time.sleep(attesa)
                    continue
                r.raise_for_status()
                data = r.json()
                txt = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                fr = data["candidates"][0].get("finishReason", "?")
                print(f"✅ Gemini OK con {modello} (finishReason={fr}, len={len(txt)})")
                return txt, None
            except Exception as e:
                ultimo_errore = f"{modello}: {str(e)[:120]}"
                print(f"❌ {ultimo_errore}")
                if tentativo < 2:
                    time.sleep(5)
    return None, ultimo_errore


def analisi_finale_ai(candidati_arricchiti):
    """Passa il dossier a Gemini e chiede UNA scelta con conviction score e tesi."""
    dossier = costruisci_dossier_per_ai(candidati_arricchiti)

    prompt = f"""Sei un analista finanziario senior con focus su small/mid cap S&P 500 e orizzonte temporale 1-2 mesi.

Hai davanti {len(candidati_arricchiti)} candidati pre-selezionati con screening tecnico. Analizza i loro fondamentali, news, earnings call recenti e scegli UNA SOLA azione con la più alta probabilità di apprezzamento nei prossimi 30-60 giorni.

Rispondi ESATTAMENTE in questo formato JSON valido (niente markdown, niente backtick, niente testo prima o dopo il JSON):

{{
  "ticker_scelto": "TICKER",
  "company_name": "Nome completo società",
  "conviction_score": <numero da 0 a 100>,
  "tesi_acquisto": "4-6 righe complete che spiegano perché questa azione salirà nei prossimi 1-2 mesi. Cita: rendimenti recenti, valutazione (P/E, DCF upside), trend medie mobili, qualità fondamentali, momentum, news rilevanti, earnings.",
  "catalisti_brevi": ["catalista 1 con data/timing se noto", "catalista 2", "catalista 3"],
  "rischi_principali": ["rischio 1", "rischio 2"],
  "target_orizzonte_giorni": <30, 45 o 60>
}}

Regole assolute:
- Scegli SOLO tra i ticker presenti nel dossier qui sotto
- conviction_score deve essere alto (>=70) solo se hai dati concreti che supportano la tesi
- Sii onesto: se nessun candidato è davvero forte, metti conviction_score basso (<70) — il bot non invierà segnale e va bene così
- Cita SEMPRE numeri concreti nella tesi (rendimenti, P/E, DCF, surprise%, ecc.)
- Non inventare dati che non sono nel dossier

DOSSIER CANDIDATI:
{dossier}
"""

    risposta, errore = chiamata_gemini(prompt)
    if not risposta:
        return None, errore

    # Parse JSON, robusto a wrappers
    try:
        # Rimuovi eventuali markdown fences
        clean = risposta.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()
        # Trova le parentesi graffe principali
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            clean = clean[start:end + 1]
        parsed = json.loads(clean)
        return parsed, None
    except Exception as e:
        return None, f"Parsing JSON fallito: {e}. Risposta grezza: {risposta[:500]}"


# ============================================================
# RISK MANAGEMENT - Stop loss e take profit
# ============================================================
def calcola_stop_target(prezzo, atr):
    """Stop loss e take profit basati su ATR con cap percentuali sensati."""
    if atr is None or atr <= 0:
        stop_pct = 7.0
        target_pct = 14.0
    else:
        stop_pct = (atr * ATR_STOP_MULTIPLIER / prezzo) * 100
        target_pct = (atr * ATR_TARGET_MULTIPLIER / prezzo) * 100
        stop_pct = max(MIN_STOP_PCT, min(MAX_STOP_PCT, stop_pct))
        target_pct = max(MIN_TARGET_PCT, min(MAX_TARGET_PCT, target_pct))

    stop_price = round(prezzo * (1 - stop_pct / 100), 2)
    target_price = round(prezzo * (1 + target_pct / 100), 2)
    rr = round(target_pct / stop_pct, 2)
    return {
        "stop_loss_price": stop_price,
        "stop_loss_pct": round(stop_pct, 2),
        "take_profit_price": target_price,
        "take_profit_pct": round(target_pct, 2),
        "risk_reward": rr,
    }


# ============================================================
# TELEGRAM
# ============================================================
def invia_telegram(messaggio):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram non configurato: salto invio")
        print("--- MESSAGGIO CHE SAREBBE STATO INVIATO ---")
        print(messaggio)
        return
    MAX_LEN = 4000
    parti = [messaggio[i:i + MAX_LEN] for i in range(0, len(messaggio), MAX_LEN)]
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for parte in parti:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": parte,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        r = requests.post(url, json=payload)
        print(f"Telegram inviato: {len(parte)} char, status {r.status_code}")


def formatta_messaggio_segnale(scelta, screening, risk):
    """Compone il messaggio Telegram per il segnale di acquisto."""
    ticker = scelta["ticker_scelto"]
    nome = scelta.get("company_name", ticker)
    conviction = scelta.get("conviction_score", 0)
    tesi = html.escape(scelta.get("tesi_acquisto", ""))
    catalisti = scelta.get("catalisti_brevi", [])
    rischi = scelta.get("rischi_principali", [])
    orizzonte = scelta.get("target_orizzonte_giorni", 45)

    oggi = datetime.now().strftime("%d/%m/%Y")

    msg = f"<b>🎯 SEGNALE DI ACQUISTO - {oggi}</b>\n"
    msg += "═══════════════════════\n\n"
    msg += f"<b>📈 {html.escape(nome)} ({ticker})</b>\n"
    msg += f"<b>Prezzo entry:</b> ${screening['prezzo']}\n"
    msg += f"<b>Conviction:</b> {conviction}/100 ⭐\n"
    msg += f"<b>Orizzonte:</b> {orizzonte} giorni\n\n"

    msg += "<b>🛡️ RISK MANAGEMENT</b>\n"
    msg += f"🔴 Stop loss: <b>${risk['stop_loss_price']}</b> (-{risk['stop_loss_pct']}%)\n"
    msg += f"🟢 Take profit: <b>${risk['take_profit_price']}</b> (+{risk['take_profit_pct']}%)\n"
    msg += f"⚖️ Risk/Reward: <b>{risk['risk_reward']}:1</b>\n\n"

    msg += "<b>📝 TESI DI ACQUISTO</b>\n"
    msg += f"{tesi}\n\n"

    if catalisti:
        msg += "<b>🚀 CATALISTI BREVE PERIODO</b>\n"
        for c in catalisti[:4]:
            msg += f"• {html.escape(str(c))}\n"
        msg += "\n"

    if rischi:
        msg += "<b>⚠️ RISCHI PRINCIPALI</b>\n"
        for r in rischi[:3]:
            msg += f"• {html.escape(str(r))}\n"
        msg += "\n"

    msg += "<b>📊 DATI TECNICI</b>\n"
    msg += (f"1M: {screening.get('ret_1m')}% | 3M: {screening.get('ret_3m')}% | "
            f"6M: {screening.get('ret_6m')}%\n")
    msg += (f"Sopra MA50: {'✅' if screening.get('sopra_ma50') else '❌'} | "
            f"Sopra MA200: {'✅' if screening.get('sopra_ma200') else '❌'}\n")
    msg += f"Volatilità 30g: {screening.get('vol_30g')}%\n\n"

    msg += (f"<i>⚠️ Analisi automatizzata, non costituisce consulenza finanziaria. "
            f"Investi solo capitale che puoi permetterti di perdere.</i>")
    return msg


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"🚀 Stock Picker avviato - {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # STEP 1: universo
    tickers = get_sp500_tickers()
    if not tickers:
        print("❌ Impossibile caricare universo")
        return

    # STEP 2-3: screening + top-N
    df = screening_universo(tickers)
    if df.empty:
        print("❌ Screening vuoto")
        return

    top_df = df.head(TOP_N_CANDIDATES)
    print(f"\n🎯 Top {TOP_N_CANDIDATES} candidati (per score):")
    print(top_df[["ticker", "prezzo", "score", "ret_1m", "ret_3m", "ret_6m"]].to_string())

    # STEP 4: arricchimento fondamentale (FMP)
    print(f"\n📚 Arricchimento fondamentale via FMP...")
    candidati_arricchiti = []
    for i, row in top_df.iterrows():
        ticker = row["ticker"]
        print(f"  [{i+1}/{len(top_df)}] {ticker}...")
        fnd = arricchisci_fondamentale(ticker) if FMP_API_KEY else {}
        candidati_arricchiti.append({
            "screening": row.to_dict(),
            "fundamentals": fnd,
            "news": [],
        })

    # STEP 5: news per i top-5
    print(f"\n📰 News via NewsAPI per top-{TOP_N_FOR_NEWS}...")
    for i, c in enumerate(candidati_arricchiti[:TOP_N_FOR_NEWS]):
        ticker = c["screening"]["ticker"]
        c["news"] = get_news(ticker) if NEWSAPI_KEY else []
        print(f"  [{i+1}/{TOP_N_FOR_NEWS}] {ticker}: {len(c['news'])} news")

    # STEP 6: sintesi AI
    print(f"\n🤖 Analisi finale con Gemini...")
    scelta, errore = analisi_finale_ai(candidati_arricchiti)

    if errore or not scelta:
        print(f"❌ AI fallita: {errore}")
        # Non invia nulla su Telegram: meglio silenzio che messaggio rotto
        return

    conviction = scelta.get("conviction_score", 0)
    ticker_scelto = scelta.get("ticker_scelto")
    print(f"\n📊 Scelta AI: {ticker_scelto} con conviction {conviction}/100")

    # Trova lo screening corrispondente
    screening_scelto = next(
        (c["screening"] for c in candidati_arricchiti if c["screening"]["ticker"] == ticker_scelto),
        None,
    )
    if not screening_scelto:
        print(f"❌ Ticker scelto {ticker_scelto} non trovato nel dossier")
        return

    # Check soglia conviction
    if conviction < CONVICTION_THRESHOLD:
        print(f"⏸️ Conviction {conviction} < soglia {CONVICTION_THRESHOLD}: nessun invio Telegram")
        return

    # Calcola stop loss / take profit
    risk = calcola_stop_target(screening_scelto["prezzo"], screening_scelto.get("atr"))

    # Compone e invia
    messaggio = formatta_messaggio_segnale(scelta, screening_scelto, risk)
    invia_telegram(messaggio)
    print("✅ Segnale inviato su Telegram")


if __name__ == "__main__":
    main()
