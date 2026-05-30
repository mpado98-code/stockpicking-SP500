"""
PERFORMANCE TRACKER
===================
Gira settimanalmente (venerdì sera post-chiusura USA).
Per ogni posizione OPEN in positions.json:
  - Recupera la storia prezzi dall'entry ad oggi via yfinance
  - Controlla se ha toccato take_profit (CLOSED_WIN), stop_loss (CLOSED_LOSS)
    o se è passato l'orizzonte (CLOSED_TIMEOUT con prezzo corrente)
  - Aggiorna lo stato e calcola P&L

Poi calcola statistiche aggregate (win rate, expectancy, breakdown per regime
e per bucket conviction) e manda report Telegram.

Output: positions.json aggiornato + messaggio Telegram.
"""

import os
import json
import html
from datetime import datetime, timedelta
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN_PICKER")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID_PICKER")

POSITIONS_FILE = Path(__file__).parent / "positions.json"


# ============================================================
# CONTROLLO STATO POSIZIONE
# ============================================================
def controlla_posizione(pos):
    """
    Ritorna dict aggiornato con status finale, close_date, close_price, pnl_pct, days_held.
    Logica:
    - Scarica history da entry_date a oggi
    - Cerca il primo giorno in cui High >= take_profit (WIN) o Low <= stop_loss (LOSS)
    - Se nessuno dei due e days_held > orizzonte → TIMEOUT a close price corrente
    - Altrimenti la posizione resta OPEN
    """
    ticker = pos["ticker"]
    entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
    today = datetime.now()
    days_held = (today - entry_date).days

    try:
        # Aggiungo buffer di 1 giorno prima per sicurezza
        start = entry_date - timedelta(days=1)
        end = today + timedelta(days=1)
        h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if h.empty:
            print(f"  ⚠️ {ticker}: nessun dato yfinance, lascio OPEN")
            return pos

        if h.index.tz is not None:
            h.index = h.index.tz_localize(None)

        # Filtra dal giorno DOPO l'entry (non includere il giorno entry stesso, perché potrei aver entrato sopra/sotto in intraday)
        h_post = h[h.index > entry_date]
        if h_post.empty:
            pos["days_held"] = days_held
            return pos

        target = pos["take_profit_price"]
        stop = pos["stop_loss_price"]
        orizzonte = pos.get("orizzonte_giorni", 45)

        # Trova prima data hit
        hit_target_dates = h_post.index[h_post["High"] >= target]
        hit_stop_dates = h_post.index[h_post["Low"] <= stop]

        prima_target = hit_target_dates[0] if len(hit_target_dates) > 0 else None
        prima_stop = hit_stop_dates[0] if len(hit_stop_dates) > 0 else None

        close_date = None
        close_price = None
        close_reason = None

        if prima_target and prima_stop:
            # Entrambi colpiti: il primo cronologicamente vince
            if prima_target <= prima_stop:
                close_date, close_price, close_reason = prima_target, target, "TARGET"
            else:
                close_date, close_price, close_reason = prima_stop, stop, "STOP"
        elif prima_target:
            close_date, close_price, close_reason = prima_target, target, "TARGET"
        elif prima_stop:
            close_date, close_price, close_reason = prima_stop, stop, "STOP"
        elif days_held >= orizzonte:
            # Timeout: chiudo al prezzo corrente
            close_date = h_post.index[-1]
            close_price = float(h_post["Close"].iloc[-1])
            close_reason = "TIMEOUT"
        else:
            # Posizione ancora aperta
            pos["days_held"] = days_held
            # Aggiungo prezzo corrente per il report (non chiude)
            pos["current_price"] = float(h_post["Close"].iloc[-1])
            pos["current_pnl_pct"] = round(
                (pos["current_price"] - pos["entry_price"]) / pos["entry_price"] * 100, 2
            )
            return pos

        # Aggiorna record
        pos["status"] = f"CLOSED_{close_reason}"
        pos["close_date"] = close_date.strftime("%Y-%m-%d")
        pos["close_price"] = round(float(close_price), 2)
        pos["close_reason"] = close_reason
        pos["pnl_pct"] = round((close_price - pos["entry_price"]) / pos["entry_price"] * 100, 2)
        pos["days_held"] = (close_date - entry_date).days
        return pos

    except Exception as e:
        print(f"  ⚠️ {ticker}: errore controllo ({e}), lascio OPEN")
        return pos


# ============================================================
# STATISTICHE AGGREGATE
# ============================================================
def calcola_statistiche(posizioni):
    chiuse = [p for p in posizioni if p["status"].startswith("CLOSED")]
    aperte = [p for p in posizioni if p["status"] == "OPEN"]

    stats = {
        "totale": len(posizioni),
        "aperte": len(aperte),
        "chiuse": len(chiuse),
        "win_rate": None,
        "n_win": 0,
        "n_loss": 0,
        "n_timeout": 0,
        "avg_winner_pct": None,
        "avg_loser_pct": None,
        "avg_trade_pct": None,
        "expectancy": None,
        "max_drawdown_pct": None,
        "cumulative_pct": None,
        "avg_days_held": None,
        "breakdown_regime": {},
        "breakdown_conviction": {},
    }

    if not chiuse:
        return stats

    wins = [p for p in chiuse if p["close_reason"] == "TARGET"]
    losses = [p for p in chiuse if p["close_reason"] == "STOP"]
    timeouts = [p for p in chiuse if p["close_reason"] == "TIMEOUT"]

    # Win count: TARGET + TIMEOUT con pnl > 0
    timeouts_win = [p for p in timeouts if (p["pnl_pct"] or 0) > 0]
    timeouts_loss = [p for p in timeouts if (p["pnl_pct"] or 0) <= 0]

    n_win = len(wins) + len(timeouts_win)
    n_loss = len(losses) + len(timeouts_loss)
    stats["n_win"] = n_win
    stats["n_loss"] = n_loss
    stats["n_timeout"] = len(timeouts)
    stats["win_rate"] = round(n_win / (n_win + n_loss) * 100, 1) if (n_win + n_loss) > 0 else 0

    all_pnl = [p["pnl_pct"] for p in chiuse if p["pnl_pct"] is not None]
    if all_pnl:
        stats["avg_trade_pct"] = round(float(np.mean(all_pnl)), 2)
        stats["cumulative_pct"] = round(float(np.sum(all_pnl)), 2)

    winner_pnl = [p["pnl_pct"] for p in wins + timeouts_win if p["pnl_pct"] is not None]
    loser_pnl = [p["pnl_pct"] for p in losses + timeouts_loss if p["pnl_pct"] is not None]
    if winner_pnl:
        stats["avg_winner_pct"] = round(float(np.mean(winner_pnl)), 2)
    if loser_pnl:
        stats["avg_loser_pct"] = round(float(np.mean(loser_pnl)), 2)

    # Expectancy = win_rate*avg_win + loss_rate*avg_loss
    if stats["avg_winner_pct"] is not None and stats["avg_loser_pct"] is not None:
        wr = stats["win_rate"] / 100
        stats["expectancy"] = round(
            wr * stats["avg_winner_pct"] + (1 - wr) * stats["avg_loser_pct"], 2
        )

    # Max drawdown sulla equity curve cumulativa (in ordine di close_date)
    chiuse_ord = sorted(chiuse, key=lambda x: x.get("close_date") or "")
    equity = np.cumsum([p["pnl_pct"] for p in chiuse_ord if p["pnl_pct"] is not None])
    if len(equity) > 0:
        running_max = np.maximum.accumulate(equity)
        dd = equity - running_max
        stats["max_drawdown_pct"] = round(float(np.min(dd)), 2)

    days_held = [p["days_held"] for p in chiuse if p.get("days_held") is not None]
    if days_held:
        stats["avg_days_held"] = round(float(np.mean(days_held)), 1)

    # Breakdown per regime
    regimi = {}
    for p in chiuse:
        r = p.get("regime_at_entry", "neutral")
        regimi.setdefault(r, []).append(p)
    for r, lista in regimi.items():
        ws = sum(1 for p in lista if p["close_reason"] == "TARGET" or
                 (p["close_reason"] == "TIMEOUT" and (p["pnl_pct"] or 0) > 0))
        avg = float(np.mean([p["pnl_pct"] for p in lista if p["pnl_pct"] is not None])) if lista else 0
        stats["breakdown_regime"][r] = {
            "n": len(lista),
            "win_rate": round(ws / len(lista) * 100, 1) if lista else 0,
            "avg_pct": round(avg, 2),
        }

    # Breakdown per bin conviction
    bins = {"60-70": [], "70-80": [], "80-90": [], "90+": []}
    for p in chiuse:
        c = p.get("conviction", 0)
        if c >= 90: bins["90+"].append(p)
        elif c >= 80: bins["80-90"].append(p)
        elif c >= 70: bins["70-80"].append(p)
        elif c >= 60: bins["60-70"].append(p)
    for bin_name, lista in bins.items():
        if not lista:
            continue
        ws = sum(1 for p in lista if p["close_reason"] == "TARGET" or
                 (p["close_reason"] == "TIMEOUT" and (p["pnl_pct"] or 0) > 0))
        avg = float(np.mean([p["pnl_pct"] for p in lista if p["pnl_pct"] is not None]))
        stats["breakdown_conviction"][bin_name] = {
            "n": len(lista),
            "win_rate": round(ws / len(lista) * 100, 1),
            "avg_pct": round(avg, 2),
        }

    return stats


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
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID, "text": p, "parse_mode": "HTML",
            "disable_web_page_preview": True
        })
        print(f"Telegram: {len(p)} char, status {r.status_code}")


def emoji_status(pos):
    s = pos.get("status", "")
    if s == "CLOSED_TARGET" or (s == "CLOSED_TIMEOUT" and (pos.get("pnl_pct") or 0) > 0):
        return "✅"
    if s == "CLOSED_STOP" or (s == "CLOSED_TIMEOUT" and (pos.get("pnl_pct") or 0) <= 0):
        return "❌"
    if s == "OPEN":
        pnl = pos.get("current_pnl_pct", 0) or 0
        if pnl > 5: return "🟢"
        if pnl < -3: return "🔴"
        return "🟡"
    return "⏰"


def formatta_report(posizioni, stats, chiuse_questa_settimana):
    oggi = datetime.now().strftime("%d/%m/%Y")
    aperte = [p for p in posizioni if p["status"] == "OPEN"]
    chiuse = sorted([p for p in posizioni if p["status"].startswith("CLOSED")],
                    key=lambda x: x.get("close_date") or "", reverse=True)

    m = f"<b>📊 PERFORMANCE REPORT - {oggi}</b>\n"
    m += "═══════════════════════\n\n"

    # Stato generale
    m += "<b>📈 STATO GENERALE</b>\n"
    m += f"Trade totali: <b>{stats['totale']}</b> "
    m += f"(aperti: {stats['aperte']}, chiusi: {stats['chiuse']})\n"
    if stats["chiuse"] > 0:
        m += f"Win rate: <b>{stats['win_rate']}%</b> ({stats['n_win']}W / {stats['n_loss']}L)\n"
        if stats["avg_trade_pct"] is not None:
            m += f"Rendimento medio/trade: <b>{stats['avg_trade_pct']:+.2f}%</b>\n"
        if stats["expectancy"] is not None:
            m += f"Expectancy: <b>{stats['expectancy']:+.2f}%</b> per trade\n"
        if stats["cumulative_pct"] is not None:
            m += f"Rendimento cumulato: <b>{stats['cumulative_pct']:+.2f}%</b>\n"
        if stats["max_drawdown_pct"] is not None:
            m += f"Max drawdown: <b>{stats['max_drawdown_pct']:+.2f}%</b>\n"
        if stats["avg_days_held"] is not None:
            m += f"Holding medio: <b>{stats['avg_days_held']} giorni</b>\n"
    m += "\n"

    # Posizioni aperte
    if aperte:
        m += f"<b>💼 POSIZIONI APERTE ({len(aperte)})</b>\n"
        for p in aperte:
            tk = p["ticker"]
            dh = p.get("days_held", 0) or 0
            oriz = p.get("orizzonte_giorni", 45)
            ep = p["entry_price"]
            cp = p.get("current_price")
            pnl = p.get("current_pnl_pct")
            conv = p.get("conviction", 0)
            if cp is not None:
                m += (f"{emoji_status(p)} <b>{tk}</b> | {dh}g/{oriz}g | "
                      f"${ep} → ${cp} ({pnl:+.2f}%) | conv {conv}\n")
            else:
                m += f"{emoji_status(p)} <b>{tk}</b> | {dh}g/{oriz}g | entry ${ep} | conv {conv}\n"
        m += "\n"

    # Chiuse questa settimana
    if chiuse_questa_settimana:
        m += f"<b>🆕 CHIUSE QUESTA SETTIMANA ({len(chiuse_questa_settimana)})</b>\n"
        for p in chiuse_questa_settimana:
            pnl = p.get("pnl_pct", 0)
            reason = p.get("close_reason", "?")
            days = p.get("days_held", 0)
            conv = p.get("conviction", 0)
            m += (f"{emoji_status(p)} <b>{p['ticker']}</b> | {pnl:+.2f}% in {days}g "
                  f"({reason}) | conv {conv}\n")
        m += "\n"

    # Ultimi 5 chiusi
    if chiuse:
        recenti = chiuse[:5]
        m += f"<b>📜 ULTIMI {len(recenti)} CHIUSI</b>\n"
        for p in recenti:
            pnl = p.get("pnl_pct", 0)
            reason = p.get("close_reason", "?")
            days = p.get("days_held", 0)
            cd = p.get("close_date", "?")
            m += f"{emoji_status(p)} <b>{p['ticker']}</b> | {pnl:+.2f}% in {days}g ({reason}) | {cd}\n"
        m += "\n"

    # Breakdown
    if stats["breakdown_regime"]:
        m += "<b>🌍 BREAKDOWN PER REGIME</b>\n"
        for r, st in stats["breakdown_regime"].items():
            m += f"  {r}: {st['n']} trade | Win {st['win_rate']}% | Avg {st['avg_pct']:+.2f}%\n"
        m += "\n"

    if stats["breakdown_conviction"]:
        m += "<b>⭐ BREAKDOWN PER CONVICTION</b>\n"
        for bn, st in stats["breakdown_conviction"].items():
            m += f"  {bn}: {st['n']} trade | Win {st['win_rate']}% | Avg {st['avg_pct']:+.2f}%\n"
        m += "\n"

    m += "<i>⚠️ Performance basata su ipotesi di esecuzione perfetta a stop/target. " \
         "I risultati reali possono differire per slippage e gap di mercato.</i>"
    return m


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"🚀 Tracker avviato - {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if not POSITIONS_FILE.exists():
        print("ℹ️ positions.json non esiste ancora. Nessuna posizione da tracciare.")
        return

    with open(POSITIONS_FILE) as f:
        posizioni = json.load(f)

    if not posizioni:
        print("ℹ️ Nessuna posizione registrata.")
        return

    print(f"📋 {len(posizioni)} posizioni totali, "
          f"{sum(1 for p in posizioni if p['status'] == 'OPEN')} aperte")

    # Snapshot pre-update per identificare nuove chiusure
    chiuse_prima = {p["id"] for p in posizioni if p["status"].startswith("CLOSED")}

    # Aggiorna ogni posizione OPEN
    print("\n🔍 Controllo posizioni aperte...")
    for i, p in enumerate(posizioni):
        if p["status"] != "OPEN":
            continue
        print(f"  [{i + 1}] {p['ticker']}")
        posizioni[i] = controlla_posizione(p)

    # Salva file aggiornato
    with open(POSITIONS_FILE, "w") as f:
        json.dump(posizioni, f, indent=2)

    # Identifica chiusure di questa settimana
    chiuse_questa_settimana = [
        p for p in posizioni
        if p["status"].startswith("CLOSED") and p["id"] not in chiuse_prima
    ]

    # Statistiche
    stats = calcola_statistiche(posizioni)
    print(f"\n📊 Stats: win_rate={stats['win_rate']}% | expectancy={stats['expectancy']}% | "
          f"cum={stats['cumulative_pct']}%")

    # Telegram report
    msg = formatta_report(posizioni, stats, chiuse_questa_settimana)
    invia_telegram(msg)
    print("✅ Report inviato")


if __name__ == "__main__":
    main()
