"""
CALIBRATION RUNNER
==================
Backtest statistico del punteggio di stockpicker.py.
Produce calibration.json con probabilità empiriche per bucket
(score_bin × regime). Lo stockpicker legge questo file per
rimpiazzare il conviction euristico con probabilità reali.

Strategia:
1) Scarica 4 anni di prezzi per tutti i ticker S&P500 (1 chiamata bulk yfinance)
2) Per ogni data check-point settimanale (~200 osservazioni):
   - Calcola regime del giorno (SPY vs MA200, VIX bin)
   - Per ~80 ticker random calcola lo score
   - Track forward 45gg return
3) Aggrega per bucket (score_bin × regime):
   - p_pos_45d: % rendimenti positivi
   - p_target_10pct: % che hanno toccato +10% entro 45gg
   - p_stop_8pct: % che hanno toccato -8% entro 45gg
   - avg_ret_45d
4) Salva calibration.json e committalo nel repo

Tempo atteso: 10-20 min su GitHub Actions.
"""

import os
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
ANNI_LOOKBACK = 4               # anni di storia da scaricare
FORWARD_DAYS = 45               # orizzonte di valutazione
TARGET_PCT = 10.0               # soglia target (+10%)
STOP_PCT = -8.0                 # soglia stop loss (-8%)
WEEKS_BETWEEN_CHECKS = 1        # checkpoint settimanale
TICKERS_PER_CHECK = 80          # quanti ticker valutare per check-point (random sampling)
MIN_SAMPLES_BUCKET = 20         # minimo per dichiarare un bucket affidabile
BATCH_SIZE = 50                 # ticker per chiamata yf.download (evita rate limit)

OUTPUT_FILE = Path(__file__).parent / "calibration.json"

# Score bins
SCORE_BINS = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]


# ============================================================
# UTILITY
# ============================================================
def get_sp500_tickers():
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return df["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        print(f"⚠️ Errore Wikipedia: {e}")
        return []


def scarica_storici(tickers):
    """
    Scarica prezzi in BATCH da BATCH_SIZE ticker per evitare rate limit/timeout di yfinance.
    Estrae direttamente Close/Volume/High/Low aggregando i batch.
    Ritorna 4 DataFrame separati (close, volume, high, low) con i ticker come colonne.
    """
    end = datetime.now()
    start = end - timedelta(days=ANNI_LOOKBACK * 365 + 100)

    n_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📥 Download {len(tickers)} ticker in {n_batches} batch da {BATCH_SIZE}, "
          f"{ANNI_LOOKBACK} anni di storia...")

    close_dict, volume_dict, high_dict, low_dict = {}, {}, {}, {}

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        n_bat = i // BATCH_SIZE + 1
        try:
            dati = yf.download(batch, start=start, end=end, auto_adjust=True,
                               progress=False, group_by="ticker", threads=True)
            if dati is None or dati.empty:
                print(f"  Batch {n_bat}/{n_batches}: vuoto, salto")
                continue

            # Caso 1: MultiIndex (ticker, field) — quando batch ha più ticker
            # Caso 2: colonne semplici (Close, High...) — quando batch ha 1 solo ticker
            estratti = 0
            for tk in batch:
                try:
                    if isinstance(dati.columns, pd.MultiIndex):
                        if tk not in dati.columns.get_level_values(0):
                            continue
                        sub = dati[tk]
                    else:
                        # Singolo ticker nel batch
                        sub = dati
                    if sub is None or sub.empty:
                        continue
                    if "Close" not in sub.columns:
                        continue
                    close_dict[tk] = sub["Close"]
                    volume_dict[tk] = sub["Volume"]
                    high_dict[tk] = sub["High"]
                    low_dict[tk] = sub["Low"]
                    estratti += 1
                except Exception:
                    continue
            print(f"  Batch {n_bat}/{n_batches}: estratti {estratti}/{len(batch)}")
        except Exception as e:
            print(f"  Batch {n_bat}/{n_batches}: errore {str(e)[:80]}")
            continue

    close = pd.DataFrame(close_dict)
    volume = pd.DataFrame(volume_dict)
    high = pd.DataFrame(high_dict)
    low = pd.DataFrame(low_dict)
    print(f"✅ Download completato: {close.shape[1]}/{len(tickers)} ticker validi, "
          f"{close.shape[0]} giorni")
    return close, volume, high, low


# ============================================================
# SCORE (riprodotto da stockpicker.py - mantienine la sincronia se modifichi)
# ============================================================
def calcola_score_storico(close_serie, volume_serie, high_serie, low_serie, idx_oggi):
    """
    Calcola score 0-100 alla data idx_oggi usando solo dati DISPONIBILI a quella data.
    Replica calcola_score() di stockpicker.py senza regime-adjustment (regime tracciato separato).
    """
    if idx_oggi < 200:
        return None

    storico_close = close_serie.iloc[:idx_oggi + 1].dropna()
    if len(storico_close) < 200:
        return None

    prezzo = float(storico_close.iloc[-1])
    if pd.isna(prezzo) or prezzo <= 0:
        return None

    def ret(d):
        if len(storico_close) <= d:
            return None
        past = storico_close.iloc[-d]
        if pd.isna(past) or past <= 0:
            return None
        return ((prezzo - past) / past) * 100

    r1m = ret(21)
    r3m = ret(63)
    r6m = ret(126)
    if r3m is None:
        return None

    ma50 = storico_close.rolling(50).mean().iloc[-1]
    ma200 = storico_close.rolling(200).mean().iloc[-1]
    if pd.isna(ma50) or pd.isna(ma200):
        return None

    high_252 = high_serie.iloc[:idx_oggi + 1].tail(252).max()
    if pd.isna(high_252) or high_252 <= 0:
        return None
    dist = ((prezzo - high_252) / high_252) * 100

    rg = storico_close.pct_change().dropna().tail(30)
    vol30 = float(rg.std() * np.sqrt(252) * 100) if len(rg) > 0 else None

    vol_serie_30 = volume_serie.iloc[:idx_oggi + 1].tail(30)
    if len(vol_serie_30) >= 10:
        vol_trend = float(vol_serie_30.tail(10).mean()) > float(vol_serie_30.mean()) * 1.1
    else:
        vol_trend = False

    # Score uguale all'euristica base in stockpicker
    score = 50.0
    if r6m is not None:
        if r6m > 20: score += 15
        elif r6m > 10: score += 10
        elif r6m > 0: score += 5
        elif r6m < -20: score -= 15
        elif r6m < -10: score -= 8
    if r3m is not None:
        if r3m > 10: score += 8
        elif r3m > 0: score += 4
        elif r3m < -15: score -= 10
    if r1m is not None and r6m is not None:
        if -10 < r1m < -2 and r6m > 10:
            score += 15
        elif r1m > 15:
            score -= 5
    if prezzo > ma200: score += 8
    if prezzo > ma50: score += 5
    if -15 < dist < -3: score += 8
    elif dist < -35: score -= 12
    if vol30 is not None:
        if vol30 > 80: score -= 12
        elif vol30 > 55: score -= 5
        elif 20 < vol30 < 40: score += 3
    if vol_trend: score += 5

    return max(0, min(100, score))


# ============================================================
# REGIME (semplificato per backtest)
# ============================================================
def rileva_regime_storico(spy_close, vix_close, idx):
    """Regime alla data idx: bull-quiet, bull-volatile, range, bear-quiet, bear-volatile."""
    if idx < 200:
        return "neutral"
    spy_oggi = spy_close.iloc[idx]
    spy_ma200 = spy_close.iloc[max(0, idx - 200):idx + 1].mean()
    if pd.isna(spy_oggi) or pd.isna(spy_ma200):
        return "neutral"
    diff = (spy_oggi - spy_ma200) / spy_ma200 * 100

    vix_val = vix_close.iloc[idx] if not pd.isna(vix_close.iloc[idx]) else 20

    if diff > 3:
        return "bull-quiet" if vix_val < 20 else "bull-volatile"
    if diff < -3:
        return "bear-volatile" if vix_val > 25 else "bear-quiet"
    return "range"


# ============================================================
# FORWARD RETURN + HIT TARGET/STOP
# ============================================================
def stat_forward(close_serie, high_serie, low_serie, idx_oggi, days=FORWARD_DAYS):
    """
    Calcola sui prossimi 'days' giorni:
    - rendimento finale
    - se ha toccato +10% (target)
    - se ha toccato -8% (stop)
    Ritorna None se dati insufficienti.
    """
    if idx_oggi + days >= len(close_serie):
        return None
    prezzo_entry = close_serie.iloc[idx_oggi]
    if pd.isna(prezzo_entry) or prezzo_entry <= 0:
        return None

    future_close = close_serie.iloc[idx_oggi + 1: idx_oggi + days + 1]
    future_high = high_serie.iloc[idx_oggi + 1: idx_oggi + days + 1]
    future_low = low_serie.iloc[idx_oggi + 1: idx_oggi + days + 1]

    if future_close.dropna().empty:
        return None

    prezzo_finale = future_close.dropna().iloc[-1]
    ret_finale = (prezzo_finale - prezzo_entry) / prezzo_entry * 100

    max_alto = future_high.max()
    min_basso = future_low.min()
    hit_target = (max_alto - prezzo_entry) / prezzo_entry * 100 >= TARGET_PCT
    hit_stop = (min_basso - prezzo_entry) / prezzo_entry * 100 <= STOP_PCT

    return {
        "ret_finale": float(ret_finale),
        "hit_target": bool(hit_target),
        "hit_stop": bool(hit_stop),
    }


# ============================================================
# BACKTEST CORE
# ============================================================
def esegui_backtest():
    tickers = get_sp500_tickers()
    if not tickers:
        print("❌ Universo vuoto")
        return None
    print(f"✅ Universo: {len(tickers)} ticker")

    # Scarica TUTTO in batch (compresi SPY e VIX per regime detection)
    close, volume, high, low = scarica_storici(tickers + ["SPY", "^VIX"])

    if close.empty or close.shape[1] < 50:
        print(f"❌ Troppi pochi dati: solo {close.shape[1]} ticker validi")
        return None

    if "SPY" not in close.columns or "^VIX" not in close.columns:
        print("⚠️ SPY o VIX mancanti — riprovo a scaricarli singolarmente")
        try:
            for sp in ["SPY", "^VIX"]:
                if sp not in close.columns:
                    h = yf.Ticker(sp).history(period=f"{ANNI_LOOKBACK}y", auto_adjust=True)
                    if not h.empty:
                        close[sp] = h["Close"]
                        volume[sp] = h["Volume"]
                        high[sp] = h["High"]
                        low[sp] = h["Low"]
        except Exception as e:
            print(f"  Errore recupero SPY/VIX: {e}")
        if "SPY" not in close.columns or "^VIX" not in close.columns:
            print("❌ SPY o VIX ancora mancanti, impossibile rilevare regime")
            return None

    spy_close = close["SPY"]
    vix_close = close["^VIX"]

    dates = close.index
    if len(dates) < 300:
        print(f"❌ Storico troppo breve: {len(dates)} giorni")
        return None
    print(f"  Range date: {dates[0].date()} → {dates[-1].date()}, {len(dates)} giorni")

    # Aggregatore
    risultati = {}  # bucket_key -> list of dict {ret, hit_target, hit_stop}

    # Check-points settimanali
    weekly_indices = list(range(200, len(dates) - FORWARD_DAYS, 5))
    random.seed(42)
    print(f"🔄 Backtest su {len(weekly_indices)} check-points settimanali")

    altri_tickers = [t for t in tickers if t in close.columns and t not in ("SPY", "^VIX")]

    for i, idx_oggi in enumerate(weekly_indices):
        regime = rileva_regime_storico(spy_close, vix_close, idx_oggi)
        # Campiona ticker random per ridurre tempo
        sample = random.sample(altri_tickers, min(TICKERS_PER_CHECK, len(altri_tickers)))

        for tk in sample:
            try:
                score = calcola_score_storico(
                    close[tk], volume[tk], high[tk], low[tk], idx_oggi
                )
                if score is None:
                    continue
                fwd = stat_forward(close[tk], high[tk], low[tk], idx_oggi)
                if fwd is None:
                    continue

                # Bin
                bin_lo, bin_hi = None, None
                for blo, bhi in SCORE_BINS:
                    if blo <= score < bhi:
                        bin_lo, bin_hi = blo, bhi
                        break
                if bin_lo is None:
                    continue

                # Bucket specifico per regime + bucket generico "all"
                for key in [
                    f"score_{bin_lo}_{bin_hi}_regime_{regime}",
                    f"score_{bin_lo}_{bin_hi}_regime_all",
                ]:
                    risultati.setdefault(key, []).append(fwd)
            except Exception:
                continue

        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(weekly_indices)} check-points processati")

    # Aggrega in probabilità
    buckets = {}
    n_totale = 0
    for key, lista in risultati.items():
        n = len(lista)
        n_totale += n if "_all" not in key else 0
        if n < MIN_SAMPLES_BUCKET:
            continue
        p_pos = sum(1 for x in lista if x["ret_finale"] > 0) / n
        p_target = sum(1 for x in lista if x["hit_target"]) / n
        p_stop = sum(1 for x in lista if x["hit_stop"]) / n
        avg_ret = float(np.mean([x["ret_finale"] for x in lista]))
        med_ret = float(np.median([x["ret_finale"] for x in lista]))
        buckets[key] = {
            "n_samples": n,
            "p_pos_45d": round(p_pos, 3),
            "p_target_10pct": round(p_target, 3),
            "p_stop_8pct": round(p_stop, 3),
            "avg_ret_45d": round(avg_ret, 2),
            "median_ret_45d": round(med_ret, 2),
        }

    output = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "data_period": f"{dates[0].date()} → {dates[-1].date()}",
        "forward_days": FORWARD_DAYS,
        "target_pct": TARGET_PCT,
        "stop_pct": STOP_PCT,
        "n_checkpoints": len(weekly_indices),
        "tickers_per_check": TICKERS_PER_CHECK,
        "n_samples": n_totale,
        "buckets": buckets,
    }
    return output


def main():
    print(f"🚀 Calibration runner avviato - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    out = None
    try:
        out = esegui_backtest()
    except Exception as e:
        import traceback
        print(f"❌ Eccezione in esegui_backtest: {e}")
        traceback.print_exc()

    # Anche se il backtest fallisce, scrivi sempre un file con metadati di stato
    if not out:
        out = {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "status": "FAILED",
            "message": "Backtest non completato. Lo stockpicker continuerà con euristica.",
            "n_samples": 0,
            "buckets": {},
        }
        with open(OUTPUT_FILE, "w") as f:
            json.dump(out, f, indent=2)
        print(f"⚠️ Backtest fallito. Scritto placeholder in {OUTPUT_FILE}")
        return

    out["status"] = "OK"
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✅ Salvato in {OUTPUT_FILE}")
    print(f"   N samples totali: {out['n_samples']}")
    print(f"   N buckets validi: {len(out['buckets'])}")

    print("\n=== BUCKET PRINCIPALI ===")
    for k in sorted(out["buckets"].keys()):
        if "_all" in k:
            b = out["buckets"][k]
            print(f"  {k}: n={b['n_samples']} | p_target={b['p_target_10pct']} | "
                  f"p_stop={b['p_stop_8pct']} | avg_ret={b['avg_ret_45d']}%")


if __name__ == "__main__":
    main()
