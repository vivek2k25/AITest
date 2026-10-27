"""
StockAI V6.2.1 — Regime & Signal Quality Diagnostic Backtest
============================================================
Diagnostic only: does not modify production scanner V6.1.

Method:
- 5 years of daily data; next-day OPEN entry
- V6.1 baseline signal: FULL_BULLISH OR (20-day breakout + volume >= 1.5x + FULL_BULLISH)
- Stop: 1.5 ATR; T1: 1.5R; T2: 2.5R; maximum hold: 20 sessions
- 0.20% round-trip costs
- One open position at a time per symbol
- Conservative daily-bar ordering: stop checked before targets
- Technical only; fundamentals/news are not included

Outputs (saved to the current working directory):
  backtest_v6_2_1_trades.csv
  backtest_v6_2_1_condition_analysis.csv
  backtest_v6_2_1_stock_summary.csv
  backtest_v6_2_1_stop_clustering.csv

Use results descriptively. Do not adopt an in-sample winner without out-of-sample validation.
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = [
    "WIPRO.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "RELIANCE.NS",
    "HFCL.NS", "TMCV.NS", "ADANIPORTS.NS", "QPOWER.NS"
]
PERIOD = "5y"
ATR_PERIOD = 14
ADX_PERIOD = 14
RSI_PERIOD = 14
ATR_STOP_MULT = 1.50
T1_R = 1.50
T2_R = 2.50
MAX_HOLD_DAYS = 20
ROUND_TRIP_COST = 0.0020
MIN_SAMPLE = 5
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def calc_atr(df, period=ATR_PERIOD):
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_adx_di(df, period=ADX_PERIOD):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs(),
    ], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_w = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_w = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_w / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_w / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def calc_supertrend_direction(df, period=10, multiplier=3.0):
    """
    Calculate SuperTrend direction without propagating initial NaN bands.

    Returns:
        +1.0 = bullish
        -1.0 = bearish
        NaN  = indicator not yet initialized
    """
    mid = (df["High"] + df["Low"]) / 2
    atr = calc_atr(df, period)

    upper = mid + multiplier * atr
    lower = mid - multiplier * atr

    final_upper = pd.Series(np.nan, index=df.index, dtype=float)
    final_lower = pd.Series(np.nan, index=df.index, dtype=float)
    direction = pd.Series(np.nan, index=df.index, dtype=float)

    valid_positions = np.flatnonzero(atr.notna().to_numpy())

    # Not enough data to initialize SuperTrend.
    if len(valid_positions) == 0:
        return direction

    start = int(valid_positions[0])

    # Initialize the bands at the first valid ATR row.
    final_upper.iloc[start] = upper.iloc[start]
    final_lower.iloc[start] = lower.iloc[start]

    direction.iloc[start] = (
        1.0
        if df["Close"].iloc[start] >= mid.iloc[start]
        else -1.0
    )

    for i in range(start + 1, len(df)):

        if pd.isna(atr.iloc[i]):
            continue

        previous_upper = final_upper.iloc[i - 1]
        previous_lower = final_lower.iloc[i - 1]

        # Defensive recovery if a prior band is unexpectedly missing.
        if pd.isna(previous_upper) or pd.isna(previous_lower):
            final_upper.iloc[i] = upper.iloc[i]
            final_lower.iloc[i] = lower.iloc[i]

            previous_direction = direction.iloc[i - 1]

            if pd.isna(previous_direction):
                direction.iloc[i] = (
                    1.0
                    if df["Close"].iloc[i] >= mid.iloc[i]
                    else -1.0
                )
            else:
                direction.iloc[i] = previous_direction

            continue

        previous_close = df["Close"].iloc[i - 1]
        close = df["Close"].iloc[i]

        # Calculate the trailing upper band.
        if (
            upper.iloc[i] < previous_upper
            or previous_close > previous_upper
        ):
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = previous_upper

        # Calculate the trailing lower band.
        if (
            lower.iloc[i] > previous_lower
            or previous_close < previous_lower
        ):
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = previous_lower

        previous_direction = direction.iloc[i - 1]

        # Determine the current trend using the previous trailing bands.
        if previous_direction < 0 and close > previous_upper:
            direction.iloc[i] = 1.0

        elif previous_direction > 0 and close < previous_lower:
            direction.iloc[i] = -1.0

        else:
            direction.iloc[i] = previous_direction

    return direction


def add_indicators(df):
    x = df.copy()
    x["ATR"] = calc_atr(x)
    x["RSI"] = calc_rsi(x["Close"])
    for span in (20, 50, 200):
        x[f"EMA{span}"] = x["Close"].ewm(span=span, adjust=False).mean()
    macd_line = x["Close"].ewm(span=12, adjust=False).mean() - x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = macd_line
    x["MACD_SIGNAL"] = macd_line.ewm(span=9, adjust=False).mean()
    x["ADX"], x["+DI"], x["-DI"] = calc_adx_di(x)
    x["ST_DIRECTION"] = calc_supertrend_direction(x)
    x["VOL_AVG20"] = x["Volume"].rolling(20).mean()
    x["VOL_RATIO"] = x["Volume"] / x["VOL_AVG20"].replace(0, np.nan)
    x["PREV_20_HIGH"] = x["High"].shift(1).rolling(20).max()
    x["PREV_20_LOW"] = x["Low"].shift(1).rolling(20).min()
    x["BREAKOUT"] = x["Close"] > x["PREV_20_HIGH"]
    x["BREAKDOWN"] = x["Close"] < x["PREV_20_LOW"]
    x["EMA_BULL"] = (x["EMA20"] > x["EMA50"]) & (x["EMA50"] > x["EMA200"])
    x["MACD_BULL"] = x["MACD"] > x["MACD_SIGNAL"]
    x["ST_BULL"] = x["ST_DIRECTION"] > 0
    x["DI_BULL"] = x["+DI"] > x["-DI"]
    x["FULL_BULLISH"] = x["EMA_BULL"] & x["MACD_BULL"] & x["ST_BULL"] & x["DI_BULL"]
    x["STRONG_VOLUME"] = x["VOL_RATIO"] >= 1.5
    # Frozen V6.1 baseline condition
    x["V61_SIGNAL"] = x["FULL_BULLISH"] | (
        x["BREAKOUT"] & x["STRONG_VOLUME"] & x["FULL_BULLISH"]
    )
    return x


def download_data(symbol):
    try:
        df = yf.download(
            symbol, period=PERIOD, interval="1d",
            auto_adjust=False, progress=False, threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in cols):
            print(f"{symbol}: required OHLCV columns missing.")
            return None
        return df[cols].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    except Exception as exc:
        print(f"{symbol}: download error: {exc}")
        return None


def build_nifty_regime(df):
    x = df.copy()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    x["RET20"] = x["Close"].pct_change(20) * 100
    bull = (x["Close"] > x["EMA50"]) & (x["EMA20"] >= x["EMA50"]) & (x["RET20"] > 1)
    bear = (x["Close"] < x["EMA50"]) & (x["EMA20"] <= x["EMA50"]) & (x["RET20"] < -1)
    x["MarketRegime"] = np.select([bull, bear], ["BULL", "BEAR"], default="SIDEWAYS")
    return x[["MarketRegime"]]


def volume_bucket(v):
    if pd.isna(v): return "N/A"
    if v < 1: return "<1.0x"
    if v < 1.5: return "1.0–1.5x"
    if v < 2: return "1.5–2.0x"
    return ">2.0x"


def adx_bucket(v):
    if pd.isna(v): return "N/A"
    if v < 20: return "<20"
    if v < 25: return "20–25"
    if v < 30: return "25–30"
    return ">30"


def hold_bucket(days):
    if days <= 5: return "1–5"
    if days <= 10: return "6–10"
    if days <= 15: return "11–15"
    return "16–20"


def component_label(r):
    if r["BREAKOUT"] and r["STRONG_VOLUME"] and r["EMA_BULL"]:
        return "Breakout + Volume + EMA"
    if r["BREAKOUT"] and r["STRONG_VOLUME"]:
        return "Breakout + Volume"
    if r["BREAKOUT"] and r["EMA_BULL"]:
        return "Breakout + EMA"
    if r["FULL_BULLISH"] and safe_float(r["ADX"]) >= 20:
        return "Full Bullish + ADX >=20"
    if r["FULL_BULLISH"] and r["STRONG_VOLUME"]:
        return "Full Bullish + Volume >=1.5x"
    if r["FULL_BULLISH"]:
        return "Full Bullish Structure"
    if r["BREAKOUT"]:
        return "Breakout"
    return "Other V6.1 Signal"


def simulate_trade(df, signal_i, symbol):
    entry_i = signal_i + 1
    if entry_i >= len(df):
        return None
    signal = df.iloc[signal_i]
    entry = safe_float(df.iloc[entry_i]["Open"])
    atr_value = safe_float(signal["ATR"])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr_value) or atr_value <= 0:
        return None

    stop = entry - ATR_STOP_MULT * atr_value
    risk = entry - stop
    t1 = entry + T1_R * risk
    t2 = entry + T2_R * risk
    last_i = min(len(df) - 1, entry_i + MAX_HOLD_DAYS - 1)
    exit_i = last_i
    exit_price = safe_float(df.iloc[last_i]["Close"])
    reason = "TIME"

    for j in range(entry_i, last_i + 1):
        hi = safe_float(df.iloc[j]["High"])
        lo = safe_float(df.iloc[j]["Low"])
        # Conservative daily-candle assumption if both stop and target are touched.
        if lo <= stop:
            exit_i, exit_price, reason = j, stop, "STOP"
            break
        if hi >= t2:
            exit_i, exit_price, reason = j, t2, "T2"
            break
        if hi >= t1:
            exit_i, exit_price, reason = j, t1, "T1"
            break

    gross = (exit_price - entry) / entry
    net = gross - ROUND_TRIP_COST
    days = exit_i - entry_i + 1
    vol = safe_float(signal["VOL_RATIO"])
    adx = safe_float(signal["ADX"])
    return {
        "Symbol": symbol,
        "SignalDate": df.index[signal_i],
        "EntryDate": df.index[entry_i],
        "ExitDate": df.index[exit_i],
        "Entry": entry, "Stop": stop, "T1": t1, "T2": t2, "Exit": exit_price,
        "ExitReason": reason, "HoldDays": days, "HoldBucket": hold_bucket(days),
        "GrossReturnPct": gross * 100, "NetReturnPct": net * 100,
        "RMultiple": (exit_price - entry) / risk, "Win": bool(net > 0),
        "MarketRegime": str(signal.get("MarketRegime", "N/A")),
        "VolumeRatio": vol, "VolumeBucket": volume_bucket(vol),
        "ADX": adx, "ADXBucket": adx_bucket(adx),
        "RSI": safe_float(signal["RSI"]), "ATR": atr_value,
        "Breakout": bool(signal["BREAKOUT"]),
        "StrongVolume": bool(signal["STRONG_VOLUME"]),
        "EMA_Bull": bool(signal["EMA_BULL"]),
        "MACD_Bull": bool(signal["MACD_BULL"]),
        "SuperTrend_Bull": bool(signal["ST_BULL"]),
        "DI_Bull": bool(signal["DI_BULL"]),
        "FullBullish": bool(signal["FULL_BULLISH"]),
        "ADX20": bool(adx >= 20),
        "Component": component_label(signal),
        "StopDistancePct": risk / entry * 100,
    }


def summarize(df, group_cols):
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(group_cols, dropna=False)
    s = g.agg(
        Trades=("NetReturnPct", "size"),
        WinRatePct=("Win", lambda a: a.mean() * 100),
        AvgNetReturnPct=("NetReturnPct", "mean"),
        MedianNetReturnPct=("NetReturnPct", "median"),
        BestNetReturnPct=("NetReturnPct", "max"),
        WorstNetReturnPct=("NetReturnPct", "min"),
        AvgR=("RMultiple", "mean"),
        StopCount=("ExitReason", lambda a: int((a == "STOP").sum())),
        T1Count=("ExitReason", lambda a: int((a == "T1").sum())),
        T2Count=("ExitReason", lambda a: int((a == "T2").sum())),
        TimeCount=("ExitReason", lambda a: int((a == "TIME").sum())),
        AvgHoldDays=("HoldDays", "mean"),
    ).reset_index()
    s["SampleOK"] = s["Trades"] >= MIN_SAMPLE
    s["Quality"] = [
        "⚪" if n < MIN_SAMPLE else
        "🟢" if avg > 0 and win >= 50 else
        "🟡" if avg > 0 or win >= 45 else
        "🟠" if avg > -0.5 else "🔴"
        for n, avg, win in zip(s["Trades"], s["AvgNetReturnPct"], s["WinRatePct"])
    ]
    return s


def stop_clustering(trades):
    stops = trades[trades["ExitReason"] == "STOP"].copy()
    cols = ["StopDistanceBucket", "Stops", "PctOfStops"]
    if stops.empty:
        return pd.DataFrame(columns=cols)
    bins = [-np.inf, 2, 4, 6, 8, 10, np.inf]
    labels = ["<2%", "2–4%", "4–6%", "6–8%", "8–10%", ">10%"]
    stops["StopDistanceBucket"] = pd.cut(stops["StopDistancePct"], bins=bins, labels=labels, right=False)
    result = stops.groupby("StopDistanceBucket", observed=False).size().reset_index(name="Stops")
    total = result["Stops"].sum()
    result["PctOfStops"] = result["Stops"] / total * 100 if total else 0
    return result


def print_table(title, s, label_col):
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)
    if s.empty:
        print("No data.")
        return
    print(f"{label_col:<34}{'N':>6}{'Win%':>9}{'Avg%':>10}{'Med%':>10}{'AvgR':>8}{'Stops':>8}{'T1':>6}{'T2':>6}{'Flag':>7}")
    print("-" * 112)
    for _, r in s.iterrows():
        print(f"{str(r[label_col]):<34}{int(r['Trades']):>6}{r['WinRatePct']:>8.2f}%"
              f"{r['AvgNetReturnPct']:>9.2f}%{r['MedianNetReturnPct']:>9.2f}%"
              f"{r['AvgR']:>8.2f}{int(r['StopCount']):>8}{int(r['T1Count']):>6}"
              f"{int(r['T2Count']):>6}{r['Quality']:>7}")


def main():
    print("=" * 112)
    print("STOCKAI V6.2.1 — REGIME & SIGNAL QUALITY ANALYSIS")
    print("=" * 112)
    print("Diagnostic only: production V6.1 is not changed.")
    print(f"Period={PERIOD} | Entry=next-day OPEN | Stop={ATR_STOP_MULT} ATR | "
          f"T1={T1_R}R | T2={T2_R}R | Max hold={MAX_HOLD_DAYS} days | "
          f"Round-trip cost={ROUND_TRIP_COST * 100:.2f}%")

    nifty_raw = download_data("^NSEI")
    nifty_regime = build_nifty_regime(nifty_raw) if nifty_raw is not None else None
    if nifty_regime is None:
        print("WARNING: NIFTY data unavailable; regime will be N/A.")

    trades_list = []
    skipped = []

    for symbol in UNIVERSE:
        print(f"\nProcessing {symbol}...")
        raw = download_data(symbol)
        if raw is None or len(raw) < 260:
            skipped.append((symbol, "insufficient data"))
            print(f"  SKIPPED: insufficient data ({0 if raw is None else len(raw)} rows)")
            continue

        df = add_indicators(raw)
        if nifty_regime is not None:
            df = df.join(nifty_regime, how="left")
        else:
            df["MarketRegime"] = "N/A"

        # One position at a time per symbol: skip signal days while an earlier
        # simulated trade remains open. Entry is on the next trading session.
        WARMUP_ROWS = 260

        next_available_signal = WARMUP_ROWS

        stock_trades = 0
        for i in range(WARMUP_ROWS, len(df) - 1):
            if i < next_available_signal:
                continue
            if not bool(df.iloc[i]["V61_SIGNAL"]):
                continue
            trade = simulate_trade(df, i, symbol)
            if trade is None:
                continue
            trades_list.append(trade)
            stock_trades += 1
            exit_date = pd.Timestamp(trade["ExitDate"])
            exit_positions = np.flatnonzero(df.index == exit_date)
            if len(exit_positions):
                next_available_signal = int(exit_positions[0]) + 1
        print(f"  V6.1 diagnostic trades: {stock_trades}")

    if not trades_list:
        print("\nNo trades generated. Check internet access, ticker symbols, and data availability.")
        return

    trades = pd.DataFrame(trades_list)
    for c in ("SignalDate", "EntryDate", "ExitDate"):
        trades[c] = pd.to_datetime(trades[c])

    group_specs = [
        ("Market Regime", ["MarketRegime"]),
        ("Volume Bucket", ["VolumeBucket"]),
        ("ADX Bucket", ["ADXBucket"]),
        ("Signal Component", ["Component"]),
        ("Holding Period", ["HoldBucket"]),
        ("Stock", ["Symbol"]),
        ("Exit Reason", ["ExitReason"]),
    ]
    parts = []
    for condition_type, cols in group_specs:
        s = summarize(trades, cols)
        if s.empty:
            continue
        s.insert(0, "ConditionType", condition_type)
        s["Condition"] = s[cols].astype(str).agg(" | ".join, axis=1)
        parts.append(s)
    condition_analysis = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    stock_summary = summarize(trades, ["Symbol"])
    stop_summary = stop_clustering(trades)

    paths = {
        "trades": OUT / "backtest_v6_2_1_trades.csv",
        "conditions": OUT / "backtest_v6_2_1_condition_analysis.csv",
        "stocks": OUT / "backtest_v6_2_1_stock_summary.csv",
        "stops": OUT / "backtest_v6_2_1_stop_clustering.csv",
    }
    trades.to_csv(paths["trades"], index=False)
    condition_analysis.to_csv(paths["conditions"], index=False)
    stock_summary.to_csv(paths["stocks"], index=False)
    stop_summary.to_csv(paths["stops"], index=False)

    print_table("STOCK-BY-STOCK SUMMARY", stock_summary, "Symbol")
    for title, col, label in [
        ("MARKET REGIME ANALYSIS", "MarketRegime", "MarketRegime"),
        ("VOLUME BUCKET ANALYSIS", "VolumeBucket", "VolumeBucket"),
        ("ADX BUCKET ANALYSIS", "ADXBucket", "ADXBucket"),
        ("SIGNAL / CONFIRMATION ANALYSIS", "Component", "Component"),
        ("HOLDING PERIOD ANALYSIS", "HoldBucket", "HoldBucket"),
        ("EXIT REASON ANALYSIS", "ExitReason", "ExitReason"),
    ]:
        print_table(title, summarize(trades, [col]), label)

    print("\n" + "=" * 112)
    print("STOP-LOSS CLUSTERING")
    print("=" * 112)
    if stop_summary.empty:
        print("No stop-loss exits.")
    else:
        print(f"{'Stop distance':<20}{'Stops':>10}{'% of stops':>14}")
        print("-" * 44)
        for _, r in stop_summary.iterrows():
            print(f"{str(r['StopDistanceBucket']):<20}{int(r['Stops']):>10}{r['PctOfStops']:>13.2f}%")

    print("\n" + "=" * 112)
    print("BEGINNER-FRIENDLY CONCLUSION")
    print("=" * 112)
    print(f"Trades: {len(trades)} | Win rate: {trades['Win'].mean() * 100:.2f}% | "
          f"Average net return: {trades['NetReturnPct'].mean():.2f}% | "
          f"Median: {trades['NetReturnPct'].median():.2f}% | "
          f"Average R: {trades['RMultiple'].mean():.2f}R")
    print("This analysis describes historical outcomes; it does not prove future profitability.")
    print("Do not select a rule solely because it looks best in this same historical sample.")
    print("Any proposed change must be tested on untouched out-of-sample data.")

    print("\nCSV files created:")
    for p in paths.values():
        print(f"  {p.resolve()}")
    if skipped:
        print("\nSkipped symbols:")
        for symbol, reason in skipped:
            print(f"  {symbol:<16} {reason}")


if __name__ == "__main__":
    main()
