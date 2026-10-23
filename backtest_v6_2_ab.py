import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# STOCKAI V6 ACTION ENGINE BACKTEST
# Technical decision-layer validation
#
# IMPORTANT:
# - This backtest validates the historical TECHNICAL action layer.
# - Fundamental/news classifications are NOT backtested because
#   yfinance current fundamentals/news are not point-in-time history.
# - No hidden score, ranking, probability or price forecast.
# - Uses the same core technical concepts as StockAI V4.7/V6.1:
#     EMA20/50/200, MACD, SuperTrend, ADX/+DI/-DI,
#     RSI, volume, 20D breakout, ATR, S/R context.
# - Entry is next trading day's OPEN after a signal.
# - Round-trip transaction cost: 0.20%.
# - Stop: 1.50 ATR from signal close.
# - Mechanical T1: 1.50R
# - Mechanical T2: 2.50R
# - Maximum holding period: 20 trading sessions.
# ============================================================

STOCKS = [
    "WIPRO.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS",
    "RELIANCE.NS", "HFCL.NS", "TMCV.NS", "ADANIPORTS.NS", "QPOWER.NS"
]

PERIOD = "5y"
INTERVAL = "1d"

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
EMA_FAST, EMA_MID, EMA_SLOW = 20, 50, 200
VOLUME_PERIOD = 20
VOLUME_THRESHOLD = 1.5
BREAKOUT_LOOKBACK = 20
SR_LOOKBACK = 20
ATR_STOP_MULT = 1.50
T1_R = 1.50
T2_R = 2.50
MAX_HOLD = 20
ROUND_TRIP_COST = 0.0020
ADX_STRONG = 20


def clean_yfinance(data):
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in data.columns:
            return None
    data = data[needed].copy()
    for c in needed:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data.dropna(subset=["Open", "High", "Low", "Close"])


def rsi_wilder(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0)
    return rsi


def atr_wilder(df, period=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx_wilder(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def supertrend(df, period=10, multiplier=3.0):
    atr = atr_wilder(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    ub = hl2 + multiplier * atr
    lb = hl2 - multiplier * atr
    upper = pd.Series(np.nan, index=df.index)
    lower = pd.Series(np.nan, index=df.index)
    direction = pd.Series(np.nan, index=df.index)
    st = pd.Series(np.nan, index=df.index)
    valid = np.where(atr.notna().values)[0]
    if len(valid) == 0:
        return st, direction
    first = int(valid[0])
    upper.iloc[first] = ub.iloc[first]
    lower.iloc[first] = lb.iloc[first]
    direction.iloc[first] = 1
    st.iloc[first] = lower.iloc[first]
    for i in range(first + 1, len(df)):
        if pd.isna(ub.iloc[i]) or pd.isna(lb.iloc[i]):
            continue
        pu = upper.iloc[i - 1]
        pl = lower.iloc[i - 1]
        pdirection = direction.iloc[i - 1]
        if pd.isna(pu): pu = ub.iloc[i]
        if pd.isna(pl): pl = lb.iloc[i]
        if pd.isna(pdirection): pdirection = 1
        pc = df["Close"].iloc[i - 1]
        upper.iloc[i] = ub.iloc[i] if ub.iloc[i] < pu or pc > pu else pu
        lower.iloc[i] = lb.iloc[i] if lb.iloc[i] > pl or pc < pl else pl
        if pdirection == -1 and df["Close"].iloc[i] > upper.iloc[i]:
            direction.iloc[i] = 1
        elif pdirection == 1 and df["Close"].iloc[i] < lower.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = pdirection
        st.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]
    return st, direction


def add_indicators(df):
    df = df.copy()
    df["RSI"] = rsi_wilder(df["Close"], RSI_PERIOD)
    df["EMA20"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=EMA_MID, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    e12 = df["Close"].ewm(span=12, adjust=False).mean()
    e26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["ATR"] = atr_wilder(df, ATR_PERIOD)
    df["ADX"], df["PLUS_DI"], df["MINUS_DI"] = adx_wilder(df, ADX_PERIOD)
    df["ST"], df["ST_DIRECTION"] = supertrend(df, 10, 3.0)
    df["VOL_AVG"] = df["Volume"].rolling(VOLUME_PERIOD).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_AVG"].replace(0, np.nan)
    df["PRIOR_20_HIGH"] = df["High"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    df["PRIOR_20_LOW"] = df["Low"].rolling(BREAKOUT_LOOKBACK).min().shift(1)
    return df



def technical_features(row):
    """Return transparent historical technical conditions used by both tests."""
    vals = [row.get(c, np.nan) for c in [
        "Close", "EMA20", "EMA50", "EMA200", "MACD", "MACD_SIGNAL",
        "ST_DIRECTION", "PLUS_DI", "MINUS_DI", "RSI", "VOL_RATIO", "ATR",
        "ADX"
    ]]
    if any(pd.isna(x) for x in vals):
        return None

    price = float(row["Close"])

    ema_bull = price > row["EMA20"] > row["EMA50"] > row["EMA200"]
    ema_bear = price < row["EMA20"] < row["EMA50"] < row["EMA200"]
    macd_bull = row["MACD"] > row["MACD_SIGNAL"]
    st_bull = row["ST_DIRECTION"] > 0
    di_bull = row["PLUS_DI"] > row["MINUS_DI"]
    adx_confirm = row["ADX"] >= ADX_STRONG and di_bull
    rsi_bull = row["RSI"] >= 50
    volume_confirm = row["VOL_RATIO"] >= VOLUME_THRESHOLD
    breakout = pd.notna(row["PRIOR_20_HIGH"]) and price > row["PRIOR_20_HIGH"]

    bullish_count = sum([ema_bull, macd_bull, st_bull, di_bull, rsi_bull])
    bearish_count = sum([
        ema_bear, not macd_bull, not st_bull, not di_bull, row["RSI"] < 50
    ])

    # A = current V6.1 BUY ON BREAKOUT logic used as the baseline.
    a_signal = (
        breakout and volume_confirm and bullish_count >= 4
    ) or (
        ema_bull and macd_bull and st_bull and di_bull and bullish_count >= 4
    )

    # B = stricter experiment:
    # A genuine breakout is required, plus strong volume, full EMA trend,
    # and ADX/+DI confirmation. This is deliberately a single controlled
    # change, not a new scoring system.
    b_signal = (
        breakout
        and volume_confirm
        and ema_bull
        and macd_bull
        and st_bull
        and di_bull
        and adx_confirm
    )

    if macd_bull and st_bull and not ema_bull and bullish_count >= 3:
        recovery = True
    else:
        recovery = False

    if ema_bear and bearish_count >= 4:
        action = "AVOID — BEARISH"
    elif bearish_count >= 4:
        action = "AVOID — BEARISH"
    elif recovery:
        action = "WATCH / RECOVERY"
    else:
        action = "WATCH — CONFIRMATION"

    return {
        "Action": action,
        "A": bool(a_signal),
        "B": bool(b_signal),
        "Breakout": bool(breakout),
        "VolumeConfirm": bool(volume_confirm),
        "EMAConfirm": bool(ema_bull),
        "ADXConfirm": bool(adx_confirm),
        "MACDConfirm": bool(macd_bull),
        "STConfirm": bool(st_bull),
        "DIConfirm": bool(di_bull),
        "RSIConfirm": bool(rsi_bull),
        "ATR": float(row["ATR"]),
        "Price": price,
        "VolumeRatio": float(row["VOL_RATIO"]),
        "ADX": float(row["ADX"]),
    }


def simulate_strategy(df, signal_i, features):
    """Enter next day open and manage the same mechanical trade rules."""
    entry_i = signal_i + 1
    if entry_i >= len(df):
        return None

    entry = float(df["Open"].iloc[entry_i])
    atr = float(features["ATR"])
    stop = entry - ATR_STOP_MULT * atr
    risk = entry - stop

    if not np.isfinite(entry) or not np.isfinite(stop) or risk <= 0:
        return None

    t1 = entry + T1_R * risk
    t2 = entry + T2_R * risk

    end_i = min(len(df) - 1, entry_i + MAX_HOLD - 1)
    exit_price = float(df["Close"].iloc[end_i])
    exit_i = end_i
    exit_reason = "TIME"
    hit_t1 = False
    hit_t2 = False

    for j in range(entry_i, end_i + 1):
        high = float(df["High"].iloc[j])
        low = float(df["Low"].iloc[j])

        # Conservative same-bar assumption: stop first if both are touched.
        if low <= stop:
            exit_price = stop
            exit_i = j
            exit_reason = "STOP"
            break

        if high >= t2:
            exit_price = t2
            exit_i = j
            exit_reason = "T2"
            hit_t2 = True
            hit_t1 = True
            break

        if high >= t1:
            exit_price = t1
            exit_i = j
            exit_reason = "T1"
            hit_t1 = True
            break

    gross_return = exit_price / entry - 1.0
    net_return = gross_return - ROUND_TRIP_COST
    r_multiple = (exit_price - entry) / risk

    return {
        "SignalDate": df.index[signal_i],
        "EntryDate": df.index[entry_i],
        "ExitDate": df.index[exit_i],
        "Entry": entry,
        "Stop": stop,
        "T1": t1,
        "T2": t2,
        "Exit": exit_price,
        "ExitReason": exit_reason,
        "GrossReturnPct": gross_return * 100,
        "NetReturnPct": net_return * 100,
        "RMultiple": r_multiple,
        "HoldDays": exit_i - entry_i + 1,
        "RiskPct": risk / entry * 100,
        "HitT1": hit_t1,
        "HitT2": hit_t2,
        "Breakout": features["Breakout"],
        "VolumeConfirm": features["VolumeConfirm"],
        "EMAConfirm": features["EMAConfirm"],
        "ADXConfirm": features["ADXConfirm"],
        "MACDConfirm": features["MACDConfirm"],
        "STConfirm": features["STConfirm"],
        "DIConfirm": features["DIConfirm"],
        "RSIConfirm": features["RSIConfirm"],
    }


def summarize(trades):
    if not trades:
        return {
            "Signals": 0, "WinRate": np.nan, "AvgReturn": np.nan,
            "MedianReturn": np.nan, "Best": np.nan, "Worst": np.nan,
            "AvgR": np.nan, "Stops": 0, "T1": 0, "T2": 0,
        }

    s = pd.Series([t["NetReturnPct"] for t in trades], dtype=float)
    r = pd.Series([t["RMultiple"] for t in trades], dtype=float)

    return {
        "Signals": len(trades),
        "WinRate": (s > 0).mean() * 100,
        "AvgReturn": s.mean(),
        "MedianReturn": s.median(),
        "Best": s.max(),
        "Worst": s.min(),
        "AvgR": r.mean(),
        "Stops": sum(t["ExitReason"] == "STOP" for t in trades),
        "T1": sum(t["ExitReason"] == "T1" for t in trades),
        "T2": sum(t["ExitReason"] == "T2" for t in trades),
    }


def backtest_stock(symbol):
    raw = yf.download(
        symbol,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    df = clean_yfinance(raw)

    if df is None or len(df) < 300:
        return None, [], [], {}

    df = add_indicators(df)

    trades_a = []
    trades_b = []
    signal_counts = {}
    component_counts = {
        "breakout": 0,
        "volume": 0,
        "ema": 0,
        "adx": 0,
        "breakout_volume": 0,
        "breakout_volume_ema": 0,
    }

    min_i = max(EMA_SLOW + 5, 260)
    i_a = min_i
    i_b = min_i

    # A and B are tested independently so one strategy's exit does not
    # suppress a later signal in the other strategy.
    while i_a < len(df) - 1:
        f = technical_features(df.iloc[i_a])
        if f is not None:
            if f["Breakout"]:
                component_counts["breakout"] += 1
            if f["VolumeConfirm"]:
                component_counts["volume"] += 1
            if f["EMAConfirm"]:
                component_counts["ema"] += 1
            if f["ADXConfirm"]:
                component_counts["adx"] += 1
            if f["Breakout"] and f["VolumeConfirm"]:
                component_counts["breakout_volume"] += 1
            if f["Breakout"] and f["VolumeConfirm"] and f["EMAConfirm"]:
                component_counts["breakout_volume_ema"] += 1

            signal_counts[f["Action"]] = signal_counts.get(f["Action"], 0) + 1

            if f["A"]:
                trade = simulate_strategy(df, i_a, f)
                if trade is not None:
                    trade["Symbol"] = symbol.replace(".NS", "")
                    trade["Strategy"] = "A_CURRENT_V6.1"
                    trades_a.append(trade)
                    exit_loc = df.index.get_loc(trade["ExitDate"])
                    i_a = max(i_a + 1, int(exit_loc) + 1)
                    continue
        i_a += 1

    while i_b < len(df) - 1:
        f = technical_features(df.iloc[i_b])
        if f is not None and f["B"]:
            trade = simulate_strategy(df, i_b, f)
            if trade is not None:
                trade["Symbol"] = symbol.replace(".NS", "")
                trade["Strategy"] = "B_STRICT_CONFIRMATION"
                trades_b.append(trade)
                exit_loc = df.index.get_loc(trade["ExitDate"])
                i_b = max(i_b + 1, int(exit_loc) + 1)
                continue
        i_b += 1

    return df, trades_a, trades_b, {
        "signals": signal_counts,
        "components": component_counts,
    }


def print_strategy_result(title, overall, stock_rows):
    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)
    print(f"Signals / trades     : {overall['Signals']}")
    print(f"Win rate             : {overall['WinRate']:.2f}%" if pd.notna(overall["WinRate"]) else "Win rate             : N/A")
    print(f"Average net return   : {overall['AvgReturn']:.2f}%" if pd.notna(overall["AvgReturn"]) else "Average net return   : N/A")
    print(f"Median net return    : {overall['MedianReturn']:.2f}%" if pd.notna(overall["MedianReturn"]) else "Median net return    : N/A")
    print(f"Best trade           : {overall['Best']:.2f}%" if pd.notna(overall["Best"]) else "Best trade           : N/A")
    print(f"Worst trade          : {overall['Worst']:.2f}%" if pd.notna(overall["Worst"]) else "Worst trade           : N/A")
    print(f"Average R-multiple   : {overall['AvgR']:.2f}" if pd.notna(overall["AvgR"]) else "Average R-multiple   : N/A")
    print(f"Stopped out          : {overall['Stops']}")
    print(f"T1 exits             : {overall['T1']}")
    print(f"T2 exits             : {overall['T2']}")

    if stock_rows:
        print("\nPER-STOCK RESULTS")
        print("-" * 132)
        hdr = f"{'Stock':<13}{'Trades':>8}{'Win%':>9}{'Avg%':>10}{'Median%':>10}{'Best%':>10}{'Worst%':>10}{'AvgR':>9}{'Stops':>8}"
        print(hdr)
        print("-" * len(hdr))
        for row in stock_rows:
            print(
                f"{row['Stock']:<13}{row['Signals']:>8}"
                f"{row['WinRate']:>8.1f}%{row['AvgReturn']:>9.2f}%"
                f"{row['MedianReturn']:>9.2f}%{row['Best']:>9.2f}%"
                f"{row['Worst']:>9.2f}%{row['AvgR']:>9.2f}{row['Stops']:>8}"
            )


def main():
    print("=" * 132)
    print("STOCKAI V6.2 — ACTION ENGINE A/B BACKTEST")
    print("Controlled experiment | A = current V6.1 | B = stricter breakout confirmation")
    print("=" * 132)
    print(
        f"Universe: {len(STOCKS)} stocks | Data: {PERIOD} | Timeframe: Daily | "
        f"Entry: next-day OPEN"
    )
    print(
        f"Stop: {ATR_STOP_MULT:.2f} ATR | T1: {T1_R:.2f}R | T2: {T2_R:.2f}R | "
        f"Max hold: {MAX_HOLD} days | Costs: {ROUND_TRIP_COST*100:.2f}% round-trip"
    )
    print()
    print("A — CURRENT V6.1 BASELINE")
    print("  BUY ON BREAKOUT if historical breakout + volume + strong bullish structure,")
    print("  OR full bullish EMA/MACD/SuperTrend/+DI structure is present.")
    print()
    print("B — STRICT CONFIRMATION EXPERIMENT")
    print("  Requires: historical 20D breakout + volume >= 1.5x + EMA20>EMA50>EMA200")
    print("  + MACD bullish + SuperTrend bullish + +DI > -DI + ADX >= 20.")
    print()
    print("IMPORTANT")
    print("  • This tests technical conditions only; fundamentals/news are excluded.")
    print("  • B is an experiment, NOT an automatic change to StockAI.")
    print("  • One position at a time per strategy; A and B are simulated independently.")
    print("  • No hidden score, ranking, probability or price forecast.")
    print()

    all_a, all_b = [], []
    rows_a, rows_b = [], []
    action_totals = {}
    component_totals = {
        "breakout": 0,
        "volume": 0,
        "ema": 0,
        "adx": 0,
        "breakout_volume": 0,
        "breakout_volume_ema": 0,
    }

    for symbol in STOCKS:
        try:
            df, trades_a, trades_b, meta = backtest_stock(symbol)
            if df is None:
                print(f"Skipping {symbol}: insufficient data")
                continue

            sa = summarize(trades_a)
            sb = summarize(trades_b)
            rows_a.append({"Stock": symbol.replace(".NS", ""), **sa})
            rows_b.append({"Stock": symbol.replace(".NS", ""), **sb})
            all_a.extend(trades_a)
            all_b.extend(trades_b)

            for k, v in meta["signals"].items():
                action_totals[k] = action_totals.get(k, 0) + v
            for k, v in meta["components"].items():
                component_totals[k] += v

            print(
                f"{symbol.replace('.NS',''):<13}"
                f"A={sa['Signals']:>3} trades, {sa['WinRate']:>5.1f}% win, {sa['AvgReturn']:>7.2f}% avg | "
                f"B={sb['Signals']:>3} trades, {sb['WinRate']:>5.1f}% win, {sb['AvgReturn']:>7.2f}% avg"
            )
        except Exception as exc:
            print(f"ERROR {symbol}: {exc}")

    print("\n" + "-" * 132)
    print("HISTORICAL TECHNICAL CONDITION COUNTS — ALL BARS")
    print(f"  20D breakout                         {component_totals['breakout']:>6}")
    print(f"  Volume >= 1.5x                      {component_totals['volume']:>6}")
    print(f"  EMA20 > EMA50 > EMA200              {component_totals['ema']:>6}")
    print(f"  ADX >= 20 and +DI > -DI             {component_totals['adx']:>6}")
    print(f"  Breakout + volume                    {component_totals['breakout_volume']:>6}")
    print(f"  Breakout + volume + EMA              {component_totals['breakout_volume_ema']:>6}")

    print("\nACTION STATE COUNTS — ALL HISTORICAL BARS")
    for action in ["BUY ON BREAKOUT", "WATCH / RECOVERY", "WATCH — CONFIRMATION", "AVOID — BEARISH"]:
        print(f"  {action:<24} {action_totals.get(action, 0):>6}")

    overall_a = summarize(all_a)
    overall_b = summarize(all_b)

    print_strategy_result("V6.2 A — CURRENT V6.1 BASELINE", overall_a, rows_a)
    print_strategy_result("V6.2 B — STRICT CONFIRMATION EXPERIMENT", overall_b, rows_b)

    print("\n" + "=" * 132)
    print("A vs B COMPARISON")
    print("=" * 132)
    metrics = [
        ("Signals / trades", "Signals", "0"),
        ("Win rate", "WinRate", "%"),
        ("Average net return", "AvgReturn", "%"),
        ("Median net return", "MedianReturn", "%"),
        ("Average R", "AvgR", "R"),
        ("Stopped out", "Stops", "0"),
        ("T1 exits", "T1", "0"),
        ("T2 exits", "T2", "0"),
    ]
    print(f"{'Metric':<25}{'A — V6.1':>18}{'B — Strict':>18}{'Difference':>18}")
    print("-" * 79)
    for label, key, unit in metrics:
        av = overall_a[key]
        bv = overall_b[key]
        if pd.isna(av) or pd.isna(bv):
            print(f"{label:<25}{str(av):>18}{str(bv):>18}{'N/A':>18}")
        else:
            diff = bv - av
            if unit == "%":
                print(f"{label:<25}{av:>17.2f}%{bv:>17.2f}%{diff:>17.2f}%")
            elif unit == "R":
                print(f"{label:<25}{av:>17.2f}R{bv:>17.2f}R{diff:>17.2f}R")
            else:
                print(f"{label:<25}{av:>18.0f}{bv:>18.0f}{diff:>18.0f}")

    if all_a or all_b:
        trade_frames = []
        if all_a:
            trade_frames.append(pd.DataFrame(all_a))
        if all_b:
            trade_frames.append(pd.DataFrame(all_b))
        trades_df = pd.concat(trade_frames, ignore_index=True)
        for c in ["SignalDate", "EntryDate", "ExitDate"]:
            trades_df[c] = pd.to_datetime(trades_df[c]).dt.strftime("%Y-%m-%d")
        trades_df.to_csv("backtest_v6_2_ab_trades.csv", index=False)
        print("\nTrade-level A/B results saved to: backtest_v6_2_ab_trades.csv")

    print("\n" + "=" * 132)
    print("V6.2 DECISION GUIDE")
    print("=" * 132)
    print("  1. Do NOT choose B just because its win rate is higher.")
    print("  2. Look for improvement in average return AND average R, with a reasonable sample size.")
    print("  3. Check whether improvement is spread across multiple stocks, not just one winner.")
    print("  4. If B is not clearly better, keep V6.1 unchanged.")
    print("  5. Even if B is better, run a separate out-of-sample test before changing production StockAI.")
    print("  6. This experiment does not prove future profitability or guarantee returns.")


if __name__ == "__main__":
    main()
