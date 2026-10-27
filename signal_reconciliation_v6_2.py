"""
STOCKAI V6.2 — SIGNAL RECONCILIATION
=====================================

Purpose:
Compare the signal definitions used by:
    A. backtest_v6_2_ab.py          -> A_CURRENT_V6.1
    B. backtest_v6_2_ab.py          -> B_STRICT_CONFIRMATION
    C. backtest_v6_2_1_regime_analysis.py -> V6.2.1 V61_SIGNAL

Important:
- Does NOT modify scanner_v6_1.py.
- Does NOT modify existing backtest scripts.
- Uses one downloaded OHLCV dataset per stock.
- Uses a common evaluation window beginning at row 260.
- Compares raw signal dates separately from executed trades.
- Writes two CSV reports to the current directory.

Required existing files:
    backtest_v6_2_ab.py
    backtest_v6_2_1_regime_analysis.py

Run:
    python signal_reconciliation_v6_2.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Reuse the existing project implementations.
import backtest_v6_2_ab as ab
import backtest_v6_2_1_regime_analysis as regime


# ============================================================
# CONFIGURATION
# ============================================================

STOCKS = [
    "WIPRO.NS",
    "INFY.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "RELIANCE.NS",
    "HFCL.NS",
    "TMCV.NS",
    "ADANIPORTS.NS",
    "QPOWER.NS",
]

PERIOD = "5y"
INTERVAL = "1d"

# Match the A/B backtest's starting evaluation index.
COMMON_WARMUP_ROWS = 260

OUTPUT_DIR = Path.cwd()

SUMMARY_FILE = OUTPUT_DIR / "signal_reconciliation_summary.csv"
DATES_FILE = OUTPUT_DIR / "signal_reconciliation_dates.csv"


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_common_data(symbol: str):
    """Download OHLCV once and share it with all three engines."""

    try:
        raw = yf.download(
            symbol,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        df = ab.clean_yfinance(raw)

        if df is None or df.empty:
            return None, "No data returned"

        required = ["Open", "High", "Low", "Close", "Volume"]

        missing = [c for c in required if c not in df.columns]

        if missing:
            return None, f"Missing columns: {missing}"

        df = df[required].copy()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

        if df.empty:
            return None, "No valid OHLC rows"

        return df, "OK"

    except Exception as exc:
        return None, f"Download error: {exc}"


# ============================================================
# SIGNAL CALCULATION
# ============================================================

def calculate_signal_sets(raw: pd.DataFrame):
    """
    Calculate signals using the actual existing functions.

    Returns:
        DataFrame containing the three signal definitions.
    """

    # Existing A/B engine.
    ab_df = ab.add_indicators(raw.copy())

    # Existing V6.2.1 regime-analysis engine.
    regime_df = regime.add_indicators(raw.copy())

    # Both engines must have the same dates.
    common_index = ab_df.index.intersection(regime_df.index)

    ab_df = ab_df.loc[common_index].copy()
    regime_df = regime_df.loc[common_index].copy()

    records = []

    # The final row cannot be entered on the following session.
    final_signal_index = len(common_index) - 1

    for i in range(COMMON_WARMUP_ROWS, final_signal_index + 1):

        date = common_index[i]

        ab_features = ab.technical_features(ab_df.iloc[i])

        if ab_features is None:
            a_signal = False
            b_signal = False
            breakout = False
            volume_confirm = False
            ema_confirm = False
            macd_confirm = False
            st_confirm = False
            di_confirm = False
            adx_confirm = False
        else:
            a_signal = bool(ab_features["A"])
            b_signal = bool(ab_features["B"])

            breakout = bool(ab_features["Breakout"])
            volume_confirm = bool(ab_features["VolumeConfirm"])
            ema_confirm = bool(ab_features["EMAConfirm"])
            macd_confirm = bool(ab_features["MACDConfirm"])
            st_confirm = bool(ab_features["STConfirm"])
            di_confirm = bool(ab_features["DIConfirm"])
            adx_confirm = bool(ab_features["ADXConfirm"])

        regime_row = regime_df.iloc[i]

        # Reproduce the existing V6.2.1 baseline definition.
        # FULL_BULLISH OR (BREAKOUT AND STRONG_VOLUME AND FULL_BULLISH)
        # simplifies logically to FULL_BULLISH.
        c_signal = bool(regime_row["V61_SIGNAL"])

        records.append({
            "SignalDate": date,

            "A_CURRENT_V6_1": a_signal,
            "B_STRICT_CONFIRMATION": b_signal,
            "V6_2_1_V61_SIGNAL": c_signal,

            "A_Breakout": breakout,
            "A_VolumeConfirm": volume_confirm,
            "A_EMAConfirm": ema_confirm,
            "A_MACDConfirm": macd_confirm,
            "A_SuperTrendConfirm": st_confirm,
            "A_DIConfirm": di_confirm,
            "A_ADXConfirm": adx_confirm,

            "A_Price": (
                float(ab_features["Price"])
                if ab_features is not None else np.nan
            ),

            "A_VolumeRatio": (
                float(ab_features["VolumeRatio"])
                if ab_features is not None else np.nan
            ),

            "A_ADX": (
                float(ab_features["ADX"])
                if ab_features is not None else np.nan
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# SIGNAL OVERLAP / DIFFERENCE ANALYSIS
# ============================================================

def summarize_signals(symbol: str, signals: pd.DataFrame):

    columns = [
        "A_CURRENT_V6_1",
        "B_STRICT_CONFIRMATION",
        "V6_2_1_V61_SIGNAL",
    ]

    if signals.empty:
        return {
            "Symbol": symbol,
            "Status": "No eligible signal rows",
            "EvaluationRows": 0,
            "A_RawSignals": 0,
            "B_RawSignals": 0,
            "C_RawSignals": 0,
            "A_B_Overlap": 0,
            "A_C_Overlap": 0,
            "B_C_Overlap": 0,
            "All_Three_Overlap": 0,
            "A_Only": 0,
            "B_Only": 0,
            "C_Only": 0,
        }

    a = signals["A_CURRENT_V6_1"]
    b = signals["B_STRICT_CONFIRMATION"]
    c = signals["V6_2_1_V61_SIGNAL"]

    return {
        "Symbol": symbol,
        "Status": "OK",
        "EvaluationRows": len(signals),

        "A_RawSignals": int(a.sum()),
        "B_RawSignals": int(b.sum()),
        "C_RawSignals": int(c.sum()),

        "A_B_Overlap": int((a & b).sum()),
        "A_C_Overlap": int((a & c).sum()),
        "B_C_Overlap": int((b & c).sum()),

        "All_Three_Overlap": int((a & b & c).sum()),

        "A_Only": int((a & ~b & ~c).sum()),
        "B_Only": int((b & ~a & ~c).sum()),
        "C_Only": int((c & ~a & ~b).sum()),
    }


# ============================================================
# EXECUTED-TRADE COUNTS
# ============================================================

def calculate_executed_counts(
    symbol: str,
    raw: pd.DataFrame,
):
    """
    Count trades under each existing engine's position-management
    approach. These are supplementary to raw signal counts.

    The engines retain their own ATR and exit calculations, so these
    counts should not be treated as perfectly identical simulations.
    """

    ab_df = ab.add_indicators(raw.copy())
    reg_df = regime.add_indicators(raw.copy())

    common_index = ab_df.index.intersection(reg_df.index)

    ab_df = ab_df.loc[common_index].copy()
    reg_df = reg_df.loc[common_index].copy()

    start_i = COMMON_WARMUP_ROWS
    stop_i = len(common_index) - 1

    counts = {
        "A_ExecutedTrades": 0,
        "B_ExecutedTrades": 0,
        "C_ExecutedTrades": 0,
    }

    if stop_i <= start_i:
        return counts

    # A and B are simulated independently.
    for key, signal_key in [
        ("A_ExecutedTrades", "A"),
        ("B_ExecutedTrades", "B"),
    ]:

        i = start_i

        while i < stop_i:

            features = ab.technical_features(ab_df.iloc[i])

            if features is None or not bool(features[signal_key]):
                i += 1
                continue

            trade = ab.simulate_strategy(
                ab_df,
                i,
                features,
            )

            if trade is None:
                i += 1
                continue

            counts[key] += 1

            exit_date = pd.Timestamp(trade["ExitDate"])
            positions = np.flatnonzero(
                common_index == exit_date
            )

            if len(positions):
                i = max(i + 1, int(positions[0]) + 1)
            else:
                i += 1

    # V6.2.1 execution logic: one open position per symbol.
    next_available_signal = start_i

    for i in range(start_i, stop_i):

        if i < next_available_signal:
            continue

        if not bool(reg_df.iloc[i]["V61_SIGNAL"]):
            continue

        trade = regime.simulate_trade(
            reg_df,
            i,
            symbol,
        )

        if trade is None:
            continue

        counts["C_ExecutedTrades"] += 1

        exit_date = pd.Timestamp(trade["ExitDate"])

        positions = np.flatnonzero(
            common_index == exit_date
        )

        if len(positions):
            next_available_signal = int(positions[0]) + 1

    return counts


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 105)
    print("STOCKAI V6.2 — SIGNAL RECONCILIATION")
    print("=" * 105)

    print(f"Universe          : {len(STOCKS)} stocks")
    print(f"Historical period : {PERIOD}")
    print(f"Evaluation starts  : row {COMMON_WARMUP_ROWS}")
    print("Production scanner : NOT MODIFIED")
    print()

    summary_rows = []
    date_frames = []

    for symbol in STOCKS:

        print(f"Processing {symbol} ...")

        raw, status = download_common_data(symbol)

        if raw is None:
            summary_rows.append({
                "Symbol": symbol,
                "Status": status,
                "DataRows": 0,
                "FirstDate": None,
                "LastDate": None,
                "EvaluationRows": 0,
                "A_RawSignals": 0,
                "B_RawSignals": 0,
                "C_RawSignals": 0,
                "A_B_Overlap": 0,
                "A_C_Overlap": 0,
                "B_C_Overlap": 0,
                "All_Three_Overlap": 0,
                "A_Only": 0,
                "B_Only": 0,
                "C_Only": 0,
                "A_ExecutedTrades": 0,
                "B_ExecutedTrades": 0,
                "C_ExecutedTrades": 0,
            })

            print(f"  SKIPPED: {status}")
            continue

        if len(raw) <= COMMON_WARMUP_ROWS + 1:
            summary_rows.append({
                "Symbol": symbol,
                "Status": "Insufficient history for common window",
                "DataRows": len(raw),
                "FirstDate": raw.index.min(),
                "LastDate": raw.index.max(),
                "EvaluationRows": 0,
                "A_RawSignals": 0,
                "B_RawSignals": 0,
                "C_RawSignals": 0,
                "A_B_Overlap": 0,
                "A_C_Overlap": 0,
                "B_C_Overlap": 0,
                "All_Three_Overlap": 0,
                "A_Only": 0,
                "B_Only": 0,
                "C_Only": 0,
                "A_ExecutedTrades": 0,
                "B_ExecutedTrades": 0,
                "C_ExecutedTrades": 0,
            })

            print(f"  SKIPPED: only {len(raw)} data rows")
            continue

        try:
            signals = calculate_signal_sets(raw)

            summary = summarize_signals(symbol, signals)

            executed = calculate_executed_counts(
                symbol,
                raw,
            )

            summary.update({
                "DataRows": len(raw),
                "FirstDate": raw.index.min(),
                "LastDate": raw.index.max(),
            })

            summary.update(executed)

            summary_rows.append(summary)

            if not signals.empty:

                signals.insert(0, "Symbol", symbol)

                # Keep rows where at least one engine signals.
                any_signal = (
                    signals[
                        [
                            "A_CURRENT_V6_1",
                            "B_STRICT_CONFIRMATION",
                            "V6_2_1_V61_SIGNAL",
                        ]
                    ]
                    .any(axis=1)
                )

                signal_dates = signals.loc[any_signal].copy()

                if not signal_dates.empty:

                    signal_dates["Agreement"] = np.select(
                        [
                            signal_dates[
                                [
                                    "A_CURRENT_V6_1",
                                    "B_STRICT_CONFIRMATION",
                                    "V6_2_1_V61_SIGNAL",
                                ]
                            ].all(axis=1),

                            (
                                signal_dates["A_CURRENT_V6_1"]
                                & signal_dates["B_STRICT_CONFIRMATION"]
                            ),

                            (
                                signal_dates["A_CURRENT_V6_1"]
                                & signal_dates["V6_2_1_V61_SIGNAL"]
                            ),

                            (
                                signal_dates["B_STRICT_CONFIRMATION"]
                                & signal_dates["V6_2_1_V61_SIGNAL"]
                            ),
                        ],
                        [
                            "ALL_THREE_AGREE",
                            "A_AND_B",
                            "A_AND_C",
                            "B_AND_C",
                        ],
                        default="SINGLE_ENGINE_OR_OTHER",
                    )

                    date_frames.append(signal_dates)

            print(
                f"  Rows={len(raw)} | "
                f"A={summary['A_RawSignals']} | "
                f"B={summary['B_RawSignals']} | "
                f"C={summary['C_RawSignals']}"
            )

        except Exception as exc:

            summary_rows.append({
                "Symbol": symbol,
                "Status": f"Calculation error: {exc}",
                "DataRows": len(raw),
                "FirstDate": raw.index.min(),
                "LastDate": raw.index.max(),
                "EvaluationRows": 0,
                "A_RawSignals": 0,
                "B_RawSignals": 0,
                "C_RawSignals": 0,
                "A_B_Overlap": 0,
                "A_C_Overlap": 0,
                "B_C_Overlap": 0,
                "All_Three_Overlap": 0,
                "A_Only": 0,
                "B_Only": 0,
                "C_Only": 0,
                "A_ExecutedTrades": 0,
                "B_ExecutedTrades": 0,
                "C_ExecutedTrades": 0,
            })

            print(f"  ERROR: {exc}")

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    if date_frames:
        dates_df = pd.concat(
            date_frames,
            ignore_index=True,
        )

        dates_df.to_csv(
            DATES_FILE,
            index=False,
        )
    else:
        dates_df = pd.DataFrame()

        dates_df.to_csv(
            DATES_FILE,
            index=False,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 105)
    print("RECONCILIATION SUMMARY")
    print("=" * 105)

    display_columns = [
        "Symbol",
        "Status",
        "A_RawSignals",
        "B_RawSignals",
        "C_RawSignals",
        "A_B_Overlap",
        "A_C_Overlap",
        "B_C_Overlap",
        "A_ExecutedTrades",
        "B_ExecutedTrades",
        "C_ExecutedTrades",
    ]

    available_columns = [
        c for c in display_columns
        if c in summary_df.columns
    ]

    if not summary_df.empty:
        print(
            summary_df[available_columns]
            .to_string(index=False)
        )

    print()
    print("Files created:")
    print(f"  {SUMMARY_FILE}")
    print(f"  {DATES_FILE}")

    print()
    print("IMPORTANT:")
    print("- Raw signal counts are not the same as executed-trade counts.")
    print("- Executed trades are affected by open-position restrictions.")
    print("- The three engines still use different indicator implementations.")
    print("- This report diagnoses differences; it does not select a winner.")
    print("=" * 105)


if __name__ == "__main__":
    main()