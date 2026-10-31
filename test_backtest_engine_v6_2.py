"""StockAI V6.2.1 offline engine tests.
Copy beside backtest_v6_2_1_regime_analysis_fixed.py and run:
python validation_v6_2\test_backtest_engine_v6_2.py
No market data download; production scanner is untouched.
"""
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE_PATH = HERE / "backtest_v6_2_1_regime_analysis_fixed.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if not ENGINE_PATH.exists():
    raise FileNotFoundError(f"Missing engine: {ENGINE_PATH}")
spec = importlib.util.spec_from_file_location("stockai_engine_under_test", ENGINE_PATH)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def frame(bars, atr=2.0):
    rows = [{
        "Open": 100., "High": 101., "Low": 99., "Close": 100.,
        "Volume": 1000., "ATR": atr, "VOL_RATIO": 1., "ADX": 22.,
        "RSI": 55., "BREAKOUT": False, "STRONG_VOLUME": False,
        "EMA_BULL": True, "MACD_BULL": True, "ST_BULL": True,
        "DI_BULL": True, "FULL_BULLISH": True,
    }]
    for b in bars:
        rows.append({
            **b, "Volume": 1000., "ATR": atr, "VOL_RATIO": 1.,
            "ADX": 22., "RSI": 55., "BREAKOUT": False,
            "STRONG_VOLUME": False, "EMA_BULL": True,
            "MACD_BULL": True, "ST_BULL": True, "DI_BULL": True,
            "FULL_BULLISH": True,
        })
    return pd.DataFrame(rows, index=pd.bdate_range("2024-01-02", periods=len(rows)))


class TestSuperTrend(unittest.TestCase):
    def test_initializes_after_atr_warmup_and_has_valid_directions(self):
        n = 80
        close = pd.Series(np.linspace(100, 125, n) + np.sin(np.arange(n)) * 1.5)
        df = pd.DataFrame({"Close": close, "High": close + 2, "Low": close - 2})
        d = engine.calc_supertrend_direction(df, period=10, multiplier=3)
        self.assertTrue(d.iloc[:9].isna().all())
        self.assertGreater(d.notna().sum(), 0)
        self.assertTrue(set(d.dropna().unique()).issubset({-1.0, 1.0}))
        self.assertTrue(d.iloc[9:].notna().all())

    def test_returns_nan_when_insufficient_data(self):
        df = pd.DataFrame({"High": [101., 102.], "Low": [99., 100.], "Close": [100., 101.]})
        self.assertTrue(engine.calc_supertrend_direction(df, period=10).isna().all())

    def test_supertrend_detects_bullish_and_bearish_directions(self):
    # Construct an uptrend followed by a sharp downtrend.
        up = np.linspace(100, 130, 60)
        down = np.linspace(130, 70, 60)
        close = np.concatenate([up, down])

        df = pd.DataFrame({
        "Close": close,
        "High": close + 1.0,
        "Low": close - 1.0,
      })

        direction = engine.calc_supertrend_direction(
        df, period=10, multiplier=3.0
     )

        valid = direction.dropna()

        self.assertIn(1.0, valid.values, "No bullish direction detected.")
        self.assertIn(-1.0, valid.values, "No bearish direction detected.")

       


class TestTradeSimulation(unittest.TestCase):
    def test_entry_uses_next_session_open(self):
        df = frame([
            {"Open": 103., "High": 104., "Low": 102., "Close": 103.},
            {"Open": 104., "High": 105., "Low": 103., "Close": 104.},
        ])
        t = engine.simulate_trade(df, 0, "TEST")
        self.assertEqual(t["Entry"], 103.)
        self.assertEqual(t["EntryDate"], df.index[1])

    def test_stop_has_priority_if_stop_and_target_touched_same_bar(self):
        df = frame([
            {"Open": 100., "High": 105., "Low": 96., "Close": 102.},
            {"Open": 102., "High": 103., "Low": 101., "Close": 102.},
        ])
        t = engine.simulate_trade(df, 0, "TEST")
        self.assertEqual(t["ExitReason"], "STOP")
        self.assertAlmostEqual(t["Exit"], 97.)

    def test_t1_exit(self):
        df = frame([
            {"Open": 100., "High": 105., "Low": 99., "Close": 103.},
            {"Open": 103., "High": 103.5, "Low": 102., "Close": 103.},
        ])
        t = engine.simulate_trade(df, 0, "TEST")
        self.assertEqual(t["ExitReason"], "T1")
        self.assertAlmostEqual(t["Exit"], 104.5)

    def test_cost_is_deducted_from_gross_return(self):
        df = frame([
            {"Open": 100., "High": 105., "Low": 99., "Close": 103.},
            {"Open": 103., "High": 103.5, "Low": 102., "Close": 103.},
        ])
        t = engine.simulate_trade(df, 0, "TEST")
        gross_pct = (104.5 - 100.) / 100. * 100.
        self.assertAlmostEqual(t["GrossReturnPct"], gross_pct)
        self.assertAlmostEqual(t["NetReturnPct"], gross_pct - engine.ROUND_TRIP_COST * 100.)

    def test_holding_period_respects_maximum(self):
        bars = [{"Open": 100., "High": 101., "Low": 99., "Close": 100.}
                for _ in range(engine.MAX_HOLD_DAYS + 5)]
        t = engine.simulate_trade(frame(bars), 0, "TEST")
        self.assertLessEqual(t["HoldDays"], engine.MAX_HOLD_DAYS)

    def test_gap_stop_fills_at_open_when_open_below_stop(self):
        # Signal on row 0.
        # Entry = 100, ATR = 2, stop = 97.
        # The next session opens at 94, below the stop.
        df = frame([
            {"Open": 100., "High": 101., "Low": 99., "Close": 100.},
            {"Open": 94., "High": 96., "Low": 92., "Close": 93.},
            {"Open": 93., "High": 94., "Low": 92., "Close": 93.},
        ])

        trade = engine.simulate_trade(df, 0, "TEST")

        self.assertEqual(trade["ExitReason"], "STOP")
        self.assertAlmostEqual(trade["Entry"], 100.0)
        self.assertAlmostEqual(trade["Stop"], 97.0)

        # Corrected execution price: gap-down opening price.
        self.assertAlmostEqual(trade["Exit"], 94.0)

        # Verify the gap was below the stop.
        self.assertLess(df.iloc[2]["Open"], trade["Stop"])

       print("\nGAP TEST PASSED: exit uses gap-down open (94.00), below stop (97.00).")

if __name__ == "__main__":
    print("STOCKAI V6.2.1 — OFFLINE ENGINE TESTS (synthetic data only)")
    unittest.main(verbosity=2)
