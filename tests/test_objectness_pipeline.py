"""Tests for the pure P1 verdict logic in experiments/objectness_pipeline.py (no torch)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.objectness_pipeline import phase1_verdict

RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _flat(val, std=0.0):
    return {r: (val, std) for r in RATES}


def test_works_when_objectness_beats_both_beyond_floor():
    # objectness +0.10 over baselines at every rate, tiny std -> WORKS
    obj = _flat(0.80, 0.01)
    rand = _flat(0.70, 0.01)
    zero = _flat(0.69, 0.01)
    v = phase1_verdict(obj, rand, zero, RATES, min_gain=0.05, corruption_floor=0.3)
    assert v["verdict"] == "WORKS" and v["works"] and v["reversal"] is False


def test_null_when_gain_below_min():
    # only +0.02 gain -> below min_gain 0.05 -> NULL
    obj = _flat(0.72, 0.01)
    rand = _flat(0.70, 0.01)
    zero = _flat(0.70, 0.01)
    v = phase1_verdict(obj, rand, zero, RATES, min_gain=0.05, corruption_floor=0.3)
    assert v["verdict"] == "NULL"


def test_null_when_gain_within_noise():
    # +0.06 gain but std huge -> not beyond pooled std -> NULL
    obj = _flat(0.76, 0.20)
    rand = _flat(0.70, 0.20)
    zero = _flat(0.70, 0.20)
    v = phase1_verdict(obj, rand, zero, RATES, min_gain=0.05, corruption_floor=0.3)
    assert v["verdict"] == "NULL"


def test_reversal_blocks_works():
    # strong at 0.3-0.5 but objectness DROPS below a baseline at 0.9 -> reversal -> NULL
    obj = {r: (0.80, 0.01) for r in RATES}
    rand = {r: (0.70, 0.01) for r in RATES}
    zero = {r: (0.69, 0.01) for r in RATES}
    obj[0.9] = (0.60, 0.01)   # reversal below rand at high corruption
    v = phase1_verdict(obj, rand, zero, RATES, min_gain=0.05, corruption_floor=0.3)
    assert v["reversal"] is True and v["verdict"] == "NULL"


def test_only_rates_at_or_above_floor_count():
    # objectness loses badly below floor, wins above -> still WORKS (sub-floor ignored)
    obj = dict(_flat(0.80, 0.01)); rand = dict(_flat(0.70, 0.01)); zero = dict(_flat(0.69, 0.01))
    obj[0.1] = (0.10, 0.01); obj[0.2] = (0.10, 0.01)   # awful below floor, must be ignored
    v = phase1_verdict(obj, rand, zero, RATES, min_gain=0.05, corruption_floor=0.3)
    assert v["verdict"] == "WORKS" and v["n_rates_considered"] == 7  # rates 0.3..0.9
