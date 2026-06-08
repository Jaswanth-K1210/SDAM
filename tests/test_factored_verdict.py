"""Tests for the pure FactoredSDAM verdict logic (no torch)."""
from experiments.factored_pipeline import classify, factored_verdict

RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def curve(vals):
    assert len(vals) == len(RATES)
    return dict(zip(RATES, vals))


FLAT_BASE = curve([0.90] * 9)
# objectness crosses ~0.55 (gain +.02 at 0.5, -.02 at 0.6)
OBJ = curve([0.95, 0.95, 0.94, 0.93, 0.92, 0.88, 0.85, 0.82, 0.80])


def test_classify_works():
    factored = curve([0.95] * 9)   # +0.05 everywhere, no crossover
    assert classify(factored, OBJ, FLAT_BASE, FLAT_BASE, RATES) == "WORKS"


def test_classify_collapse():
    factored = curve([0.905] * 9)  # within collapse_tol (0.02) of zeroed
    assert classify(factored, OBJ, FLAT_BASE, FLAT_BASE, RATES) == "COLLAPSE"


def test_classify_partial_fix():
    # factored crosses ~0.73 — later than objectness's ~0.55
    factored = curve([0.95, 0.95, 0.95, 0.94, 0.93, 0.92, 0.91, 0.88, 0.85])
    assert classify(factored, OBJ, FLAT_BASE, FLAT_BASE, RATES) == "PARTIAL_FIX"


def test_classify_still_hurts():
    # factored crosses at ~the same point as objectness
    factored = curve([0.95, 0.95, 0.94, 0.93, 0.92, 0.88, 0.85, 0.82, 0.80])
    assert classify(factored, OBJ, FLAT_BASE, FLAT_BASE, RATES) == "STILL_HURTS"


def test_verdict_unanimous():
    v = factored_verdict({0.5: "WORKS", 1.0: "WORKS", 2.0: "WORKS"})
    assert v["verdict"] == "WORKS" and v["passed"] is True


def test_verdict_inconclusive_when_w_flips():
    # works at 1x/2x but collapses at 0.5x -> channel scaling drives it -> INCONCLUSIVE
    v = factored_verdict({0.5: "COLLAPSE", 1.0: "WORKS", 2.0: "WORKS"})
    assert v["verdict"] == "INCONCLUSIVE" and v["passed"] is False
