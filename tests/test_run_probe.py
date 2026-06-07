"""Unit tests for the pure verdict logic in probe/run_probe.py (no torch/data)."""
from probe.decodability import DecodabilityResult
from probe.run_probe import compute_verdict, load_config

CFG = load_config()  # uses the real probe/config.yaml thresholds


def _dec(factor, acc, margin):
    return DecodabilityResult(
        factor=factor, balanced_acc=acc, permutation_acc=acc - margin, margin=margin,
        best_C=1.0, class_counts={},
    )


def _all_decodable():
    return {f: _dec(f, 0.85, 0.40) for f in ["shape", "count", "layout"]}


def test_verdict_green():
    decod = _all_decodable()
    gate = {"per_factor": {"shape": 8.0, "count": 6.0, "layout": 5.0}, "combined": 0.30}
    v = compute_verdict(decod, gate, CFG)
    assert v["verdict"] == "GREEN" and v["gate_pass"] and v["n_decodable"] == 3


def test_verdict_yellow_dead_factor():
    """All decodable, combined healthy, but one factor dead -> YELLOW."""
    decod = _all_decodable()
    gate = {"per_factor": {"shape": 8.0, "count": 6.0, "layout": 1.0}, "combined": 0.30}
    v = compute_verdict(decod, gate, CFG)
    assert v["verdict"] == "YELLOW"
    assert v["dead_factors"] == ["layout"] and not v["gate_pass"]


def test_verdict_yellow_one_factor_not_decodable():
    decod = {"shape": _dec("shape", 0.85, 0.4), "count": _dec("count", 0.85, 0.4),
             "layout": _dec("layout", 0.55, 0.05)}  # layout below bar
    gate = {"per_factor": {"shape": 8.0, "count": 6.0, "layout": 5.0}, "combined": 0.30}
    v = compute_verdict(decod, gate, CFG)
    assert v["verdict"] == "YELLOW" and v["n_decodable"] == 2


def test_verdict_red_two_not_decodable():
    decod = {"shape": _dec("shape", 0.85, 0.4), "count": _dec("count", 0.50, 0.02),
             "layout": _dec("layout", 0.52, 0.03)}  # two below bar
    gate = {"per_factor": {"shape": 8.0, "count": 6.0, "layout": 5.0}, "combined": 0.30}
    v = compute_verdict(decod, gate, CFG)
    assert v["verdict"] == "RED" and v["n_decodable"] == 1


def test_verdict_red_collapsed_combined():
    """Even if decodable, a (near) collapsed combined span -> RED (untestable)."""
    decod = _all_decodable()
    gate = {"per_factor": {"shape": 8.0, "count": 6.0, "layout": 5.0}, "combined": 0.03}
    v = compute_verdict(decod, gate, CFG)
    assert v["verdict"] == "RED"
