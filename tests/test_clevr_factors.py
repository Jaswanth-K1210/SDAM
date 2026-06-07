"""Tests for probe/clevr_factors.py on synthetic scene dicts (no real CLEVR)."""
import numpy as np

from probe.clevr_factors import (
    shape_factor, count_factor, layout_factor, build_factor_arrays, SHAPE_CLASSES,
)


def _obj(shape, x):
    return {"shape": shape, "pixel_coords": [x, 100, 9.0]}


def test_shape_factor_dominant():
    scene = {"objects": [_obj("cube", 10), _obj("cube", 20), _obj("cube", 30), _obj("sphere", 40)]}
    assert shape_factor(scene) == SHAPE_CLASSES["cube"]  # 0


def test_shape_factor_tie_broken_by_order():
    scene = {"objects": [_obj("cube", 10), _obj("cube", 20), _obj("sphere", 30), _obj("sphere", 40)]}
    # 2 cubes vs 2 spheres -> cube wins (lower SHAPE_CLASSES index)
    assert shape_factor(scene) == SHAPE_CLASSES["cube"]  # 0


def test_count_factor():
    scene = {"objects": [_obj("cube", i) for i in range(7)]}
    assert count_factor(scene) == 7


def test_layout_factor_spread():
    wide = {"objects": [_obj("cube", 10), _obj("cube", 470), _obj("cube", 240)]}
    tight = {"objects": [_obj("cube", 230), _obj("cube", 240), _obj("cube", 250)]}
    assert layout_factor(wide) > layout_factor(tight)


def test_layout_fallback_to_relationships():
    scene = {
        "objects": [{"shape": "cube"}, {"shape": "sphere"}],  # no pixel_coords
        "relationships": {"left": [[0], [0, 1]], "right": [[1], []]},
    }
    val = layout_factor(scene)
    assert isinstance(val, float)
    # left total = 1 + 2 = 3 ; right total = 1 + 0 = 1 ; sum = 4
    assert val == 4.0


def test_build_factor_arrays_aligned():
    scenes = [
        {"objects": [_obj("cube", 10), _obj("sphere", 20)]},
        {"objects": [_obj("cylinder", 5)]},
        {"objects": [_obj("sphere", 1), _obj("sphere", 2), _obj("sphere", 400)]},
        {"objects": [_obj("cube", 100), _obj("cube", 110), _obj("cylinder", 120)]},
        {"objects": [_obj("cylinder", 200), _obj("cylinder", 210)]},
    ]
    arr = build_factor_arrays(scenes)
    assert arr["shape"].shape == (5,) and arr["count"].shape == (5,) and arr["layout"].shape == (5,)
    assert arr["shape"].dtype == np.int64 and arr["count"].dtype == np.int64
    assert arr["layout"].dtype == np.float64
    # order preserved: counts in input order
    assert arr["count"].tolist() == [2, 1, 3, 3, 2]
    # shapes: cube-tie? scene0 1cube+1sphere -> cube(0); scene1 cylinder(2);
    # scene2 sphere(1); scene3 2cube+1cyl -> cube(0); scene4 cylinder(2)
    assert arr["shape"].tolist() == [0, 2, 1, 0, 2]
