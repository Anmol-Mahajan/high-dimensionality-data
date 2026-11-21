"""Tests for brush geometry, including exact parity with the browser implementation."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import JsBridge
from matplotlib.path import Path as MplPath

from kpi_explorer.selection import (
    describe_pipeline,
    filter_mask,
    points_in_polygon,
    selection_mask,
)

SQUARE = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
CONCAVE = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [2.0, 1.0], [0.0, 4.0]]
SELF_CROSSING = [[0.0, 0.0], [4.0, 4.0], [4.0, 0.0], [0.0, 4.0]]
FIXED_POLYGON = [
    [-1.5, -1.25],
    [0.25, -2.0],
    [1.75, -0.75],
    [0.5, -0.25],
    [2.0, 1.0],
    [0.75, 2.25],
    [-0.25, 0.5],
    [-1.75, 1.5],
]


@pytest.fixture
def matrix() -> np.ndarray:
    """Return random points plus adversarial points on vertices, edges and NaN."""
    rng = np.random.default_rng(11)
    random_points = rng.normal(0, 1.2, size=(40_000, 3)).astype(np.float32)
    vertices = np.array(FIXED_POLYGON, dtype=np.float32)
    midpoints = (vertices + np.roll(vertices, 1, axis=0)) / 2
    grid = np.stack(np.meshgrid(np.arange(-2, 2.25, 0.25), np.arange(-2, 2.25, 0.25)), -1)
    special = np.vstack([vertices, midpoints, grid.reshape(-1, 2).astype(np.float32)])
    special = np.column_stack([special, np.zeros(len(special), dtype=np.float32)])
    missing = np.array([[np.nan, 0.0, 0.0], [0.0, np.nan, 0.0]], dtype=np.float32)
    return np.vstack([random_points, special, missing])


def lasso(polygon: list[list[float]]) -> dict:
    return {
        "type": "lasso",
        "x_index": 0,
        "y_index": 1,
        "x_name": "a",
        "y_name": "b",
        "polygon": polygon,
    }


def test_square_contains_interior_only() -> None:
    inside = points_in_polygon([1.0, 3.0, -1.0], [1.0, 1.0, 1.0], SQUARE)
    assert inside.tolist() == [True, False, False]


def test_concave_notch_is_outside() -> None:
    inside = points_in_polygon([2.0, 0.5, 3.5], [3.0, 1.0, 1.0], CONCAVE)
    assert inside.tolist() == [False, True, True]


def test_self_crossing_polygon_uses_even_odd_rule() -> None:
    inside = points_in_polygon([1.0, 3.0, 2.0], [2.0, 2.0, 3.5], SELF_CROSSING)
    assert inside.tolist() == [True, True, False]


def test_nan_is_never_inside() -> None:
    assert not points_in_polygon([np.nan, 1.0], [1.0, np.nan], SQUARE).any()


def test_degenerate_polygon_selects_nothing() -> None:
    assert not points_in_polygon([0.5], [0.5], [[0.0, 0.0], [1.0, 1.0]]).any()


def test_matches_matplotlib_away_from_edges() -> None:
    rng = np.random.default_rng(3)
    points = rng.uniform(-2.5, 2.5, size=(20_000, 2))
    path = MplPath(np.array(FIXED_POLYGON))
    margin = path.contains_points(points, radius=0.01) == path.contains_points(points, radius=-0.01)
    ours = points_in_polygon(points[:, 0], points[:, 1], FIXED_POLYGON)
    assert np.array_equal(ours[margin], path.contains_points(points)[margin])


def test_box_and_range_bounds_are_inclusive() -> None:
    data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    box = {"type": "box", "x_index": 0, "y_index": 1, "bounds": [1.0, 2.0, 1.0, 2.0]}
    span = {"type": "range", "x_index": 0, "y_index": None, "bounds": [2.0, 3.0]}
    assert selection_mask(box, data).tolist() == [True, True, False]
    assert selection_mask(span, data).tolist() == [False, True, True]


def test_unknown_selection_type_raises() -> None:
    data = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown selection type"):
        selection_mask({"type": "circle", "x_index": 0, "y_index": 1}, data)


def test_filters_compose_keep_then_exclude() -> None:
    data = np.arange(10, dtype=np.float32).reshape(-1, 1)
    keep = {"type": "range", "x_index": 0, "y_index": None, "bounds": [2.0, 7.0]}
    exclude = {"type": "range", "x_index": 0, "y_index": None, "bounds": [4.0, 5.0]}
    mask = filter_mask(
        [{"op": "keep", "selection": keep}, {"op": "exclude", "selection": exclude}], data
    )
    assert np.flatnonzero(mask).tolist() == [2, 3, 6, 7]


def test_describe_pipeline_sentence() -> None:
    steps = [
        {
            "op": "exclude",
            "selection": {"type": "lasso", "x_name": "sinr_db", "y_name": "rsrp_dbm"},
        },
        {"op": "keep", "selection": {"type": "range", "x_name": "latency_ms", "y_name": None}},
    ]
    assert describe_pipeline(steps, 41_616, 200_000) == (
        "lasso in `sinr_db × rsrp_dbm` → exclude, box on `latency_ms` → keep → "
        "41,616 of 200,000 rows remain"
    )


@pytest.mark.parametrize("polygon", [FIXED_POLYGON, CONCAVE, SELF_CROSSING, SQUARE])
def test_js_and_python_agree_on_lasso(
    js_bridge: JsBridge, matrix: np.ndarray, polygon: list[list[float]]
) -> None:
    selection = lasso(polygon)
    [from_js] = js_bridge.evaluate(matrix, selections=[selection])
    from_python = selection_mask(selection, matrix)
    assert from_python.any()
    assert np.array_equal(from_js, from_python)


def test_js_and_python_agree_on_box_and_range(js_bridge: JsBridge, matrix: np.ndarray) -> None:
    selections = [
        {"type": "box", "x_index": 0, "y_index": 1, "bounds": [-0.5, 1.25, -1.0, 0.75]},
        {"type": "range", "x_index": 1, "y_index": None, "bounds": [-0.25, 0.5]},
    ]
    masks = js_bridge.evaluate(matrix, selections=selections)
    for selection, from_js in zip(selections, masks, strict=True):
        assert np.array_equal(from_js, selection_mask(selection, matrix))


def test_js_and_python_agree_on_filter_stack(js_bridge: JsBridge, matrix: np.ndarray) -> None:
    stack = [
        {"op": "keep", "selection": lasso(FIXED_POLYGON)},
        {
            "op": "exclude",
            "selection": {"type": "range", "x_index": 2, "y_index": None, "bounds": [-0.5, 0.5]},
        },
        {
            "op": "keep",
            "selection": {
                "type": "box",
                "x_index": 0,
                "y_index": 2,
                "bounds": [-1.0, 1.5, -3.0, 3.0],
            },
        },
    ]
    [from_js] = js_bridge.evaluate(matrix, stacks=[stack])
    from_python = filter_mask(stack, matrix)
    assert from_python.any()
    assert np.array_equal(from_js, from_python)
