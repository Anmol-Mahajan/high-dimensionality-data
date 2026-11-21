"""Tests for the PairPlot Python API and its wire format."""

from __future__ import annotations

import json

import marimo as mo
import numpy as np
import polars as pl
import pytest
from conftest import JsBridge

from kpi_explorer import PairPlot

COLUMNS = ["a", "b", "c"]
BOX = {
    "type": "box",
    "x_index": 0,
    "y_index": 1,
    "x_name": "a",
    "y_name": "b",
    "bounds": [-1.0, 1.0, -1.0, 1.0],
}
RANGE = {
    "type": "range",
    "x_index": 2,
    "y_index": None,
    "x_name": "c",
    "y_name": None,
    "bounds": [0.0, 5.0],
}
LASSO = {
    "type": "lasso",
    "x_index": 0,
    "y_index": 2,
    "x_name": "a",
    "y_name": "c",
    "polygon": [[-1.5, -1.5], [1.5, -1.0], [1.0, 1.5], [-1.0, 0.75]],
}


@pytest.fixture
def frame() -> pl.DataFrame:
    """Return a small frame with numeric, categorical and integer columns."""
    rng = np.random.default_rng(5)
    rows = 5_000
    return pl.DataFrame(
        {
            "a": rng.normal(size=rows).astype(np.float32),
            "b": rng.normal(size=rows).astype(np.float32),
            "c": rng.normal(size=rows).astype(np.float32),
            "state": rng.choice(["ok", "bad", "ugly"], size=rows),
            "cluster": rng.integers(0, 4, size=rows).astype(np.int8),
        }
    )


@pytest.fixture
def widget(frame: pl.DataFrame) -> PairPlot:
    """Return a widget over the sample frame."""
    return PairPlot(frame, columns=COLUMNS, color="state", color_options=["state", "cluster"])


def expected_box(frame: pl.DataFrame) -> pl.Series:
    return frame["a"].is_between(-1, 1) & frame["b"].is_between(-1, 1)


def test_columns_are_sent_as_float32_bytes(widget: PairPlot, frame: pl.DataFrame) -> None:
    assert isinstance(widget._columns, bytes)
    assert len(widget._columns) == frame.height * len(COLUMNS) * 4
    assert len(widget._groups) == frame.height * 2


def test_metadata_contains_no_row_data(widget: PairPlot) -> None:
    assert len(json.dumps(widget._meta)) < 500


def test_sent_rows_are_a_permutation(widget: PairPlot) -> None:
    sent = np.frombuffer(widget._columns, dtype=np.float32).reshape(3, -1)
    np.testing.assert_array_equal(sent[0], widget._matrix[widget._order, 0])
    assert not np.array_equal(sent[0], widget._matrix[:, 0])


def test_group_labels_and_cluster_prefix(widget: PairPlot) -> None:
    groups = {group["name"]: group["labels"] for group in widget._meta["groups"]}
    assert groups["state"] == ["bad", "ok", "ugly"]
    assert groups["cluster"] == ["cluster 0", "cluster 1", "cluster 2", "cluster 3"]


def test_enum_keeps_category_order() -> None:
    order = ["z", "m", "a"]
    data = pl.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["a", "z", "m"]}).with_columns(
        pl.col("g").cast(pl.Enum(order))
    )
    assert PairPlot(data, columns=["x"], color="g")._meta["groups"][0]["labels"] == order


def test_no_selection_returns_all_rows(widget: PairPlot, frame: pl.DataFrame) -> None:
    assert widget.selection is None
    assert widget.filters == []
    assert widget.selected_indices().size == frame.height
    assert widget.filtered_df().height == frame.height


def test_selection_indices_match_direct_filter(widget: PairPlot, frame: pl.DataFrame) -> None:
    widget.selection = BOX
    expected = np.flatnonzero(expected_box(frame).to_numpy())
    np.testing.assert_array_equal(widget.selected_indices(), expected)
    assert widget.selected_df().height == expected.size
    assert widget.filtered_df().height == frame.height


def test_keep_then_exclude_compose(widget: PairPlot, frame: pl.DataFrame) -> None:
    widget.filters = [{"op": "keep", "selection": BOX}, {"op": "exclude", "selection": RANGE}]
    expected = expected_box(frame) & ~frame["c"].is_between(0, 5)
    np.testing.assert_array_equal(widget.filtered_indices(), np.flatnonzero(expected.to_numpy()))


def test_selection_intersects_filters(widget: PairPlot, frame: pl.DataFrame) -> None:
    widget.filters = [{"op": "keep", "selection": BOX}]
    widget.selection = RANGE
    expected = expected_box(frame) & frame["c"].is_between(0, 5)
    np.testing.assert_array_equal(widget.selected_indices(), np.flatnonzero(expected.to_numpy()))


def test_filter_can_leave_zero_rows(widget: PairPlot) -> None:
    empty = {**RANGE, "bounds": [100.0, 200.0]}
    widget.filters = [{"op": "keep", "selection": empty}]
    assert widget.filtered_df().height == 0
    assert set(widget.group_counts().values()) == {0}


def test_group_counts_follow_colour_by(widget: PairPlot, frame: pl.DataFrame) -> None:
    widget.selection = BOX
    selected = frame.filter(expected_box(frame))
    assert sum(widget.group_counts().values()) == selected.height
    assert widget.group_counts()["ok"] == (selected["state"] == "ok").sum()
    widget.color = "cluster"
    assert widget.group_counts()["cluster 2"] == (selected["cluster"] == 2).sum()


def test_pipeline_text(widget: PairPlot, frame: pl.DataFrame) -> None:
    assert widget.pipeline_text() == f"No filters applied → {frame.height:,} of 5,000 rows remain"
    widget.filters = [{"op": "exclude", "selection": BOX}]
    widget.selection = RANGE
    text = widget.pipeline_text()
    assert text.startswith("box in `a × b` → exclude → ")
    assert "Live selection: box on `c`" in text


def test_max_points_maps_back_to_source_rows(frame: pl.DataFrame) -> None:
    widget = PairPlot(frame, columns=COLUMNS, color="state", max_points=1_000)
    assert widget._meta["rows"] == 1_000
    widget.selection = BOX
    source = frame[widget.selected_indices()]
    assert source["a"].is_between(-1, 1).all()
    assert source["b"].is_between(-1, 1).all()
    assert source.height == widget.selected_df().height


def test_unknown_column_raises(frame: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown columns"):
        PairPlot(frame, columns=["a", "nope"], color="state")


def test_high_cardinality_colour_raises(frame: pl.DataFrame) -> None:
    data = frame.with_row_index("row_id")
    with pytest.raises(ValueError, match="categories"):
        PairPlot(data, columns=COLUMNS, color="row_id")


def test_marimo_wrapper_exposes_public_api(widget: PairPlot, frame: pl.DataFrame) -> None:
    wrapped = mo.ui.anywidget(widget)
    wrapped.widget.selection = BOX
    assert wrapped.selection == BOX
    assert wrapped.filters == []
    assert wrapped.selected_indices().size == expected_box(frame).sum()
    assert wrapped.filtered_df().height == frame.height


def test_sent_buffers_replay_to_same_rows_in_js(widget: PairPlot, js_bridge: JsBridge) -> None:
    stack = [
        {"op": "keep", "selection": LASSO},
        {"op": "exclude", "selection": {**BOX, "bounds": [-0.5, 0.5, -0.5, 0.5]}},
    ]
    widget.filters = stack
    sent = np.frombuffer(widget._columns, dtype=np.float32).reshape(3, -1).T
    [from_js] = js_bridge.evaluate(sent, stacks=[stack])
    assert from_js.sum() > 0
    np.testing.assert_array_equal(np.sort(widget._order[from_js]), widget.filtered_indices())
