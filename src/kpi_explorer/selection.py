"""Turn brush geometry into row masks with vectorised NumPy.

The arithmetic deliberately mirrors ``pointInPolygon`` and ``selectRows`` in
``static/widget.js``. Both sides evaluate the same float64 expressions on the same float32
values, so the browser and Python agree on every row, including points on an edge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoolArray = NDArray[np.bool_]
Selection = Mapping[str, object]
FilterStep = Mapping[str, object]


def points_in_polygon(x: ArrayLike, y: ArrayLike, polygon: ArrayLike) -> BoolArray:
    """Return which points fall inside a polygon, using even-odd ray casting.

    Args:
        x: Point x coordinates.
        y: Point y coordinates.
        polygon: Sequence of ``[x, y]`` vertices. The polygon is closed implicitly.

    Returns:
        A boolean array with one entry per point. NaN coordinates are never inside.
    """
    px = np.asarray(x, dtype=np.float64)
    py = np.asarray(y, dtype=np.float64)
    vertices = np.asarray(polygon, dtype=np.float64)
    inside = np.zeros(px.shape, dtype=bool)
    if vertices.shape[0] < 3:
        return inside

    # Rows outside the polygon's y-span cannot straddle any edge.
    candidates = np.flatnonzero((py >= vertices[:, 1].min()) & (py < vertices[:, 1].max()))
    cx = px[candidates]
    cy = py[candidates]
    toggles = np.zeros(candidates.shape, dtype=bool)
    previous = np.roll(vertices, 1, axis=0)
    for (xi, yi), (xj, yj) in zip(vertices, previous, strict=True):
        straddles = np.flatnonzero((yi > cy) != (yj > cy))
        crossing = cx[straddles] < (xj - xi) * (cy[straddles] - yi) / (yj - yi) + xi
        toggles[straddles[crossing]] ^= True
    inside[candidates] = toggles
    return inside


def selection_mask(selection: Selection, matrix: NDArray[np.floating]) -> BoolArray:
    """Evaluate one selection against a ``rows x features`` matrix.

    Args:
        selection: A ``range``, ``box`` or ``lasso`` selection dict.
        matrix: Feature values, with columns in the widget's feature order.

    Returns:
        A boolean mask with one entry per row.

    Raises:
        ValueError: If the selection type is unknown.
    """
    kind = selection["type"]
    x = matrix[:, int(selection["x_index"])].astype(np.float64)
    if kind == "range":
        low, high = selection["bounds"]
        return (x >= low) & (x <= high)

    y = matrix[:, int(selection["y_index"])].astype(np.float64)
    if kind == "box":
        x_low, x_high, y_low, y_high = selection["bounds"]
        return (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
    if kind == "lasso":
        return points_in_polygon(x, y, selection["polygon"])
    raise ValueError(f"Unknown selection type: {kind!r}")


def filter_mask(filters: Sequence[FilterStep], matrix: NDArray[np.floating]) -> BoolArray:
    """Apply a stack of keep and exclude steps, returning the rows that remain.

    Raises:
        ValueError: If a step's operation is unknown.
    """
    mask = np.ones(matrix.shape[0], dtype=bool)
    for step in filters:
        inside = selection_mask(step["selection"], matrix)
        if step["op"] == "keep":
            mask &= inside
        elif step["op"] == "exclude":
            mask &= ~inside
        else:
            raise ValueError(f"Unknown filter operation: {step['op']!r}")
    return mask


def describe_selection(selection: Selection) -> str:
    """Describe a selection in words, for example ``lasso in `x × y```."""
    if selection["type"] == "range":
        return f"box on `{selection['x_name']}`"
    return f"{selection['type']} in `{selection['x_name']} × {selection['y_name']}`"


def describe_pipeline(filters: Sequence[FilterStep], remaining: int, total: int) -> str:
    """Summarise a filter stack as one sentence ending in the remaining row count."""
    counts = f"{remaining:,} of {total:,} rows remain"
    if not filters:
        return f"No filters applied → {counts}"
    steps = ", ".join(f"{describe_selection(s['selection'])} → {s['op']}" for s in filters)
    return f"{steps} → {counts}"
