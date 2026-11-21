"""Crossfiltering scatter-plot matrix widget for marimo and other anywidget hosts."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import anywidget
import numpy as np
import polars as pl
import traitlets

from kpi_explorer.selection import (
    describe_pipeline,
    describe_selection,
    filter_mask,
    selection_mask,
)

_STATIC = Path(__file__).parent / "static"
_MAX_CATEGORIES = 32
_DEFAULT_MAX_COLUMNS = 8
_DRAW_ORDER_SEED = 0


def _encode_groups(series: pl.Series) -> tuple[np.ndarray, list[str]]:
    """Encode a categorical-like column as ``uint8`` codes plus display labels."""
    if isinstance(series.dtype, pl.Enum):
        categories = series.dtype.categories.to_list()
        present = set(series.drop_nulls().unique().to_list())
        labels = [label for label in categories if label in present]
    elif series.dtype.is_numeric():
        labels = [str(value) for value in series.drop_nulls().unique().sort().to_list()]
    else:
        labels = sorted(series.drop_nulls().cast(pl.String).unique().to_list())

    text = series.cast(pl.String)
    if series.null_count():
        labels = [*labels, "null"]
        text = text.fill_null("null")
    if len(labels) > _MAX_CATEGORIES:
        raise ValueError(
            f"Column {series.name!r} has {len(labels)} categories; "
            f"the limit for colouring is {_MAX_CATEGORIES}."
        )

    lookup = {label: code for code, label in enumerate(labels)}
    codes = text.replace_strict(lookup, return_dtype=pl.UInt8).to_numpy()
    if series.dtype.is_numeric():
        labels = [f"{series.name} {label}" for label in labels]
    return codes, labels


def _infer_color_options(frame: pl.DataFrame, color: str | None) -> list[str]:
    """Return the low-cardinality columns that can be used for colouring."""
    options: list[str] = [] if color is None else [color]
    for name, dtype in frame.schema.items():
        if name == color or dtype.is_float():
            continue
        if frame[name].n_unique() <= min(_MAX_CATEGORIES, 12):
            options.append(name)
    return options


class PairPlot(anywidget.AnyWidget):
    """Interactive scatter-plot matrix with linked brushing and a filter stack.

    Args:
        data: A polars DataFrame, or anything ``polars.DataFrame`` can construct from.
        columns: Numeric columns to plot. Defaults to the first eight numeric columns.
        color: Categorical column used for the initial colouring.
        max_points: Optionally downsample to this many rows (seeded, without replacement).
        color_options: Columns offered by the colour-by switch. Inferred when omitted.
    """

    _esm = _STATIC / "widget.js"
    _css = _STATIC / "widget.css"

    _meta = traitlets.Dict().tag(sync=True)
    _columns = traitlets.Bytes().tag(sync=True)
    _groups = traitlets.Bytes().tag(sync=True)
    color = traitlets.Unicode("").tag(sync=True)
    selection = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)
    filters = traitlets.List(trait=traitlets.Dict()).tag(sync=True)

    def __init__(
        self,
        data: pl.DataFrame | Mapping[str, Sequence[object]],
        columns: Sequence[str] | None = None,
        color: str | None = None,
        max_points: int | None = None,
        color_options: Sequence[str] | None = None,
    ) -> None:
        frame = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)
        rng = np.random.default_rng(_DRAW_ORDER_SEED)

        if max_points is not None and max_points < frame.height:
            rows = np.sort(rng.choice(frame.height, size=max_points, replace=False))
        else:
            rows = np.arange(frame.height)

        options = (
            list(color_options) if color_options is not None else _infer_color_options(frame, color)
        )
        if color is not None and color not in options:
            options.insert(0, color)
        names = list(columns) if columns is not None else self._default_columns(frame, options)
        missing = [name for name in [*names, *options] if name not in frame.columns]
        if missing:
            raise ValueError(f"Unknown columns: {missing}")

        self._frame = frame
        self._rows = rows
        self._names = names
        self._matrix = np.column_stack(
            [
                frame[name].cast(pl.Float32).fill_null(float("nan")).to_numpy()[rows]
                for name in names
            ]
        )

        # Rows are sent in random order so that no group is systematically drawn on top.
        order = rng.permutation(rows.size)
        encoded = [_encode_groups(frame[name][rows]) for name in options]
        self._order = order
        self._encoded_groups = dict(zip(options, encoded, strict=True))
        super().__init__(
            _meta={
                "rows": int(rows.size),
                "names": names,
                "groups": [
                    {"name": name, "labels": labels}
                    for name, (_, labels) in zip(options, encoded, strict=True)
                ],
                "dev": os.environ.get("KPI_EXPLORER_DEV") == "1",
            },
            _columns=self._matrix[order].tobytes(order="F"),
            _groups=b"".join(codes[order].tobytes() for codes, _ in encoded),
            color=color or (options[0] if options else ""),
        )

    @staticmethod
    def _default_columns(frame: pl.DataFrame, exclude: Sequence[str]) -> list[str]:
        """Pick the first numeric columns that are not used for colouring."""
        numeric = [
            name
            for name, dtype in frame.schema.items()
            if dtype.is_numeric() and name not in exclude
        ]
        return numeric[:_DEFAULT_MAX_COLUMNS]

    def _filter_mask(self) -> np.ndarray:
        """Return a mask over the widget's rows for the current filter stack."""
        return filter_mask(self.filters, self._matrix)

    def _selected_mask(self) -> np.ndarray:
        """Return a mask of rows in the live selection and the filter stack."""
        mask = self._filter_mask()
        if self.selection:
            mask &= selection_mask(self.selection, self._matrix)
        return mask

    def filtered_indices(self) -> np.ndarray:
        """Return row indices of the source frame that pass the filter stack."""
        return self._rows[self._filter_mask()]

    def selected_indices(self) -> np.ndarray:
        """Return source-frame row indices in the selection and the filter stack.

        When nothing is selected, this is every row that passes the filter stack.
        """
        return self._rows[self._selected_mask()]

    def filtered_df(self) -> pl.DataFrame:
        """Return the rows that pass the filter stack."""
        return self._frame[self.filtered_indices()]

    def selected_df(self) -> pl.DataFrame:
        """Return the rows in the selection and the filter stack."""
        return self._frame[self.selected_indices()]

    def group_counts(self) -> dict[str, int]:
        """Count selected rows per label of the current colour-by column."""
        codes, labels = self._encoded_groups[self.color]
        counts = np.bincount(codes[self._selected_mask()], minlength=len(labels))
        return {label: int(count) for label, count in zip(labels, counts, strict=True)}

    def pipeline_text(self) -> str:
        """Describe the filter pipeline and any live selection in one sentence."""
        remaining = int(self._filter_mask().sum())
        text = describe_pipeline(self.filters, remaining, int(self._rows.size))
        if self.selection:
            selected = int(self._selected_mask().sum())
            text += f". Live selection: {describe_selection(self.selection)} ({selected:,} rows)"
        return text
