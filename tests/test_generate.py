"""Tests for the synthetic cell KPI generator."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from generate import CORRELATION, STATES, frame_digest, generate_kpis

ROWS = 30_000


@pytest.fixture(scope="module")
def kpis() -> pl.DataFrame:
    """Return a seeded sample of the dataset."""
    return generate_kpis(rows=ROWS, seed=7)


def test_row_count(kpis: pl.DataFrame) -> None:
    assert kpis.height == ROWS


def test_correlation_is_positive_definite() -> None:
    assert np.linalg.eigvalsh(CORRELATION).min() > 0


@pytest.mark.parametrize(
    ("column", "low", "high"),
    [
        ("hour", 0, 23),
        ("packet_loss_pct", 0, 10),
        ("rsrp_dbm", -125, -70),
        ("sinr_db", -5, 30),
        ("prb_util_pct", 0, 100),
    ],
)
def test_value_ranges(kpis: pl.DataFrame, column: str, low: float, high: float) -> None:
    assert kpis[column].min() >= low
    assert kpis[column].max() <= high


def test_no_missing_values(kpis: pl.DataFrame) -> None:
    assert kpis.null_count().to_numpy().sum() == 0


@pytest.mark.parametrize("state", STATES, ids=lambda state: state.name)
def test_state_proportions(kpis: pl.DataFrame, state) -> None:
    share = (kpis["true_state"] == state.name).mean()
    assert share == pytest.approx(state.share, abs=0.02)


def test_same_seed_gives_same_hash(kpis: pl.DataFrame) -> None:
    assert frame_digest(kpis) == frame_digest(generate_kpis(rows=ROWS, seed=7))


def test_different_seed_gives_different_hash(kpis: pl.DataFrame) -> None:
    assert frame_digest(kpis) != frame_digest(generate_kpis(rows=ROWS, seed=8))


def test_congested_rows_exceed_prb_threshold(kpis: pl.DataFrame) -> None:
    assert kpis.filter(pl.col("true_state") == "Congested")["prb_util_pct"].min() > 85


def test_congestion_concentrates_in_busy_hours(kpis: pl.DataFrame) -> None:
    busy = pl.col("hour").is_in([7, 8, 9, 10, 17, 18, 19, 20])
    congested = kpis.filter(pl.col("true_state") == "Congested")
    healthy = kpis.filter(pl.col("true_state") == "Healthy")
    assert congested.filter(busy).height / congested.height > (
        healthy.filter(busy).height / healthy.height + 0.15
    )


def test_interference_has_good_rsrp_but_low_sinr(kpis: pl.DataFrame) -> None:
    medians = kpis.group_by("true_state").agg(pl.col("rsrp_dbm", "sinr_db").median())
    lookup = {row["true_state"]: row for row in medians.to_dicts()}
    assert abs(lookup["Interference"]["rsrp_dbm"] - lookup["Healthy"]["rsrp_dbm"]) < 6
    assert abs(lookup["Interference"]["sinr_db"] - lookup["Poor coverage"]["sinr_db"]) < 3
    assert lookup["Interference"]["rsrp_dbm"] - lookup["Poor coverage"]["rsrp_dbm"] > 20


def test_clusters_disagree_with_truth_in_places(kpis: pl.DataFrame) -> None:
    assert kpis["cluster"].n_unique() == 5
    counts = kpis.group_by("cluster", "true_state").len()
    majority = counts.group_by("cluster").agg(pl.col("len").max())["len"].sum()
    assert 0.5 < majority / kpis.height < 0.95


def test_kmeans_merges_interference_into_poor_coverage(kpis: pl.DataFrame) -> None:
    poor = kpis.filter(pl.col("true_state") == "Poor coverage")
    poor_cluster = poor["cluster"].mode().item()
    interference = kpis.filter(pl.col("true_state") == "Interference")
    assert (interference["cluster"] == poor_cluster).mean() > 0.7


def test_interference_box_is_clean_in_rsrp_sinr(kpis: pl.DataFrame) -> None:
    box = kpis.filter((pl.col("rsrp_dbm") > -100) & (pl.col("sinr_db") < 7))
    is_interference = box["true_state"] == "Interference"
    assert is_interference.mean() > 0.85
    assert is_interference.sum() / (kpis["true_state"] == "Interference").sum() > 0.85
