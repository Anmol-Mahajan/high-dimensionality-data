"""Generate the synthetic mobile-network cell KPI dataset.

Each row is one cell-hour measurement drawn from one of five latent operating states.
The state means are chosen so that states overlap in every 1D histogram but separate in
specific 2D projections, most notably Interference in ``rsrp_dbm x sinr_db`` and
Backhaul degradation in ``latency_ms x sinr_db``.

All data is synthetic. Usage::

    uv run python data/generate.py --rows 200000 --seed 7
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_OUTPUT = Path(__file__).parent / "cell_kpis.parquet"

KPI_COLUMNS: tuple[str, ...] = (
    "dl_throughput_mbps",
    "ul_throughput_mbps",
    "latency_ms",
    "jitter_ms",
    "packet_loss_pct",
    "rsrp_dbm",
    "sinr_db",
    "prb_util_pct",
    "active_users",
)

REGIONS: tuple[tuple[str, str, float], ...] = (
    ("London", "LDN", 0.40),
    ("Manchester", "MAN", 0.20),
    ("Birmingham", "BHM", 0.18),
    ("Glasgow", "GLA", 0.12),
    ("Bristol", "BRS", 0.10),
)

N_SITES = 834
SECTORS = ("A", "B", "C")
KMEANS_FIT_ROWS = 50_000


@dataclass(frozen=True)
class OperatingState:
    """Latent state parameters, in the transformed space the KPIs are sampled in.

    ``mean`` and ``sd`` follow the order: log DL, log UL, log latency, log jitter,
    logit loss, RSRP, SINR, logit PRB, log users.
    """

    name: str
    share: float
    mean: tuple[float, ...]
    sd: tuple[float, ...]
    busy_hour_power: float
    prb_floor: float = 0.0
    cell_concentration: float = 1.2


STATES: tuple[OperatingState, ...] = (
    OperatingState(
        name="Healthy",
        share=0.45,
        mean=(4.44, 3.18, 3.09, 1.39, -3.66, -88.0, 19.0, -0.10, 4.25),
        sd=(0.70, 0.72, 0.32, 0.45, 0.70, 5.5, 3.8, 0.95, 0.70),
        busy_hour_power=0.3,
        cell_concentration=8.0,
    ),
    OperatingState(
        name="Congested",
        share=0.20,
        mean=(2.64, 1.61, 4.22, 2.71, -1.52, -90.0, 13.0, 0.30, 5.56),
        sd=(0.45, 0.50, 0.30, 0.38, 0.50, 6.0, 3.3, 0.80, 0.30),
        busy_hour_power=3.0,
        prb_floor=85.0,
    ),
    OperatingState(
        name="Poor coverage",
        share=0.15,
        mean=(2.56, 1.16, 4.06, 2.56, -0.16, -112.0, 1.5, -0.35, 3.64),
        sd=(0.50, 0.55, 0.32, 0.40, 0.45, 4.0, 2.6, 0.60, 0.45),
        busy_hour_power=0.5,
        cell_concentration=0.8,
    ),
    OperatingState(
        name="Interference",
        share=0.12,
        mean=(2.62, 1.25, 4.02, 2.52, -0.35, -86.0, 3.0, -0.32, 3.70),
        sd=(0.50, 0.55, 0.32, 0.40, 0.50, 4.5, 2.5, 0.60, 0.45),
        busy_hour_power=0.8,
    ),
    OperatingState(
        name="Backhaul degradation",
        share=0.08,
        mean=(3.95, 2.71, 4.91, 3.53, -2.31, -88.0, 18.5, -0.10, 4.19),
        sd=(0.42, 0.45, 0.24, 0.30, 0.55, 5.5, 3.6, 0.55, 0.40),
        busy_hour_power=0.6,
        cell_concentration=0.9,
    ),
)

STATE_NAMES: tuple[str, ...] = tuple(state.name for state in STATES)

# Within-state correlation of the latent dimensions, in the same order as ``mean``.
CORRELATION = np.array(
    [
        [1.00, 0.72, -0.25, -0.18, -0.25, 0.20, 0.45, -0.30, -0.20],
        [0.72, 1.00, -0.18, -0.12, -0.20, 0.18, 0.35, -0.22, -0.15],
        [-0.25, -0.18, 1.00, 0.78, 0.25, -0.05, -0.12, 0.15, 0.12],
        [-0.18, -0.12, 0.78, 1.00, 0.28, -0.05, -0.12, 0.12, 0.10],
        [-0.25, -0.20, 0.25, 0.28, 1.00, -0.15, -0.25, 0.10, 0.06],
        [0.20, 0.18, -0.05, -0.05, -0.15, 1.00, 0.35, 0.00, 0.00],
        [0.45, 0.35, -0.12, -0.12, -0.25, 0.35, 1.00, -0.10, -0.08],
        [-0.30, -0.22, 0.15, 0.12, 0.10, 0.00, -0.10, 1.00, 0.62],
        [-0.20, -0.15, 0.12, 0.10, 0.06, 0.00, -0.08, 0.62, 1.00],
    ]
)


def _diurnal_load(hour: np.ndarray) -> np.ndarray:
    """Return relative network load for each hour, peaking at the two commute windows."""
    morning = np.exp(-0.5 * ((hour - 8.5) / 1.8) ** 2)
    evening = 1.25 * np.exp(-0.5 * ((hour - 18.5) / 2.4) ** 2)
    return 0.2 + morning + evening


def _reflect(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Fold values back into ``[low, high]`` so bounds do not create pile-ups at the edges."""
    span = high - low
    folded = np.mod(values - low, 2 * span)
    return low + np.where(folded > span, 2 * span - folded, folded)


def _expit(values: np.ndarray) -> np.ndarray:
    """Return the logistic function of ``values``."""
    return 1.0 / (1.0 + np.exp(-values))


def _build_cells(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Create the cell catalogue, returning cell ids and each cell's region index."""
    weights = np.array([share for _, _, share in REGIONS])
    site_regions = rng.choice(len(REGIONS), size=N_SITES, p=weights / weights.sum())
    cell_ids: list[str] = []
    cell_regions: list[int] = []
    next_site = [1] * len(REGIONS)
    for region in site_regions:
        code = REGIONS[region][1]
        site = next_site[region]
        next_site[region] += 1
        for sector in SECTORS:
            cell_ids.append(f"{code}-{site:04d}-{sector}")
            cell_regions.append(int(region))
    return np.array(cell_ids), np.array(cell_regions, dtype=np.int8)


def _sample_state_rows(
    state: OperatingState, count: int, cholesky: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample hours and latent KPI vectors for ``count`` rows of one state."""
    hours = np.arange(24)
    hour_weights = _diurnal_load(hours) ** state.busy_hour_power
    hour = rng.choice(hours, size=count, p=hour_weights / hour_weights.sum())

    noise = rng.standard_normal((count, len(state.mean))) @ cholesky.T
    latent = np.asarray(state.mean) + noise * np.asarray(state.sd)

    load = _diurnal_load(hour) - _diurnal_load(hours).mean()
    latent[:, 8] += 0.22 * load
    if state.prb_floor == 0.0:
        latent[:, 7] += 0.25 * load
    return hour, latent


def _latent_to_kpis(latent: np.ndarray, prb_floor: float) -> dict[str, np.ndarray]:
    """Map latent samples to bounded, physically plausible KPI values."""
    return {
        "dl_throughput_mbps": np.round(np.exp(latent[:, 0]), 2),
        "ul_throughput_mbps": np.round(np.exp(latent[:, 1]), 2),
        "latency_ms": np.round(np.exp(latent[:, 2]), 2),
        "jitter_ms": np.round(np.exp(latent[:, 3]), 2),
        "packet_loss_pct": np.round(10.0 * _expit(latent[:, 4]), 3),
        "rsrp_dbm": np.round(_reflect(latent[:, 5], -125.0, -70.0), 2),
        "sinr_db": np.round(_reflect(latent[:, 6], -5.0, 30.0), 2),
        "prb_util_pct": np.round(prb_floor + (100.0 - prb_floor) * _expit(latent[:, 7]), 2),
        "active_users": np.maximum(np.round(np.exp(latent[:, 8])), 1.0),
    }


def _sample_cells(states: np.ndarray, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    """Assign rows to cells so that problem states recur on the same subset of cells."""
    assigned = np.zeros(states.size, dtype=np.int32)
    for index, state in enumerate(STATES):
        mask = states == index
        shape = state.cell_concentration
        affinity = rng.gamma(shape, 1.0 / shape, size=n_cells)
        assigned[mask] = rng.choice(n_cells, size=int(mask.sum()), p=affinity / affinity.sum())
    return assigned


def _fit_clusters(kpis: np.ndarray, seed: int) -> np.ndarray:
    """Fit K-means (k=5) on standardised KPIs, labelling clusters by descending size."""
    scaled = StandardScaler().fit_transform(kpis)
    rng = np.random.default_rng(seed)
    fit_rows = rng.choice(len(scaled), size=min(KMEANS_FIT_ROWS, len(scaled)), replace=False)
    model = KMeans(n_clusters=len(STATES), n_init=5, random_state=seed).fit(scaled[fit_rows])
    labels = model.predict(scaled)
    order = np.argsort(-np.bincount(labels, minlength=len(STATES)), kind="stable")
    relabel = np.empty_like(order)
    relabel[order] = np.arange(len(STATES))
    return relabel[labels]


def generate_kpis(rows: int = 200_000, seed: int = 7) -> pl.DataFrame:
    """Generate the cell KPI table.

    Args:
        rows: Number of cell-hour rows to generate.
        seed: Seed for all random draws. The same seed always gives the same table.

    Returns:
        A DataFrame with cell identifiers, KPI columns, ``true_state`` and ``cluster``.
    """
    rng = np.random.default_rng(seed)
    cell_ids, cell_regions = _build_cells(rng)
    cholesky = np.linalg.cholesky(CORRELATION)

    shares = np.array([state.share for state in STATES])
    states = rng.choice(len(STATES), size=rows, p=shares / shares.sum())

    hour = np.zeros(rows, dtype=np.int8)
    columns = {name: np.zeros(rows, dtype=np.float32) for name in KPI_COLUMNS}
    for index, state in enumerate(STATES):
        mask = states == index
        count = int(mask.sum())
        if count == 0:
            continue
        state_hour, latent = _sample_state_rows(state, count, cholesky, rng)
        hour[mask] = state_hour
        for name, values in _latent_to_kpis(latent, state.prb_floor).items():
            columns[name][mask] = values

    cells = _sample_cells(states, len(cell_ids), rng)
    kpi_matrix = np.column_stack([columns[name] for name in KPI_COLUMNS])
    clusters = _fit_clusters(kpi_matrix, seed)

    region_names = [name for name, _, _ in REGIONS]
    return pl.DataFrame(
        {
            "cell_id": cell_ids[cells],
            "region": np.array(region_names)[cell_regions[cells]],
            "hour": hour,
            **{name: columns[name] for name in KPI_COLUMNS if name != "active_users"},
            "active_users": columns["active_users"].astype(np.int32),
            "true_state": np.array(STATE_NAMES)[states],
            "cluster": clusters.astype(np.int8),
        }
    ).with_columns(
        pl.col("region").cast(pl.Enum(region_names)),
        pl.col("true_state").cast(pl.Enum(list(STATE_NAMES))),
    )


def frame_digest(df: pl.DataFrame) -> str:
    """Return a SHA-256 digest of a frame's column names and values."""
    digest = hashlib.sha256()
    for name in df.columns:
        digest.update(name.encode())
        series = df[name]
        if series.dtype in (pl.String, pl.Enum):
            digest.update("\n".join(series.cast(pl.String).to_list()).encode())
        else:
            digest.update(series.to_numpy().tobytes())
    return digest.hexdigest()


def save_sanity_plot(df: pl.DataFrame, path: Path) -> None:
    """Save side-by-side scatter plots of the two projections that separate hidden states."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#2F6FDB", "#E4572E", "#2BAE66", "#E8A33D", "#D65DB1")
    sample = df.sample(n=min(40_000, df.height), seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=130)
    projections = (("rsrp_dbm", "sinr_db"), ("latency_ms", "sinr_db"))
    for axis, (x_name, y_name) in zip(axes, projections, strict=True):
        for name, color in zip(STATE_NAMES, colors, strict=True):
            subset = sample.filter(pl.col("true_state") == name)
            axis.scatter(subset[x_name], subset[y_name], s=1, alpha=0.3, color=color, label=name)
        axis.set(xlabel=x_name, ylabel=y_name)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].set_xscale("log")
    legend = axes[0].legend(markerscale=8, frameon=False, loc="upper left", fontsize=8)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sanity-plot", type=Path, default=None)
    args = parser.parse_args()

    df = generate_kpis(rows=args.rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"wrote {df.height:,} rows to {args.out} (sha256 {frame_digest(df)[:12]})")
    if args.sanity_plot is not None:
        save_sanity_plot(df, args.sanity_plot)
        print(f"wrote sanity plot to {args.sanity_plot}")


if __name__ == "__main__":
    main()
