import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Telecom KPI Explorer")


@app.cell(hide_code=True)
def _():
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from kpi_explorer import PairPlot

    return Path, PairPlot, mo, pl, subprocess, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Telecom KPI Explorer

    **200,000 cell-hour measurements** from a *synthetic* mobile network. No real customer or
    operator data is used. Each row is in one of five hidden operating states, and some of those
    states only separate when two KPIs are viewed together.

    **Try this:** switch to **Lasso** and circle the orange blob at good RSRP and low SINR in the
    `rsrp_dbm × sinr_db` cell. That is cell interference, which neither KPI reveals alone.
    """)
    return


@app.cell(hide_code=True)
def _(Path, pl, subprocess, sys):
    root = Path(__file__).resolve().parent.parent
    parquet = root / "data" / "cell_kpis.parquet"
    if not parquet.exists():
        subprocess.run([sys.executable, str(root / "data" / "generate.py")], check=True)
    df = pl.read_parquet(parquet)
    return (df,)


@app.cell(hide_code=True)
def _(PairPlot, df, mo):
    columns = [
        "latency_ms",
        "rsrp_dbm",
        "sinr_db",
        "jitter_ms",
        "packet_loss_pct",
        "dl_throughput_mbps",
        "prb_util_pct",
        "active_users",
    ]
    pp = mo.ui.anywidget(
        PairPlot(
            df,
            columns=columns,
            color="true_state",
            color_options=["true_state", "cluster", "region"],
        )
    )
    pp
    return (pp,)


@app.cell(hide_code=True)
def _(mo, pp):
    table = "\n".join(
        [
            "| group | selected rows |",
            "|---|---:|",
            *(f"| {label} | {count:,} |" for label, count in pp.group_counts().items()),
        ]
    )
    mo.md(f"### Filter pipeline\n\n{pp.pipeline_text()}\n\n{table}")
    return


@app.cell(hide_code=True)
def _(mo, pl, pp):
    worst = (
        pp.selected_df()
        .group_by("cell_id", "region")
        .agg(
            pl.len().alias("rows"),
            pl.col("packet_loss_pct").mean().round(2),
            pl.col("latency_ms").mean().round(1),
            pl.col("sinr_db").mean().round(1),
            pl.col("true_state").mode().sort().first().alias("typical_state"),
        )
        .filter(pl.col("rows") >= 3)
        .sort(["packet_loss_pct", "cell_id"], descending=[True, False])
        .head(10)
    )
    worst_table = "\n".join(
        [
            "| cell | region | rows | packet loss % | latency ms | SINR dB | typical state |",
            "|---|---|---:|---:|---:|---:|---|",
            *(
                f"| `{row['cell_id']}` | {row['region']} | {row['rows']:,} "
                f"| {row['packet_loss_pct']:.2f} | {row['latency_ms']:.1f} "
                f"| {row['sinr_db']:.1f} | {row['typical_state']} |"
                for row in worst.iter_rows(named=True)
            ),
        ]
    )
    body = "No cell has 3 or more rows in this selection." if worst.is_empty() else worst_table
    mo.md(
        "### 10 worst cells in the current selection\n\n"
        "Ranked by mean packet loss, among cells with at least 3 selected rows.\n\n"
        f"{body}"
    )
    return


if __name__ == "__main__":
    app.run()
