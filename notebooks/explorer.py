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


if __name__ == "__main__":
    app.run()
