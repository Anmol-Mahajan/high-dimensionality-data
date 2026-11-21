# CLAUDE.md — Telecom KPI Explorer

An interactive, crossfiltering **scatter-plot matrix (SPLOM) widget** for high-dimensional data, built as an
anywidget and run inside a **marimo** notebook, demoed on **200,000 rows of mobile-network cell KPIs**.
The finished project is a LinkedIn portfolio piece, so it is judged on three things: technical depth,
usability, and how clean it looks in short screen recordings.

Read this whole file before writing code. Work through the phases in order and confirm each phase's
"Done when" checks before moving on.

---

## 1. What we are building (reference behaviour)

The interaction model is copied from a marimo demo of a pair-plot widget. Reproduce these behaviours exactly:

**Layout (top to bottom)**
1. **Legend row**: one coloured dot and a label per group (e.g. `● Healthy ● Congested …`).
2. **Status line**: `40,092 of 200,000 rows · 20%` with the count in bold. When filters are active, add
   `· 2 filters`. When nothing is selected, show a hint for the cell under the cursor instead, e.g.
   `prb_util × latency_ms · drag to lasso`, `sinr_db · drag a range`, or `drag a box in any cell`.
3. **Toolbar**: four segmented button groups:
   `[Keep | Exclude | Clear]` `[Reset]` `[Box | Lasso]` `[Full | Lower]`.
   - Disable Keep/Exclude/Clear while there is no active selection.
   - Disable Reset while the filter stack is empty.
   - The active mode button is filled black with white text. Inactive buttons are plain text.
4. **Matrix grid** (N×N cells for N features):
   - **Diagonal cell**: a histogram of that feature, **stacked by group colour**, with the feature name
     top-left in bold.
   - **Off-diagonal cell**: a scatter plot of feature_j (x) against feature_i (y).
   - Y-axis ticks on the left column only. Cell backgrounds are a soft off-white (`#F3F3EF`), with a
     1 px gap between cells and no heavy gridlines.
   - **Full** shows the whole matrix. **Lower** shows the lower triangle plus the diagonal only.

**Selection (brushing)**
- **Box mode, scatter cell**: drag a 2D rectangle.
- **Box mode, diagonal cell**: drag a 1D x-range (full-height band).
- **Lasso mode, scatter cell**: freehand polygon. Draw the outline as a 1.5 px black stroke with a light
  translucent fill.
- Only one selection is live at a time. A new drag replaces it.
- **Linked highlighting**: points inside the selection keep their group colour in **every** cell, and all
  other points turn light grey (`#C9C9C4`, alpha ≈ 0.35). Histograms redraw so the selected portion is
  coloured and stacked, with the remainder in grey behind it.
- The status line updates live while dragging, throttled to one update per animation frame.

**Filtering (the stack)**
- **Keep** pushes "keep only the selected rows" onto the filter stack. **Exclude** pushes "remove the
  selected rows". **Clear** drops the current selection without filtering.
- Filters compose, so you can lasso → Keep → lasso → Exclude → box → Keep.
- After every push, **axes rescale to the remaining data** (zooming into the subset) and the status line
  shows `N of 200,000 rows · k filters`.
- **Reset** empties the stack and restores the original axes.

**Python side (the marimo part of the demo)**
- Wrap the widget with `mo.ui.anywidget(...)` so selections re-run downstream cells reactively.
- Expose these on the Python object:
  - `.selection`: a dict such as
    `{"type": "lasso"|"box"|"range", "x_index": int, "y_index": int|None, "x_name": str, "y_name": str|None, "bounds": [...] | "polygon": [[x, y], ...]}`
  - `.filters`: the list of applied filter steps, each with its operation and selection
  - `.selected_indices()` → `np.ndarray` of row indices (selection ∩ filters)
  - `.filtered_df()` / `.selected_df()` → DataFrames of those rows
- A downstream notebook cell renders a markdown summary: a sentence describing the current filter
  pipeline (e.g. "lasso in `sinr_db × rsrp_dbm` → exclude, box on `latency_ms` → keep → 41,616 of 200,000
  rows remain"), followed by a table of `| group | selected rows |` counts.

---

## 2. Dataset: synthetic mobile-network cell KPIs

This dataset is relevant to telecom/MSP work, safe to publish because it is synthetic, and has structure
that only shows up in multiple dimensions. Generate it with `data/generate.py` (seeded, deterministic) and
save it to `data/cell_kpis.parquet`.

**200,000 rows.** Each row is one cell-hour measurement.

| column | unit / type | notes |
|---|---|---|
| `cell_id` | str | ~2,500 cells, e.g. `LDN-0412-B` |
| `region` | category | London, Manchester, Birmingham, Glasgow, Bristol |
| `hour` | int 0–23 | busy-hour effects drive congestion |
| `dl_throughput_mbps` | float | log-normal |
| `ul_throughput_mbps` | float | correlated with DL |
| `latency_ms` | float | |
| `jitter_ms` | float | correlated with latency |
| `packet_loss_pct` | float | bounded 0–10 |
| `rsrp_dbm` | float | roughly −125 to −70 |
| `sinr_db` | float | roughly −5 to 30 |
| `prb_util_pct` | float | 0–100 |
| `active_users` | int | |
| `true_state` | category | latent ground truth (below) |
| `cluster` | int | K-means / GMM label fitted on standardised KPIs |

**Latent operating states** (multivariate Gaussians with realistic correlations, then clipped/transformed):
1. **Healthy** (~45%): high throughput, low latency, good RSRP and SINR, moderate PRB.
2. **Congested** (~20%): PRB > 85%, many users, throughput collapses, latency up. Concentrated in busy hours.
3. **Poor coverage** (~15%): low RSRP *and* low SINR, high packet loss.
4. **Interference** (~12%): **good RSRP but low SINR**. This state is only separable in the
   `rsrp_dbm × sinr_db` cell, which is the "aha" moment for the demo.
5. **Backhaul degradation** (~8%): radio KPIs fine, but latency and jitter high. Separable only in
   `latency_ms × sinr_db`.

Design the means so that **some states overlap in every 1D histogram but separate in specific 2D
projections**. That makes the demo worth watching. Fit `cluster` with K-means (k=5) so it disagrees with
`true_state` in places. The widget's colour-by switch (see §4) then lets the viewer lasso a region and see
the model mislabelling it.

**Optional second dataset ("embedding mode")**: generate 40 raw per-cell counters, reduce them to 6 dims with
PCA (and UMAP if installed), and load the components into the same widget. This shows the widget handles
reduced embeddings as well as raw tables.

---

## 3. Tech stack

- Python ≥ 3.11, managed with **uv** (`uv init`, `uv add …`). Do not use pip directly.
- `marimo`, `anywidget`, `traitlets`, `numpy`, `polars` (pandas only where a library needs it),
  `scikit-learn`, `pyarrow`. Optional: `umap-learn`.
- Frontend: **plain ES module JS + CSS** loaded through anywidget's `_esm` / `_css`. No React and no bundler
  unless one proves necessary.
- Dev tools: `pytest`, `ruff`, and `playwright` (for GIF capture), plus `ffmpeg` on PATH.

Before writing the widget, quickly check whether marimo or `wigglystuff` already publishes this pair-plot
widget. If one exists, note it in the README as prior art. **Still build our own**: building it is the
portfolio value. Do not copy its code.

---

## 4. Architecture and performance rules

200k points × up to 36 cells means up to 7.2M point draws per frame. These rules are mandatory:

1. **Binary data transfer.** Send columns from Python to JS as `Float32Array` buffers through anywidget's
   binary support (traitlets `Bytes`, or the `buffers` argument of custom messages). **Never send 200k
   rows as JSON.** Group labels go as `Uint8Array`.
2. **Selection is computed in JS** so brushing feels instant. Store the result as a `Uint8Array` mask.
3. **Sync back to Python only on mouse-up** (and on Keep/Exclude). Send the selection *geometry*, not the
   mask. Python recomputes the indices with vectorised NumPy (`matplotlib.path.Path.contains_points` or a
   NumPy ray-casting function for lasso polygons). Add a pytest that confirms JS and Python agree on a
   fixed polygon.
4. **Rendering**: one `<canvas>` per cell, or one large canvas with cell viewports (preferred, because it
   means fewer elements). Draw scatter points by writing directly into an `ImageData` buffer (1 px points,
   alpha-blended) rather than calling `arc()` 200k times. Draw grey (unselected) points first and coloured
   points on top. If any frame takes more than 50 ms at 8 features, switch to WebGL (`regl` from a CDN ESM
   import is fine).
5. **Histograms**: compute 30 bins per feature in JS from the current filtered rows. Recompute counts on
   selection change.
6. **Overlay layer**: a separate transparent canvas for the box/lasso outline and the hover crosshair, so
   brushing never forces a full redraw of the points.
7. **Throttle** redraws with `requestAnimationFrame`. Debounce the status-line text.
8. **Colour-by switch**: an extra small dropdown on the toolbar (`true_state` / `cluster` / `region`) to
   re-colour without re-sending the data.

**Repository layout**
```
telecom-kpi-explorer/
├── CLAUDE.md
├── README.md                    # what it is, GIFs, how to run, design notes
├── pyproject.toml
├── data/
│   └── generate.py              # seeded synthetic data → cell_kpis.parquet
├── src/kpi_explorer/
│   ├── __init__.py
│   ├── widget.py                # PairPlot(anywidget.AnyWidget)
│   ├── selection.py             # polygon/box/range → indices (NumPy)
│   └── static/
│       ├── widget.js
│       └── widget.css
├── notebooks/
│   └── explorer.py              # the marimo app used in the demo
├── scripts/
│   └── record_gifs.py           # Playwright capture → GIF + MP4
├── tests/
│   ├── test_generate.py
│   ├── test_selection.py
│   └── test_widget_state.py
└── assets/gifs/                 # outputs for LinkedIn/README
```

**Public API**
```python
from kpi_explorer import PairPlot
pp = mo.ui.anywidget(PairPlot(df, columns=[...], color="true_state", max_points=None))
pp.selection; pp.filters; pp.selected_indices(); pp.filtered_df()
```

---

## 5. Visual design (clean and professional)

- Font: system UI stack (`-apple-system, "Segoe UI", Inter, sans-serif`), 12–13 px. Tabular figures for
  counts (`font-variant-numeric: tabular-nums`).
- Palette (colour-blind safe, muted): Healthy `#2F6FDB`, Congested `#E4572E`, Poor coverage `#2BAE66`,
  Interference `#E8A33D`, Backhaul `#D65DB1`. Unselected `#C9C9C4`. Text `#1A1A1A`, secondary `#8A8A85`.
- Buttons: 28 px tall, 6 px radius, 1 px `#E2E2DD` border around each segmented group, no shadows.
- Lots of whitespace. No chart titles inside cells apart from the diagonal feature name.
- Must look right in both marimo light and dark themes. Read marimo's CSS variables where available, and
  fall back to the light palette.
- Notebook narrative in `explorer.py`: a short markdown intro (what the data is and what to try), then the
  widget, then the reactive summary cell, then one "drill-down" cell (a table of the 10 worst cells in the
  current selection, ranked by packet loss). Hide code by default when running as an app.

---

## 6. Phases

**Phase 1: Data**
- Write `data/generate.py` (CLI: `--rows 200000 --seed 7`).
- Tests: row count, value ranges, state proportions ±2%, and determinism (same seed gives the same hash).
- *Done when*: a seaborn/matplotlib sanity plot of `rsrp_dbm × sinr_db` clearly separates Interference.
  Save it to `assets/sanity.png` and look at it.

**Phase 2: Static widget**
- Draw the matrix, histograms, legend, status line, and toolbar (no interaction yet).
- Full/Lower toggle.
- *Done when*: `uv run marimo edit notebooks/explorer.py` renders 8 features × 200k rows in under 1.5 s.

**Phase 3: Brushing**
- Box (2D + 1D range) and Lasso, linked highlighting, live status line, hint text.
- *Done when*: dragging a lasso stays smooth (no visible lag) at 8 features. Measure frame time with
  `performance.now()` and log it in dev mode only.

**Phase 4: Filter stack and Python sync**
- Keep / Exclude / Clear / Reset, axis rescaling, `.selection`, `.filters`, `.selected_indices()`.
- Reactive markdown summary cell.
- *Done when*: `tests/test_widget_state.py` and `test_selection.py` pass, and JS/Python index parity holds.

**Phase 5: Polish**
- Colour-by dropdown, dark mode, keyboard shortcuts (`B` box, `L` lasso, `K` keep, `X` exclude,
  `Esc` clear, `R` reset) shown as tooltips, empty-state message when a filter leaves 0 rows.
- README with GIFs, a "Design decisions" section (binary transfer, JS-side selection, geometry sync),
  and run instructions.

**Phase 6: GIF capture** (see §7)

---

## 7. GIF capture for LinkedIn (`scripts/record_gifs.py`)

Record **real** screen captures of the running app using scripted Playwright mouse movements, so every
take is smooth and repeatable.

1. Start the app: `uv run marimo run notebooks/explorer.py --headless --port 2718`.
2. Playwright Chromium: viewport **1200×1200** (square crops well on LinkedIn), `device_scale_factor=2`,
   `record_video_dir=...`. Hide the cursor? No: **keep the cursor visible**, because it shows the interaction.
3. Drive the mouse with eased paths (≥ 40 intermediate `mouse.move` steps per stroke, small pauses between
   steps) so the lasso looks hand-drawn.
4. Convert each take with a two-pass palette for clean, small GIFs:
   ```
   ffmpeg -i take.webm -vf "fps=15,scale=900:-1:flags=lanczos,palettegen=stats_mode=diff" pal.png
   ffmpeg -i take.webm -i pal.png -lavfi "fps=15,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4" out.gif
   ```
   Also export an MP4 (`-c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p`) of each take, as a
   fallback if a GIF doesn't animate on upload. Keep each GIF short (6–10 s) and small.
5. **Shot list** (one GIF each, saved to `assets/gifs/`):
   - `01_overview.gif`: page loads, and the camera holds on the full 8×8 matrix of 200k points for 2 s.
     Toggle Full → Lower → Full.
   - `02_lasso_interference.gif`: Lasso mode. Draw around the good-RSRP / low-SINR blob in
     `rsrp_dbm × sinr_db`. Every other cell lights up with the selection, and the status line reads
     ~12% of rows.
   - `03_filter_stack.gif`: Box on the `latency_ms` histogram → Keep → lasso a sub-cloud → Exclude. Axes
     zoom in after each step and the filter counter increments. End on Reset.
   - `04_python_sync.gif`: split view. Make a selection, then scroll to the reactive markdown cell
     showing the pipeline sentence and per-group counts updating.
   - `05_cluster_vs_truth.gif`: switch colour-by from `true_state` to `cluster`, lasso the Interference
     region, and show the K-means label mix in the summary table.
6. Look at every GIF (extract a mid-frame PNG and check it) before calling the phase done.

---

## 8. Conventions

- Type hints everywhere. Docstrings on public functions. Run `ruff format` and `ruff check` clean.
- No secrets or real customer data. The dataset is synthetic, so say so in the README and the notebook intro.
- Keep `widget.js` readable: small named functions (`drawScatterCell`, `drawHistogramCell`,
  `pointInPolygon`, `applyFilters`, `rescaleAxes`) rather than one big render loop.
- Commit after each phase with a descriptive message.
- If a performance target is missed, measure first and change one thing at a time. Record what you tried
  in `README.md → Design decisions`.
