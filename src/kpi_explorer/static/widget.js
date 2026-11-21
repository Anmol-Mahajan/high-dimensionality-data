// Telecom KPI Explorer: a crossfiltering scatter-plot matrix rendered on canvas.

const PALETTE = [
  "#2F6FDB", "#E4572E", "#2BAE66", "#E8A33D", "#D65DB1",
  "#3AA6B9", "#8C6BD1", "#A8783F", "#5E7D3A", "#C2476B",
];
const HIST_BINS = 30;
const NO_BIN = 255;
const NO_PIXEL = 0xffff;
const OPAQUE = 0xff000000 | 0;
const GUTTER = { left: 44, bottom: 22, right: 2, top: 2 };
const CELL_GAP = 1;
const MIN_CELL = 64;
const DOMAIN_PADDING = 0.04;
const HIST_HEADROOM_CSS = 20;

// ---------------------------------------------------------------------------
// Data decoding
// ---------------------------------------------------------------------------

function toBytes(value) {
  if (value instanceof Uint8Array) return value;
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  return new Uint8Array(value);
}

function decodeColumns(value, rows, count) {
  let bytes = toBytes(value);
  if (bytes.byteOffset % 4 !== 0) bytes = bytes.slice();
  const flat = new Float32Array(bytes.buffer, bytes.byteOffset, rows * count);
  return Array.from({ length: count }, (_, i) => flat.subarray(i * rows, (i + 1) * rows));
}

function decodeGroupings(value, rows, groups) {
  const bytes = toBytes(value);
  return groups.map((group, i) => ({
    name: group.name,
    labels: group.labels,
    codes: bytes.subarray(i * rows, (i + 1) * rows),
  }));
}

// ---------------------------------------------------------------------------
// Scales and ticks
// ---------------------------------------------------------------------------

function columnExtent(column, active, count) {
  let min = Infinity;
  let max = -Infinity;
  for (let k = 0; k < count; k++) {
    const value = column[active[k]];
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return [min, max];
}

function paddedDomain([min, max]) {
  if (!Number.isFinite(min)) return [0, 1];
  if (min === max) return [min - 0.5, max + 0.5];
  const pad = (max - min) * DOMAIN_PADDING;
  return [min - pad, max + pad];
}

function rescaleAxes(view) {
  const { columns } = view.data;
  view.domains = columns.map((column) =>
    paddedDomain(columnExtent(column, view.active, view.activeCount)),
  );
}

function niceTicks(min, max, target) {
  const rough = (max - min) / target;
  const power = 10 ** Math.floor(Math.log10(rough));
  const ratio = rough / power;
  const step = (ratio >= 5 ? 10 : ratio >= 2 ? 5 : ratio >= 1.5 ? 2 : 1) * power;
  const ticks = [];
  for (let n = Math.ceil(min / step); n * step <= max; n++) ticks.push(n * step);
  return ticks;
}

function formatTick(value) {
  const size = Math.abs(value);
  if (size >= 1000) return `${+(value / 1000).toFixed(1)}k`;
  if (size >= 10 || value === 0) return String(Math.round(value));
  return String(+value.toFixed(1));
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

function computeGrid(width, count) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const usable = width - GUTTER.left - GUTTER.right - CELL_GAP * (count - 1);
  const cell = Math.max(MIN_CELL, Math.floor(usable / count));
  const size = cell * count + CELL_GAP * (count - 1);
  return {
    count,
    dpr,
    cell,
    size,
    cellPx: Math.round(cell * dpr),
    stepPx: Math.round((cell + CELL_GAP) * dpr),
    leftPx: Math.round(GUTTER.left * dpr),
    topPx: Math.round(GUTTER.top * dpr),
    cssWidth: GUTTER.left + size + GUTTER.right,
    cssHeight: GUTTER.top + size + GUTTER.bottom,
    width: Math.round((GUTTER.left + size + GUTTER.right) * dpr),
    height: Math.round((GUTTER.top + size + GUTTER.bottom) * dpr),
  };
}

function isCellVisible(view, row, col) {
  return view.layoutMode === "full" || col <= row;
}

function sizeCanvas(canvas, grid) {
  canvas.width = grid.width;
  canvas.height = grid.height;
  canvas.style.width = `${grid.cssWidth}px`;
  canvas.style.height = `${grid.cssHeight}px`;
}

function applyLayout(view) {
  const width = view.stage.clientWidth;
  if (!width) return;
  const grid = computeGrid(width, view.data.names.length);
  view.grid = grid;
  view.stage.style.height = `${grid.cssHeight}px`;
  for (const canvas of [view.pointsCanvas, view.labelsCanvas, view.overlayCanvas]) {
    sizeCanvas(canvas, grid);
  }
  view.frame = view.pointsContext.createImageData(grid.width, grid.height);
  view.pixels = new Int32Array(view.frame.data.buffer);
  computePixelCoords(view);
  computeBins(view);
}

// ---------------------------------------------------------------------------
// Per-feature caches
// ---------------------------------------------------------------------------

function computePixelCoords(view) {
  const { cellPx } = view.grid;
  const top = cellPx - 1;
  view.pixelCoords = view.data.columns.map((column, f) => {
    const out = view.pixelCoords?.[f] ?? new Uint16Array(column.length);
    const [min, max] = view.domains[f];
    const scale = top / (max - min);
    for (let r = 0; r < column.length; r++) {
      const px = (column[r] - min) * scale;
      out[r] = px >= 0 && px <= top ? px : NO_PIXEL;
    }
    return out;
  });
}

function computeBins(view) {
  view.bins = view.data.columns.map((column, f) => {
    const out = view.bins?.[f] ?? new Uint8Array(column.length);
    const [min, max] = view.domains[f];
    const scale = HIST_BINS / (max - min);
    for (let r = 0; r < column.length; r++) {
      const bin = Math.floor((column[r] - min) * scale);
      out[r] = bin >= 0 && bin < HIST_BINS ? bin : NO_BIN;
    }
    return out;
  });
}

function computeHistograms(view, rows, count) {
  const groups = view.grouping.labels.length;
  const features = view.data.names.length;
  const counts = new Uint32Array(features * HIST_BINS * groups);
  const codes = view.grouping.codes;
  for (let f = 0; f < features; f++) {
    const bins = view.bins[f];
    const base = f * HIST_BINS;
    for (let k = 0; k < count; k++) {
      const r = rows[k];
      const bin = bins[r];
      if (bin !== NO_BIN) counts[(base + bin) * groups + codes[r]]++;
    }
  }
  return counts;
}

// ---------------------------------------------------------------------------
// Raster drawing
// ---------------------------------------------------------------------------

function parseColor(hex) {
  const value = parseInt(hex.slice(1), 16);
  return { r: (value >> 16) & 255, g: (value >> 8) & 255, b: value & 255 };
}

function packColor({ r, g, b }) {
  return (0xff000000 | (b << 16) | (g << 8) | r) >>> 0;
}

function blendTables(colors, alpha) {
  const a = Math.max(1, Math.min(256, Math.round(alpha * 256)));
  const rb = new Int32Array(colors.length);
  const g = new Int32Array(colors.length);
  colors.forEach((color, i) => {
    rb[i] = Math.imul((color.b << 16) | color.r, a);
    g[i] = Math.imul(color.g << 8, a);
  });
  return { rb, g, inverse: 256 - a };
}

function pointAlpha(grid, count) {
  const density = count / (grid.cellPx * grid.cellPx);
  return Math.max(0.09, Math.min(0.75, 0.42 / Math.sqrt(density + 0.05)));
}

function cellOrigin(view, row, col) {
  const { leftPx, topPx, stepPx, width } = view.grid;
  return (topPx + row * stepPx) * width + leftPx + col * stepPx;
}

function fillPixels(view, x, y, w, h, color) {
  const { pixels } = view;
  const { width } = view.grid;
  for (let j = y; j < y + h; j++) {
    pixels.fill(color, j * width + x, j * width + x + w);
  }
}

function clearCells(view) {
  const { count, leftPx, topPx, stepPx, cellPx } = view.grid;
  const background = packColor(view.theme.cell);
  view.pixels.fill(0);
  for (let row = 0; row < count; row++) {
    for (let col = 0; col < count; col++) {
      if (!isCellVisible(view, row, col)) continue;
      fillPixels(view, leftPx + col * stepPx, topPx + row * stepPx, cellPx, cellPx, background);
    }
  }
}

function drawScatterCell(view, row, col, rows, count, tables, codes) {
  const { pixels } = view;
  const { width, cellPx } = view.grid;
  const xs = view.pixelCoords[col];
  const ys = view.pixelCoords[row];
  const origin = cellOrigin(view, row, col);
  const top = cellPx - 1;
  const { rb, g, inverse } = tables;
  for (let k = 0; k < count; k++) {
    const r = rows[k];
    const px = xs[r];
    const py = ys[r];
    if (px === NO_PIXEL || py === NO_PIXEL) continue;
    const p = origin + (top - py) * width + px;
    const d = pixels[p];
    const c = codes[r];
    pixels[p] =
      OPAQUE |
      (((Math.imul(d & 0xff00ff, inverse) + rb[c]) >>> 8) & 0xff00ff) |
      (((Math.imul(d & 0xff00, inverse) + g[c]) >>> 8) & 0xff00);
  }
}

// Cell (col, row) plots the same points as (row, col) with axes swapped, so the upper
// triangle is copied from the lower one by reflecting across the anti-diagonal.
function mirrorScatterCell(view, row, col) {
  const { pixels } = view;
  const { width, cellPx } = view.grid;
  const top = cellPx - 1;
  const source = cellOrigin(view, row, col);
  const target = cellOrigin(view, col, row);
  for (let y = 0; y < cellPx; y++) {
    const targetLine = target + y * width;
    const sourceColumn = top - y;
    for (let x = 0; x < cellPx; x++) {
      pixels[targetLine + x] = pixels[source + (top - x) * width + sourceColumn];
    }
  }
}

function drawHistogramCell(view, feature, counts) {
  const { cellPx, stepPx, leftPx, topPx, dpr } = view.grid;
  const groups = view.grouping.labels.length;
  const x0 = leftPx + feature * stepPx;
  const y0 = topPx + feature * stepPx;
  const usable = cellPx - Math.round(HIST_HEADROOM_CSS * dpr);
  const base = feature * HIST_BINS;

  let peak = 1;
  for (let bin = 0; bin < HIST_BINS; bin++) {
    let total = 0;
    for (let g = 0; g < groups; g++) total += counts[(base + bin) * groups + g];
    if (total > peak) peak = total;
  }

  const barGap = Math.max(1, Math.round(dpr / 2));
  for (let bin = 0; bin < HIST_BINS; bin++) {
    const left = x0 + Math.round((bin * cellPx) / HIST_BINS);
    const right = x0 + Math.round(((bin + 1) * cellPx) / HIST_BINS) - barGap;
    let cumulative = 0;
    for (let g = 0; g < groups; g++) {
      const value = counts[(base + bin) * groups + g];
      if (!value) continue;
      const lower = Math.round((cumulative / peak) * usable);
      cumulative += value;
      const upper = Math.round((cumulative / peak) * usable);
      if (upper > lower) {
        fillPixels(
          view, left, y0 + cellPx - upper, Math.max(1, right - left), upper - lower,
          view.groupPacked[g],
        );
      }
    }
  }
}

function renderPoints(view) {
  const started = performance.now();
  const { count } = view.grid;
  clearCells(view);
  const cleared = performance.now();
  const tables = blendTables(view.groupColors, pointAlpha(view.grid, view.activeCount));
  for (let row = 1; row < count; row++) {
    for (let col = 0; col < row; col++) {
      drawScatterCell(view, row, col, view.active, view.activeCount, tables, view.grouping.codes);
      if (view.layoutMode === "full") mirrorScatterCell(view, row, col);
    }
  }
  const scattered = performance.now();
  const counts = computeHistograms(view, view.active, view.activeCount);
  for (let f = 0; f < count; f++) drawHistogramCell(view, f, counts);
  const histogrammed = performance.now();
  view.pointsContext.putImageData(view.frame, 0, 0);
  const finished = performance.now();
  recordFrame(view, finished - started, {
    clear: cleared - started,
    scatter: scattered - cleared,
    histogram: histogrammed - scattered,
    put: finished - histogrammed,
  });
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

function renderLabels(view) {
  const { labelsContext: ctx, grid, theme } = view;
  const { count, dpr, cell } = grid;
  const names = view.data.names;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, grid.cssWidth, grid.cssHeight);
  const family = '-apple-system, "Segoe UI", Inter, sans-serif';

  ctx.fillStyle = theme.text;
  ctx.font = `600 11px ${family}`;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  names.forEach((name, f) => {
    const offset = f * (cell + CELL_GAP);
    ctx.save();
    ctx.beginPath();
    ctx.rect(GUTTER.left + offset, GUTTER.top + offset, cell, cell);
    ctx.clip();
    ctx.fillText(name, GUTTER.left + offset + 5, GUTTER.top + offset + 5);
    ctx.restore();
  });

  ctx.fillStyle = theme.muted;
  ctx.font = `10px ${family}`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let row = 0; row < count; row++) {
    const [min, max] = view.domains[row];
    const y0 = GUTTER.top + row * (cell + CELL_GAP);
    for (const tick of niceTicks(min, max, 3)) {
      const y = y0 + cell - ((tick - min) / (max - min)) * cell;
      ctx.fillText(formatTick(tick), GUTTER.left - 5, Math.max(y0 + 5, Math.min(y0 + cell - 5, y)));
    }
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const bottom = GUTTER.top + grid.size + 5;
  for (let col = 0; col < count; col++) {
    const [min, max] = view.domains[col];
    const x0 = GUTTER.left + col * (cell + CELL_GAP);
    for (const tick of niceTicks(min, max, 3)) {
      const x = x0 + ((tick - min) / (max - min)) * cell;
      ctx.fillText(formatTick(tick), Math.max(x0 + 9, Math.min(x0 + cell - 9, x)), bottom);
    }
  }
}

// ---------------------------------------------------------------------------
// DOM chrome
// ---------------------------------------------------------------------------

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function buildSegment(view, entries) {
  const segment = element("div", "kpi-segment");
  for (const { id, label, onClick } of entries) {
    const button = element("button", "kpi-btn", label);
    button.type = "button";
    button.addEventListener("click", onClick);
    view.buttons[id] = button;
    segment.append(button);
  }
  return segment;
}

function buildToolbar(view) {
  const noop = () => {};
  view.toolbar.append(
    buildSegment(view, [
      { id: "keep", label: "Keep", onClick: noop },
      { id: "exclude", label: "Exclude", onClick: noop },
      { id: "clear", label: "Clear", onClick: noop },
    ]),
    buildSegment(view, [{ id: "reset", label: "Reset", onClick: noop }]),
    buildSegment(view, [
      { id: "box", label: "Box", onClick: () => setBrushMode(view, "box") },
      { id: "lasso", label: "Lasso", onClick: () => setBrushMode(view, "lasso") },
    ]),
    buildSegment(view, [
      { id: "full", label: "Full", onClick: () => setLayoutMode(view, "full") },
      { id: "lower", label: "Lower", onClick: () => setLayoutMode(view, "lower") },
    ]),
  );
}

function updateToolbar(view) {
  const { buttons } = view;
  for (const id of ["keep", "exclude", "clear", "reset"]) buttons[id].disabled = true;
  buttons.box.classList.toggle("is-active", view.brushMode === "box");
  buttons.lasso.classList.toggle("is-active", view.brushMode === "lasso");
  buttons.full.classList.toggle("is-active", view.layoutMode === "full");
  buttons.lower.classList.toggle("is-active", view.layoutMode === "lower");
}

function renderLegend(view) {
  view.legend.replaceChildren(
    ...view.grouping.labels.map((label, i) => {
      const item = element("span", "kpi-legend-item");
      const dot = element("span", "kpi-legend-dot");
      dot.style.background = PALETTE[i % PALETTE.length];
      item.append(dot, label);
      return item;
    }),
  );
}

function renderStatus(view) {
  view.status.textContent = "drag a box in any cell";
}

function setBrushMode(view, mode) {
  view.brushMode = mode;
  updateToolbar(view);
}

function setLayoutMode(view, mode) {
  view.layoutMode = mode;
  updateToolbar(view);
  renderPoints(view);
}

function recordFrame(view, ms, stages) {
  if (!view.data.dev) return;
  const stats = (window.__kpiStats ??= { frames: [] });
  stats.frames.push(ms);
  stats.last = ms;
  stats.stages = stages;
  if (stats.firstPaint === undefined) {
    stats.firstPaint = performance.now() - view.startedAt;
    stats.firstPaintSinceNavigation = performance.now();
  }
}

// ---------------------------------------------------------------------------
// Widget lifecycle
// ---------------------------------------------------------------------------

function selectGrouping(view, name) {
  view.grouping = view.data.groupings.find((g) => g.name === name) ?? view.data.groupings[0];
  view.groupColors = view.grouping.labels.map((_, i) => parseColor(PALETTE[i % PALETTE.length]));
  view.groupPacked = view.groupColors.map(packColor);
}

function createView(model, el) {
  const meta = model.get("_meta");
  const names = meta.names;
  const view = {
    model,
    startedAt: performance.now(),
    data: {
      rows: meta.rows,
      names,
      dev: meta.dev,
      columns: decodeColumns(model.get("_columns"), meta.rows, names.length),
      groupings: decodeGroupings(model.get("_groups"), meta.rows, meta.groups),
    },
    theme: { cell: parseColor("#F3F3EF"), text: "#1A1A1A", muted: "#8A8A85" },
    brushMode: "box",
    layoutMode: "full",
    buttons: {},
    root: element("div", "kpi-root"),
    legend: element("div", "kpi-legend"),
    status: element("div", "kpi-status"),
    toolbar: element("div", "kpi-toolbar"),
    stage: element("div", "kpi-stage"),
    pointsCanvas: element("canvas"),
    labelsCanvas: element("canvas"),
    overlayCanvas: element("canvas", "kpi-overlay"),
  };
  view.pointsContext = view.pointsCanvas.getContext("2d", { alpha: true });
  view.labelsContext = view.labelsCanvas.getContext("2d");
  view.overlayContext = view.overlayCanvas.getContext("2d");
  view.active = new Uint32Array(meta.rows);
  for (let r = 0; r < meta.rows; r++) view.active[r] = r;
  view.activeCount = meta.rows;

  view.stage.append(view.pointsCanvas, view.labelsCanvas, view.overlayCanvas);
  view.root.append(view.legend, view.status, view.toolbar, view.stage);
  el.append(view.root);
  return view;
}

function renderAll(view) {
  applyLayout(view);
  if (!view.grid) return;
  renderPoints(view);
  renderLabels(view);
}

function render({ model, el }) {
  const view = createView(model, el);
  selectGrouping(view, model.get("color"));
  rescaleAxes(view);
  buildToolbar(view);
  updateToolbar(view);
  renderLegend(view);
  renderStatus(view);
  renderAll(view);

  let lastWidth = view.stage.clientWidth;
  const observer = new ResizeObserver(() => {
    const width = view.stage.clientWidth;
    if (width === lastWidth) return;
    lastWidth = width;
    renderAll(view);
  });
  observer.observe(view.stage);
  return () => observer.disconnect();
}

export default { render };
