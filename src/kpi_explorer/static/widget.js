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
const UNSELECTED_ALPHA = 0.35;
const LIVE_POINT_BUDGET = 70_000;
const POLYGON_BUCKETS = 128;
const MIN_DRAG_CSS = 3;
const LASSO_STEP_CSS = 2;
const HINT_DEBOUNCE_MS = 50;
const LIGHT_TOKENS = {
  cellColor: "#F3F3EF",
  unselectedColor: "#C9C9C4",
  text: "#1A1A1A",
  muted: "#8A8A85",
  border: "#E2E2DD",
  activeBg: "#1A1A1A",
  activeText: "#FFFFFF",
  hover: "#F3F3EF",
  brushStroke: "#000000",
  brushFill: "rgba(26, 26, 26, 0.08)",
};

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
  if (view.activeCount === 0 && view.domains) return;
  const { columns } = view.data;
  view.domains = columns.map((column) =>
    paddedDomain(columnExtent(column, view.active, view.activeCount)),
  );
}

function niceTicks(min, max, target) {
  const rough = (max - min) / target;
  const power = 10 ** Math.floor(Math.log10(rough));
  const ratio = rough / power;
  const step = (ratio >= 7 ? 10 : ratio >= 3 ? 5 : ratio >= 1.5 ? 2 : 1) * power;
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
  invalidateLayers(view);
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

function drawScatterCell(view, row, col, rows, count, tables, codes, stride) {
  const { pixels } = view;
  const { width, cellPx } = view.grid;
  const xs = view.pixelCoords[col];
  const ys = view.pixelCoords[row];
  const origin = cellOrigin(view, row, col);
  const top = cellPx - 1;
  const { rb, g, inverse } = tables;
  for (let k = 0; k < count; k += stride) {
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

function histogramPeak(counts, feature, groups) {
  const base = feature * HIST_BINS;
  let peak = 1;
  for (let bin = 0; bin < HIST_BINS; bin++) {
    let total = 0;
    for (let g = 0; g < groups; g++) total += counts[(base + bin) * groups + g];
    if (total > peak) peak = total;
  }
  return peak;
}

function drawHistogramCell(view, feature, counts, colors, peak) {
  const { cellPx, stepPx, leftPx, topPx, dpr } = view.grid;
  const groups = view.grouping.labels.length;
  const x0 = leftPx + feature * stepPx;
  const y0 = topPx + feature * stepPx;
  const usable = cellPx - Math.round(HIST_HEADROOM_CSS * dpr);
  const base = feature * HIST_BINS;

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
          colors[g],
        );
      }
    }
  }
}

function drawMatrix(view, rows, count, tables, codes, stride) {
  const features = view.grid.count;
  for (let row = 1; row < features; row++) {
    for (let col = 0; col < row; col++) {
      drawScatterCell(view, row, col, rows, count, tables, codes, stride);
      if (view.layoutMode === "full") mirrorScatterCell(view, row, col);
    }
  }
}

function storeLayer(view, layer) {
  const out = layer?.length === view.pixels.length ? layer : new Int32Array(view.pixels.length);
  out.set(view.pixels);
  return out;
}

function computeTotals(view) {
  const groups = view.grouping.labels.length;
  view.totals = computeHistograms(view, view.active, view.activeCount);
  view.peaks = view.data.names.map((_, f) => histogramPeak(view.totals, f, groups));
}

function invalidateLayers(view) {
  view.colorLayerValid = false;
  view.greyLayerValid = false;
}

function buildColorLayer(view) {
  clearCells(view);
  const tables = blendTables(view.groupColors, pointAlpha(view.grid, view.activeCount));
  drawMatrix(view, view.active, view.activeCount, tables, view.grouping.codes, 1);
  view.data.names.forEach((_, f) => {
    drawHistogramCell(view, f, view.totals, view.groupPacked, view.peaks[f]);
  });
  view.colorLayer = storeLayer(view, view.colorLayer);
  view.colorLayerValid = true;
}

// Layers the current frame did not need are prepared while the browser is idle, so the
// next brush or Clear does not pay for them.
function scheduleIdleLayers(view) {
  if (view.idlePending) return;
  view.idlePending = true;
  const idle = window.requestIdleCallback ?? ((callback) => setTimeout(callback, 1));
  idle(() => {
    view.idlePending = false;
    if (view.drag) return;
    if (!view.greyLayerValid) buildGreyLayer(view);
    else if (!view.colorLayerValid) buildColorLayer(view);
    if (!view.colorLayerValid || !view.greyLayerValid) scheduleIdleLayers(view);
  });
}

function buildGreyLayer(view) {
  clearCells(view);
  const tables = blendTables([view.theme.unselected], UNSELECTED_ALPHA);
  drawMatrix(view, view.active, view.activeCount, tables, view.zeroCodes, 1);
  const grey = view.grouping.labels.map(() => packColor(view.theme.unselected));
  view.data.names.forEach((_, f) => drawHistogramCell(view, f, view.totals, grey, view.peaks[f]));
  view.greyLayer = storeLayer(view, view.greyLayer);
  view.greyLayerValid = true;
}

// While dragging, large selections are drawn with a stride and a higher alpha so the
// frame stays cheap. Counts and histograms always use every selected row.
function renderFrame(view, interactive) {
  const started = performance.now();
  if (view.selection === null) {
    if (!view.colorLayerValid) buildColorLayer(view);
    view.pixels.set(view.colorLayer);
  } else {
    if (!view.greyLayerValid) buildGreyLayer(view);
    view.pixels.set(view.greyLayer);
    const stride = interactive ? Math.max(1, Math.ceil(view.selectedCount / LIVE_POINT_BUDGET)) : 1;
    const drawn = view.selectedCount / stride;
    const tables = blendTables(view.groupColors, pointAlpha(view.grid, drawn));
    drawMatrix(view, view.selected, view.selectedCount, tables, view.grouping.codes, stride);
    const counts = computeHistograms(view, view.selected, view.selectedCount);
    view.data.names.forEach((_, f) => {
      drawHistogramCell(view, f, counts, view.groupPacked, view.peaks[f]);
    });
  }
  view.pointsContext.putImageData(view.frame, 0, 0);
  recordFrame(view, performance.now() - started);
  if (!view.colorLayerValid || !view.greyLayerValid) scheduleIdleLayers(view);
}

// ---------------------------------------------------------------------------
// Selection geometry
// ---------------------------------------------------------------------------

export function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// Buckets edges by y so each point only tests the few edges near its scanline. Edges that
// do not straddle y never toggle the result, so this matches pointInPolygon exactly.
function buildPolygonIndex(polygon) {
  const ys = polygon.map((vertex) => vertex[1]);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scale = POLYGON_BUCKETS / (maxY - minY || 1);
  const lists = Array.from({ length: POLYGON_BUCKETS }, () => []);
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    const first = Math.min(POLYGON_BUCKETS - 1, Math.floor((Math.min(yi, yj) - minY) * scale));
    const last = Math.min(POLYGON_BUCKETS - 1, Math.floor((Math.max(yi, yj) - minY) * scale));
    for (let b = first; b <= last; b++) lists[b].push(xi, yi, xj, yj);
  }
  return { minY, maxY, scale, buckets: lists.map((list) => Float64Array.from(list)) };
}

function indexedPointInPolygon(index, x, y) {
  if (!(y >= index.minY && y < index.maxY)) return false;
  const bucket = Math.min(POLYGON_BUCKETS - 1, Math.floor((y - index.minY) * index.scale));
  const edges = index.buckets[bucket];
  let inside = false;
  for (let e = 0; e < edges.length; e += 4) {
    const xi = edges[e];
    const yi = edges[e + 1];
    const xj = edges[e + 2];
    const yj = edges[e + 3];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function collectRange(xs, [low, high], active, count, mask, out) {
  let matched = 0;
  for (let k = 0; k < count; k++) {
    const r = active[k];
    const x = xs[r];
    if (x >= low && x <= high) {
      mask[r] = 1;
      out[matched++] = r;
    }
  }
  return matched;
}

function collectBox(xs, ys, [xLow, xHigh, yLow, yHigh], active, count, mask, out) {
  let matched = 0;
  for (let k = 0; k < count; k++) {
    const r = active[k];
    const x = xs[r];
    const y = ys[r];
    if (x >= xLow && x <= xHigh && y >= yLow && y <= yHigh) {
      mask[r] = 1;
      out[matched++] = r;
    }
  }
  return matched;
}

function collectLasso(xs, ys, polygon, active, count, mask, out) {
  const index = buildPolygonIndex(polygon);
  let matched = 0;
  for (let k = 0; k < count; k++) {
    const r = active[k];
    if (indexedPointInPolygon(index, xs[r], ys[r])) {
      mask[r] = 1;
      out[matched++] = r;
    }
  }
  return matched;
}

export function selectRows(selection, columns, active, activeCount, mask, out) {
  mask.fill(0);
  const xs = columns[selection.x_index];
  if (selection.type === "range") {
    return collectRange(xs, selection.bounds, active, activeCount, mask, out);
  }
  const ys = columns[selection.y_index];
  if (selection.type === "box") {
    return collectBox(xs, ys, selection.bounds, active, activeCount, mask, out);
  }
  return collectLasso(xs, ys, selection.polygon, active, activeCount, mask, out);
}

function applyFilterStep(op, mask, active, count) {
  const keepValue = op === "keep" ? 1 : 0;
  let kept = 0;
  for (let k = 0; k < count; k++) {
    const r = active[k];
    if (mask[r] === keepValue) active[kept++] = r;
  }
  return kept;
}

export function replayFilters(filters, columns, active, mask, scratch) {
  let count = active.length;
  for (let r = 0; r < count; r++) active[r] = r;
  for (const { op, selection } of filters) {
    selectRows(selection, columns, active, count, mask, scratch);
    count = applyFilterStep(op, mask, active, count);
  }
  mask.fill(0);
  return count;
}

function applyFilters(view) {
  view.activeCount = replayFilters(
    view.filters, view.data.columns, view.active, view.mask, view.selected,
  );
}

function fractionToData(view, feature, fraction) {
  const [min, max] = view.domains[feature];
  return min + fraction * (max - min);
}

function dataToFraction(view, feature, value) {
  const [min, max] = view.domains[feature];
  return (value - min) / (max - min);
}

function buildSelection(view, drag) {
  const { row, col, kind, start, current, points } = drag;
  const { cell } = view.grid;
  const { names } = view.data;
  const dataX = (x) => fractionToData(view, col, x / cell);
  const dataY = (y) => fractionToData(view, row, 1 - y / cell);
  if (kind === "range") {
    const [a, b] = [dataX(start.x), dataX(current.x)];
    return {
      type: "range", x_index: col, y_index: null, x_name: names[col], y_name: null,
      bounds: [Math.min(a, b), Math.max(a, b)],
    };
  }
  const axes = { x_index: col, y_index: row, x_name: names[col], y_name: names[row] };
  if (kind === "box") {
    const [x0, x1] = [dataX(start.x), dataX(current.x)];
    const [y0, y1] = [dataY(start.y), dataY(current.y)];
    return {
      type: "box", ...axes,
      bounds: [Math.min(x0, x1), Math.max(x0, x1), Math.min(y0, y1), Math.max(y0, y1)],
    };
  }
  if (points.length < 3) return null;
  return { type: "lasso", ...axes, polygon: points.map(([x, y]) => [dataX(x), dataY(y)]) };
}

function applyDragSelection(view, drag) {
  const selection = buildSelection(view, drag);
  if (selection === null) return false;
  view.selection = selection;
  view.selectedCount = selectRows(
    selection, view.data.columns, view.active, view.activeCount, view.mask, view.selected,
  );
  return true;
}

function clearSelection(view) {
  view.selection = null;
  view.selectedCount = 0;
  view.mask.fill(0);
}

// ---------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------

function cellLeft(view, col) {
  return GUTTER.left + col * (view.grid.cell + CELL_GAP);
}

function cellTop(view, row) {
  return GUTTER.top + row * (view.grid.cell + CELL_GAP);
}

function traceSelection(view, ctx, selection) {
  const { cell } = view.grid;
  const col = selection.x_index;
  const row = selection.y_index ?? col;
  const left = cellLeft(view, col);
  const top = cellTop(view, row);
  const toX = (value) => left + dataToFraction(view, col, value) * cell;
  const toY = (value) => top + (1 - dataToFraction(view, row, value)) * cell;
  ctx.beginPath();
  if (selection.type === "range") {
    const [low, high] = selection.bounds;
    ctx.rect(toX(low), top, toX(high) - toX(low), cell);
  } else if (selection.type === "box") {
    const [xLow, xHigh, yLow, yHigh] = selection.bounds;
    ctx.rect(toX(xLow), toY(yHigh), toX(xHigh) - toX(xLow), toY(yLow) - toY(yHigh));
  } else {
    selection.polygon.forEach(([x, y], i) => {
      if (i === 0) ctx.moveTo(toX(x), toY(y));
      else ctx.lineTo(toX(x), toY(y));
    });
    ctx.closePath();
  }
}

function drawSelectionOutline(view, ctx, selection) {
  const { cell } = view.grid;
  const col = selection.x_index;
  const row = selection.y_index ?? col;
  ctx.save();
  ctx.beginPath();
  ctx.rect(cellLeft(view, col), cellTop(view, row), cell, cell);
  ctx.clip();
  traceSelection(view, ctx, selection);
  ctx.fillStyle = view.theme.brushFill;
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.strokeStyle = view.theme.brushStroke;
  ctx.stroke();
  ctx.restore();
}

function drawCrosshair(view, ctx, hover) {
  const { cell } = view.grid;
  const left = cellLeft(view, hover.col);
  const top = cellTop(view, hover.row);
  ctx.save();
  ctx.strokeStyle = view.theme.muted;
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = 1;
  ctx.beginPath();
  const x = Math.round(left + hover.x) + 0.5;
  ctx.moveTo(x, top);
  ctx.lineTo(x, top + cell);
  if (hover.row !== hover.col) {
    const y = Math.round(top + hover.y) + 0.5;
    ctx.moveTo(left, y);
    ctx.lineTo(left + cell, y);
  }
  ctx.stroke();
  ctx.restore();
}

function renderOverlay(view) {
  const { overlayContext: ctx, grid } = view;
  ctx.setTransform(grid.dpr, 0, 0, grid.dpr, 0, 0);
  ctx.clearRect(0, 0, grid.cssWidth, grid.cssHeight);
  if (view.selection) drawSelectionOutline(view, ctx, view.selection);
  if (!view.drag && view.hover) drawCrosshair(view, ctx, view.hover);
}

// ---------------------------------------------------------------------------
// Pointer interaction
// ---------------------------------------------------------------------------

function locateCell(view, event) {
  const { grid } = view;
  if (!grid) return null;
  const rect = view.overlayCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left - GUTTER.left;
  const y = event.clientY - rect.top - GUTTER.top;
  const step = grid.cell + CELL_GAP;
  const col = Math.floor(x / step);
  const row = Math.floor(y / step);
  if (col < 0 || row < 0 || col >= grid.count || row >= grid.count) return null;
  if (!isCellVisible(view, row, col)) return null;
  const cellX = x - col * step;
  const cellY = y - row * step;
  if (cellX >= grid.cell || cellY >= grid.cell) return null;
  return { row, col, x: cellX, y: cellY };
}

function clampToDragCell(view, drag, event) {
  const rect = view.overlayCanvas.getBoundingClientRect();
  const { cell } = view.grid;
  const x = event.clientX - rect.left - cellLeft(view, drag.col);
  const y = event.clientY - rect.top - cellTop(view, drag.row);
  return { x: Math.max(0, Math.min(cell, x)), y: Math.max(0, Math.min(cell, y)) };
}

function scheduleFrame(view) {
  if (view.frameId) return;
  view.frameId = requestAnimationFrame(() => {
    view.frameId = 0;
    flushFrame(view);
  });
}

function flushFrame(view) {
  const { drag } = view;
  if (drag?.dirty && drag.moved) {
    const started = performance.now();
    drag.dirty = false;
    applyDragSelection(view, drag);
    renderFrame(view, true);
    renderOverlay(view);
    renderStatus(view);
    recordBrushFrame(view, performance.now() - started);
    return;
  }
  renderOverlay(view);
}

function scheduleHint(view) {
  clearTimeout(view.hintTimer);
  view.hintTimer = setTimeout(() => renderStatus(view), HINT_DEBOUNCE_MS);
}

function onPointerDown(view, event) {
  if (event.button !== 0) return;
  const hit = locateCell(view, event);
  if (!hit) return;
  event.preventDefault();
  view.root.focus({ preventScroll: true });
  view.overlayCanvas.setPointerCapture(event.pointerId);
  const start = { x: hit.x, y: hit.y };
  view.drag = {
    row: hit.row,
    col: hit.col,
    kind: hit.row === hit.col ? "range" : view.brushMode,
    start,
    current: start,
    points: [[start.x, start.y]],
    moved: false,
    dirty: false,
  };
  scheduleFrame(view);
}

function onPointerMove(view, event) {
  const { drag } = view;
  if (!drag) {
    const previous = view.hover;
    view.hover = locateCell(view, event);
    if (previous?.row !== view.hover?.row || previous?.col !== view.hover?.col) scheduleHint(view);
    scheduleFrame(view);
    return;
  }
  const point = clampToDragCell(view, drag, event);
  drag.current = point;
  if (Math.hypot(point.x - drag.start.x, point.y - drag.start.y) >= MIN_DRAG_CSS) {
    drag.moved = true;
  }
  if (drag.kind === "lasso") {
    const [lastX, lastY] = drag.points[drag.points.length - 1];
    if (Math.hypot(point.x - lastX, point.y - lastY) >= LASSO_STEP_CSS) {
      drag.points.push([point.x, point.y]);
    }
  }
  drag.dirty = true;
  scheduleFrame(view);
}

function onPointerUp(view, event) {
  const { drag } = view;
  if (!drag) return;
  view.drag = null;
  if (!drag.moved || !applyDragSelection(view, drag)) clearSelection(view);
  view.hover = locateCell(view, event);
  renderFrame(view, false);
  renderOverlay(view);
  renderStatus(view);
  updateToolbar(view);
  syncToPython(view);
  logBrushStats(view);
}

function onPointerLeave(view) {
  if (view.drag) return;
  view.hover = null;
  scheduleHint(view);
  scheduleFrame(view);
}

function cancelDrag(view) {
  view.drag = null;
  dropSelection(view);
  renderOverlay(view);
}

const SHORTCUTS = {
  b: (view) => setBrushMode(view, "box"),
  l: (view) => setBrushMode(view, "lasso"),
  k: (view) => pushFilter(view, "keep"),
  x: (view) => pushFilter(view, "exclude"),
  r: (view) => resetFilters(view),
  escape: (view) => dropSelection(view),
};

function onKeyDown(view, event) {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.target.tagName === "SELECT") return;
  const key = event.key.toLowerCase();
  const action = SHORTCUTS[key];
  if (!action) return;
  event.preventDefault();
  event.stopPropagation();
  if (view.drag) {
    if (key === "escape") cancelDrag(view);
    return;
  }
  action(view);
}

function attachPointerEvents(view) {
  const canvas = view.overlayCanvas;
  canvas.addEventListener("pointerdown", (event) => onPointerDown(view, event));
  canvas.addEventListener("pointermove", (event) => onPointerMove(view, event));
  canvas.addEventListener("pointerup", (event) => onPointerUp(view, event));
  canvas.addEventListener("pointercancel", (event) => onPointerUp(view, event));
  canvas.addEventListener("pointerleave", () => onPointerLeave(view));
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
  for (const { id, label, title, onClick } of entries) {
    const button = element("button", "kpi-btn", label);
    button.type = "button";
    if (title) button.title = title;
    button.addEventListener("click", () => {
      onClick();
      // A disabled button drops focus, which would stop the keyboard shortcuts working.
      if (button.disabled) view.root.focus({ preventScroll: true });
    });
    view.buttons[id] = button;
    segment.append(button);
  }
  return segment;
}

function buildToolbar(view) {
  view.toolbar.append(
    buildSegment(view, [
      {
        id: "keep",
        label: "Keep",
        title: "Keep only the selected rows (K)",
        onClick: () => pushFilter(view, "keep"),
      },
      {
        id: "exclude",
        label: "Exclude",
        title: "Remove the selected rows (X)",
        onClick: () => pushFilter(view, "exclude"),
      },
      {
        id: "clear",
        label: "Clear",
        title: "Clear the selection (Esc)",
        onClick: () => dropSelection(view),
      },
    ]),
    buildSegment(view, [
      {
        id: "reset",
        label: "Reset",
        title: "Remove all filters (R)",
        onClick: () => resetFilters(view),
      },
    ]),
    buildSegment(view, [
      { id: "box", label: "Box", title: "Box select (B)", onClick: () => setBrushMode(view, "box") },
      {
        id: "lasso",
        label: "Lasso",
        title: "Lasso select (L)",
        onClick: () => setBrushMode(view, "lasso"),
      },
    ]),
    buildSegment(view, [
      {
        id: "full",
        label: "Full",
        title: "Show the full matrix",
        onClick: () => setLayoutMode(view, "full"),
      },
      {
        id: "lower",
        label: "Lower",
        title: "Show the lower triangle",
        onClick: () => setLayoutMode(view, "lower"),
      },
    ]),
    buildColorSelect(view),
  );
}

function buildColorSelect(view) {
  const wrapper = element("label", "kpi-color", "Colour by");
  const select = element("select", "kpi-select");
  for (const { name } of view.data.groupings) {
    const option = element("option", "", name);
    option.value = name;
    select.append(option);
  }
  select.value = view.grouping.name;
  select.addEventListener("change", () => setColorBy(view, select.value));
  wrapper.append(select);
  return wrapper;
}

function updateToolbar(view) {
  const { buttons } = view;
  for (const id of ["keep", "exclude", "clear"]) buttons[id].disabled = view.selection === null;
  buttons.reset.disabled = view.filters.length === 0;
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

function hintText(view) {
  const verb = view.brushMode === "lasso" ? "lasso" : "box";
  if (!view.hover) return `drag a ${verb} in any cell`;
  const { row, col } = view.hover;
  const { names } = view.data;
  if (row === col) return `${names[col]} · drag a range`;
  const action = view.brushMode === "lasso" ? "drag to lasso" : "drag a box";
  return `${names[col]} × ${names[row]} · ${action}`;
}

function renderStatus(view) {
  const total = view.data.rows;
  const filterCount = view.filters.length;
  const filterText = filterCount ? ` · ${filterCount} filter${filterCount === 1 ? "" : "s"}` : "";
  let count = null;
  let suffix = hintText(view);
  if (view.selection) {
    count = view.selectedCount;
    const share = Math.round((count / total) * 100);
    suffix = ` of ${total.toLocaleString("en-US")} rows · ${share}%${filterText}`;
  } else if (filterCount) {
    count = view.activeCount;
    suffix = ` of ${total.toLocaleString("en-US")} rows${filterText} · ${suffix}`;
  }
  const key = `${count}|${suffix}`;
  if (key === view.statusKey) return;
  view.statusKey = key;
  if (count === null) {
    view.status.textContent = suffix;
  } else {
    view.status.replaceChildren(element("strong", "", count.toLocaleString("en-US")), suffix);
  }
}

function syncToPython(view) {
  view.model.set("selection", view.selection);
  view.model.set("filters", view.filters);
  view.model.save_changes();
}

function refreshAfterFilterChange(view) {
  const started = performance.now();
  applyFilters(view);
  rescaleAxes(view);
  computePixelCoords(view);
  computeBins(view);
  computeTotals(view);
  invalidateLayers(view);
  renderStatus(view);
  updateToolbar(view);
  updateEmptyState(view);
  recordRebuild(view, "filter", performance.now() - started);
  syncToPython(view);
  scheduleRepaint(view);
}

// Axes and counts are prepared in one frame and points are painted in the next, so
// neither frame exceeds the budget. Labels wait so they change together with the points.
function scheduleRepaint(view) {
  cancelAnimationFrame(view.repaintId);
  view.repaintId = requestAnimationFrame(() => {
    const started = performance.now();
    renderFrame(view, false);
    renderLabels(view);
    renderOverlay(view);
    recordRebuild(view, "repaint", performance.now() - started);
  });
}

function updateEmptyState(view) {
  view.empty.hidden = view.activeCount > 0;
}

function setColorBy(view, name) {
  const started = performance.now();
  selectGrouping(view, name);
  computeTotals(view);
  view.colorLayerValid = false;
  renderLegend(view);
  renderFrame(view, false);
  recordRebuild(view, "color", performance.now() - started);
  view.model.set("color", name);
  view.model.save_changes();
}

function pushFilter(view, op) {
  if (view.selection === null) return;
  view.filters = [...view.filters, { op, selection: view.selection }];
  clearSelection(view);
  refreshAfterFilterChange(view);
}

function resetFilters(view) {
  if (view.filters.length === 0) return;
  view.filters = [];
  clearSelection(view);
  refreshAfterFilterChange(view);
}

function dropSelection(view) {
  if (view.selection === null) return;
  clearSelection(view);
  renderFrame(view, false);
  renderOverlay(view);
  renderStatus(view);
  updateToolbar(view);
  syncToPython(view);
}

function restoreSelection(view) {
  const selection = view.model.get("selection");
  if (!selection) return;
  view.selection = selection;
  view.selectedCount = selectRows(
    selection, view.data.columns, view.active, view.activeCount, view.mask, view.selected,
  );
}

function setBrushMode(view, mode) {
  view.brushMode = mode;
  updateToolbar(view);
  renderStatus(view);
}

function setLayoutMode(view, mode) {
  const started = performance.now();
  view.layoutMode = mode;
  view.hover = null;
  updateToolbar(view);
  invalidateLayers(view);
  renderFrame(view, false);
  renderOverlay(view);
  renderStatus(view);
  recordRebuild(view, "layout", performance.now() - started);
}

function recordRebuild(view, kind, ms) {
  if (!view.data.dev) return;
  const stats = window.__kpiStats;
  (stats.rebuilds ??= []).push({ kind, ms });
}

function themeKey() {
  return `${document.body?.dataset.theme}|${document.body?.className}`;
}

function cssColorToRgb(value) {
  const context = (cssColorToRgb.context ??= document
    .createElement("canvas")
    .getContext("2d", { willReadFrequently: true }));
  context.fillStyle = "#000000";
  context.fillStyle = value;
  context.fillRect(0, 0, 1, 1);
  const [r, g, b] = context.getImageData(0, 0, 1, 1).data;
  return { r, g, b };
}

function readTokens(root) {
  const body = document.body;
  if (body?.dataset.theme !== "dark" && !body?.classList.contains("dark")) return LIGHT_TOKENS;
  const style = getComputedStyle(root);
  const read = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
  const background = read("--background", "#181C1A");
  const foreground = read("--foreground", "#ECEEED");
  return {
    cellColor: read("--card", "#252927"),
    unselectedColor: "#5A605D",
    text: foreground,
    muted: read("--muted-foreground", "#AAB2AF"),
    border: read("--border", "#3B403E"),
    activeBg: foreground,
    activeText: background,
    hover: read("--card", "#252927"),
    brushStroke: foreground,
    brushFill: "rgba(255, 255, 255, 0.10)",
  };
}

function applyTheme(view) {
  const tokens = readTokens(view.root);
  view.themeKey = themeKey();
  view.theme = {
    ...tokens,
    cell: cssColorToRgb(tokens.cellColor),
    unselected: cssColorToRgb(tokens.unselectedColor),
  };
  const { style } = view.root;
  style.setProperty("--kpi-text", tokens.text);
  style.setProperty("--kpi-muted", tokens.muted);
  style.setProperty("--kpi-border", tokens.border);
  style.setProperty("--kpi-cell", tokens.cellColor);
  style.setProperty("--kpi-active-bg", tokens.activeBg);
  style.setProperty("--kpi-active-text", tokens.activeText);
  style.setProperty("--kpi-hover", tokens.hover);
}

function watchTheme(view) {
  const observer = new MutationObserver(() => {
    if (themeKey() !== view.themeKey) renderAll(view);
  });
  observer.observe(document.body, { attributes: true, attributeFilter: ["class", "data-theme"] });
  return observer;
}

function recordFrame(view, ms) {
  if (!view.data.dev) return;
  const stats = (window.__kpiStats ??= { frames: [], brush: [] });
  window.__kpiView = view;
  stats.frames.push(ms);
  stats.last = ms;
  if (stats.firstPaint === undefined) {
    stats.firstPaint = performance.now() - view.startedAt;
    stats.firstPaintSinceNavigation = performance.now();
  }
}

function recordBrushFrame(view, ms) {
  if (view.data.dev) window.__kpiStats.brush.push(ms);
}

function logBrushStats(view) {
  if (!view.data.dev) return;
  const frames = [...window.__kpiStats.brush].sort((a, b) => a - b);
  if (!frames.length) return;
  const median = frames[Math.floor(frames.length / 2)];
  console.debug(
    `[kpi] brush frames: ${frames.length}, median ${median.toFixed(1)} ms, ` +
      `max ${frames[frames.length - 1].toFixed(1)} ms`,
  );
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
    brushMode: "box",
    layoutMode: "full",
    selection: null,
    selectedCount: 0,
    filters: model.get("filters") ?? [],
    drag: null,
    hover: null,
    frameId: 0,
    buttons: {},
    root: element("div", "kpi-root"),
    legend: element("div", "kpi-legend"),
    status: element("div", "kpi-status"),
    toolbar: element("div", "kpi-toolbar"),
    stage: element("div", "kpi-stage"),
    pointsCanvas: element("canvas"),
    labelsCanvas: element("canvas"),
    overlayCanvas: element("canvas", "kpi-overlay"),
    empty: element("div", "kpi-empty"),
  };
  view.empty.append(
    element("span", "kpi-empty-message", "No rows match these filters. Press Reset to start over."),
  );
  view.root.tabIndex = 0;
  view.pointsContext = view.pointsCanvas.getContext("2d", { alpha: true });
  view.labelsContext = view.labelsCanvas.getContext("2d");
  view.overlayContext = view.overlayCanvas.getContext("2d");
  view.active = new Uint32Array(meta.rows);
  for (let r = 0; r < meta.rows; r++) view.active[r] = r;
  view.activeCount = meta.rows;
  view.mask = new Uint8Array(meta.rows);
  view.selected = new Uint32Array(meta.rows);
  view.zeroCodes = new Uint8Array(meta.rows);

  view.stage.append(view.pointsCanvas, view.labelsCanvas, view.overlayCanvas, view.empty);
  view.root.append(view.legend, view.status, view.toolbar, view.stage);
  el.append(view.root);
  return view;
}

function renderAll(view) {
  applyTheme(view);
  applyLayout(view);
  if (!view.grid) return;
  computeTotals(view);
  renderFrame(view, false);
  renderLabels(view);
  renderOverlay(view);
}

function render({ model, el }) {
  const view = createView(model, el);
  selectGrouping(view, model.get("color"));
  applyFilters(view);
  rescaleAxes(view);
  restoreSelection(view);
  buildToolbar(view);
  updateToolbar(view);
  renderLegend(view);
  renderStatus(view);
  renderAll(view);
  updateEmptyState(view);
  attachPointerEvents(view);
  view.root.addEventListener("keydown", (event) => onKeyDown(view, event));
  const themeObserver = watchTheme(view);

  let lastWidth = view.stage.clientWidth;
  const observer = new ResizeObserver(() => {
    const width = view.stage.clientWidth;
    if (width === lastWidth) return;
    lastWidth = width;
    renderAll(view);
  });
  observer.observe(view.stage);
  return () => {
    observer.disconnect();
    themeObserver.disconnect();
    cancelAnimationFrame(view.frameId);
    cancelAnimationFrame(view.repaintId);
    clearTimeout(view.hintTimer);
  };
}

export default { render };
