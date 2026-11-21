"""Shared fixtures, including a bridge that runs the real widget.js under Node."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

WIDGET_JS = Path(__file__).parent.parent / "src" / "kpi_explorer" / "static" / "widget.js"

DRIVER = """
import { readFileSync, writeFileSync } from "node:fs";
import { pointInPolygon, replayFilters, selectRows } from "./widget.mjs";

const spec = JSON.parse(readFileSync(process.argv[2], "utf8"));
const raw = readFileSync(spec.matrix);
const flat = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const columns = Array.from({ length: spec.features }, (_, i) =>
  flat.subarray(i * spec.rows, (i + 1) * spec.rows),
);
const everyRow = Uint32Array.from({ length: spec.rows }, (_, i) => i);
const scratch = new Uint32Array(spec.rows);
const masks = [];

for (const selection of spec.selections) {
  const mask = new Uint8Array(spec.rows);
  selectRows(selection, columns, everyRow, spec.rows, mask, scratch);
  if (selection.type === "lasso") {
    const xs = columns[selection.x_index];
    const ys = columns[selection.y_index];
    for (let r = 0; r < spec.rows; r++) {
      if (pointInPolygon(xs[r], ys[r], selection.polygon) !== (mask[r] === 1)) {
        throw new Error(`indexed lasso disagrees with pointInPolygon at row ${r}`);
      }
    }
  }
  masks.push(mask);
}

for (const filters of spec.stacks) {
  const active = new Uint32Array(spec.rows);
  const mask = new Uint8Array(spec.rows);
  const count = replayFilters(filters, columns, active, mask, scratch);
  for (let k = 0; k < count; k++) mask[active[k]] = 1;
  masks.push(mask);
}

writeFileSync(spec.output, Buffer.concat(masks.map((mask) => Buffer.from(mask))));
"""


class JsBridge:
    """Evaluate selections and filter stacks with the browser code."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        shutil.copy(WIDGET_JS, directory / "widget.mjs")
        (directory / "driver.mjs").write_text(DRIVER)

    def evaluate(
        self,
        matrix: np.ndarray,
        selections: Sequence[dict] = (),
        stacks: Sequence[Sequence[dict]] = (),
    ) -> list[np.ndarray]:
        """Return one boolean mask per selection, then one per filter stack."""
        rows, features = matrix.shape
        matrix_path = self.directory / "matrix.bin"
        output_path = self.directory / "masks.bin"
        spec_path = self.directory / "spec.json"
        matrix_path.write_bytes(matrix.astype(np.float32).tobytes(order="F"))
        spec_path.write_text(
            json.dumps(
                {
                    "rows": rows,
                    "features": features,
                    "matrix": str(matrix_path),
                    "output": str(output_path),
                    "selections": list(selections),
                    "stacks": [list(stack) for stack in stacks],
                }
            )
        )
        subprocess.run(
            ["node", "driver.mjs", str(spec_path)],
            cwd=self.directory,
            check=True,
            capture_output=True,
            text=True,
        )
        flat = np.frombuffer(output_path.read_bytes(), dtype=np.uint8).astype(bool)
        return list(flat.reshape(len(selections) + len(stacks), rows))


@pytest.fixture
def js_bridge(tmp_path: Path) -> JsBridge:
    """Return a bridge to widget.js, skipping when Node is unavailable."""
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JS/Python parity tests")
    return JsBridge(tmp_path)
