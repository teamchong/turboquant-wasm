/**
 * Reconstruct a valid INRIA v1 PLY binary from parsed Gaussian data.
 * GaussianSplats3D expects specific property names and order.
 */

import type { ParsedGaussians } from "./ply-parser.js";

const TOTAL_SH_REST = 45; // degree 3 = 45 rest coefficients (we zero-pad unused)

export function buildPlyBinary(data: ParsedGaussians): Uint8Array {
  const n = data.numGaussians;

  // Build header
  const props = [
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    ...Array.from({ length: TOTAL_SH_REST }, (_, i) => `f_rest_${i}`),
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
  ];

  const headerLines = [
    "ply",
    "format binary_little_endian 1.0",
    `element vertex ${n}`,
    ...props.map((p) => `property float ${p}`),
    "end_header",
    "", // trailing newline
  ];
  const headerStr = headerLines.join("\n");
  const headerBytes = new TextEncoder().encode(headerStr);

  const floatsPerVertex = props.length; // 62
  const vertexBytes = floatsPerVertex * 4;
  const totalSize = headerBytes.length + n * vertexBytes;

  const result = new Uint8Array(totalSize);
  result.set(headerBytes, 0);

  const view = new DataView(result.buffer, headerBytes.length);

  for (let i = 0; i < n; i++) {
    const base = i * vertexBytes;
    let off = 0;

    function writeFloat(val: number) {
      view.setFloat32(base + off, val, true);
      off += 4;
    }

    // position
    writeFloat(data.positions[i * 3 + 0]);
    writeFloat(data.positions[i * 3 + 1]);
    writeFloat(data.positions[i * 3 + 2]);

    // normals (zeros)
    writeFloat(0);
    writeFloat(0);
    writeFloat(0);

    // f_dc
    writeFloat(data.shDC[i * 3 + 0]);
    writeFloat(data.shDC[i * 3 + 1]);
    writeFloat(data.shDC[i * 3 + 2]);

    // f_rest (pad with zeros beyond available data)
    for (let j = 0; j < TOTAL_SH_REST; j++) {
      if (j < data.shRestCount) {
        writeFloat(data.shRest[i * data.shRestCount + j]);
      } else {
        writeFloat(0);
      }
    }

    // opacity
    writeFloat(data.opacity[i]);

    // scale
    writeFloat(data.scales[i * 3 + 0]);
    writeFloat(data.scales[i * 3 + 1]);
    writeFloat(data.scales[i * 3 + 2]);

    // rotation
    writeFloat(data.rotations[i * 4 + 0]);
    writeFloat(data.rotations[i * 4 + 1]);
    writeFloat(data.rotations[i * 4 + 2]);
    writeFloat(data.rotations[i * 4 + 3]);
  }

  return result;
}
