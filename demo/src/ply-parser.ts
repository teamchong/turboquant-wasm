/**
 * Parse INRIA v1 binary PLY files from 3D Gaussian Splatting training output.
 *
 * Expected properties per vertex (62 floats = 248 bytes):
 *   x, y, z, nx, ny, nz,
 *   f_dc_0, f_dc_1, f_dc_2,
 *   f_rest_0 .. f_rest_44,
 *   opacity,
 *   scale_0, scale_1, scale_2,
 *   rot_0, rot_1, rot_2, rot_3
 */

export interface PlyData {
  numVertices: number;
  /** Property names in order as declared in the PLY header. */
  properties: string[];
  /** byte offset of each property within a vertex record */
  offsets: Map<string, number>;
  /** bytes per vertex */
  vertexStride: number;
  /** raw vertex data (binary, little-endian float32) */
  vertexData: DataView;
}

export interface ParsedGaussians {
  numGaussians: number;
  /** xyz positions, interleaved [x0,y0,z0, x1,y1,z1, ...] */
  positions: Float32Array;
  /** SH DC coefficients (3 per Gaussian) */
  shDC: Float32Array;
  /** SH rest coefficients (shRestCount per Gaussian, up to 45) */
  shRest: Float32Array;
  shRestCount: number;
  /** log-space opacity */
  opacity: Float32Array;
  /** log-space scale */
  scales: Float32Array;
  /** rotation quaternion */
  rotations: Float32Array;
  /** original file size in bytes */
  originalFileSize: number;
}

export function parsePlyHeader(buffer: ArrayBuffer): PlyData {
  const bytes = new Uint8Array(buffer);
  // Find end_header
  let headerEnd = -1;
  const decoder = new TextDecoder("ascii");
  for (let i = 0; i < Math.min(bytes.length, 8192); i++) {
    if (
      bytes[i] === 0x65 && // 'e'
      decoder.decode(bytes.slice(i, i + 10)) === "end_header"
    ) {
      headerEnd = i + 10;
      // skip the newline after end_header
      while (headerEnd < bytes.length && bytes[headerEnd] === 0x0a)
        headerEnd++;
      if (headerEnd < bytes.length && bytes[headerEnd - 1] !== 0x0a) {
        // no newline was found after end_header text, skip one byte for \n
      }
      break;
    }
  }
  if (headerEnd === -1) throw new Error("PLY: could not find end_header");

  // Ensure we land right after the \n following "end_header\n"
  const headerText = decoder.decode(bytes.slice(0, headerEnd));
  const lines = headerText.split("\n").map((l) => l.trim());

  if (!lines[0].startsWith("ply")) throw new Error("PLY: not a PLY file");

  let format = "";
  let numVertices = 0;
  const properties: string[] = [];

  for (const line of lines) {
    if (line.startsWith("format ")) {
      format = line;
    } else if (line.startsWith("element vertex ")) {
      numVertices = parseInt(line.split(" ")[2], 10);
    } else if (line.startsWith("property float ")) {
      properties.push(line.split(" ")[2]);
    }
  }

  if (!format.includes("binary_little_endian"))
    throw new Error(`PLY: unsupported format: ${format}`);
  if (numVertices === 0) throw new Error("PLY: no vertices found");

  const vertexStride = properties.length * 4; // all float32
  const offsets = new Map<string, number>();
  for (let i = 0; i < properties.length; i++) {
    offsets.set(properties[i], i * 4);
  }

  const vertexData = new DataView(
    buffer,
    headerEnd,
    numVertices * vertexStride,
  );

  return { numVertices, properties, offsets, vertexStride, vertexData };
}

export function extractGaussians(
  ply: PlyData,
  maxShDegree: number = 2,
): ParsedGaussians {
  const n = ply.numVertices;
  const { offsets, vertexStride, vertexData } = ply;

  // Determine available SH rest count
  let shRestCount = 0;
  for (let i = 0; i < 45; i++) {
    if (offsets.has(`f_rest_${i}`)) shRestCount = i + 1;
    else break;
  }

  // Clamp to requested SH degree
  // degree 1 = 9 rest, degree 2 = 24 rest, degree 3 = 45 rest
  const maxRestForDegree = [0, 9, 24, 45][maxShDegree] ?? 45;
  shRestCount = Math.min(shRestCount, maxRestForDegree);

  const positions = new Float32Array(n * 3);
  const shDC = new Float32Array(n * 3);
  const shRest = new Float32Array(n * shRestCount);
  const opacity = new Float32Array(n);
  const scales = new Float32Array(n * 3);
  const rotations = new Float32Array(n * 4);

  function getFloat(vertexIdx: number, prop: string): number {
    const off = offsets.get(prop);
    if (off === undefined) return 0;
    return vertexData.getFloat32(vertexIdx * vertexStride + off, true);
  }

  for (let i = 0; i < n; i++) {
    positions[i * 3 + 0] = getFloat(i, "x");
    positions[i * 3 + 1] = getFloat(i, "y");
    positions[i * 3 + 2] = getFloat(i, "z");

    shDC[i * 3 + 0] = getFloat(i, "f_dc_0");
    shDC[i * 3 + 1] = getFloat(i, "f_dc_1");
    shDC[i * 3 + 2] = getFloat(i, "f_dc_2");

    for (let j = 0; j < shRestCount; j++) {
      shRest[i * shRestCount + j] = getFloat(i, `f_rest_${j}`);
    }

    opacity[i] = getFloat(i, "opacity");

    scales[i * 3 + 0] = getFloat(i, "scale_0");
    scales[i * 3 + 1] = getFloat(i, "scale_1");
    scales[i * 3 + 2] = getFloat(i, "scale_2");

    rotations[i * 4 + 0] = getFloat(i, "rot_0");
    rotations[i * 4 + 1] = getFloat(i, "rot_1");
    rotations[i * 4 + 2] = getFloat(i, "rot_2");
    rotations[i * 4 + 3] = getFloat(i, "rot_3");
  }

  return {
    numGaussians: n,
    positions,
    shDC,
    shRest,
    shRestCount,
    opacity,
    scales,
    rotations,
    originalFileSize: ply.vertexData.byteLength + 512, // approximate header
  };
}
