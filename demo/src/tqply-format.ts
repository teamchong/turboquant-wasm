/**
 * .tqply binary container format for TurboQuant-compressed 3DGS scenes.
 *
 * Header (32 bytes):
 *   magic:         4 bytes  "TQPL"
 *   version:       1 byte   (1)
 *   sh_degree:     1 byte   (0-2)
 *   num_gaussians: 4 bytes  uint32 LE
 *   tq_seed:       4 bytes  uint32 LE
 *   tq_dim:        2 bytes  uint16 LE
 *   tq_blob_size:  2 bytes  uint16 LE (bytes per compressed SH vector)
 *   non_sh_stride: 2 bytes  uint16 LE (bytes per non-SH vertex data)
 *   reserved:      12 bytes (zeros)
 *
 * Body:
 *   [non_sh_block]  pos(3)+dc(3)+opacity(1)+scale(3)+rot(4) = 14 floats * 4 bytes * N
 *   [tq_sh_block]   tq_blob_size * N bytes of concatenated TQ blobs
 */

export const TQPLY_MAGIC = "TQPL";
export const TQPLY_VERSION = 1;
export const TQPLY_HEADER_SIZE = 32;
export const NON_SH_FLOATS = 14; // pos(3) + dc(3) + opacity(1) + scale(3) + rot(4)
export const NON_SH_STRIDE = NON_SH_FLOATS * 4;

export interface TqplyHeader {
  version: number;
  shDegree: number;
  numGaussians: number;
  tqSeed: number;
  tqDim: number;
  tqBlobSize: number;
  nonShStride: number;
}

export function writeTqplyHeader(header: TqplyHeader): Uint8Array {
  const buf = new ArrayBuffer(TQPLY_HEADER_SIZE);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);

  // Magic
  u8[0] = 0x54; // T
  u8[1] = 0x51; // Q
  u8[2] = 0x50; // P
  u8[3] = 0x4c; // L

  view.setUint8(4, header.version);
  view.setUint8(5, header.shDegree);
  view.setUint32(6, header.numGaussians, true);
  view.setUint32(10, header.tqSeed, true);
  view.setUint16(14, header.tqDim, true);
  view.setUint16(16, header.tqBlobSize, true);
  view.setUint16(18, header.nonShStride, true);
  // bytes 20-31 reserved (zeros)

  return u8;
}

export function readTqplyHeader(buffer: ArrayBuffer): TqplyHeader {
  if (buffer.byteLength < TQPLY_HEADER_SIZE) {
    throw new Error("TQPLY: file too small for header");
  }

  const view = new DataView(buffer);
  const u8 = new Uint8Array(buffer);

  // Check magic
  if (u8[0] !== 0x54 || u8[1] !== 0x51 || u8[2] !== 0x50 || u8[3] !== 0x4c) {
    throw new Error("TQPLY: invalid magic bytes");
  }

  const version = view.getUint8(4);
  if (version !== TQPLY_VERSION) {
    throw new Error(`TQPLY: unsupported version ${version}`);
  }

  return {
    version,
    shDegree: view.getUint8(5),
    numGaussians: view.getUint32(6, true),
    tqSeed: view.getUint32(10, true),
    tqDim: view.getUint16(14, true),
    tqBlobSize: view.getUint16(16, true),
    nonShStride: view.getUint16(18, true),
  };
}

export function writeTqply(
  header: TqplyHeader,
  nonShBlock: Uint8Array,
  tqShBlock: Uint8Array,
): Uint8Array {
  const headerBytes = writeTqplyHeader(header);
  const total = TQPLY_HEADER_SIZE + nonShBlock.byteLength + tqShBlock.byteLength;
  const result = new Uint8Array(total);
  result.set(headerBytes, 0);
  result.set(nonShBlock, TQPLY_HEADER_SIZE);
  result.set(tqShBlock, TQPLY_HEADER_SIZE + nonShBlock.byteLength);
  return result;
}

export function readTqplyBody(
  buffer: ArrayBuffer,
  header: TqplyHeader,
): { nonShBlock: DataView; tqShBlock: Uint8Array } {
  const nonShSize = header.nonShStride * header.numGaussians;
  const tqShSize = header.tqBlobSize * header.numGaussians;
  const expectedSize = TQPLY_HEADER_SIZE + nonShSize + tqShSize;

  if (buffer.byteLength < expectedSize) {
    throw new Error(
      `TQPLY: file too small (${buffer.byteLength} < ${expectedSize})`,
    );
  }

  const nonShBlock = new DataView(buffer, TQPLY_HEADER_SIZE, nonShSize);
  const tqShBlock = new Uint8Array(
    buffer,
    TQPLY_HEADER_SIZE + nonShSize,
    tqShSize,
  );

  return { nonShBlock, tqShBlock };
}
