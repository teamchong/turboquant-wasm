/**
 * Multi-branch system-cache container.
 *
 * Wraps N independent TQKV KV-cache blobs (each produced by
 * Engine.dumpCache) into one file keyed by branch name. The runtime
 * picks which branch to load as the initial cache (the "router") and
 * keeps references to the others for later mountKV() swaps.
 *
 * Format (little-endian):
 *
 *   magic          4B  ASCII "TQKC"  (0x434B5154)
 *   version        2B  u16, starts at 1
 *   branch_count   2B  u16
 *   for each branch:
 *     name_len     1B  u8 (max 32)
 *     name         name_len bytes, UTF-8
 *     offset       4B  u32, bytes from file start to the branch's TQKV blob
 *     length       4B  u32, bytes occupied by the branch's TQKV blob
 *     token_count  4B  u32, convenience copy (also in the branch's TQKV header)
 *   for each branch (in index order, blobs are concatenated after the index):
 *     TQKV blob    length bytes, exactly as Engine.dumpCache() produced it
 *
 * Single-branch TQKV files (today's system-cache.bin) stay valid on
 * their own — the runtime detects magic TQKV vs TQKC and picks the
 * loader path.
 */

export const CONTAINER_MAGIC = 0x434b5154; // "TQKC" little-endian
export const CONTAINER_VERSION = 1;
export const MAX_BRANCH_NAME_LEN = 32;

const SINGLE_BLOB_MAGIC = 0x564b5154; // "TQKV"

export interface BranchEntry {
  name: string;
  blob: Uint8Array;
  tokenCount: number;
}

export interface ParsedContainer {
  version: number;
  branches: Map<string, { blob: Uint8Array; tokenCount: number }>;
}

/** Pack a set of per-branch TQKV dumps into one container. */
export function packContainer(entries: BranchEntry[]): Uint8Array {
  if (entries.length === 0) throw new Error("packContainer: need at least one branch");
  if (entries.length > 0xffff) throw new Error(`packContainer: too many branches (${entries.length})`);

  const enc = new TextEncoder();
  const nameBytes: Uint8Array[] = [];
  let indexSize = 0;
  for (const e of entries) {
    const bytes = enc.encode(e.name);
    if (bytes.length === 0) throw new Error("packContainer: branch name cannot be empty");
    if (bytes.length > MAX_BRANCH_NAME_LEN) {
      throw new Error(`packContainer: branch name "${e.name}" exceeds ${MAX_BRANCH_NAME_LEN} bytes`);
    }
    nameBytes.push(bytes);
    indexSize += 1 + bytes.length + 4 + 4 + 4;
  }

  const headerSize = 4 + 2 + 2; // magic + version + count
  const firstBlobOffset = headerSize + indexSize;
  let total = firstBlobOffset;
  for (const e of entries) total += e.blob.byteLength;

  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, CONTAINER_MAGIC, true);
  dv.setUint16(4, CONTAINER_VERSION, true);
  dv.setUint16(6, entries.length, true);

  let indexCursor = headerSize;
  let blobCursor = firstBlobOffset;
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    const nb = nameBytes[i];
    dv.setUint8(indexCursor, nb.length);
    indexCursor += 1;
    out.set(nb, indexCursor);
    indexCursor += nb.length;
    dv.setUint32(indexCursor, blobCursor, true);       // offset
    indexCursor += 4;
    dv.setUint32(indexCursor, e.blob.byteLength, true); // length
    indexCursor += 4;
    dv.setUint32(indexCursor, e.tokenCount, true);      // token_count
    indexCursor += 4;

    out.set(e.blob, blobCursor);
    blobCursor += e.blob.byteLength;
  }

  if (indexCursor !== firstBlobOffset) {
    throw new Error(`packContainer: index size mismatch (${indexCursor} vs ${firstBlobOffset})`);
  }
  if (blobCursor !== total) {
    throw new Error(`packContainer: blob size mismatch (${blobCursor} vs ${total})`);
  }
  return out;
}

/** Parse a container (magic "TQKC"). Throws if magic/version is wrong. */
export function unpackContainer(buffer: ArrayBuffer): ParsedContainer {
  const dv = new DataView(buffer);
  if (buffer.byteLength < 8) throw new Error("unpackContainer: buffer too small for header");
  const magic = dv.getUint32(0, true);
  if (magic !== CONTAINER_MAGIC) {
    throw new Error(`unpackContainer: bad magic 0x${magic.toString(16)} (expected 0x${CONTAINER_MAGIC.toString(16)} "TQKC")`);
  }
  const version = dv.getUint16(4, true);
  if (version !== CONTAINER_VERSION) {
    throw new Error(`unpackContainer: unsupported version ${version} (max ${CONTAINER_VERSION})`);
  }
  const count = dv.getUint16(6, true);

  const dec = new TextDecoder();
  const branches = new Map<string, { blob: Uint8Array; tokenCount: number }>();
  let cursor = 8;
  for (let i = 0; i < count; i++) {
    if (cursor >= buffer.byteLength) throw new Error("unpackContainer: truncated index");
    const nameLen = dv.getUint8(cursor);
    cursor += 1;
    if (nameLen === 0 || nameLen > MAX_BRANCH_NAME_LEN) {
      throw new Error(`unpackContainer: invalid name length ${nameLen} at entry ${i}`);
    }
    if (cursor + nameLen + 12 > buffer.byteLength) throw new Error("unpackContainer: truncated index entry");
    const name = dec.decode(new Uint8Array(buffer, cursor, nameLen));
    cursor += nameLen;
    const offset = dv.getUint32(cursor, true); cursor += 4;
    const length = dv.getUint32(cursor, true); cursor += 4;
    const tokenCount = dv.getUint32(cursor, true); cursor += 4;
    if (offset + length > buffer.byteLength) {
      throw new Error(`unpackContainer: branch "${name}" extends past file end (offset=${offset}, len=${length}, total=${buffer.byteLength})`);
    }
    const blob = new Uint8Array(buffer, offset, length);
    branches.set(name, { blob, tokenCount });
  }
  return { version, branches };
}

/** Auto-detect: TQKV single-blob vs TQKC container. Returns a unified
 *  shape where a single-blob file surfaces as a one-entry map under
 *  the name "system" (back-compat with today's system-cache.bin). */
export function parseSystemCache(
  buffer: ArrayBuffer,
): ParsedContainer | { version: 0; branches: Map<string, { blob: Uint8Array; tokenCount: number }> } {
  if (buffer.byteLength < 4) throw new Error("parseSystemCache: buffer too small for magic");
  const dv = new DataView(buffer);
  const magic = dv.getUint32(0, true);
  if (magic === CONTAINER_MAGIC) {
    return unpackContainer(buffer);
  }
  if (magic === SINGLE_BLOB_MAGIC) {
    // Back-compat: treat a single TQKV blob as a one-branch container
    // keyed "system". Token count is the cached length at bytes [8..12]
    // (Engine.dumpCache header slot 2).
    const tokenCount = dv.getUint32(8, true);
    const branches = new Map<string, { blob: Uint8Array; tokenCount: number }>();
    branches.set("system", { blob: new Uint8Array(buffer), tokenCount });
    return { version: 0, branches };
  }
  throw new Error(`parseSystemCache: unknown magic 0x${magic.toString(16)}`);
}
