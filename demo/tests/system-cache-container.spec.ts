import { test, expect } from "@playwright/test";
import {
  packContainer,
  unpackContainer,
  parseSystemCache,
  CONTAINER_MAGIC,
  CONTAINER_VERSION,
  MAX_BRANCH_NAME_LEN,
} from "../src/draw/system-cache-container";

// Unit-ish test for the multi-branch container format. Pure TS, runs
// without a browser. Execute with:
//   cd demo && npx playwright test tests/system-cache-container.spec.ts

function makeFakeBlob(len: number, fillByte: number): Uint8Array {
  const b = new Uint8Array(len);
  // Realistic-enough shape: first 4 bytes are TQKV magic so parseSystemCache's
  // back-compat path works for single-blob-shaped inputs. Byte [8..12] is
  // token_count (cached length).
  const dv = new DataView(b.buffer);
  dv.setUint32(0, 0x564b5154, true);
  dv.setUint32(8, fillByte * 100, true);
  for (let i = 12; i < len; i++) b[i] = fillByte;
  return b;
}

test.describe("system-cache-container", () => {
  test("pack/unpack round-trips 3 branches", () => {
    const router = makeFakeBlob(64, 0x11);
    const sequence = makeFakeBlob(128, 0x22);
    const architecture = makeFakeBlob(256, 0x33);

    const packed = packContainer([
      { name: "router", blob: router, tokenCount: 200 },
      { name: "sequence", blob: sequence, tokenCount: 800 },
      { name: "architecture", blob: architecture, tokenCount: 2200 },
    ]);

    // Header sanity
    const dv = new DataView(packed.buffer);
    expect(dv.getUint32(0, true)).toBe(CONTAINER_MAGIC);
    expect(dv.getUint16(4, true)).toBe(CONTAINER_VERSION);
    expect(dv.getUint16(6, true)).toBe(3);

    const { branches } = unpackContainer(packed.buffer);
    expect(branches.size).toBe(3);

    const r = branches.get("router")!;
    expect(r.tokenCount).toBe(200);
    expect(r.blob.byteLength).toBe(64);
    expect([...r.blob.slice(12, 16)]).toEqual([0x11, 0x11, 0x11, 0x11]);

    const s = branches.get("sequence")!;
    expect(s.tokenCount).toBe(800);
    expect(s.blob.byteLength).toBe(128);

    const a = branches.get("architecture")!;
    expect(a.tokenCount).toBe(2200);
    expect(a.blob.byteLength).toBe(256);
  });

  test("parseSystemCache detects TQKV single-blob back-compat", () => {
    const single = makeFakeBlob(200, 0x55);
    const parsed = parseSystemCache(single.buffer);
    expect(parsed.version).toBe(0); // back-compat marker
    expect(parsed.branches.size).toBe(1);
    const entry = parsed.branches.get("system")!;
    expect(entry.blob.byteLength).toBe(200);
    expect(entry.tokenCount).toBe(0x55 * 100);
  });

  test("parseSystemCache detects TQKC container", () => {
    const packed = packContainer([
      { name: "only", blob: makeFakeBlob(32, 0x77), tokenCount: 50 },
    ]);
    const parsed = parseSystemCache(packed.buffer);
    expect(parsed.version).toBe(CONTAINER_VERSION);
    expect(parsed.branches.size).toBe(1);
    expect(parsed.branches.get("only")!.tokenCount).toBe(50);
  });

  test("rejects unknown magic", () => {
    const bad = new Uint8Array([0xde, 0xad, 0xbe, 0xef, 0, 0, 0, 0]);
    expect(() => parseSystemCache(bad.buffer)).toThrow(/unknown magic/);
  });

  test("rejects empty branch list", () => {
    expect(() => packContainer([])).toThrow(/at least one branch/);
  });

  test("rejects oversize branch name", () => {
    expect(() =>
      packContainer([
        { name: "x".repeat(MAX_BRANCH_NAME_LEN + 1), blob: makeFakeBlob(16, 0), tokenCount: 0 },
      ]),
    ).toThrow(/exceeds/);
  });

  test("rejects empty branch name", () => {
    expect(() =>
      packContainer([
        { name: "", blob: makeFakeBlob(16, 0), tokenCount: 0 },
      ]),
    ).toThrow(/cannot be empty/);
  });
});
