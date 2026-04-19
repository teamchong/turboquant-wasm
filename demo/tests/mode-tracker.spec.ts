import { test, expect } from "@playwright/test";
import {
  ModeTracker,
  SDK_MODE_UNSET,
  SDK_MODE_SEQUENCE,
  SDK_MODE_ARCHITECTURE,
} from "../src/draw/grammar";

// Pure-TS tests for the SDK ModeTracker. No browser needed.
//   cd demo && npx playwright test tests/mode-tracker.spec.ts

test.describe("ModeTracker", () => {
  test("starts UNSET", () => {
    const t = new ModeTracker();
    expect(t.mode).toBe(SDK_MODE_UNSET);
  });

  test("detects sequence setType in one blob", () => {
    const t = new ModeTracker();
    const fired = t.observe('setType("sequence");');
    expect(fired).toBe(true);
    expect(t.mode).toBe(SDK_MODE_SEQUENCE);
  });

  test("detects architecture setType in one blob", () => {
    const t = new ModeTracker();
    const fired = t.observe('setType("architecture");\nconst x');
    expect(fired).toBe(true);
    expect(t.mode).toBe(SDK_MODE_ARCHITECTURE);
  });

  test("accumulates across token fragments (realistic streaming)", () => {
    const t = new ModeTracker();
    // Gemma BPE tends to split setType into a few tokens
    expect(t.observe("set")).toBe(false);
    expect(t.observe("Type")).toBe(false);
    expect(t.observe('("sequ')).toBe(false);
    expect(t.observe('ence")')).toBe(true);
    expect(t.mode).toBe(SDK_MODE_SEQUENCE);
  });

  test("only fires once per generation", () => {
    const t = new ModeTracker();
    expect(t.observe('setType("sequence");')).toBe(true);
    // A second setType later in the stream is ignored.
    expect(t.observe('setType("architecture");')).toBe(false);
    expect(t.mode).toBe(SDK_MODE_SEQUENCE);
  });

  test("reset() restores UNSET and clears buffer", () => {
    const t = new ModeTracker();
    t.observe('setType("sequence");');
    expect(t.mode).toBe(SDK_MODE_SEQUENCE);
    t.reset();
    expect(t.mode).toBe(SDK_MODE_UNSET);
    // After reset the detector should re-fire on a fresh setType.
    expect(t.observe('setType("architecture");')).toBe(true);
    expect(t.mode).toBe(SDK_MODE_ARCHITECTURE);
  });

  test("ignores unknown setType args", () => {
    const t = new ModeTracker();
    expect(t.observe('setType("flowchart");\nsetType("er");')).toBe(false);
    expect(t.mode).toBe(SDK_MODE_UNSET);
  });

  test("tolerant of whitespace variations", () => {
    const t1 = new ModeTracker();
    expect(t1.observe('setType( "sequence" );')).toBe(true);
    expect(t1.mode).toBe(SDK_MODE_SEQUENCE);

    const t2 = new ModeTracker();
    expect(t2.observe('setType ("architecture")')).toBe(true);
    expect(t2.mode).toBe(SDK_MODE_ARCHITECTURE);
  });

  test("does not fire on partial setType within thinking text", () => {
    const t = new ModeTracker();
    // Thinking mode might emit prose that MENTIONS setType — but since the
    // regex requires ("...") with a valid arg, prose won't trigger.
    expect(t.observe('I should call setType with the right mode...')).toBe(false);
    expect(t.observe(' then proceed.')).toBe(false);
    expect(t.mode).toBe(SDK_MODE_UNSET);
    // Actual call in code still fires.
    expect(t.observe('\nsetType("sequence");')).toBe(true);
    expect(t.mode).toBe(SDK_MODE_SEQUENCE);
  });
});
