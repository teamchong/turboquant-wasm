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

  test("routes all non-sequence types to ARCHITECTURE", () => {
    // flowchart / state / orgchart / er / class / swimlane all collapse
    // to the architecture branch because they share Graphviz layout.
    for (const arg of ["flowchart", "state", "orgchart", "er", "class", "swimlane", "architecture"]) {
      const t = new ModeTracker();
      expect(t.observe(`setType("${arg}");`)).toBe(true);
      expect(t.mode).toBe(SDK_MODE_ARCHITECTURE);
    }
  });

  test("ignores truly-unknown setType args", () => {
    // Not in the DiagramType union → regex doesn't match → no fire.
    const t = new ModeTracker();
    expect(t.observe('setType("calendar");\nsetType("mindmap");')).toBe(false);
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

  test("onEnter handler fires synchronously on transition", () => {
    const t = new ModeTracker();
    const calls: number[] = [];
    t.onEnter(SDK_MODE_SEQUENCE, (m) => calls.push(m));
    t.onEnter(SDK_MODE_ARCHITECTURE, (m) => calls.push(m));

    t.observe('setType("sequence");');
    expect(calls).toEqual([SDK_MODE_SEQUENCE]);
  });

  test("multiple onEnter handlers run in registration order", () => {
    const t = new ModeTracker();
    const order: string[] = [];
    t.onEnter(SDK_MODE_ARCHITECTURE, () => order.push("first"));
    t.onEnter(SDK_MODE_ARCHITECTURE, () => order.push("second"));
    t.onEnter(SDK_MODE_ARCHITECTURE, () => order.push("third"));

    t.observe('setType("architecture");');
    expect(order).toEqual(["first", "second", "third"]);
  });

  test("onEnter does not fire on truly-unknown setType args", () => {
    // "calendar" isn't in the DiagramType union → regex doesn't match
    // → no handler fires and mode stays UNSET. (Real types from the
    // union like "flowchart" fire ARCHITECTURE — see the routing test.)
    const t = new ModeTracker();
    let fired = false;
    t.onEnter(SDK_MODE_SEQUENCE, () => { fired = true; });
    t.onEnter(SDK_MODE_ARCHITECTURE, () => { fired = true; });

    t.observe('setType("calendar");');
    expect(fired).toBe(false);
    expect(t.mode).toBe(SDK_MODE_UNSET);
  });

  test("onEnter(UNSET) rejects at registration time", () => {
    const t = new ModeTracker();
    expect(() => t.onEnter(SDK_MODE_UNSET, () => {})).toThrow(/never fires/);
  });

  test("handlers persist across reset()", () => {
    const t = new ModeTracker();
    const calls: number[] = [];
    t.onEnter(SDK_MODE_SEQUENCE, (m) => calls.push(m));

    t.observe('setType("sequence");');
    expect(calls).toEqual([SDK_MODE_SEQUENCE]);

    t.reset();
    t.observe('setType("sequence");');
    expect(calls).toEqual([SDK_MODE_SEQUENCE, SDK_MODE_SEQUENCE]);
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
