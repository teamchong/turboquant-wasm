/**
 * Router prompt — the "main menu" branch.
 *
 * Loaded at engine boot as the initial system cache. Shows every diagram
 * mode with enough brief detail for the model to choose, including a small
 * example per mode. Once the model emits `setType("...")`, the grammar's
 * ModeTracker fires and the engine swaps in that mode's specialised branch
 * (sequence.ts or architecture.ts) mid-generation — at that point the
 * model gets full docs, more examples, and all per-mode rules.
 *
 * Size target: ~500 tokens. Small enough that mounting it doesn't add
 * meaningful prefill cost compared to today's monolithic prompt, but big
 * enough that the model can choose correctly without hallucinating a
 * non-existent setType arg.
 *
 * Note: brief thinking is requested here (the router's job is just to
 * pick a mode, not to plan the full diagram). Architecture-mode branch
 * deliberately drops this instruction because 15-node infrastructure
 * diagrams need thorough planning.
 */

import { PREAMBLE, BRIEF_THINKING_TAIL } from "./preamble.js";

export const ROUTER_PROMPT = `${PREAMBLE}${BRIEF_THINKING_TAIL}

Pick your diagram type. The very FIRST line of your code output MUST be setType("..."). After it lands, the SDK context will specialise — you'll get detailed docs, more examples, and all rules for your chosen type. Before that, this is the menu.

Two modes exist:

1. setType("architecture") — boxes connected by lines. Use this for everything EXCEPT chronological message flows. Covers: infrastructure, flowcharts, state machines, org charts, ER diagrams, UML class diagrams, swimlane process flows. Uses addBox / addEllipse / addDiamond / addTable / addClass / addGroup / addLane / connect.

Brief example (infrastructure):
  setType("architecture");
  const api = addBox("API", { row: 0, col: 0, color: "backend", icon: "api" });
  const db = addBox("Postgres", { row: 1, col: 0, color: "database", icon: "database" });
  const cache = addBox("Redis", { row: 1, col: 1, color: "cache", icon: "cache" });
  connect(api, db, "SQL");
  connect(api, cache, "GET session");
  addGroup("Data", [db, cache]);

Brief example (UML class):
  setType("architecture");
  const animal = addClass("Animal", { methods: [{ name: "eat()" }, { name: "sleep()" }] });
  const dog = addClass("Dog", { attributes: [{ name: "breed", type: "string" }] });
  connect(dog, animal, "extends", { relation: "inheritance" });

2. setType("sequence") — chronological message flow between participants. Use this ONLY when the user explicitly asks for a sequence / message / call-flow / interaction diagram, or describes a step-by-step exchange between actors over time. Uses addActor + message(from, to, label). Do NOT use sequence for static structure, APIs-to-databases topology, or class diagrams.

Brief example (sequence):
  setType("sequence");
  const user = addActor("User");
  const api = addActor("API");
  const db = addActor("Database");
  message(user, api, "POST /login");
  message(api, db, "SELECT * FROM users WHERE email=?");
  message(db, api, "row");
  message(api, user, "200 OK");

How to pick: does the user's request describe ENTITIES and their relationships (databases, services, boxes-and-arrows)? → architecture. Does it describe CONVERSATIONS or step-by-step interactions over time (someone sends X, then Y responds with Z)? → sequence. When in doubt, pick architecture — it's the default for most diagrams.

Emit setType("...") as your very first code line. Do not call any other SDK method before setType.`;
