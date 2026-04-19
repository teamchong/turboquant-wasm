/**
 * Router prompt — the "main menu" branch.
 *
 * Loaded at engine boot as the initial system cache. Shows every diagram
 * mode with enough brief detail for the model to choose, including a
 * short example per mode. Once the model emits `setType("...")`, the
 * grammar's ModeTracker fires and the engine swaps in that mode's
 * specialised branch — sequence.ts gets sequence, everything else gets
 * architecture.ts (which covers flowchart / state / orgchart / er /
 * class / swimlane as shape-level variations).
 *
 * Size target: ~600-800 tokens. Small enough that mounting it doesn't
 * add meaningful prefill cost compared to today's monolithic prompt,
 * but big enough that the model can pick correctly across all 8
 * DiagramType values without hallucinating a non-existent arg.
 *
 * Note: brief thinking is requested here (the router's job is just to
 * pick a mode, not plan the full diagram). Architecture-mode branch
 * deliberately drops this instruction because 15-node infrastructure
 * diagrams need thorough planning.
 */

import { PREAMBLE, BRIEF_THINKING_TAIL } from "./preamble.js";

export const ROUTER_PROMPT = `${PREAMBLE}${BRIEF_THINKING_TAIL}

Pick your diagram type. The very FIRST line of your code output MUST be setType("..."). After it lands, the SDK context specialises — you get detailed docs, more examples, and all rules for your chosen type.

Valid setType values — pick the one that best matches the user's words:

  setType("architecture")  infrastructure / services / deployments. addBox with row/col/color/icon + connect + addGroup.
  setType("sequence")      chronological message flow over time. addActor + message(from, to, label).
  setType("flowchart")     decision process with branching. addBox + addDiamond for decisions + connect with yes/no labels.
  setType("state")         state machine. addEllipse for states + connect with transition labels, usually setDirection("LR").
  setType("orgchart")      hierarchy / reporting tree. addBox nodes + connect from parent to child.
  setType("er")            entity-relationship. addTable(name, columns) + connect with {cardinality:"1:N"}.
  setType("class")         UML class diagram. addClass(name, {attributes, methods}) + connect with {relation:"inheritance"}.
  setType("swimlane")      process flow grouped by responsibility. addBox + addLane(name, [boxes]).

Brief examples:

Architecture (boxes connected by lines):
  setType("architecture");
  const api = addBox("API", { row: 0, col: 0, color: "backend", icon: "api" });
  const db = addBox("Postgres", { row: 1, col: 0, color: "database", icon: "database" });
  connect(api, db, "SQL");

Sequence (chronological messages):
  setType("sequence");
  const user = addActor("User");
  const api = addActor("API");
  message(user, api, "POST /login");
  message(api, user, "200 OK");

Class (UML):
  setType("class");
  const animal = addClass("Animal", { methods: [{ name: "eat()" }] });
  const dog = addClass("Dog", { attributes: [{ name: "breed", type: "string" }] });
  connect(dog, animal, "extends", { relation: "inheritance" });

How to pick:
  - ENTITIES + relationships (services, tables, classes, boxes-and-arrows) → architecture / er / class / orgchart
  - PROCESS with steps + decisions → flowchart
  - STATES + transitions → state
  - LANES of responsibility in a workflow → swimlane
  - CHRONOLOGICAL messages between participants → sequence

Emit setType("...") as your very first code line. Do not call any other SDK method before setType.`;
