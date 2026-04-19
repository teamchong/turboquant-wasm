/**
 * UML-class-mode branch. Mounted after the grammar recognises
 * `setType("class")`.
 *
 * UML class diagrams: every class is an addClass with attributes +
 * methods; relationships between classes go through connect with a
 * { relation: "..." } opt (inheritance / composition / aggregation /
 * dependency / association).
 */

import { PREAMBLE } from "./preamble.js";
import { SDK_TYPES } from "../drawmode/sdk-types.js";

export const CLASS_PROMPT = `${PREAMBLE}

You are in UML CLASS mode (setType("class") has been called).
Use:
  addClass(name, { attributes?, methods? }, opts?)
  connect(from, to, "label", { relation: "..." })

attributes / methods are arrays of objects. Member shapes — one of:
  { name: "<n>" }
  { name: "<n>", type: "<t>" }
  { name: "<n>", type: "<t>", visibility: "public" | "private" | "protected" | "package" }

Brace-matching is the #1 way this mode fails. Each member object is exactly ONE pair of "{...}". Each attributes / methods array is exactly ONE pair of "[...]". Each addClass call closes with exactly "});". After the methods array's closing "]", the next character is "}" (to close the { attributes, methods } object) and then ");" — not "]," "}," or ",});".

relation values: "inheritance" | "composition" | "aggregation" | "dependency" | "association".

Do NOT use addBox, addEllipse, addDiamond, addActor, message, addTable, addGroup, or addLane.

UML CLASS example — abstract base + 4 subclasses pattern (mirror this
structure exactly when the prompt asks for a base class with multiple
subtypes; every class gets its own const + addClass, then one connect
per inheritance edge at the end):
setType("class");
const Shape = addClass("Shape", {
  attributes: [{ name: "color", type: "string" }],
  methods: [
    { name: "area()", type: "number" },
    { name: "draw()", type: "void" },
  ],
});
const Circle = addClass("Circle", {
  attributes: [{ name: "radius", type: "number" }],
  methods: [
    { name: "area()", type: "number" },
    { name: "draw()", type: "void" },
  ],
});
const Square = addClass("Square", {
  attributes: [{ name: "side", type: "number" }],
  methods: [
    { name: "area()", type: "number" },
    { name: "draw()", type: "void" },
  ],
});
const Triangle = addClass("Triangle", {
  attributes: [
    { name: "base", type: "number" },
    { name: "height", type: "number" },
  ],
  methods: [
    { name: "area()", type: "number" },
    { name: "draw()", type: "void" },
  ],
});
const Rectangle = addClass("Rectangle", {
  attributes: [
    { name: "width", type: "number" },
    { name: "height", type: "number" },
  ],
  methods: [
    { name: "area()", type: "number" },
    { name: "draw()", type: "void" },
  ],
});
connect(Circle, Shape, "extends", { relation: "inheritance" });
connect(Square, Shape, "extends", { relation: "inheritance" });
connect(Triangle, Shape, "extends", { relation: "inheritance" });
connect(Rectangle, Shape, "extends", { relation: "inheritance" });

connect arg order is ALWAYS (from, to, label, opts). Not
(from, label, to, opts). For inheritance edges the child is "from"
and the parent is "to": connect(Subclass, Base, "extends", { relation: "inheritance" }).

Rules:
- EVERY addClass call MUST be assigned to a const — on EVERY class, not just the first few. Writing bare "PayPal = addClass(...)" without "const" fails with "PayPal is not defined" at the next connect that references it.
- Attributes and methods ALWAYS go inside the addClass call's { attributes, methods } object. NEVER as separate connect() calls — connect is ONLY for class-to-class relationships.
- Every relationship connect MUST include { relation: "..." } with a valid value. Unrelated edges are invalid.
- Inheritance arrows: connect(child, parent, "extends", { relation: "inheritance" }). Direction is child → parent.
- Composition (whole owns part): connect(whole, part, "owns", { relation: "composition" }).
- Visibility markers are optional; default is "public" if omitted.
- 4+ classes; every class has at least one attribute OR method (empty classes are invalid); every class is referenced by at least one connect.

${SDK_TYPES}`;
