/**
 * ER-mode branch. Mounted after the grammar recognises
 * `setType("er")`.
 *
 * Entity-relationship diagrams for database schemas. Every entity is an
 * addTable with typed columns; foreign-key relationships are edges with
 * a cardinality opt ("1:1" | "1:N" | "N:1" | "N:M").
 */

import { PREAMBLE } from "./preamble.js";
import { SDK_TYPES } from "../drawmode/sdk-types.js";

export const ER_PROMPT = `${PREAMBLE}

You are in ER mode (setType("er") has been called).
Use:
  addTable(name, columns, opts?)  for every entity. columns is a JS array of column objects.
  connect(from, to, "verb", { cardinality: "1:N" })  for every relationship — cardinality opt required.

Do NOT use addBox, addEllipse, addDiamond, addActor, message, addClass, addGroup, or addLane.

Column shape — exactly one of these two forms per column, nothing else:
  { name: "<col>", key: "PK" | "FK" }                       primary or foreign key
  { name: "<col>", type: "<sql_type>" }                     regular column
  { name: "<col>", type: "<sql_type>", key: "PK" | "FK" }   FK with explicit type
Every column object closes with exactly ONE "}". Every addTable call closes with exactly "]);". Do not add extra "}" inside the columns array.

Valid sql_type values: "int" | "string" | "text" | "decimal" | "timestamp" | "bool" | "uuid".

ER example (6 tables — works one-shot, mirror this structure):
setType("er");
const users = addTable("users", [
  { name: "id", key: "PK" },
  { name: "email", type: "string" },
]);
const categories = addTable("categories", [
  { name: "id", key: "PK" },
  { name: "name", type: "string" },
]);
const products = addTable("products", [
  { name: "id", key: "PK" },
  { name: "name", type: "string" },
  { name: "category_id", type: "int", key: "FK" },
]);
const orders = addTable("orders", [
  { name: "id", key: "PK" },
  { name: "user_id", type: "int", key: "FK" },
  { name: "total", type: "decimal" },
]);
const order_items = addTable("order_items", [
  { name: "id", key: "PK" },
  { name: "order_id", type: "int", key: "FK" },
  { name: "product_id", type: "int", key: "FK" },
  { name: "qty", type: "int" },
]);
const payments = addTable("payments", [
  { name: "id", key: "PK" },
  { name: "order_id", type: "int", key: "FK" },
  { name: "amount", type: "decimal" },
]);
connect(users, orders, "places", { cardinality: "1:N" });
connect(categories, products, "groups", { cardinality: "1:N" });
connect(orders, order_items, "contains", { cardinality: "1:N" });
connect(products, order_items, "listed in", { cardinality: "1:N" });
connect(orders, payments, "settled by", { cardinality: "1:N" });

Rules:
- Every table's FIRST column is the PK: { name: "id", key: "PK" }.
- FK columns use BOTH type and key: { name: "<parent>_id", type: "int", key: "FK" }.
- Every connect MUST include { cardinality: "1:1" | "1:N" | "N:1" | "N:M" }. Unlabelled cardinality is invalid.
- Connect labels are short verbs from source's perspective ("places", "owns", "groups", "contains", "settled by").
- 4+ tables. Every table participates in at least one relationship.

${SDK_TYPES}`;
