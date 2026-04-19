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
  addTable(name, columns, opts?)  for every entity. columns is an array of { name, type?, key? }.
  connect(from, to, "verb", { cardinality: "1:N" })  for every relationship — cardinality opt required.

Do NOT use addBox, addEllipse, addDiamond, addActor, message, addClass, addGroup, or addLane.

ER example:
setType("er");
const users = addTable("users", [
  { name: "id", key: "PK" },
  { name: "email", type: "string" },
  { name: "created_at", type: "timestamp" },
]);
const orders = addTable("orders", [
  { name: "id", key: "PK" },
  { name: "user_id", type: "int", key: "FK" },
  { name: "total", type: "decimal" },
]);
const items = addTable("order_items", [
  { name: "id", key: "PK" },
  { name: "order_id", type: "int", key: "FK" },
  { name: "product_id", type: "int", key: "FK" },
  { name: "qty", type: "int" },
]);
const products = addTable("products", [
  { name: "id", key: "PK" },
  { name: "name", type: "string" },
  { name: "price", type: "decimal" },
]);
connect(users, orders, "places", { cardinality: "1:N" });
connect(orders, items, "contains", { cardinality: "1:N" });
connect(products, items, "listed in", { cardinality: "1:N" });

Rules:
- Every table MUST have an "id" column with key: "PK" as its first column. Tables without a primary key are invalid.
- Foreign-key columns are marked with key: "FK" and reference another table's PK.
- Every connect MUST include { cardinality: "..." } with value "1:1" | "1:N" | "N:1" | "N:M". Unlabelled cardinality is invalid.
- Connect labels describe the relationship as a verb from source's perspective ("places", "owns", "has", "belongs to").
- 4+ tables. Every table has at least one relationship (no isolated tables).
- Column types are short SQL-ish tokens: "string" | "int" | "decimal" | "timestamp" | "bool" | "uuid" | "text" | etc.

${SDK_TYPES}`;
