import json
from collections import defaultdict
from gqlalchemy import Memgraph

db = Memgraph(host="127.0.0.1", port=7687)

# --- Load nodes, grouped by type (becomes the Cypher label) ---
with open("nodes.json") as f:
    nodes = json.load(f)

nodes_by_type = defaultdict(list)
for n in nodes:
    nodes_by_type[n["type"]].append({
        "id": n["id"],
        "properties": n.get("properties", {})
    })

for label, batch in nodes_by_type.items():
    db.execute(
        f"""
        UNWIND $batch AS n
        MERGE (x:`{label}` {{id: n.id}})
        SET x += n.properties
        """,
        parameters={"batch": batch}
    )

# --- Load relationships, grouped by type ---
with open("relationships.json") as f:
    rels = json.load(f)

rels_by_type = defaultdict(list)
for r in rels:
    rels_by_type[r["type"]].append({
        "source_id": r["source"],
        "target_id": r["target"],
        "properties": r.get("properties", {})
    })

for rel_type, batch in rels_by_type.items():
    db.execute(
        f"""
        UNWIND $batch AS r
        MATCH (a {{id: r.source_id}}), (b {{id: r.target_id}})
        MERGE (a)-[rel:`{rel_type}`]->(b)
        SET rel += r.properties
        """,
        parameters={"batch": batch}
    )

# --- Sanity check for dangling references ---
node_ids = {n["id"] for n in nodes}
rel_ids = {r["source"] for r in rels} | {r["target"] for r in rels}
missing = rel_ids - node_ids
if missing:
    print(f"Warning: {len(missing)} entities referenced in relationships but missing from nodes: {missing}")

print(f"Loaded {len(nodes)} nodes and {len(rels)} relationships.")
