"""Validate exported JSON and explicitly load it into Memgraph."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def _identifier(value):
    """Escape dynamic Cypher identifiers; values are passed as parameters."""
    return "`" + value.replace("`", "``") + "`"


def _validate_record(record, required):
    if not isinstance(record, dict):
        raise ValueError("Each graph record must be an object.")
    for field in required:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"{field} must be a nonempty string: {record!r}")
    props = record.get("properties", {})
    if not isinstance(props, dict):
        raise ValueError("properties must be an object.")
    if "id" in props:
        raise ValueError("properties must not overwrite the reserved id field.")


def validate_graph(nodes, relationships):
    """Return typed relationships; reject dangling or ambiguous references."""
    if not isinstance(nodes, list) or not isinstance(relationships, list):
        raise ValueError("Nodes and relationships must be JSON arrays.")
    types = defaultdict(set)
    seen = set()
    for node in nodes:
        _validate_record(node, ("id", "type"))
        key = (node["id"], node["type"])
        if key in seen:
            raise ValueError(f"Duplicate node: {key}")
        seen.add(key)
        types[node["id"]].add(node["type"])
    resolved = []
    for rel in relationships:
        _validate_record(rel, ("source", "target", "type"))
        record = dict(rel)
        for endpoint in ("source", "target"):
            candidates = types.get(rel[endpoint], set())
            label = rel.get(f"{endpoint}_type")
            if label is None and len(candidates) == 1:
                label = next(iter(candidates))
            if not isinstance(label, str) or label not in candidates:
                raise ValueError(f"Missing or ambiguous {endpoint} endpoint: {rel[endpoint]!r}")
            record[f"{endpoint}_type"] = label
        resolved.append(record)
    return resolved


def load_graph(db, nodes, relationships):
    """Validate before writing; merge nodes by label and ID."""
    relationships = validate_graph(nodes, relationships)
    batches = defaultdict(list)
    for node in nodes:
        batches[node["type"]].append({"id": node["id"], "properties": node.get("properties", {})})
    for label, batch in batches.items():
        db.execute(f"UNWIND $batch AS n MERGE (x:{_identifier(label)} {{id: n.id}}) SET x += n.properties",
                   parameters={"batch": batch})
    batches = defaultdict(list)
    for rel in relationships:
        batches[rel["source_type"], rel["target_type"], rel["type"]].append(
            {**rel, "properties": rel.get("properties", {})})
    for (source, target, kind), batch in batches.items():
        db.execute(
            f"UNWIND $batch AS r MATCH (a:{_identifier(source)} {{id: r.source}}), "
            f"(b:{_identifier(target)} {{id: r.target}}) "
            f"MERGE (a)-[rel:{_identifier(kind)}]->(b) SET rel += r.properties",
            parameters={"batch": batch},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--relationships", type=Path, required=True)
    parser.add_argument("--host", default=os.getenv("MEMGRAPH_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MEMGRAPH_PORT", "7687")))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    nodes = json.loads(args.nodes.read_text(encoding="utf-8"))
    relationships = json.loads(args.relationships.read_text(encoding="utf-8"))
    validate_graph(nodes, relationships)
    if args.validate_only:
        print(f"Valid: {len(nodes)} nodes, {len(relationships)} relationships.")
        return
    from gqlalchemy import Memgraph

    db = Memgraph(host=args.host, port=args.port,
                  username=os.getenv("MEMGRAPH_USERNAME", ""),
                  password=os.getenv("MEMGRAPH_PASSWORD", ""))
    load_graph(db, nodes, relationships)
    print(f"Loaded {len(nodes)} node records and {len(relationships)} relationship records.")


if __name__ == "__main__":
    main()
