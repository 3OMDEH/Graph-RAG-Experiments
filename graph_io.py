"""Shared JSON export and HTML visualization for notebook experiments."""

import json
from pathlib import Path


def export_graph(nodes, relationships, output_dir, prefix):
    """Write validated UTF-8 records using predictable filenames."""
    from load_memgraph import validate_graph

    nodes, relationships = list(nodes), list(relationships)
    validate_graph(nodes, relationships)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, records in (("nodes", nodes), ("relationships", relationships)):
        path = output_dir / f"{prefix}_{suffix}.json"
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(path)
        print(f"Exported {len(records)} {suffix} to {path}")
    return tuple(paths)


def graph_document_records(graph):
    """Preserve endpoint types so identical IDs in different labels stay distinct."""
    nodes = [dict(id=n.id, type=n.type, properties=n.properties) for n in graph.nodes]
    relationships = [dict(source=r.source.id, source_type=r.source.type,
                          target=r.target.id, target_type=r.target.type,
                          type=r.type, properties=r.properties) for r in graph.relationships]
    return nodes, relationships


def visualize_graph(nodes, relationships, output_path):
    """Save an interactive graph with composite node identities."""
    from html import escape
    from pyvis.network import Network
    from load_memgraph import validate_graph

    relationships = validate_graph(nodes, relationships)
    net = Network(height="850px", width="100%", directed=True, cdn_resources="in_line")
    identities = {(n["id"], n["type"]): i for i, n in enumerate(nodes)}
    for node in nodes:
        net.add_node(identities[node["id"], node["type"]], label=str(node["id"]),
                     title=escape(node["type"]), group=node["type"])
    for rel in relationships:
        net.add_edge(identities[rel["source"], rel["source_type"]],
                     identities[rel["target"], rel["target_type"]], label=rel["type"])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(path))
    return path
