"""
merge_graph_documents.py

Post-processing utility for LangChain's LLMGraphTransformer output.

When you run llm_transformer.convert_to_graph_documents(chunks), you get back
one GraphDocument PER CHUNK. Because each chunk is processed independently by
the LLM, the same real-world entity can surface as multiple, slightly
different Node objects across chunks (e.g. "Marie Curie" in chunk 3 and
"marie curie" in chunk 12 -- Ollama/local models tend to be less consistent
about this than GPT-4-class models).

merge_graph_documents() collapses those into a single deduplicated graph,
using an explicit (normalized id, exact type) key. This identity rule is
local to this utility and can differ from database-level merge semantics. Relationships are deduplicated the same way, remapped to
point at the canonical node objects.
"""

from copy import deepcopy
from typing import Any, Dict, List, Tuple

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship


def _normalize_id(node_id: Any, case_sensitive: bool) -> str:
    """Normalize a node id for use as a merge key (does not change the stored id)."""
    text = str(node_id).strip()
    return text if case_sensitive else text.casefold()


def _merge_properties(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Combine two property dicts without silently dropping conflicting values."""
    merged = deepcopy(existing)
    for key, value in new.items():
        if key not in merged:
            merged[key] = deepcopy(value)
        elif merged[key] != value:
            existing_values = merged[key] if isinstance(merged[key], list) else [merged[key]]
            incoming = value if isinstance(value, list) else [value]
            for item in incoming:
                if item not in existing_values:
                    existing_values = existing_values + [deepcopy(item)]
            merged[key] = existing_values
    return merged


def merge_graph_documents(
    graph_documents: List[GraphDocument],
    case_sensitive: bool = False,
) -> GraphDocument:
    """
    Merge a list of per-chunk GraphDocuments into one deduplicated GraphDocument.

    Args:
        graph_documents: output of llm_transformer.convert_to_graph_documents(chunks)
        case_sensitive: if False (default), "Marie Curie" and "marie curie" are
            treated as the same node. First-seen spelling/casing is kept.

    Returns:
        A single GraphDocument with deduplicated nodes and relationships.
        Its `source` is set to the first chunk's source document as a
        placeholder -- a merged graph no longer maps to one chunk, so don't
        rely on it for provenance. Per-entity provenance is not tracked. Empty input raises ValueError;
        relationship-only endpoints are included in the merged node list.
    """
    if not graph_documents:
        raise ValueError("At least one graph document is required.")

    node_lookup: Dict[Tuple[str, str], Node] = {}

    # --- 1. Merge nodes, keyed by (normalized id, type) ---
    for gd in graph_documents:
        candidates = list(gd.nodes)
        candidates.extend(n for r in gd.relationships for n in (r.source, r.target))
        for node in candidates:
            key = (_normalize_id(node.id, case_sensitive), node.type)
            if key in node_lookup:
                existing = node_lookup[key]
                node_lookup[key] = Node(
                    id=existing.id,  # keep first-seen casing/spelling
                    type=existing.type,
                    properties=_merge_properties(existing.properties, node.properties),
                )
            else:
                node_lookup[key] = node.model_copy(deep=True)

    # --- 2. Merge relationships, remapped to the canonical (deduplicated) nodes ---
    rel_lookup: Dict[Tuple[str, str, str, str, str], Relationship] = {}

    for gd in graph_documents:
        for rel in gd.relationships:
            src_key = (_normalize_id(rel.source.id, case_sensitive), rel.source.type)
            tgt_key = (_normalize_id(rel.target.id, case_sensitive), rel.target.type)

            canonical_source = node_lookup[src_key]
            canonical_target = node_lookup[tgt_key]

            rel_key = (
                _normalize_id(canonical_source.id, case_sensitive),
                canonical_source.type,
                _normalize_id(canonical_target.id, case_sensitive),
                canonical_target.type,
                rel.type,
            )

            if rel_key in rel_lookup:
                existing = rel_lookup[rel_key]
                rel_lookup[rel_key] = Relationship(
                    source=canonical_source,
                    target=canonical_target,
                    type=existing.type,
                    properties=_merge_properties(existing.properties, rel.properties),
                )
            else:
                rel_lookup[rel_key] = Relationship(
                    source=canonical_source,
                    target=canonical_target,
                    type=rel.type,
                    properties=deepcopy(rel.properties),
                )

    merged_source = graph_documents[0].source if graph_documents else None

    return GraphDocument(
        nodes=list(node_lookup.values()),
        relationships=list(rel_lookup.values()),
        source=merged_source,
    )
