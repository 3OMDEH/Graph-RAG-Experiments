"""Regression checks for merging, export, and database query construction."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from graph_io import export_graph, graph_document_records
from load_memgraph import load_graph, validate_graph
from merge_graph_documents import merge_graph_documents


class GraphTests(unittest.TestCase):
    def test_merge_conflicts_endpoints_and_input_preservation(self):
        a = Node(id="Abel", type="Person", properties={"alias": ["boy"]})
        b = Node(id=" abel ", type="Person", properties={"alias": ["child", "boy"]})
        fox = Node(id="Fox", type="Animal")
        docs = [GraphDocument(nodes=[a], relationships=[], source=Document(page_content="a")),
                GraphDocument(nodes=[b], relationships=[Relationship(source=b, target=fox, type="MEETS")],
                              source=Document(page_content="b"))]
        merged = merge_graph_documents(docs)
        self.assertEqual(len(merged.nodes), 2)
        self.assertEqual(merged.nodes[0].properties["alias"], ["boy", "child"])
        self.assertEqual(merged.relationships[0].source.id, "Abel")
        self.assertEqual(a.properties["alias"], ["boy"])
        self.assertEqual(len(merge_graph_documents(docs, case_sensitive=True).nodes), 3)
        validate_graph(*graph_document_records(merged))

    def test_relationship_deduplication_and_conflicts(self):
        a, b = Node(id="A", type="Entity"), Node(id="B", type="Entity")
        docs = [GraphDocument(nodes=[a, b], relationships=[
            Relationship(source=a, target=b, type="KNOWS", properties={"evidence": text})],
            source=Document(page_content=text)) for text in ["one", "two"]]
        merged = merge_graph_documents(docs)
        self.assertEqual(len(merged.relationships), 1)
        self.assertEqual(merged.relationships[0].properties["evidence"], ["one", "two"])

    def test_empty_merge(self):
        with self.assertRaises(ValueError):
            merge_graph_documents([])

    def test_validation_before_database_writes(self):
        db = Mock()
        with self.assertRaises(ValueError):
            load_graph(db, [{"id": "A", "type": "Person"}],
                       [{"source": "A", "target": "missing", "type": "MEETS"}])
        db.execute.assert_not_called()

    def test_ambiguous_ids_require_types_and_queries_escape_labels(self):
        nodes = [{"id": "A", "type": label} for label in ["Person`Label", "Place"]]
        rel = {"source": "A", "target": "A", "type": "AT"}
        with self.assertRaises(ValueError):
            validate_graph(nodes, [rel])
        rel.update(source_type="Person`Label", target_type="Place")
        db = Mock()
        load_graph(db, nodes, [rel])
        query = db.execute.call_args.args[0]
        self.assertIn("a:`Person``Label`", query)
        self.assertIn("b:`Place`", query)

    def test_legacy_schema_export_and_reserved_id(self):
        nodes = [{"id": "Abel", "type": "Person"}]
        rels = [{"source": "Abel", "target": "Abel", "type": "KNOWS"}]
        self.assertEqual(validate_graph(nodes, rels)[0]["source_type"], "Person")
        with tempfile.TemporaryDirectory() as directory:
            paths = export_graph(nodes, rels, directory, "sample")
            self.assertEqual(json.loads(Path(paths[0]).read_text(encoding="utf-8")), nodes)
        with self.assertRaises(ValueError):
            validate_graph([dict(id="A", type="Person", properties={"id": "B"})], [])


if __name__ == "__main__":
    unittest.main()
