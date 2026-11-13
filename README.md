# Graph RAG Experiments

Experiments in extracting knowledge graphs from a PDF story using local language models through Ollama. Compare open-schema LangChain extraction, constrained extraction, and KGGen entity clustering, then inspect the resulting graph or load it into Memgraph.

This repository covers the **graph construction stage** of GraphRAG. Retrieval, question answering, and evaluation are not implemented yet.

## Workflow

```text
PDF -> text chunks -> Ollama extraction -> merge / cluster
                                         |
                                         v
                              JSON + interactive HTML
                                         |
                                         v
                                Memgraph (optional)
```

## File guide

| File | Purpose |
| --- | --- |
| [LLMGraphTransformer.ipynb](LLMGraphTransformer.ipynb) | Open-schema extraction with LangChain; merges per-chunk graphs before exporting. A good starting point. |
| [Creating KG.ipynb](Creating%20KG.ipynb) | Constrained LangChain experiment with explicit allowed node and relationship types. |
| [KGGEN.ipynb](KGGEN.ipynb) | Sequential KGGen extraction, optional clustering with parsing retries, and export. Nodes use the generic `Entity` type. |
| [merge_graph_documents.py](merge_graph_documents.py) | Deduplicates LangChain nodes and directed relationships and combines conflicting properties. |
| [graph_io.py](graph_io.py) | Shared graph-record conversion, validated UTF-8 JSON export, and PyVis HTML visualization. |
| [load_memgraph.py](load_memgraph.py) | Command-line validation and Memgraph loading; checks endpoints before writing. |
| [JsonToMemgraph2.py](JsonToMemgraph2.py) | Compatibility entry point for the loader; accepts the same explicit arguments. |
| [abel-and-his-great-adventure.pdf](abel-and-his-great-adventure.pdf) | Existing sample story used by all three notebooks. |
| [requirements.txt](requirements.txt) | Dependencies for extraction, visualization, and Jupyter. |
| [requirements-memgraph.txt](requirements-memgraph.txt) | Optional database client dependency. |
| [tests/test_graph_utilities.py](tests/test_graph_utilities.py) | Regression tests for merge behavior, JSON validation/export, and query construction without a live database. |

Notebook filenames are retained so existing links remain useful. Shared operations live in Python modules to avoid maintaining separate copies in each notebook.

## Setup

Use a dedicated Python environment; Python 3.11 or 3.12 is a suggested starting point. Dependency versions are currently unpinned, and a complete fresh-install/model compatibility matrix has not been verified.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate` instead.

Install [Ollama](https://ollama.com/) separately, start its service, and download the default model:

```sh
ollama pull gpt-oss:20b
ollama serve
```

Run `ollama serve` only if the service is not already running. The default model requires substantial local resources; choose an installed model appropriate for your machine by editing `MODEL` in the notebook or setting `OLLAMA_MODEL`. The LangChain workflow needs a model compatible with its structured extraction.

Start Jupyter **from the repository root**, select a Python kernel with the installed dependencies, open a notebook, and run its cells in order:

```sh
python -m jupyterlab
```

Each notebook begins with configuration for the PDF, model, chunk size, overlap, and output directory. Changing the input PDF requires a fresh extraction run.

## Generated files

Outputs go to `outputs/`, which Git ignores. Each experiment writes:

| Experiment | JSON files | Visualization |
| --- | --- | --- |
| Open-schema LangChain | `llm_transformer_nodes.json`, `llm_transformer_relationships.json` | `llm_transformer_graph.html` |
| Constrained LangChain | `constrained_nodes.json`, `constrained_relationships.json` | `constrained_graph.html` |
| KGGen | `kggen_nodes.json`, `kggen_relationships.json` | `kggen_graph.html` |

Open the HTML file in a browser to explore nodes and edges. Re-running an experiment overwrites its own outputs.

Nodes use `(id, type)` as their identity. Relationships preserve endpoint types so two nodes with the same name but different labels remain distinct:

```json
{
  "nodes": [
    {"id": "Abel", "type": "Person", "properties": {}},
    {"id": "Village", "type": "Place", "properties": {}}
  ],
  "relationships": [
    {"source": "Abel", "source_type": "Person", "target": "Village", "target_type": "Place", "type": "LIVES_IN", "properties": {}}
  ]
}
```

This example shows both arrays together for readability; export writes them to **two separate files**. Legacy relationships without endpoint types are accepted only when each endpoint ID identifies exactly one node type.

## Optional: load into Memgraph

Use an existing Memgraph instance, then install the optional client:

```sh
python -m pip install -r requirements-memgraph.txt
python load_memgraph.py --nodes outputs/llm_transformer_nodes.json --relationships outputs/llm_transformer_relationships.json --validate-only
python load_memgraph.py --nodes outputs/llm_transformer_nodes.json --relationships outputs/llm_transformer_relationships.json
```

The default connection is `127.0.0.1:7687`. Override it with `--host` / `--port` or `MEMGRAPH_HOST` / `MEMGRAPH_PORT`. Supply authentication through `MEMGRAPH_USERNAME` and `MEMGRAPH_PASSWORD` environment variables. Environment variables are read directly; `.env` files are not loaded automatically.

Importing either loader module does not connect to the database. Loading merges nodes by exact label and ID and edges by typed endpoints and relationship type. Existing properties can be updated. Loading uses multiple statements rather than one atomic transaction; database errors can leave a partial import. Use a separate database for independent experiments if you do not want their graphs combined.

## Validation

Run the regression suite from the repository root after installing the main dependencies:

```sh
python -m unittest discover -s tests -v
```

These tests do not require Ollama or Memgraph. Full notebook execution additionally requires the configured Ollama model; database integration requires a running Memgraph instance.

## Interpretation and limitations

- Extraction can omit or invent facts. Check the PDF before treating graph relationships as evidence.
- LangChain merging ignores ID case and surrounding whitespace but requires exact node types. It preserves the first ID spelling and combines conflicting property values. It does not resolve aliases or conflicting entity types.
- A merged LangChain graph keeps the first chunk's source as a placeholder; it does not retain per-entity evidence or provenance.
- KGGen extraction reports failed chunks and raises before export. Fix the cause and rerun extraction before continuing. Clustering can fall back to the unclustered graph after parsing retries are exhausted.
- PDFs containing scanned images may need OCR before these text-loading workflows can use them.
- Dependency manifests describe the required packages but are not a locked, reproducible environment.

## References

- [LangChain LLMGraphTransformer source](https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/graph_transformers/llm.py) documents the transformer used by the LangChain notebooks.
- [KGGen project and usage](https://github.com/stair-lab/kg-gen) describes generation and clustering APIs.
