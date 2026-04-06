import os
import json
import pickle
import logging
import numpy as np
import faiss
import networkx as nx
from graph.encoder import GeminiEncoder
from langsmith import traceable
from langsmith_tracing import setup_langsmith

log = logging.getLogger(__name__)
setup_langsmith()

class Retriever:
    """
    Loads build artifacts and performs dual path search at query time.
    Dual path = search entity index + hyperedge index simultaneously.
    """

    @traceable(name="retriever_init", run_type="chain")
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.encoder = GeminiEncoder()
        self.artifacts_dir = artifacts_dir
        self._load()

    @traceable(name="retriever_load", run_type="chain")
    def _load(self):
        # load FAISS indexes
        self.index_entity = faiss.read_index(
            os.path.join(self.artifacts_dir, "index_entity.bin")
        )
        self.index_hyperedge = faiss.read_index(
            os.path.join(self.artifacts_dir, "index_hyperedge.bin")
        )

        # load metadata
        with open(os.path.join(self.artifacts_dir, "metadata.json"), encoding="utf-8") as f:
            metadata = json.load(f)
        self.entities = metadata["entities"]
        self.hyperedges = metadata["hyperedges"]

        # load graph
        with open(os.path.join(self.artifacts_dir, "graph.gpickle"), "rb") as f:
            self.G = pickle.load(f)

        log.info(f"Retriever loaded: {len(self.entities)} entities, "
                 f"{len(self.hyperedges)} hyperedges")

    @traceable(name="retriever_search_index", run_type="retriever")
    def _search_index(
        self,
        index: faiss.Index,
        query_vector: np.ndarray,
        k: int
    ) -> list[int]:
        # FAISS expects 2D input — (1, dim) not (dim,)
        query_2d = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = index.search(query_2d, k)
        # indices shape: (1, k) — flatten to 1D list
        return indices[0].tolist()

    @traceable(name="retriever_search", run_type="retriever")
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> dict:
        # step 1 — encode query into vector
        query_vector = self.encoder.encode(query)

        # step 2 — search both indexes simultaneously
        entity_ids = self._search_index(self.index_entity, query_vector, top_k)
        hyperedge_ids = self._search_index(self.index_hyperedge, query_vector, top_k)

        # step 3 — map integer IDs back to actual content
        matched_entities = [
            self.entities[i] for i in entity_ids
            if i < len(self.entities) and i >= 0
        ]
        matched_hyperedges = [
            self.hyperedges[i] for i in hyperedge_ids
            if i < len(self.hyperedges) and i >= 0
        ]

        # step 4 — one hop graph enrichment
        # for each matched entity, grab connected hyperedges from graph
        enriched_facts = []
        for entity in matched_entities:
            if entity["name"] in self.G:
                neighbors = self.G.neighbors(entity["name"])
                for neighbor in neighbors:
                    node_data = self.G.nodes[neighbor]
                    if node_data.get("type") == "hyperedge":
                        enriched_facts.append(node_data["fact"])

        # step 5 — combine and deduplicate
        all_facts = list({
            h["fact"] for h in matched_hyperedges
        } | set(enriched_facts))

        return {
            "entities": matched_entities,
            "facts": all_facts,
            "entity_count": len(matched_entities),
            "fact_count": len(all_facts)
        }
