# =============================================================================
# Incremental Knowledge Base Updater
# Adds new data to existing artifacts without rebuilding from scratch.
# React virtual DOM pattern applied to knowledge graphs.
# Fixes Graph-R1's static hypergraph limitation.
# =============================================================================

import os
import json
import pickle
import logging
import faiss
import networkx as nx
from dotenv import load_dotenv
from graph.encoder import GeminiEncoder
from graph.builder import (
    chunk_document,
    extract_entities,
    encode_entities,
    encode_hyperedges,
)

load_dotenv()
log = logging.getLogger(__name__)

encoder = GeminiEncoder()


class Updater:
    """
    Incrementally updates existing artifacts with new documents.
    Only adds genuinely new entities and hyperedges — never rebuilds.
    This is the React virtual DOM pattern applied to knowledge graphs.
    Fixes Graph-R1's static hypergraph limitation.
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self._load_artifacts()

    def _load_artifacts(self):
        # load FAISS indexes
        self.index_entity = faiss.read_index(
            os.path.join(self.artifacts_dir, "index_entity.bin")
        )
        self.index_hyperedge = faiss.read_index(
            os.path.join(self.artifacts_dir, "index_hyperedge.bin")
        )

        # load metadata
        with open(
            os.path.join(self.artifacts_dir, "metadata.json"),
            encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        self.entities = metadata["entities"]
        self.hyperedges = metadata["hyperedges"]

        # load graph
        with open(
            os.path.join(self.artifacts_dir, "graph.gpickle"),
            "rb"
        ) as f:
            self.G = pickle.load(f)

        log.info(
            f"Updater loaded: {len(self.entities)} entities, "
            f"{len(self.hyperedges)} hyperedges"
        )

    def _compute_diff(
        self,
        new_entities: list,
        new_hyperedges: list
    ) -> tuple:
        """
        Compares new entities/hyperedges against existing ones.
        Returns only the genuinely new ones — the diff.
        """
        # O(1) lookup sets
        existing_entity_names = {e["name"] for e in self.entities}
        existing_hyperedge_facts = {h["fact"] for h in self.hyperedges}

        fresh_entities = [
            e for e in new_entities
            if e["name"] not in existing_entity_names
        ]
        fresh_hyperedges = [
            h for h in new_hyperedges
            if h["fact"] not in existing_hyperedge_facts
        ]

        log.info(
            f"Diff: {len(fresh_entities)} new entities, "
            f"{len(fresh_hyperedges)} new hyperedges "
            f"(skipped {len(new_entities) - len(fresh_entities)} duplicates)"
        )
        return fresh_entities, fresh_hyperedges

    def _update_faiss(
        self,
        fresh_entities: list,
        fresh_hyperedges: list
    ) -> None:
        """
        Appends new vectors to existing FAISS indexes.
        Never rebuilds — only appends.
        """
        if fresh_entities:
            new_entity_embeddings = encode_entities(fresh_entities)
            self.index_entity.add(new_entity_embeddings)

        if fresh_hyperedges:
            new_hyperedge_embeddings = encode_hyperedges(fresh_hyperedges)
            self.index_hyperedge.add(new_hyperedge_embeddings)

    def _update_graph(
        self,
        fresh_entities: list,
        fresh_hyperedges: list
    ) -> None:
        """
        Appends new nodes and edges to existing NetworkX graph.
        """
        # combined set for edge validation
        entity_name_set = {e["name"] for e in self.entities + fresh_entities}

        for e in fresh_entities:
            self.G.add_node(
                e["name"],
                type="entity",
                entity_id=e["id"],
                entity_type=e.get("type", "unknown")
            )

        for h in fresh_hyperedges:
            self.G.add_node(
                h["id"],
                type="hyperedge",
                fact=h["fact"]
            )
            for entity_name in h["connects"]:
                if entity_name in entity_name_set:
                    self.G.add_edge(h["id"], entity_name)

    def _save_artifacts(self) -> None:
        """
        Persists updated artifacts back to disk.
        Overwrites existing files with updated versions.
        """
        faiss.write_index(
            self.index_entity,
            os.path.join(self.artifacts_dir, "index_entity.bin")
        )
        faiss.write_index(
            self.index_hyperedge,
            os.path.join(self.artifacts_dir, "index_hyperedge.bin")
        )

        metadata = {
            "entities": self.entities,
            "hyperedges": self.hyperedges,
            "entity_count": len(self.entities),
            "hyperedge_count": len(self.hyperedges)
        }
        with open(
            os.path.join(self.artifacts_dir, "metadata.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        with open(
            os.path.join(self.artifacts_dir, "graph.gpickle"),
            "wb"
        ) as f:
            pickle.dump(self.G, f, protocol=pickle.HIGHEST_PROTOCOL)

        log.info(
            f"Artifacts updated: {len(self.entities)} entities, "
            f"{len(self.hyperedges)} hyperedges"
        )

    def update(self, input_path: str) -> dict:
        """
        Main entry point. Call this when new data arrives.
        Returns count of what was added.
        """
        log.info(f"Updating knowledge base with: {input_path}")

        # step 1 — chunk new document
        chunks, image_paths = chunk_document(input_path)

        # step 2 — extract entities and hyperedges
        new_entities, new_hyperedges = extract_entities(chunks, image_paths)

        # step 3 — compute diff
        fresh_entities, fresh_hyperedges = self._compute_diff(
            new_entities,
            new_hyperedges
        )

        # nothing new — exit early
        if not fresh_entities and not fresh_hyperedges:
            log.info("No new knowledge found — artifacts unchanged")
            return {"added_entities": 0, "added_hyperedges": 0}

        # step 4 — update FAISS
        self._update_faiss(fresh_entities, fresh_hyperedges)

        # step 5 — update NetworkX graph
        self._update_graph(fresh_entities, fresh_hyperedges)

        # step 6 — extend metadata
        self.entities.extend(fresh_entities)
        self.hyperedges.extend(fresh_hyperedges)

        # step 7 — save everything
        self._save_artifacts()

        log.info(
            f"Update complete: +{len(fresh_entities)} entities, "
            f"+{len(fresh_hyperedges)} hyperedges"
        )
        return {
            "added_entities": len(fresh_entities),
            "added_hyperedges": len(fresh_hyperedges)
        }


# =============================================================================
# REUSABLE UPDATER PATTERN — for future projects
# =============================================================================
#
# 1. load existing artifacts from disk
# 2. process new document using same pipeline as builder
# 3. compute diff — compare new vs existing by semantic identity
#    entities   → compare by name
#    hyperedges → compare by fact string
# 4. update FAISS — append new vectors only
# 5. update NetworkX — add new nodes and edges only
# 6. extend metadata lists
# 7. save artifacts back to disk
#
# key principle: never rebuild, only append
# same as React virtual DOM — diff and patch, not full re-render
#
# what changes per project:
#   identity field for deduplication (name, id, hash, url...)
#   artifact format (FAISS + JSON vs Milvus vs Neo4j)
#   everything else stays identical
# =============================================================================