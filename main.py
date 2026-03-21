import argparse
import logging
from graph.builder import build
from agent.retriever import Retriever
from agent.controller import Controller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

def run_build(input_path: str, save_dir: str = "artifacts"):
    log.info(f"Build mode — input: {input_path}")
    build(input_path, save_dir)

def run_query(question: str, artifacts_dir: str = "artifacts"):
    log.info(f"Query mode — question: {question}")
    retriever = Retriever(artifacts_dir)
    controller = Controller(max_turns=3)
    answer = controller.run(question, retriever)
    print("\n" + "="*50)
    print("ANSWER:")
    print("="*50)
    print(answer)
    print("="*50 + "\n")

def run_update(input_path: str, artifacts_dir: str = "artifacts"):
    from graph.updater import Updater
    log.info(f"Update mode — input: {input_path}")
    updater = Updater(artifacts_dir)
    result = updater.update(input_path)
    print(f"\nUpdate complete: +{result['added_entities']} entities, +{result['added_hyperedges']} hyperedges\n")

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Hypergraph RAG System"
    )
    parser.add_argument(
        "--mode",
        choices=["build", "query", "update"],
        required=True,
        help="build: index a document | query: ask a question | update: incrementally update artifacts"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="path to document (required for build mode)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="question to ask (required for query mode)"
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="artifacts",
        help="directory to save/load artifacts (default: artifacts)"
    )

    args = parser.parse_args()

    if args.mode == "build":
        if not args.input:
            parser.error("--input is required for build mode")
        run_build(args.input, args.artifacts)

    elif args.mode == "query":
        if not args.question:
            parser.error("--question is required for query mode")
        run_query(args.question, args.artifacts)

    elif args.mode == "update":
        if not args.input:
            parser.error("--input is required for update mode")
        run_update(args.input, args.artifacts)

if __name__ == "__main__":
    main()