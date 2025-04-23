import argparse
from utils import load_llm
from utils import load_vector_store
from utils import run_query

def main():
    parser = argparse.ArgumentParser(description="RAG QA over PDF docs using local models.")
    parser.add_argument("query", type=str, help="Your natural language question.")
    args = parser.parse_args()

    # Config
    INDEX_PATH = "faiss_index_local"
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "google/flan-t5-base"

    print(f"[INFO] Loading models...")
    vector_store = load_vector_store(INDEX_PATH, EMBED_MODEL)
    llm = load_llm(LLM_MODEL)

    print(f"[INFO] Processing query: {args.query}")
    result = run_query(args.query, vector_store, llm)

    print("\n✅ Answer:\n", result["result"])
    print("\n📄 Sources:")
    for doc in result["source_documents"]:
        print(" -", doc.metadata.get("source", "Unknown source"))

if __name__ == "__main__":
    main()