import argparse
import time
from utils import load_llm, load_vector_store, run_query

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
    start_time = time.time()
    result = run_query(args.query, vector_store, llm)
    end_time = time.time()
    
    latency = end_time - start_time #Calculate latency

    print("\n✅ Answer:\n", result["result"])
    print("\n📄 Sources:")
    for doc in result["source_documents"]:
        print(" -", doc.metadata.get("source", "Unknown source"))
    
    print(f"\nRetrieval latency: {latency: .3f} seconds")

if __name__ == "__main__":
    main()