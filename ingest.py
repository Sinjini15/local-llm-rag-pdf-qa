import logging
from dotenv import load_dotenv

from utils import download_pdf_if_needed,load_and_split_pdf, embed_and_store

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Config
PDF_URL = "https://www.apple.com/compliance/pdfs/Business-Conduct-Policy.pdf"
PDF_PATH = "docs/company_policy.pdf"
INDEX_DIR = "faiss_index_local"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def main():
    load_dotenv()  # Optional: use for future config like PDF_URL from .env
    download_pdf_if_needed(PDF_PATH, PDF_URL)
    chunks = load_and_split_pdf(PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    embed_and_store(chunks, EMBED_MODEL_NAME, INDEX_DIR)

if __name__ == "__main__":
    main()
