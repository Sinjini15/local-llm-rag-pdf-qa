import os
import logging
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA


def download_pdf_if_needed(pdf_path: str, url: str):
    """Download the PDF if it doesn't exist locally."""
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    if not os.path.exists(pdf_path):
        logging.info("PDF not found. Downloading...")
        response = requests.get(url)
        if response.status_code != 200 or "application/pdf" not in response.headers.get("Content-Type", ""):
            raise ValueError("Failed to download a valid PDF file.")
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        logging.info("Download complete.")
    else:
        logging.info("PDF already exists. Skipping download.")

def load_and_split_pdf(pdf_path: str, chunk_size: int, chunk_overlap: int):
    """Load PDF and split into chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    logging.info(f"Loaded and split {len(chunks)} document chunks.")
    return chunks

def embed_and_store(chunks, model_name: str, output_dir: str):
    """Embed the chunks and save to a FAISS vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local(output_dir)
    logging.info(f"Stored FAISS index at '{output_dir}'.")

def load_vector_store(index_path: str, embedding_model_name: str):
    """Load the FAISS vector store and embedding model."""
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = FAISS.load_local(
        folder_path=index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return vector_store

def load_llm(model_name: str):
    """Load a HuggingFace LLM pipeline for QA."""
    llm_pipeline = pipeline("text2text-generation", model=model_name, max_length=512)
    return HuggingFacePipeline(pipeline=llm_pipeline)

def run_query(query: str, vector_store, llm):
    """Execute the RAG pipeline with a given query."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})
    return result