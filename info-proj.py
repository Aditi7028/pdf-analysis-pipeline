# Import necessary libraries
import os
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoTokenizer
import numpy as np
import faiss
import fitz

# PDF File Ingestion
def load_pdf(uploaded_file):
    """
    Load the PDF file.
    """
    start_time = time.time()
    with open("temp_pdf_file.pdf", "wb") as f:
        f.write(uploaded_file)
    docs = fitz.open("temp_pdf_file.pdf")
    end_time = time.time()
    print(f"PDF loading time: {end_time - start_time:.2f} seconds")
    return docs

# PDF File Extraction
def extract_text(docs):
    """
    Extract text from PDF documents.
    """
    start_time = time.time()
    extracted_text = []
    for page in docs:
        extracted_text.append(page.get_text())
    end_time = time.time()
    print(f"Text extraction time: {end_time - start_time:.2f} seconds")
    return extracted_text

# Chunking function for Extracted Data
def chunk_text(extracted_text):
    """
    Split extracted text into smaller, overlapping chunks.
    """
    start_time = time.time()
    # Concatenate the list of strings into a single string
    text = " ".join(extracted_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    end_time = time.time()
    print(f"Text chunking time: {end_time - start_time:.2f} seconds")
    return splits

# Embedding for Chunked Data
def create_embeddings(splits):
    """
    Create and save embeddings for text chunks.
    """
    start_time = time.time()
    model_name = "BAAI/bge-small-en"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedded_docs = []
    for split in splits:
        inputs = tokenizer(split, return_tensors="pt")
        outputs = model(**inputs)
        embedded_docs.append(outputs.last_hidden_state[:, 0, :].detach().numpy()[0])
    end_time = time.time()
    print(f"Embedding time: {end_time - start_time:.2f} seconds")
    return embedded_docs
# Vector Store - Local
def create_local_vector_store(embedded_docs):
    """
    Create a local FAISS index and store embeddings.
    """
    start_time = time.time()
    embedded_docs = np.array(embedded_docs)
    index = faiss.IndexFlatL2(embedded_docs.shape[1])
    index.add(embedded_docs)
    faiss.write_index(index, "local_vector_store.index")
    end_time = time.time()
    print(f"Local vector store creation time: {end_time - start_time:.2f} seconds")
    return index


# Main function
def main():
    # Replace with your PDF file ingestion logic
    file_path = r"C:\Users\Dell\Downloads\book.pdf"
    with open(file_path, "rb") as f:
        uploaded_file = f.read()
    
    docs = load_pdf(uploaded_file)
    extracted_text = extract_text(docs)
    splits = chunk_text(extracted_text)
    embedded_docs = create_embeddings(splits)
    local_vector_store = create_local_vector_store(embedded_docs)
    # remote_vector_store = create_remote_vector_store(embedded_docs)

if __name__ == "__main__":
    main()
