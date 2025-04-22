"""
**********************************************************************************************************************************************
*****************************************************       FILE INGESTOR       **************************************************************
**********************************************************************************************************************************************

This module provides functions to extract text from various file formats (PDF, DOCX, CSV, XLSX) and add them to a vector database.
It includes functions to generate unique IDs for documents, process files, and add them to the vector database while checking for duplicates.

The module uses the following libraries:
- os: for file path manipulation
- hashlib: for generating unique IDs
- uuid: for generating unique IDs (currently not used)
- fitz: from PyMuPDF, used for PDF extraction
- docx: for DOCX extraction (even if Document is not used, it's imported for avoiding import errors)
- pandas: for CSV and XLSX extraction
"""

import os                              
import hashlib                          
import uuid                             
import fitz                             
from docx import Document               
import pandas as pd                     



def generate_deterministic_id(text: str) -> str:
    """Generate a SHA256 hash of the document content to use as a stable ID."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_pdf_text(file_path):
    """Extract text from a PDF file and return it as a list of documents and metadata."""
    documents, metadatas = [], []
    doc = fitz.open(file_path)
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            documents.append(text)
            metadatas.append({"source": os.path.basename(file_path), "page": i + 1})
    return documents, metadatas


def extract_docx_text(file_path):
    """Extract text from a DOCX file and return it as a list of documents and metadata."""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return [text], [{"source": os.path.basename(file_path)}] if text.strip() else ([], [])


def extract_csv_text(file_path):
    """Extract text from a CSV file and return it as a list of documents and metadata."""
    df = pd.read_csv(file_path)
    documents = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
    metadatas = [{"source": os.path.basename(file_path), "row": i} for i in range(len(documents))]
    return documents, metadatas


def extract_xlsx_text(file_path):
    """Extract text from an XLSX file and return it as a list of documents and metadata."""
    docs, metas = [], []
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        for i, row in df.iterrows():
            text = " | ".join(map(str, row))
            if text.strip():
                docs.append(text)
                metas.append({"source": os.path.basename(file_path), "sheet": sheet_name, "row": i})
    return docs, metas


def process_file_for_vdb(file_path):
    """Process a file and extract text based on its type."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pdf_text(file_path)
    elif ext == ".docx":
        return extract_docx_text(file_path)
    elif ext == ".csv":
        return extract_csv_text(file_path)
    elif ext in [".xlsx", ".xls"]:
        return extract_xlsx_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    


def add_file_safely_to_vector_db(file_path, vdb_instance, collection_name=None):
    """
            ----------------------CURRENTLY NOT USED----------------------

    Add a file to the vector database after processing it and checking for duplicates.
    
    """
    print(f"\n\n📂 Processing file: {file_path}\n\n")
    documents, metadatas = process_file_for_vdb(file_path)
    print(f"\n\n📄 Extracted {len(documents)} document(s) from {file_path}.\n\n")
    for doc in documents:
        print(f"\n\nDocument: {doc[:50]}...\n\n")
    for meta in metadatas:
        print(f"\n\nMetadata: {meta}\n\n")
    
    if not documents:
        print(f"⚠️ No content extracted from {file_path}.")
        return

    print(f"\n\nGenerating Unique IDs for {len(documents)} document(s)...\n\n")
    ids = [generate_deterministic_id(doc) for doc in documents]
    print(f"\n\nGenerated IDs: {ids}\n\n")

    print(f"\n\nChecking for existing documents in the vector database...\n\n")
    existing = vdb_instance.current_collection.get(ids=ids)
    existing_ids = set(existing['ids'])

    filtered_docs, filtered_ids, filtered_metas = [], [], []
    for doc, doc_id, meta in zip(documents, ids, metadatas):
        if doc_id not in existing_ids:
            filtered_docs.append(doc)
            filtered_ids.append(doc_id)
            filtered_metas.append(meta)
    print(f"\n\nFiltered {len(filtered_docs)} new document(s) to add.\n\n")

    if not filtered_docs:
        print(f"🔁 No new documents to add from {file_path}.")
        return

    print(f"\n\nAdding {len(filtered_docs)} new document(s) to the vector database...\n\n")
    vdb_instance.add_documents(
        documents=filtered_docs,
        ids=filtered_ids,
        metadatas=filtered_metas,
        collection_name=collection_name
    )
    print(f"✅ Added {len(filtered_docs)} document(s) from {os.path.basename(file_path)}.")
