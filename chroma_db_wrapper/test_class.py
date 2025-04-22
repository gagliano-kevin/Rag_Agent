"""
**********************************************************************************************************************************************
*****************************************************        TEST CLASS        ***************************************************************
**********************************************************************************************************************************************

This module provides a test for the VectorDB class, which is a wrapper around the ChromaDB vector database.
It includes:
- test_add_file: Tests the add_file_safely_to_vector_db function and the add_file method of the VectorDB class.
- test_vector_db: Tests the VectorDB class, including creating collections, adding documents, performing similarity searches, and deleting collections.
- probe_embedding_time: Tests the embedding time of the Ollama and Chroma default embeddings.
"""

from db_wrapper import VectorDB
from file_ingestor import add_file_safely_to_vector_db
from langchain_ollama import OllamaEmbeddings
from chromadb.utils import embedding_functions

import time



def test_add_file(add_safely=False):
    """
    Test the add_file_safely_to_vector_db function
    and the add_file method of the VectorDB class.
    """

    # Initialize vector DB - create a new vector store
    print("Creating vector DB...")
    db = VectorDB(db_location="./test_add_chroma_db", load_db=False, verbose=True)
    print("Vector DB created.\n\n")

    # Add file to the vector DB
    if add_safely:
        print("Adding file to the vector DB safely...")
        add_file_safely_to_vector_db("sample.pdf", db)
        print("File added safely.\n\n")
    else:
        print("Adding file to the vector DB...")
        db.add_file("sample.pdf")
        print("File added.\n\n")



def test_vector_db(test_multiple_add=False):
    """
    Test the VectorDB class.
    """

    # Initialize vector DB - create a new vector store
    print("Creating vector DB...")
    db = VectorDB(db_location="./test_vector_db", load_db=False)
    print("Vector DB created.\n\n")

    # Printing the collection info
    print("Collection info:")
    print("Current collection name: ", db.current_collection_name)
    print("Current collection metadata: ", db.get_current_collection_metadata())
    print("Collection names: ", db.collection_names)
    print("Embedding info: ", db.embedding_info, "\n\n")

    # Add 10 documents to the current collection
    print("Creating 10 documents to the current collection: ", db.current_collection_name)
    docs = ["Document " + str(i) for i in range(10)]
    ids = ["doc" + str(i) for i in range(10)]
    metadatas = [{"metadata": "metadata_" + str(i)} for i in range(10)]

    # Print documents, ids, and metadatas
    print("Listing documents, ids, and metadatas:")
    print("Documents: ", docs)
    print("IDs: ", ids)
    print("Metadatas: ", metadatas)
    print("\n\n")

    # Add documents to the current collection
    print("Adding documents to the current collection...")
    db.add_documents(docs, ids=ids, metadatas=metadatas)
    print("Documents added.\n\n")

    # Perform a similarity search
    print("Performing a similarity search...")
    query_texts = ["Document 0"]
    k = 5
    print("Query texts: ", query_texts, ", k: ", k)
    print("Retrieving documents...")
    results = db.retrieve_documents(query_texts, k=k)
    print("Documents retrieved.\n\n")

    # Print the results
    print("Results:")
    print("Number of results: ", len(results))
    for doc in results:
        print(f"Document: {doc.page_content}\nMetadata: {doc.metadata}\n{'-' * 40}")

    # Test multiple add of the same documents (it should fail)
    if test_multiple_add:
        print("\n\n\nTesting multiple add of the same documents...")
        try:
            db.add_documents(docs, ids=ids, metadatas=metadatas)
        except Exception as e:
            print("Failed to add the same documents again: ", e)
            print(f"\n\n\n{'*' * 80} TEST PASSED: Documents not added again. {'*' * 80}\n\n")
        else:
            print(f"\n\n\n{'*' * 80} TEST FAILED: Documents added again. {'*' * 80}\n\n")



    
    # Test creating a new collection
    print("\n\nCreating a new collection...")
    db.create_collection("new_collection", description="New collection for testing")
    print("New collection created.")
    print("Automatically switched to the new collection.")
    print("Collection info:")
    print("Current collection name: ", db.current_collection_name)
    print("Current collection metadata: ", db.get_current_collection_metadata())
    print("Collection names: ", db.collection_names)
    print("Embedding info: ", db.embedding_info, "\n\n")
    
    # Add 20 documents to the new collection
    print("Creating 20 documents to the new collection: ", db.current_collection_name)
    docs = ["New Document " + str(i) for i in range(20)]
    ids = ["new_doc" + str(i) for i in range(20)]
    metadatas = [{"metadata": "new_metadata_" + str(i)} for i in range(20)]

    # Print documents, ids, and metadatas
    print("Listing documents, ids, and metadatas:")
    print("Documents: ", docs)
    print("IDs: ", ids)
    print("Metadatas: ", metadatas)
    print("\n\n")

    # Add documents to the new collection
    print("Adding documents to the new collection...")
    db.add_documents(docs, ids=ids, metadatas=metadatas)
    print("Documents added.\n\n")

    # Perform a similarity search
    print("Performing a similarity search...")
    query_texts = ["New Document 0"]
    k = 10
    print("Query texts: ", query_texts, ", k: ", k)
    print("Retrieving documents...")
    results = db.retrieve_documents(query_texts, k=k)
    print("Documents retrieved.\n\n")

    # Print the results
    print("Results:")
    print("Number of results: ", len(results))
    for doc in results:
        print(f"Document: {doc.page_content}\nMetadata: {doc.metadata}\n{'-' * 40}")

    # Perform a metadata search
    k = 7
    print("\n\nPerforming a metadata search...")
    metadata_search = {"metadata": "new_metadata_0"}
    print("Metadata search: ", metadata_search, ", k: ", k)
    print("Retrieving documents...")
    results = db.retrieve_documents(query_texts, k=k, metadata_search=metadata_search)
    print("Documents retrieved.\n\n")

    # Print the results
    print("Number of results: ", len(results))
    print("Results:")
    for doc in results:
        print(f"Document: {doc.page_content}\nMetadata: {doc.metadata}\n{'-' * 40}")

    # Perform a string search
    print("\n\nPerforming a string search...")
    string_search = {"$contains": "New Document"}
    print("String search: ", string_search, ", k: ", k)
    print("Retrieving documents...")
    results = db.retrieve_documents(query_texts, k=5, string_search=string_search)
    print("Documents retrieved.\n\n")

    # Print the results
    print("Number of results: ", len(results))
    print("Results:")
    for doc in results:
        print(f"Document: {doc.page_content}\nMetadata: {doc.metadata}\n{'-' * 40}")

    
    # Perform an ID search
    print("\n\nPerforming an ID search...")
    ids = ["new_doc0", "new_doc1"]
    print("IDs: ", ids, ", k: ", k)
    print("Retrieving documents...")
    results = db.retrieve_documents(query_texts, k=5, id_search=True, ids=ids)
    print("Documents retrieved.\n\n")

    # Print the results
    print("Number of results: ", len(results))
    print("Results:")
    for doc in results:
        print(f"Document: {doc.page_content}\nMetadata: {doc.metadata}\n{'-' * 40}")

    # Delete the current collection
    print("\n\nDeleting the current collection...")
    db.delete_current_collection()
    print("Current collection deleted.")
    print("Collection names: ", db.collection_names)
    print("Current collection name: ", db.current_collection_name)
    print("Current collection metadata: ", db.get_current_collection_metadata())

    # Delete all collections
    print("\n\nDeleting all collections...")
    db.delete_all_collections()
    print("All collections deleted.")
    print("Collection names: ", db.collection_names)
    print("Current collection name: ", db.current_collection_name)

    # Delete the vector store
    print("\n\nDeleting the vector store...")
    db.delete_vector_store()
    print("Vector store deleted.")
    print("Collection names: ", db.collection_names)
    print("Current collection name: ", db.current_collection_name)


def probe_embedding_time():
    # ---------------------------- Test with Ollama embedding ----------------------------
    print(f"{'*' * 80} OLLAMA EMBEDDING TEST {'*' * 80} \n\n")
    embedder = OllamaEmbeddings(model="llama3.2")

    # Creating a dummy document
    partial_text= "This is a single page of text with numbers 456789123 and characters, this text is used in order to check if the llama3.2 embedder could cause possible delays..."
    test_doc = 10 * partial_text
    doc_len = len(test_doc)
    print(f"Document length: {doc_len} characters")
    start_time = time.time()
    # Embed the document
    embedding = embedder.embed_documents([test_doc])
    end_time = time.time()
    embedding_time = end_time - start_time
    print(f"Embedding time: {embedding_time:.4f} seconds")

    # print the avg time per character
    avg_time_per_char = embedding_time / doc_len
    print(f"Average time per character: {avg_time_per_char:.4f} seconds")

    # ---------------------------- Test with Chroma default embedding ----------------------------
    print(f"\n\n{'*' * 80} CHROMA DEFAULT EMBEDDING TEST {'*' * 80} \n\n")
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # Use a sample from your PDF
    partial_text= "This is a single page of text with numbers 456789123 and characters, this text is used in order to check if the llama3.2 embedder could cause possible delays..."
    test_doc = 10 * partial_text
    doc_len = len(test_doc)
    print(f"Document length: {doc_len} characters")
    start_time = time.time()
    # Embed the document
    embedding = embedder([test_doc])
    end_time = time.time()
    embedding_time = end_time - start_time
    print(f"Embedding time: {embedding_time:.4f} seconds")

    # print the avg time per character
    avg_time_per_char = embedding_time / doc_len
    print(f"Average time per character: {avg_time_per_char:.4f} seconds")


if __name__ == "__main__":
    #test_vector_db()
    test_vector_db(test_multiple_add=True)