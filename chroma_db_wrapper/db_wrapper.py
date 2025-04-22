"""
**********************************************************************************************************************************************
*****************************************************        DB WRAPPER        ***************************************************************
**********************************************************************************************************************************************

This module provides a wrapper around the ChromaDB vector database, allowing for easy management of collections and documents.
It includes functionality for creating, loading, and deleting collections, as well as adding and retrieving documents.
It also includes logging functionality to track operations performed on the database.
"""

from typing import List
import os
import shutil
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from .file_ingestor import process_file_for_vdb, generate_deterministic_id
from .logger_factory import setup_logger, log_operation



DEFAULT_CHROMA_DB = "./default_chroma_db"
DEFAULT_COLLECTION_NAME = "default_collection"
#DEFAULT_EBBEDING_FUNCTION = OllamaEmbeddings(model="llama3.2")
DEFAULT_EBBEDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")     # default embedding function for ChromaDB



class ChromaEmbeddingWrapper:
    """
    Wrapper class for embedding functions to be used with ChromaDB.
    This class is used to ensure that the embedding function is compatible with ChromaDB.
    """
    def __init__(self, langchain_embedding):
        self.langchain_embedding = langchain_embedding

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.langchain_embedding.embed_documents(input)



class VectorDB:
    def __init__(self, db_location: str = DEFAULT_CHROMA_DB, embedding_function: object = DEFAULT_EBBEDING_FUNCTION, init_collection_name: str = DEFAULT_COLLECTION_NAME, load_db: bool = False, verbose: bool = True):
        """
        Initialize the VectorDB class.

        Args:
            db_location (str): Directory path for persistent Chroma DB (defaults to DEFAULT_CHROMA_DB).
            embedding_function (object): Embedding function to use (defaults to DEFAULT_EBBEDING_FUNCTION).
            init_collection_name (str): Name of the initial collection (defaults to DEFAULT_COLLECTION_NAME).
            load_db (bool): Flag to load an existing database (defaults to False).
            
        """
        self.db_location = db_location
        self.embedding_function = ChromaEmbeddingWrapper(embedding_function) if isinstance(embedding_function, OllamaEmbeddings) else embedding_function
        self.default_collection_name = init_collection_name
        self.current_collection_name = init_collection_name

        # Set up logging
        self.logger = setup_logger(name=self.db_location[2:], console_output=verbose)
        self.logger.info(f"Initializing VectorDB with db_location: {self.db_location}, embedding_function: {self.embedding_function.__class__.__name__}, collection_name: {self.default_collection_name}")

        # Attrubutes for Chroma client and collection (setted by create_vector_store or load_vector_store)
        self.current_collection = None
        self.client = None
        self.collection_names = None
        self.embedding_info = None
        self.create_vector_store() if not load_db else self.load_vector_store()


    @log_operation
    def create_vector_store(self):
        """
        Create a persistent vector store.   
        """
        self.logger.info(f"Creating vector store at {self.db_location} with collection name: {self.default_collection_name}")
        # Check if the database directory already exists
        if os.path.exists(self.db_location):
            log = f"Database path {self.db_location} already exists. It could mean that the database is already created. If you want to load an existing database, set load_db=True."
            self.logger.error(log)
            raise ValueError(log)
        
        self.logger.info(f"Creating a new client for the vector store at {self.db_location}")
        self.client = chromadb.PersistentClient(path=self.db_location, settings=Settings(anonymized_telemetry=False))

        # Attempt to capture info about the embedding function (class name, model name if available)
        if hasattr(self.embedding_function, 'model'):
            self.embedding_info = "embedding_class: " + self.embedding_function.__class__.__name__ + ", model: " + self.embedding_function.model
        else:
            self.embedding_info = "embedding_class: " + str(self.embedding_function.__class__)

        if self.default_collection_name in self.get_collection_names():
            log = f"Collection '{self.default_collection_name}' already exists. Please choose a different name."
            self.logger.error(log)
            raise ValueError(log)
        
        self.logger.info(f"Creating collection '{self.default_collection_name}' with embedding function: {self.embedding_function.__class__.__name__}")
        self.current_collection = self.client.create_collection(
            name=self.default_collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "name": self.default_collection_name,
                "description": "Default collection.",
                "created": str(datetime.now()),
                "embedding_info": self.embedding_info
            }
        )

        self.collection_names = self.get_collection_names()
        self.logger.info(f"Collection names updated: {self.collection_names}")


    @log_operation
    def load_vector_store(self):
        """
        Load an existing vector store.
        """
        self.logger.info(f"Loading vector store from {self.db_location} with collection name: {self.default_collection_name}")
        if not os.path.exists(self.db_location):
            log = f"Database path {self.db_location} does not exist."
            self.logger.error(log)
            raise ValueError(log)
        
        if hasattr(self.embedding_function, 'model'):
            self.embedding_info = "embedding_class: " + self.embedding_function.__class__.__name__ + ", model: " + self.embedding_function.model
        else:
            self.embedding_info = "embedding_class: " + str(self.embedding_function.__class__)

        self.logger.info(f"Creating a new client for the vector store at {self.db_location}")
        self.client = chromadb.PersistentClient(path=self.db_location, settings=Settings(anonymized_telemetry=False))

        self.logger.info(f"Retrieving collection {self.default_collection_name} from the vector store")
        self.current_collection = self.client.get_collection(name=self.default_collection_name, embedding_function=self.embedding_function)

        self.collection_names = self.get_collection_names()
        self.logger.info(f"Collection names updated: {self.collection_names}")



    def get_collection_names(self) -> List[str]:
        """
        List all existing Chroma collection names.

        Returns:
            List of collection names.
        """
        collections =  self.client.list_collections()
        collection_names = []
        for collection in collections:
            collection_names.append(collection.name)
        return collection_names
    


    def get_current_collection_metadata(self) -> dict:
        """
        Get metadata of the current collection.

        Returns:
            Metadata of the current collection.
        """
        if self.current_collection is None:
            raise ValueError("No current collection set.")
        
        return self.current_collection.metadata
    

    @log_operation
    def switch_collection(self, collection_name: str = None):
        """
        Switch to a different collection.

        Args:
            collection_name (str): Name of the collection to switch to.
        """
        self.logger.info(f"Switching to collection '{collection_name}'")

        if collection_name is None:
            log = "Collection name must be provided."
            self.logger.error(log)
            raise ValueError(log)
        
        if collection_name not in self.collection_names:
            log = f"Collection '{collection_name}' does not exist."
            self.logger.error(log)
            raise ValueError(log)
        
        self.logger.info(f"Retrieving collection '{collection_name}' from the vector store")
        self.current_collection = self.client.get_collection(name=collection_name, embedding_function=self.embedding_function)
        self.current_collection_name = collection_name


    @log_operation
    def create_collection(self, collection_name: str = None, description: str = None):
        """
        Add a new collection to the vector store and switch to it.

        Args:
            collection_name (str): Name of the new collection.
        """
        self.logger.info(f"Creating collection '{collection_name}' with description: {description}")

        if collection_name is None:
            log = "Collection name must be provided."
            self.logger.error(log)
            raise ValueError(log)
        
        if collection_name in self.collection_names:
            log = f"Collection '{collection_name}' already exists."
            self.logger.error(log)
            raise ValueError(log)
        
        self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "name": collection_name,
                "description": f"{description}" if description is not None else "No description provided.",
                "created": str(datetime.now()),
                "embedding_info": self.embedding_info

            }
        )

        self.collection_names = self.get_collection_names()
        self.logger.info(f"Collection names updated: {self.collection_names}")

        self.switch_collection(collection_name)


    @log_operation
    def delete_collection(self, collection_name: str = None):
        """
        Delete a specified collection from the vector store.
        """
        self.logger.info(f"Deleting collection '{collection_name}'")

        if collection_name is None:
            log = "Delete fail. Collection name must be provided to delete."
            self.logger.error(log)
            raise ValueError(log)
        
        try:
            self.client.delete_collection(name=collection_name)
        except Exception as e:
            log = f"Warning: Could not delete collection '{collection_name}': {e}"
            self.logger.warning(log)
            print(f"Warning: Could not delete collection '{collection_name}': {e}")

        self.collection_names = self.get_collection_names()
        self.logger.info(f"Collection names updated: {self.collection_names}")

        if self.current_collection_name == collection_name:
            if self.default_collection_name in self.collection_names:
                self.switch_collection(self.default_collection_name)
            elif len(self.collection_names) > 0:
                self.switch_collection(self.collection_names[0])
            else:
                log = "No collections available. Resetting current collection to None."
                self.logger.warning(log)
                print(log)
                self.current_collection_name = None
                self.current_collection = None


    @log_operation
    def delete_current_collection(self, new_current_collection_name: str = None):
        """
        Delete the current collection from the vector store.
        """
        self.logger.info(f"Deleting current collection '{self.current_collection_name}'")

        if self.current_collection_name is None:
            log = "No current collection to delete."
            self.logger.error(log)
            raise ValueError(log)

        self.delete_collection(self.current_collection_name)

        if new_current_collection_name is not None:
            self.switch_collection(new_current_collection_name)


    @log_operation
    def delete_all_collections(self):
        """
        Delete all collections in the vector store.
        """
        self.logger.info("Deleting all collections in the vector store")

        if len(self.collection_names) != 0:
            for collection_name in self.collection_names:
                try:
                    self.client.delete_collection(name=collection_name)
                    self.logger.info(f"Deleted collection '{collection_name}'")
                except Exception as e:
                    log = f"Warning: Could not delete collection '{collection_name}': {e}"
                    self.logger.warning(log)
                    print(log)

        self.logger.info("All collections deleted.")
        self.logger.info("Resetting current collection and current collection name to None.")
        self.logger.info("Resetting collection names to empty list.")
        self.collection_names = []
        self.current_collection_name = None
        self.current_collection = None


    @log_operation
    def delete_vector_store(self):
        """
        Clean up the vector store and remove the database directory.
        """
        self.logger.info(f"Deleting vector store at {self.db_location}")

        self.delete_all_collections()

        self.logger.info(f"Deleting database directory at {self.db_location}")
        if os.path.exists(self.db_location):
            shutil.rmtree(self.db_location)
        self.collection_names = None


    @log_operation
    def add_documents(self, documents: List[str], ids: List[str] = None, metadatas: List[dict] = None, collection_name: str = None):
        """
        Add text documents to the vector store (in this context, a document is a string).

        Args:
            documents (list): List of strings.
            ids (list): Optional list of unique document IDs.
            metadatas (list): Optional list of metadata dictionaries.
            collection_name (str): Name of the collection to add documents to.
        """
        self.logger.info(f"Adding {len(documents)} document(s) to the vector store")

        if collection_name is not None:
            if collection_name in self.collection_names:
                self.switch_collection(collection_name)
            else:
                log = f'Error: Collection "{collection_name}" does not exist.'
                self.logger.error(log)
                raise ValueError(log)
        else:
            if self.current_collection_name is None:
                self.switch_collection(self.default_collection_name)
        #"""
        #Check if the documents are already in the collection
        self.logger.info(f"Checking if the documents already exist in the collection: {self.current_collection_name}")
        query = self.current_collection.query(
            query_texts=documents,
            n_results=1,
        )   
        if query['documents'][0] != []:
            log = "Documents already exist in the collection."
            self.logger.error(log)
            raise ValueError(log)

        # Check if the IDs are already in the collection
        self.logger.info(f"Checking if the IDs already exist in the collection: {self.current_collection_name}")
        query_by_ids = self.current_collection.get(
            ids=ids,
        )
        if query_by_ids['documents'] != []:
            log = "IDs already exist in the collection."
            self.logger.error(log)
            raise ValueError(log)
        #"""
        self.logger.info(f"Adding {len(documents)} document(s) to the collection: {self.current_collection_name}")
        self.current_collection.add(
            documents = documents,
            ids = ids,
            metadatas = metadatas,
        )



    @log_operation
    def add_file(self, file_path: str, collection_name: str = None):
        """
        Add a file to the vector store.

        Args:
            file_path (str): Path to the file.
            collection_name (str): Name of the collection to add the file to.
        """
        self.logger.info(f"Adding file '{file_path}' to the vector store")

        if collection_name is not None:
            if collection_name in self.collection_names:
                self.switch_collection(collection_name)
            else:
                log = f'Error: Collection "{collection_name}" does not exist.'
                self.logger.error(log)
                raise ValueError(log)
        else:
            if self.current_collection_name is None:
                self.switch_collection(self.default_collection_name)

        self.logger.info(f"Processing file '{file_path}'")
        documents, metadatas = process_file_for_vdb(file_path)

        if not documents:
            log = f"No content extracted from {file_path}."
            self.logger.warning(log)
            print(log)
            return
        
        self.logger.info("Generating unique IDs for documents")
        ids = [generate_deterministic_id(doc) for doc in documents]

        self.logger.info(f"Checking by IDs for existing documents in the vector store")
        existing = self.current_collection.get(ids=ids)
        existing_ids = set(existing['ids'])

        filtered_docs, filtered_ids, filtered_metas = [], [], []
        for doc, doc_id, meta in zip(documents, ids, metadatas):
            if doc_id not in existing_ids:
                filtered_docs.append(doc)
                filtered_ids.append(doc_id)
                filtered_metas.append(meta)

        if not filtered_docs:
            log = f"No new documents to add from {file_path}."
            self.logger.info(log)
            print(log)
            return

        self.add_documents(
            documents=filtered_docs,
            ids=filtered_ids,
            metadatas=filtered_metas,
            collection_name=collection_name
    )

    @log_operation
    def retrieve_documents(self, query_texts: list[str], k: int = 5, collection_name: str = None, metadata_search: dict = None, string_search: dict = None, id_search: bool = False, ids: list[str] = None) -> List[Document]: 
        """
        Retrieve top-k similar documents from the vector store.

        Args:
            query (str): The search query.
            k (int): Number of results to return.
            collection_name (str): Name of the collection to search in.
            metadata_search (dict): Optional metadata filter. e.g. {"metadata_field_name": "metadata_value"}
            string_search (dict): Optional string filter. e.g. {"$contains":"search_string"}

        Returns:
            List of Document objects.
        """
        self.logger.info(f"Retrieving documents from the vector store")

        if collection_name is not None:
            if collection_name in self.collection_names:
                self.switch_collection(collection_name)
            else:
                log = f'Error: Collection "{collection_name}" does not exist.'
                self.logger.error(log)
                raise ValueError(log)
        else:
            if self.current_collection_name is None:
                self.switch_collection(self.default_collection_name)
    
        if id_search is False:
            self.logger.info(f"Performing a standard similarity search with query: {query_texts}, k: {k}")
            results = self.current_collection.query(
                query_texts=query_texts,
                n_results=k,
                where = metadata_search,
                where_document=string_search
            )
        else:
            if ids is None:
                log = "IDs must be provided for ID search."
                self.logger.error(log)
                raise ValueError(log)
            
            self.logger.info(f"Performing an ID-based search with IDs: {ids}, k: {k}")
            results = self.current_collection.get(
                ids=ids,
                where = metadata_search,
                where_document=string_search
            )

        documents = []

        # For a standard similarity search using query_texts
        self.logger.info(f"Using results from the query to create output Document objects")
        if id_search is False:
            # Chroma returns a dict with keys like: "documents", "ids", "metadatas"
            # The results['documents'] from query() is a list of lists (one sublist per query), so you're accessing [0] to get the first query result. 

            # Handling multiple queries 
            for i in range(len(query_texts)):
                for doc_text, metadata in zip(results['documents'][i], results['metadatas'][i]):
                    if metadata is None:
                        metadata = {}
                    documents.append(Document(page_content=doc_text, metadata=metadata))

        # For direct ID-based search
        else:
            for doc_text, metadata in zip(results['documents'], results['metadatas']):
                if metadata is None:
                    metadata = {}
                documents.append(Document(page_content=doc_text, metadata=metadata))

        return documents
    



