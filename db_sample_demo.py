"""
This script demonstrates how to use the VectorDB class to create collections and add files to them.
It's ment to be run as a standalone script, just one time, in order to create the collections and add the files.
"""

# import for the vector database wrapper
from chroma_db_wrapper.db_wrapper import VectorDB

# import to extract text from files
from chroma_db_wrapper.file_ingestor import process_file_for_vdb

vector_db = VectorDB()

vector_db.create_collection(collection_name="NLP_collection", description="This is a collection containing a NLP book.")

vector_db.add_file(file_path="/home/kevin/Desktop/Speech_and_Language_Processing.pdf", collection_name="NLP_collection")      # if collection_name is not provided, the current collection will be used

vector_db.create_collection(collection_name="RL_collection", description="This is a collection containing a RL book.")
vector_db.add_file(file_path="/home/kevin/Desktop/reinforcement_learning.pdf", collection_name="RL_collection")      

# takes approximately 2 minutes for loading 2 pdf books with ~500 pages each (good result)