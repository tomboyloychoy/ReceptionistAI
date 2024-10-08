import logging
import sys

import openai
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from sqlalchemy import URL
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


openai.api_key = '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'


tidb_connection_url = URL(
    "mysql+pymysql",
    username='3gWTC3urj9AQht2.root',
    password='02F3s7AwdQIh9dmg',
    host='gateway01.us-east-1.prod.aws.tidbcloud.com',
    port=4000,
    database="receptionistai",
    query={"ssl_verify_cert": True, "ssl_verify_identity": True},
)
tidbvec = TiDBVectorStore(
    connection_string=tidb_connection_url,
    table_name="data",
    distance_strategy="cosine",
    vector_dimension=1536,  # Length of the vectors returned by the model
    drop_existing_table=False,
)
tidb_vec_index = VectorStoreIndex.from_vector_store(tidbvec)
storage_context = StorageContext.from_defaults(vector_store=tidbvec)
query_engine = tidb_vec_index.as_query_engine(streaming=True)
logger.info("TiDB Vector Store initialized successfully")

def get_meta(file_path):
    return {"foo": "bar", "file_path": file_path}

def do_prepare_data_from_directory(directory_path):
    logger.info("Preparing the data for the application")
    reader = SimpleDirectoryReader(input_dir=directory_path, recursive=True, filename_as_id=True)
    documents = reader.load_data()

    # add metadata
    for document in documents:
        document.metadata = {"business_id": "CINYNAIL", "business_name": "Ciny Nails and Spa", "file_name": document.id_}
    tidb_vec_index.from_documents(documents, storage_context=storage_context, show_progress=True)
    logger.info("Data preparation complete")

def do_prepare_data_from_file(file_path):
    logger.info("Preparing the data for the application")
    with open(file_path, 'r') as file:
        content = file.read()
        #print(content)
    document = Document(text=content)
    file_name = os.path.basename(file_path)
    # add metadata

    document.metadata = {"business_id": "CINYNAIL", "business_name": "Ciny Nails and Spa", "file_name": file_path}
    tidb_vec_index.from_documents([document],storage_context=storage_context, show_progress=True)
    logger.info("Data preparation complete")


# do_prepare_data_from_directory("./data")
# do_prepare_data_from_file("data\Service - Acrylic.txt")