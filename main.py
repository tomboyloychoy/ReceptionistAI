import logging
import sys

import openai
import streamlit as st
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from sqlalchemy import URL

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
openai.api_key = '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'
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info(
    "Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)",
    icon="📃")

tidb_connection_url = URL(
    "mysql+pymysql",
    username='3gWTC3urj9AQht2.root',
    password='Eh0IB03XaZqSJZpF',
    host='gateway01.us-east-1.prod.aws.tidbcloud.com',
    port=4000,
    database="test",
    query={"ssl_verify_cert": True, "ssl_verify_identity": True},
)
tidbvec = TiDBVectorStore(
    connection_string=tidb_connection_url,
    table_name="llama_index_rag_test",
    distance_strategy="cosine",
    vector_dimension=1536,  # Length of the vectors returned by the model
    drop_existing_table=False,
)
tidb_vec_index = VectorStoreIndex.from_vector_store(tidbvec)
storage_context = StorageContext.from_defaults(vector_store=tidbvec)
query_engine = tidb_vec_index.as_query_engine(streaming=True)
logger.info("TiDB Vector Store initialized successfully")


def do_prepare_data():
    logger.info("Preparing the data for the application")
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    documents = reader.load_data()
    tidb_vec_index.from_documents(documents, storage_context=storage_context, show_progress=True)
    logger.info("Data preparation complete")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello there! How can I help you today?",
        }
    ]

def get_system_message(filename):
    with open('system_prompt/' + filename, 'r') as file:
    # Read the entire content
        content = file.read()
    return content

SYSTEM_MESSAGE = get_system_message("nail_salon_ciny_nail.txt")
@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-4o",
        temperature=1.2,
        system_prompt="You are a nail salon receptionist. When user ask about a service, generate a question about that service to get information from nail salon, and what higher service you can siggest to the customer plus why they should do it",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index


#index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = tidb_vec_index.as_chat_engine(
        chat_mode="condense_plus_context", verbose=True, streaming=True
    )

# Initialize messages if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_MESSAGE}
    ]

if prompt := st.chat_input(
        "question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
