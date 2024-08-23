import os
import time
from io import StringIO

from streamlit_js_eval import streamlit_js_eval
import streamlit as st
import logging
import sys

import openai
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from sqlalchemy import URL
import mysql.connector
from mysql.connector import MySQLConnection
from mysql.connector.cursor import MySQLCursor


# logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# openai 
openai.api_key = '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'


# connect to TiDB Vector Store
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
logger.info("Connect to TiDB Vector Store successfully")

# Connect to TiDB MySQL
def get_connection(autocommit: bool = True) -> MySQLConnection:
    db_conf = {
        "host": 'gateway01.us-east-1.prod.aws.tidbcloud.com',
        "port": 4000,
        "user": '3gWTC3urj9AQht2.root',
        "password": '02F3s7AwdQIh9dmg',
        "database": "receptionistai",
        "autocommit": autocommit,
        "use_pure": True,
    }

    if "isrgrootx1.pem":
        print("get ca")
        db_conf["ssl_verify_cert"] = True
        db_conf["ssl_verify_identity"] = True
        db_conf["ssl_ca"] = "isrgrootx1.pem"
    return mysql.connector.connect(**db_conf)

def run_mysql_query(query: str):
    result = None
    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()[0]
            cur.close()
        conn.close()
    return result

def run_mysql_query_tolist(query: str):
    result = None
    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            res = cur.fetchall()
            print("res=", res)
            result = [row[0] for row in res]
            cur.close()
        conn.close()
    return result



st.session_state.chat_system_message = 'You are an AI assitant that help user answer question' # default
st.session_state.search_system_message  = 'system message' # default






# Chat interface
def do_prepare_data():
    logger.info("Preparing the data for the application")
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    documents = reader.load_data()
    tidb_vec_index.from_documents(documents, storage_context=storage_context, show_progress=True)
    logger.info("Data preparation complete")


BUSINESS_NAME_LIST = [] # to be displayed in selection list
BUSINESS_NAME_LIST = run_mysql_query_tolist(query="SELECT DISTINCT BUSINESS_NAME FROM BUSINESS;")
# Get the list of Business Code
def select_business():
    st.title("Welcome to Our Receptionist AI Chatbot Service")
    st.write("Please select a business you would like to chat with:")

    selected_business = st.selectbox("Choose a business:", BUSINESS_NAME_LIST)


    if st.button("Proceed to Chat"):
        if selected_business:
            # Save the selected business to session state
            st.session_state.selected_business = selected_business

            # find selected business code
            st.session_state.selected_business_id = run_mysql_query(f"SELECT DISTINCT BUSINESS_ID FROM BUSINESS WHERE BUSINESS_NAME = '{st.session_state.selected_business}';")
            st.session_state.chat_system_message = run_mysql_query(f"select distinct chat_system_prompt from business where business_id = '{st.session_state.selected_business_id}';")
            st.session_state.search_system_message = run_mysql_query(f"select distinct search_system_prompt from business where business_id = '{st.session_state.selected_business_id}';")
            
            st.rerun()
        else:
            st.error("Please select a business before proceeding.")
    
def chat():

    st.title(f"Welcome to {st.session_state.selected_business}")

        # Initialize messages if not already initialized

    if "messages" not in st.session_state.keys():  # Initialize the chat messages history
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello there! How can I help you today?",
            }
        ]


    # @st.cache_resource(show_spinner=False)
    # def load_data():
    #     reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    #     docs = reader.load_data()
    #     Settings.llm = OpenAI(
    #         model="gpt-4o",
    #         temperature=1.2,
    #         system_prompt=st.session_state.search_system_message
    #         ,
    #     )
    #     index = VectorStoreIndex.from_documents(docs)
    #     return index


    #index = load_data()

    if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
        st.session_state.chat_engine = tidb_vec_index.as_chat_engine(
            chat_mode="condense_plus_context", verbose=True, streaming=True
        )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": st.session_state.chat_system_message}
        ]

    if prompt := st.chat_input(
            "Typing a question..."
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

# Upload document
def track_document():
    st.title("Document tracker")

    # Initialize session state if not already done
    if "signed_in" not in st.session_state:
        st.session_state.signed_in = False
        st.session_state.username = ""

    # If not signed in, show the sign-in form
    if not st.session_state.signed_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Sign In"):
            if username == 'username1' and password == 'password':
                st.session_state.signed_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}! You have successfully signed in!")
                time.sleep(2)
                st.rerun()  # Re-run the app to hide the sign-in box
            else:
                st.error("Incorrect username or password. Please try again.")
    else:
        st.success(f"You are already signed in as {st.session_state.username}!")

        selected_user = st.session_state.username

        st.button('Refresh file')

        if selected_user != "":

            directory_path = f"./data/{selected_user}"

            # Create the directory if it doesn't exist
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            files = os.listdir(directory_path)

            if files:
                st.subheader(f"Files uploaded by {selected_user}:")
                for file in files:
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.text(file)

                    with col2:
                        remove_button = st.button(f"Remove {file}", key=file)

                        if remove_button:
                            # Remove the file from the directory
                            file_path = os.path.join(directory_path, file)
                            os.remove(file_path)
                            st.rerun()  # Re-run the app to hide the sign-in box

            else:
                st.info(f"No files found for {selected_user}.")

            # Add a black space between the header and the username box
            st.markdown(
                """
                <style>
                .white-space {
                    height: 200px; /* Adjust the height as needed */
                    background-color: white;
                    margin-bottom: 20px; /* Space below the black space */
                }
                </style>
                <div class="white-space"></div>
                """,
                unsafe_allow_html=True
            )


            uploaded_file = st.file_uploader("Choose a file")

            if uploaded_file is not None:
                # Define the full path including the filename
                save_path = os.path.join(directory_path, uploaded_file.name)

                # Save the file to the local machine
                with open(save_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                # Notify the user that the file has been saved
                alert = st.success(f"File saved to {save_path}", icon="✅")
                time.sleep(2)
                alert.empty()


page_names_to_funcs = {
    "Chatbot": chat,
    "Document Tracker": track_document,
    "Select Business": select_business
}
if "selected_business" in st.session_state:
    demo_name = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()
else:
    select_business()