import os
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

from config import Config

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

# SYSTEM_MESSAGE = get_system_message("nail_salon_ciny_nail.txt")
with get_connection(autocommit=True) as conn:
    with conn.cursor() as cur:
        cur.execute("select distinct chat_system_prompt from business where business_id = 'CINYNAIL';")
        CHAT_SYSTEM_MESSAGE = cur.fetchone()[0]
        cur.execute("select distinct search_system_prompt from business where business_id = 'CINYNAIL';")
        SEARCH_SYSTEM_MESSAGE = cur.fetchone()[0]
        cur.close()


# Chat interface
def do_prepare_data():
    logger.info("Preparing the data for the application")
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    documents = reader.load_data()
    tidb_vec_index.from_documents(documents, storage_context=storage_context, show_progress=True)
    logger.info("Data preparation complete")
def intro():
    if "messages" not in st.session_state.keys():  # Initialize the chat messages history
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello there! How can I help you today?",
            }
        ]


    @st.cache_resource(show_spinner=False)
    def load_data():
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        Settings.llm = OpenAI(
            model="gpt-4o",
            temperature=1.2,
            system_prompt=SEARCH_SYSTEM_MESSAGE,
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
            {"role": "system", "content": CHAT_SYSTEM_MESSAGE}
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

# Upload document

def mapping_demo():


    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
to display geospatial data.
"""
    )
    # List of user IDs with a placeholder as the first option
    user_ids = ['-- Select a User ID --', 'User1', 'User2', 'User3', 'User4']

    # Dropdown for user selection
    selected_user = st.selectbox("Select your User ID", user_ids, index=0)

    # Check if a valid user is selected (i.e., not the placeholder)
    if selected_user != '-- Select a User ID --':
        # Define the directory path where the file will be saved
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

        else:
            st.info(f"No files found for {selected_user}.")

        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
            # Define the full path including the filename
            save_path = os.path.join(directory_path, uploaded_file.name)

            # Save the file to the local machine
            with open(save_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            # Notify the user that the file has been saved
            st.success(f"File saved to {save_path}")
    else:
        st.info("Please select a User ID to upload a file.")

        # # To read file as bytes:
        # bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)
        #
        # # To convert to a string based IO:
        # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # st.write(stringio)
        #
        # # To read file as string:
        # string_data = stringio.read()
        # st.write(string_data)
        #
        # # Can be used wherever a "file-like" object is accepted:
        # dataframe = pd.read_csv(uploaded_file)
        # st.write(dataframe)


def plotting_demo():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!
"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


def data_frame_demo():
    import streamlit as st
    import pandas as pd
    import altair as alt

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo shows how to use `st.write` to visualize Pandas DataFrames.

(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
"""
    )

    @st.cache_data
    def get_UN_data():
        AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
        df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
        return df.set_index("Region")

    try:
        df = get_UN_data()
        countries = st.multiselect(
            "Choose countries", list(df.index), ["China", "United States of America"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write("### Gross Agricultural Production ($B)", data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )


page_names_to_funcs = {
    "Welcome": intro,
    "Plotting Demo": plotting_demo,
    "Mapping Demo": mapping_demo,
    "DataFrame Demo": data_frame_demo
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
