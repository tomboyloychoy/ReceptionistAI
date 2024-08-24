# Business Assistants

A chatbot that serves as a front desk assistant for a small business environment

## Overview of the App

- Takes user queries via Streamlit's `st.chat_input` and displays both user queries and model responses with `st.chat_message` to create a meaningful conversation between the user and the business representative.
- Allow business to upload their data which will improve the correctness and customized to each business
- Uses `LlamaIndex` to load and index data and create a chat engine that will retrieve context from `TiDB VectorStore`

#### Testing Instruction ####

## Get an OpenAI API key

You can get your own OpenAI API key by following the following instructions:
1. Go to https://platform.openai.com/account/api-keys.
2. Click on the `+ Create new secret key` button.
3. Next, enter an identifier name (optional) and click on the `Create secret key` button.
4. Add your API key to your `secrets.toml` file. If you don't already have a `secrets.toml` file, add a folder named `.streamlit`, create a file called `secrets.toml` within the folder, and add the following to it:
``` openai_key = <your key here> ```
   
Alternatively, you can use [Streamlit Community Cloud's secrets management feature](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) to add your API key via the web interface.

> [!CAUTION]
> Don't commit your secrets file to your GitHub repository. The `.gitignore` file in this repo includes `.streamlit/secrets.toml` and `secrets.toml`. 

## for this hackathon, we are not sure if we should provide open ai and other key so we just add it in the code
## Try out the app

# Test Account:

# Start the app
1. Create virtual environment
py -m venv venv
2. Activate virtual environment
venv\Scripts\activate
3. Install requirements.txt
pip install -r requirements.txt
4. Run the app
steamlit run home.py
5. Test with data

5.1> Upload data
You will act as a business owner (currently hard code to Ciny Nail and Spa) to add data into the chatbot. 
- On the left hand side, choose "Document Tracker" in the drop down. 
- Sign in as the business owner:
        username: username1
        password: password
- Upload the data by drag the file into the "Upload File", or select "Choose a File".
  The test documents are in the folder "test_data".
From here, you can choose to chat with the document, or remove the document to test the remove feature.

