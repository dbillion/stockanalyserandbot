
import streamlit as st
import pandas as pd

import streamlit as stconda
import os
import tempfile
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI

# Load OpenAI API key from Streamlit secrets
llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"], temperature=0)

# Set up Streamlit app
st.title("CSV Interpreter Chatbot")

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type='csv')
if uploaded_file is not None:
    # Load the uploaded CSV file into a DataFrame
    data_frame = pd.read_csv(uploaded_file)
    st.write(data_frame.head())

    # Create a LangChain agent for the DataFrame
    p_agent = create_pandas_dataframe_agent(llm=llm, df=data_frame, verbose=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Ask a question about the data
        response = p_agent.run(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


# Set up Streamlit app
st.title("PDF Interpreter Chatbot")

# Load OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Step 1: Upload PDF file
uploaded_file = st.file_uploader("Upload your PDF file", type='pdf')
if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_file_name = tmp.name

    # Load document
    loader = PyPDFLoader(tmp_file_name)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Select which embeddings we want to use
    embeddings = OpenAIEmbeddings()

    # Create the vectorstore to use as the index
    db = Chroma.from_documents(texts, embeddings)

    # Expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

    # Create a chain to answer questions 
    qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        role, text = message
        with st.chat_message(role):
            st.markdown(text)

    # Step 2: Ask a question about the data
    prompt = st.text_input("What would you like to know about the data?")
    if prompt:
        # Ask a question about the data
        result = qa({"question": prompt, "chat_history": st.session_state.chat_history})

        # Display the user's question
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display the bot's response
        with st.chat_message("assistant"):
            st.markdown(result["answer"])

        # Update chat history
        st.session_state.chat_history.append(("user", prompt))
        st.session_state.chat_history.append(("assistant", result["answer"]))
