import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import os

# Set up Streamlit app
st.title("fin-💵💸bot 🤖ADVISOR")

# Load OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize session state for messages and model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle chat input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type='csv')
    if uploaded_file is not None:
        # Load the uploaded CSV file into a DataFrame
        data_frame = pd.read_csv(uploaded_file)
        st.write(data_frame.head())

        # Create a LangChain agent for the DataFrame
        p_agent = create_pandas_dataframe_agent(llm=client, df=data_frame, verbose=False)

        # Ask a question about the data
        response = p_agent.run("which column carries the highest value")

        # Display the response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Append the assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": response})
