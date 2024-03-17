# imports
import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import vaex
from openai import OpenAI
import streamlit as st


import streamlit as st
import pandas as pd

import streamlit as st
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
## set offline mode for cufflinks
cf.go_offline()

# data functions
@st.cache_data
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(
        zip(df["Symbol"], df["Security"])
    )
    return tickers, tickers_companies_dict

@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start, end)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")

# sidebar

## inputs for downloading data
st.sidebar.header("Stock Parameters")

available_tickers, tickers_companies_dict = get_sp500_components()

ticker = st.sidebar.selectbox(
    "Ticker", 
    available_tickers, 
    format_func=tickers_companies_dict.get
)
start_date = st.sidebar.date_input(
    "Start date", 
    datetime.date(2019, 1, 1)
)
end_date = st.sidebar.date_input(
    "End date", 
    datetime.date.today()
)

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

## inputs for technical analysis
st.sidebar.header("Technical Analysis Parameters")

volume_flag = st.sidebar.checkbox(label="Add volume")

exp_sma = st.sidebar.expander("SMA")
sma_flag = exp_sma.checkbox(label="Add SMA")
sma_periods= exp_sma.number_input(
    label="SMA Periods", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)

exp_bb = st.sidebar.expander("Bollinger Bands")
bb_flag = exp_bb.checkbox(label="Add Bollinger Bands")
bb_periods= exp_bb.number_input(label="BB Periods", 
                                min_value=1, max_value=50, 
                                value=20, step=1)
bb_std= exp_bb.number_input(label="# of standard deviations", 
                            min_value=1, max_value=4, 
                            value=2, step=1)

exp_rsi = st.sidebar.expander("Relative Strength Index")
rsi_flag = exp_rsi.checkbox(label="Add RSI")
rsi_periods= exp_rsi.number_input(
    label="RSI Periods", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)
rsi_upper= exp_rsi.number_input(label="RSI Upper", 
                                min_value=50, 
                                max_value=90, value=70, 
                                step=1)
rsi_lower= exp_rsi.number_input(label="RSI Lower", 
                                min_value=10, 
                                max_value=50, value=30, 
                                step=1)

# main body

st.title("Stock 📈 technical📉 Analysis📊 App💹 and Chat BOT 🤖")
st.write("""
### User manual
* you can select any of the companies that is a component of the S&P index
* you can select the time period of your interest
* you can download the selected data as a CSV file
* you can add the following Technical Indicators to the plot: Simple Moving 
Average, Bollinger Bands, Relative Strength Index
* you can experiment with different parameters of the indicators
""")

df = load_data(ticker, start_date, end_date)

## data preview part
data_exp = st.expander("Preview data")
available_cols = df.columns.tolist()
columns_to_show = data_exp.multiselect(
    "Columns", 
    available_cols, 
    default=available_cols
)
data_exp.dataframe(df[columns_to_show])

csv_file = convert_df_to_csv(df[columns_to_show])
data_exp.download_button(
    label="Download selected as CSV",
    data=csv_file,
    file_name=f"{ticker}_stock_prices.csv",
    mime="text/csv",
)

## technical analysis plot
title_str = f"{tickers_companies_dict[ticker]}'s stock price"
qf = cf.QuantFig(df, title=title_str)
if volume_flag:
    qf.add_volume()
if sma_flag:
    qf.add_sma(periods=sma_periods)
if bb_flag:
    qf.add_bollinger_bands(periods=bb_periods,
                           boll_std=bb_std)
if rsi_flag:
    qf.add_rsi(periods=rsi_periods,
               rsi_upper=rsi_upper,
               rsi_lower=rsi_lower,
               showbands=True)

fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)


# Generate and display the profiling report
# profile = ProfileReport(df, explorative=True)
# st_profile_report(profile)


# Generate and display the profiling report at the end

# st.title("data 🔢🍔 shape")
# vaex_df = vaex.from_pandas(df)
# describe_output = vaex_df.describe()
# st.write(describe_output)


st.title("fin-💵💸bot 🤖ADVISOR")

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
