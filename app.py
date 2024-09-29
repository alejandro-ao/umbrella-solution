from langchain_groq import ChatGroq
from dotenv import load_dotenv
from prompts import ENGIE_SYSTEM_PROMPT, ENGIE_WELCOME_MESSAGE
from synthetic_data import generate_employee_data
from engie_assistant import EngieAssistant
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import logging


if __name__ == "__main__":

    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)

    st.set_page_config(page_title="Umbrella Onboarding", page_icon="☂", layout="wide")

    @st.cache_data(ttl=3600, show_spinner="Generating User Data...")
    def get_user_data():
        return generate_employee_data(1)[0]

    @st.cache_resource(ttl=3600, show_spinner="Loading Vector Store...")
    def init_vector_store(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

            embedding_function = OpenAIEmbeddings()

            # Use Chroma instead of FAISS
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding_function,
                persist_directory="./chroma_db",  # Specify a persistence directory
            )

            return vectorstore
        except Exception as e:
            logging.error(f"Error initializing vector store: {str(e)}")
            st.error(f"Failed to initialize vector store: {str(e)}")
            return None

    # Usage
    vector_store = init_vector_store("data/umbrella_corp_policies.pdf")
    if vector_store is None:
        st.error(
            "Failed to initialize vector store. Please check the logs for more information."
        )
        st.stop()

    llm = ChatGroq()
    system_prompt = ENGIE_SYSTEM_PROMPT
    welcome_message = ENGIE_WELCOME_MESSAGE
    customer_data = get_user_data()
    vector_store = init_vector_store("./data/umbrella_corp_policies.pdf")

    if "customer" not in st.session_state:
        st.session_state.customer = customer_data
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": welcome_message}]

    assistant = EngieAssistant(
        system_prompt=system_prompt,
        llm=llm,
        customer_data=st.session_state.customer,
        history=st.session_state.messages,
        vector_store=vector_store,
    )
    assistant.render()
