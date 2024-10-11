import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_mistralai import MistralAIEmbeddings
import streamlit as st
from streamlit_chat import message

# Set API Keys
os.environ["GROQ_API_KEY"] = st.secrets["Groq_API"]
os.environ["MISTRALAI_API_KEY"] = st.secrets["Mistral_API"]

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192")

st.title("Multi-URL Chatbot")

# Sidebar input for multiple URLs
urls_input = st.sidebar.text_area("Enter URLs (one per line)", placeholder="https://example.com\nhttps://another.com")
urls = urls_input.splitlines()

if urls:
    selected_url = st.sidebar.selectbox("Select URL to interact with", urls)

    if selected_url:
        st.write(f"Selected URL: {selected_url}")

        if 'responses' not in st.session_state:
            st.session_state['responses'] = ["Welcome to the chatbot! How can I assist you?"]

        if 'requests' not in st.session_state:
            st.session_state['requests'] = []

        if "vector" not in st.session_state:
            st.session_state.embedding =  MistralAIEmbeddings(model="mistral-embed", api_key=st.secrets["Mistral_API"])
            st.session_state.documents = []

        # Load documents based on the selected URL
        st.session_state.loader = WebBaseLoader(str(selected_url))
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.doc = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.db = FAISS.from_documents(documents=st.session_state.doc, embedding=st.session_state.embedding)

        prompt = ChatPromptTemplate.from_template("""Read the complete context, give information about context and solve questions strictly related to context.
        <context>
        {context}
        </context>
        Question: {input}""")

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.db.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response_container = st.container()
        textcontainer = st.container()

        with textcontainer:
            query = st.text_input("Query: ")
            if query:
                with st.spinner("typing..."):
                    response = retrieval_chain.invoke({"input": query})
                st.session_state.requests.append(query)
                st.session_state.responses.append(response['answer'])

        with response_container:
            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])):
                    message(st.session_state['responses'][i], key=str(i))
                    if i < len(st.session_state['requests']):
                        message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
