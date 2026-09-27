import os

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")

model = ChatGroq(model=groq_model, temperature=0.7, api_key=groq_api_key)

st.title("GEX")

isclear = st.button("clear state")    

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a helpful programming tutor named Gex.
        Explain technical concepts simply and give examples.

        Use the provided context to answer questions about the document.

        If the answer cannot be found in the context, say that you
        could not find the answer in the provided document.

        Context:
        {context}
        """),
    ("placeholder","{message}")
])

parser = StrOutputParser()

chain = prompt | model | parser

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_1"

history = RedisChatMessageHistory(
    session_id = st.session_state.session_id,
    redis_url="redis://localhost:6379"
)

if isclear:
    st.session_state.clear()
    history.clear()
    st.rerun()


for message in history.messages:

    if isinstance(message, HumanMessage):
        st.chat_message('user').write(message.content)

    if isinstance(message, AIMessage):
        st.chat_message('assistant').write(message.content)


loader = PyPDFLoader('resources/pdfs/attention.pdf')
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
chunks = text_splitter.split_documents(documents=document)

embedding = HuggingFaceEmbeddings(
    model_name = embedding_model
)

vector_store = Chroma.from_documents(
    embedding=embedding, 
    documents=chunks
)

retriever = vector_store.as_retriever(
    search_kwargs = {"k":3}
)


input = st.chat_input("input")

if input:
    human_message = HumanMessage(content=input)

    history.add_message(human_message)

    st.chat_message('user').write(input)

    relevant_docs = retriever.invoke(input)

    content_from_docs = "/n/n".join(
        docs.page_content for docs in relevant_docs 
    )

    print(content_from_docs)

    with st.chat_message('assistant'):
        response = st.write_stream(
            chain.stream({
                "context":content_from_docs,
                "message":history.messages
            })
        )
    
    ai_message = AIMessage(content=response)
    history.add_message(ai_message)





