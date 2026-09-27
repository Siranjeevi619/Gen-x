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
    (
        "system",
        """You are a helpful programming tutor named Gex.
        Explain technical concepts simply and give examples.

        Use the provided context to answer questions.

        Each context section contains a source and page number.

        If the answer is found in the context, explain the answer
        and mention the relevant page number.

        If the answer cannot be found in the context, say:
        "I could not find the answer in the provided document."

        Do not make up information.

        Context:
        {context}
        """
    ),
    ("placeholder", "{message}")
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

@st.cache_resource
def get_retriever():
    """Load PDF, create embeddings, and return retriever - cached across reruns."""
    loader = PyPDFLoader('resources/pdfs/attention.pdf')
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""])
    
    chunks = text_splitter.split_documents(documents=document)

    embedding = HuggingFaceEmbeddings(
        model_name = embedding_model
    )

    vector_store = Chroma(
        collection_name="Attention",
        embedding_function= embedding,
        persist_directory="./chromadb"
    )
    if vector_store.get()["ids"] == []:
        vector_store.add_documents(chunks)

    return vector_store

vector_store  = get_retriever()


input = st.chat_input("input")

if input:
    human_message = HumanMessage(content=input)

    history.add_message(human_message)

    st.chat_message('user').write(input)

    results = vector_store.similarity_search_with_score(
        input,
        k=5
    )

    threshold = 1.5

    relevant_docs = [
        doc
        for doc, score in results
        if score <= threshold
    ]

    for doc, score in results:
        print("Score:", score)
        print("Content:", doc.page_content[:200])
        print("----------------")

    if not relevant_docs:

        response  = (
            "I could not find relevant information in the provided document."
        )

        with st.chat_message("assistant"):
            st.write(response)

    else:
        content_from_docs = "\n\n".join(
            f"Source: {doc.metadata.get('source')}\n"
            f"Page: {doc.metadata.get('page', 0) + 1}\n"
            f"Content: {doc.page_content}"
            for doc in relevant_docs
        )
        
        print(content_from_docs)
        
        with st.chat_message('assistant'):
            response = st.write_stream(
                    chain.stream({
                        "context": content_from_docs,
                        "message": history.messages
                    })
                )
            
    ai_message = AIMessage(content=response)
    history.add_message(ai_message)
