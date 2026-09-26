import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model='openai/gpt-oss-20b', temperature=0.7, api_key=groq_api_key)

st.title("GEX")

isclear = st.button("clear state")

if isclear:
    st.session_state.clear()
    st.rerun()


prompt = ChatPromptTemplate.from_messages([
    ("system",
    "You are a helpful programming tutor. Explain technical concepts simply and give examples. If user asked about any different topics like cooking, movies and other things. just return like invalid query"),
    ("placeholder","{message}")
])

parser = StrOutputParser()

chain = prompt | model | parser

if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()

for message in st.session_state.history:

    if isinstance(message, HumanMessage):
        st.chat_message('user').write(message.conent)

    if isinstance(message, AIMessage):
        st.chat_message('assistant').write(message.content)
    

input = st.chat_input("input")

if input:
    human_message = HumanMessage(content=input)

    st.session_state.history.add_message(human_message)

    st.chat_message('user').write(input)


    with st.chat_message('assistant'):
        response = st.write_stream(
            chain.stream({
                "message":st.session_state.history.messages
            })
        )
    
    ai_message = AIMessage(content=response)
    st.session_state.history.add_message(ai_message)





