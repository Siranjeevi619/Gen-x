import os 
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage , AIMessage
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model='openai/gpt-oss-20b', temperature=0.7, api_key=groq_api_key)

st.title("GEX")

prompt = ChatPromptTemplate.from_messages([
    ("system",
    "You are a helpful programming tutor. Explain technical concepts simply and give examples. If user asked about any different topics like cooking, movies and other things. just return like invalid query"),
    ("placeholder",
    "{message}")
])

chain = prompt | model

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):
        st.chat_message('user').write(message)

    if isinstance(message, AIMessage):
        st.chat_message('assistant')
    

input = st.chat_input("input")

if input:
    human_message = HumanMessage(content=input)
    st.session_state.messages.append(human_message)
    st.chat_message('user').write(input)

    response = chain.invoke({
        "message":st.session_state.messages
    })

    print(response)

    ai_message = AIMessage(
        content=response.content
        )
    st.session_state.messages.append(ai_message)

    st.chat_message('assistant').write(response.content)




