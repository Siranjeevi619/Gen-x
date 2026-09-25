import os 
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model='openai/gpt-oss-20b', temperature=0.7, api_key=groq_api_key)

st.title("GEX")
input = st.chat_input("input")

if input:
    st.chat_message("user").write(input)
    response = model.invoke(input)
    print(response)
    st.chat_message('assistant').write(response.content)



