from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

chat_model = ChatOpenAI(model='gpt-4')

st.header("Reserch Assistant Tool")

user_input = st.text_input("Enter your prompt here")

if st.button("Summarize"):
    response = chat_model.invoke(user_input)
    st.write(response.content)
    