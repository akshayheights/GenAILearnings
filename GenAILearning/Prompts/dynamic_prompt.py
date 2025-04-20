from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

chat_model = ChatOpenAI(model='gpt-4')

st.header("Research Assistant Tool")

paper_input = st.selectbox("select research paper name",["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers",
                                                         "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style",["Beginner-Friendly", "Technical", "Code-Oriented", "Matematical"])

length_input = st.selectbox("Select Explanation Length",["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

template = load_prompt('template.json')

# prompt = template.invoke({
#     'paper_input':paper_input,
#     'style_input':style_input,
#     'length_input':length_input
# })

# if st.button("Summarize"):
#     response = chat_model.invoke(prompt)
#     st.write(response.content)

if st.button("Summarize"):
    chain = template | chat_model
    response = chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    })
    st.write(response.content)