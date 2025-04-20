from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

url = 'https://www.amazon.in/Lanterns-Hanging-Decoration-Japanese-Decorations/dp/B0DR857F4J/?_encoding=UTF8&pd_rd_w=yd34M&content-id=amzn1.sym.0e03aefb-8b93-49f8-beeb-6d21836a1b3d&pf_rd_p=0e03aefb-8b93-49f8-beeb-6d21836a1b3d&pf_rd_r=H99W5DB13AZWF33SJGMV&pd_rd_wg=DO2ZY&pd_rd_r=04f7555d-4fe8-4437-aa3e-9f5ead1a33a9&ref_=pd_hp_d_atf_dealz_cs&th=1'

loader = WebBaseLoader(url)

web_doc = loader.load()

parser = StrOutputParser()

chat_model = ChatOpenAI()

prompt = PromptTemplate(
    template="please answer the following question /n {question} based on the text {text}",
    input_variables=['text','question']
)

chain = prompt | chat_model | parser

print(chain.invoke({'question':'what is the product we aare talking about?', 'text':web_doc[0].page_content}))