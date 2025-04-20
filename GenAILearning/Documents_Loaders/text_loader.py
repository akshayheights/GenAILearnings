from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

prompt = PromptTemplate(
    template="write a one line feedback about {text}",
    input_variables=['text']
)

chat_model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt | chat_model | parser

print(chain.invoke({'text':docs[0].page_content}))

