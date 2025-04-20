'''
We will ask user for topic. we will send it LLM saying we need detailed report on topic.
After that this detailed report sent to same LLM again and ask for 5 important points in reports
'''

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

chat_model = ChatOpenAI()

prompt_1= PromptTemplate(
    template = "Generate a detailed report on {topic}",
    input_variables=['topic']
)


prompt_2= PromptTemplate(
    template = "Generate  5 pointers summary from the following {text}",
    input_variables=['txt']
)

parser = StrOutputParser()

chain = prompt_1 | chat_model | parser | prompt_2 | chat_model | parser

result = chain.invoke({'topic':'unemployement in india'})

print(result)

print(chain.get_graph().print_ascii())