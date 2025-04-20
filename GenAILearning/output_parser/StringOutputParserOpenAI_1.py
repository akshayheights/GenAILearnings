from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

chat_model = ChatOpenAI()

# prompt to get detailed report
template_1 = PromptTemplate(
    template = 'write detailed report on {topic}',
    input_variables=['topic']
)


# prompt to get summary report

template_2 = PromptTemplate(
    template = 'write 5 line summary for {text} with seperate line for each sentence with line number',
    input_variables=['text']
)


parser = StrOutputParser()

chain = template_1 | chat_model | parser | template_2 | chat_model | parser

result = chain.invoke({'topic':'black hole'})

print(result)
