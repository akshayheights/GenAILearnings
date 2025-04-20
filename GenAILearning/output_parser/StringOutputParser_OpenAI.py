from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

chat_model = ChatOpenAI()

# prompt to get detailed report
template_1 = PromptTemplate(
    template = 'write detailed report on {topic}',
    input_variables=['topic']
)

prompt_1 = template_1.invoke({'topic':'blackwhole'})

response = chat_model.invoke(prompt_1)

print(response.content)

# prompt to get summary report

template_2 = PromptTemplate(
    template = 'write 5 line summary for {text}',
    input_variables=['text']
)

prompt_2 = template_2.invoke({'text':response.content})

response2 = chat_model.invoke(prompt_2)

print(response2.content)