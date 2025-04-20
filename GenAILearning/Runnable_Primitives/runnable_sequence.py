'''
It is used to connect two components/task specific runnables together sequentially.

we will create joke from topic and then provide explnation of that joke.
'''
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt_1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="Explain the following joke {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chat_model = ChatOpenAI()

chain = RunnableSequence(prompt_1, chat_model, parser, prompt_2, chat_model, parser)

result = chain.invoke({'topic':'AI'})

print(result)