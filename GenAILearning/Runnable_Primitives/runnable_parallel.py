'''
we have topic and we want to make application where you want send this topic two different LLM's. 
First LLM should generate one tweet and another LLM should genrate linked in post

RunnableParallel is a runnable primitive that allows multiple runnables to execute in parallel.
Each runnable receives the same input and process it independently producing dictionary outputs
'''

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt_1 = PromptTemplate(
    template = "Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template = "Generate a linkedin post about {topic}",
    input_variables=['topic']
)

chat_model = ChatOpenAI()

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet':RunnableSequence(prompt_1 | chat_model | parser),
        'linkedin' : RunnableSequence(prompt_2 | chat_model | parser)
    }
)

# output {'tweet':'', 'linkedin':''}

result = parallel_chain.invoke({'topic':'sex'})

print(result['tweet'])
print(result['linkedin'])