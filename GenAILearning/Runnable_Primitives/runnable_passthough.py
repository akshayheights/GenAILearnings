'''
RunnablePassThrough is special runnable primitive that simply returns the input as output 
without modifying it.

where it is useful?
when we genrate joke from topic and then explanation of joke in which just explanation of joke printed but not joke.
'''

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

prompt_1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=['topic']
)


prompt_2 = PromptTemplate(
    template="Explain the following joke {text}",
    input_variables=['text']
)

chat_model = ChatOpenAI()

parser = StrOutputParser()


joke_generator = RunnableSequence(prompt_1, chat_model, parser)

parallel_chain = RunnableParallel(
    {
        'joke' : RunnablePassthrough(),
        'explanation': RunnableSequence(prompt_2, chat_model, parser)
    }
)

final_chain = RunnableSequence(joke_generator, parallel_chain)

result = final_chain.invoke({'topic':'data engineers'})

print('joke:',result['joke'])
print('explanation:',result['explanation'])
