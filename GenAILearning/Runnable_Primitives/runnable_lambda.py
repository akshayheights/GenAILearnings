'''
Runnable lambda:
It allows you to apply cutsom python functions with AI pipeline.

It acts as middleware between different AI components, enabling preprocessing, transformation, API calls, filetring, post processing
in langchain workflow

If we convert any python function into runnable it can chain with other components in chain.

e.g.:
customers are sending reviews we need send to LLM and fidn the sentiments. But reviews are not clean we need to preprocess this data

I want to generate joke from topic but at same time I need know word count in joke not using LLM
'''

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

prompt_1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=['topic']
)

chat_model = ChatOpenAI()

parser = StrOutputParser()

joke_generation_chain = RunnableSequence(prompt_1, chat_model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(lambda x : len(x.split()))
})

final_chain = RunnableSequence(joke_generation_chain, parallel_chain)


result = final_chain.invoke({'topic':'sex'})

print("joke:", result['joke'])
print("word-count:", result['word_count'])

