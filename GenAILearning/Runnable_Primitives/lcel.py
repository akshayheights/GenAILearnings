'''
if-else statment of langchain universe

It is control flow component in langchain that allows you to conditionally route input data to different chains or runnables based
on custom logic.

consider we have email. if email about complaint transfer it to customer service, if email about refund transfer it to account team.
'''

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

prompt_1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="summarize the following text \n  {text}",
    input_variables=['text']
)

chat_model = ChatOpenAI()

parser = StrOutputParser()


report_generation_chain = prompt_1 | chat_model | parser

branch_chain = RunnableBranch(
    (lambda x : len(x.split())>200, RunnableSequence(prompt_2, chat_model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generation_chain, branch_chain)

print(final_chain.invoke({'topic':'x-ray'}))