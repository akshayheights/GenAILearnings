'''
We need to design application where we need analyze the feedback of user.
If sentiment is positive we will send it to model to generate response accordingly.
If it is negative response will send it to model to generate response accordingly
'''
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

chat_model_1 = ChatOpenAI()

parser_1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback either positive or negative')
    
parser_2 = PydanticOutputParser(pydantic_object=Feedback)

prompt_1 = PromptTemplate(
    template='classify the sentiment of feedback either into positive or negative for following feedback \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser_2.get_format_instructions()}
)

classifier_chain = prompt_1 | chat_model_1 | parser_2


prompt_2 = PromptTemplate(
    template='write an appropriate response to positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt_3 = PromptTemplate(
    template='write an appropriate response to negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment=="positive", prompt_2 | chat_model_1 | parser_1),
    (lambda x:x.sentiment=="negative", prompt_3 | chat_model_1 | parser_1),
    RunnableLambda(lambda x:"could not find relevent sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':'this is worst phone i have ver used'})

print(result)


chain.get_graph().print_ascii()



