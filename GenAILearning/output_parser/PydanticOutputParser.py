from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

chat_model = ChatOpenAI()

class Person(BaseModel):
    name : str = Field(description="name of the person")
    age : int = Field(description="age of the person", gt=18)
    city : str = Field(description="name of the city person belongs to")
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Genreate the name, age and city of a fictional {place} person \n {format_instructions}",
    input_variables=['place'],
    partial_variables={'format_instructions':parser.get_format_instructions}
)


chain = template | chat_model | parser

result = chain.invoke({'place':'india'})

print(result)
    