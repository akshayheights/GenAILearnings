from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

chat_model = ChatOpenAI()

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me name, age and city of fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.format()

# result = chat_model.invoke(prompt)

# print(result)

# parsed_result = parser.parse(result.content)

# print(parsed_result)

chain = template | chat_model | parser

result=chain.invoke({})

print(result)