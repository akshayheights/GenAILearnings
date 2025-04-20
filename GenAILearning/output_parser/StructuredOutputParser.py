from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

chat_model = ChatOpenAI()

Schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(Schema)

template_1 = PromptTemplate(
    template = "Tell me about the {topic} /n {format_instructions}",
    input_variables = ['topic'],
    partial_variables = {'format_instructions': parser.get_format_instructions}
)

# prompt = template_1.invoke({'topic':'black hole'})

# print(prompt)

# result = chat_model.invoke(prompt)

# print("result_content", result.content)

# final_result = parser.parse(result.content)

# print("final_result",final_result)


chain = template_1 | chat_model | parser

result = chain.invoke({'topic':'black hole'})

print(result)