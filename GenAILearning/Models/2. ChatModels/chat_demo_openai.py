from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model='gpt-4', temperature=0, max_completion_tokens=50)

response = chat_model.invoke("suggest me 5 indian names")

print(response)