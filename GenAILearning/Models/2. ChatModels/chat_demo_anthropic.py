from langchain_anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

chat_model = Anthropic(model="claude-3-sonnet-20240229")

response = chat_model.invoke("suggest me 5 names")

print(response.content)

