from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.9,max_tokens=30)

response = chat_model.invoke('What is capital of india?')

print(response.content)