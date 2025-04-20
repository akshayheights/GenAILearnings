from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


chat_history = []

chat_model = ChatOpenAI(model="gpt-4")

while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input=='exit':
        break
    response = chat_model.invoke(chat_history)
    chat_history.append(response)
    print("AI Assistant: ", response.content)
    
print(chat_history)