from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

load_dotenv()


chat_history = [SystemMessage(content="You are really helpful asistant")]

chat_model = ChatOpenAI(model="gpt-4")

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    response = chat_model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI Assistant: ", response.content)
    
print(chat_history)