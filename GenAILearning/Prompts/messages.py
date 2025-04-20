from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model="gpt-4")

messages = [
    SystemMessage(content="You are really intelligent Assistant"),
    HumanMessage(content="Tell me about langchain")
]

result = chat_model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
