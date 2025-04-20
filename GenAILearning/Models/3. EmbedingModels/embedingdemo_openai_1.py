from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding  = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=30)

documents =[
    "Delhi is capital of india",
    "Mumbai is capital of Maharashtra"
]

response_vector = embedding.embed_documents(documents)

print(str(response_vector))

