from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding  = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=30)

response_vector = embedding.embed_query("Delhi is capital of india")

print(str(response_vector))

