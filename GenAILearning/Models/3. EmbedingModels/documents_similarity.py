'''
We will bulding application for semantic search where we will load 5 documents and then check sentence exist in which document using 
semantic search.
'''

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity # this is to find similarity between vectors
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=300)

documents = ["Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."]

query = "tell me about sachin"

doc_embedding = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

score = cosine_similarity([query_embedding],doc_embedding)[0]

index, score = sorted(list(enumerate(score)), key = lambda x:x[1])[-1]

print(query)

print(documents[index])

print(f"Similarity score is {score}")

'''
Here we are generating embeding again and again which is costly opreation.
So embeding for document must be stored in vector DB.
'''