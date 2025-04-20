from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# create FAISS vector store

embedding_model = OpenAIEmbeddings()

vector_store = FAISS.from_documents(
    documents = all_docs,
    embedding = embedding_model
)

# create simple retriver
similarity_retriver = vector_store.as_retriever(search_type='similarity', search_kwargs={"k":5})

# create multiquery retriver
multiquery_retriver = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k":5}),
    llm=ChatOpenAI()
)

query = "How to improve energy levels and maintain balance in person?"

similarity_results = similarity_retriver.invoke(query)
multiquery_results = multiquery_retriver.invoke(query)

print("**********************************similarity_results*********************************\n")
for i, doc in enumerate(similarity_results):
    print(f"\n results: {i+1}")
    print(doc.page_content)
    
print("**********************************multiquery_results*********************************\n")
for i, doc in enumerate(multiquery_results):
    print(f"\n results: {i+1}")
    print(doc.page_content)


