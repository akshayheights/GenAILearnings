from langchain_chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document


documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]


# create chroma vector store in memory

vector_store = Chroma(
    embedding_function = OpenAIEmbeddings(),
    persist_directory='my_chroma_db',
    collection_name='my_chroma_db_1'  
)

vector_store.add_documents(documents)

# convert vector store to retriver
retriver = vector_store.as_retriever(search_kwargs={"k":2})

# search for query
query = "what is chroma used for"
result = retriver.invoke(query)

# retrive the result
for i, doc in enumerate(result):
    print(f"----Result : {i+1}---")
    print(doc.page_content)
    

'''
We can also use similarity search to do this kind of serach but with retirivers we can use diffrent kind of searching techniques that
is not possible with similarity search.
'''