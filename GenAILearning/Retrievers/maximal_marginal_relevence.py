from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Step 1: Initialize embedding model
embedding_model = OpenAIEmbeddings()

# Step 2: Create sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# Step 3: Create FAISS vector store from documents
vector_store = FAISS.from_documents(docs, embedding_model)

retriver = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs = {"k":3, "lambda_mult":1}
)

query = "what is langchain?"
result = retriver.invoke(query)

for i, doc in enumerate(result):
    print(f"-----result: {i+1}")
    print(f"content:\n {doc}")
