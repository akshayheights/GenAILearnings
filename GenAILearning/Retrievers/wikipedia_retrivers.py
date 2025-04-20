from langchain_community.retrievers import WikipediaRetriever

# intialize the retriver
retriver = WikipediaRetriever(top_k_results=2, lang="en")

# define your query
query = "virat kohli"

# get relevant wikipedia documents
docs = retriver.invoke(query)

# print(docs)

for i, doc in enumerate(docs):
    print(f"\n---------result:{i+1}---------")
    print(f"\n content:\n{doc.page_content}---------")

