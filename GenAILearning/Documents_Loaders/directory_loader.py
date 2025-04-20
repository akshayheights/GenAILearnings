from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

directory_loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = directory_loader.load()

print(docs[0].page_content)

