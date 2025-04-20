from langchain_community.document_loaders import PyPDFLoader


pdf_loader = PyPDFLoader('dl-curriculum.pdf')

pdf_doc = pdf_loader.load()

print(pdf_doc[0])