from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

pdf_doc = loader.load()


splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

# chunk overlap is how much overlap of characters betweem two chunks
# it helps to avoid if there some gap between two chunks second chunk will get context.
# if we increase chunk overlap too much then we will have lot of chunks.
# if chunk size is 100 the chunk_overlap should be between 10 to 20 words.

splitted_text = splitter.split_documents(pdf_doc)


print(splitted_text[0])

