from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

loader = TextLoader("documents/coffee_guide.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(
    model="llama3"
)

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

vectorstore.save_local("faiss_index")

print("Documents indexed successfully!")