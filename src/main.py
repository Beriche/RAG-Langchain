from celery import chunks
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from networkx import add_star

DATA_PATH = "data/"


# Fonction qui permet de récuperer les documents

def load_documents():
    loader = DirectoryLoader(DATA_PATH,glob="*.pdf")
    documents = loader.load()
    return documents 


# Découpage des documents en plusieurs chunk 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True
)

chunks = text_splitter.split_documents(documents)
