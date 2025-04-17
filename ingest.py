""""
This program stores the text data in vector database. Here we have used ChromaDB for storing the text data and semantic chunker to split the texts.
"""
from langchain_community.document_loaders import TextLoader , DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


data_path = "data/"
path_to_save = "vectorstores/chroma"
model_name = "nomic-ai/nomic-embed-text-v1.5"
def ingest_data():
    """
    This function performs data ingestion
    """
    text_loader_kwargs = {"encoding" : 'utf-8'}
    loader = DirectoryLoader(data_path , glob = "**/*.txt" , loader_cls=TextLoader , loader_kwargs=text_loader_kwargs)
    documents = loader.load()
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = {"device" : "cuda:0" , "trust_remote_code" : True},
       
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap= 50)
    text_data = text_splitter.split_documents(documents)

    database = Chroma.from_documents(
        text_data , embeddings, persist_directory=path_to_save
    )


if __name__ == "__main__":
    ingest_data()
