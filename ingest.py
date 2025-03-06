import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Carica le variabili d'ambiente (in particolare, OPENAI_API_KEY)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La variabile OPENAI_API_KEY non è stata caricata correttamente.")

def load_all_pdfs_from_folder(folder_path):
    """
    Carica tutti i PDF presenti nella cartella 'folder_path' e restituisce una lista di documenti.
    """
    docs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(pdf_path)
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
    return docs

def ingest_pdfs_to_vectordb(folder_path, persist_directory):
    """
    Elabora tutti i PDF in 'folder_path' e salva il vector DB nella cartella 'persist_directory'.
    """
    print("Caricamento dei PDF dalla cartella:", folder_path)
    docs = load_all_pdfs_from_folder(folder_path)
    
    print("Esecuzione del text splitting...")
    text_splitter = TokenTextSplitter(model_name="gpt-4o-mini", chunk_size=1000, chunk_overlap=100, add_start_index=True,)
    docs = text_splitter.split_documents(docs)
    
    print("Creazione delle embeddings e del vector store...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    
    print("Salvataggio del vector DB in locale nella cartella:", persist_directory)
    vector_store.save_local(persist_directory)
    print("Ingestione completata con successo!")

if __name__ == '__main__':
    folder_path = "data"            # Cartella contenente i PDF
    persist_directory = "vectordb"    # Cartella in cui salvare il vector DB
    ingest_pdfs_to_vectordb(folder_path, persist_directory)
