import os
import logging
import fitz  # PyMuPDF
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carica le variabili d'ambiente
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La variabile OPENAI_API_KEY non è stata caricata correttamente.")

def load_pdf_with_pymupdf(file_path):
    """
    Carica un PDF usando PyMuPDF (fitz) che può essere più robusto con PDF problematici.
    """
    documents = []
    try:
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            if text.strip():  # Assicurati che ci sia del testo nella pagina
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_num}
                ))
        pdf_document.close()
    except Exception as e:
        logger.error(f"Errore nel caricamento del PDF {file_path} con PyMuPDF: {str(e)}")
    
    return documents

def load_txt_with_encoding_fallback(file_path):
    """
    Carica un file TXT provando diverse codifiche in caso di errore.
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            
            # Controlla se il contenuto è vuoto o contiene solo spazi
            if not content.strip():
                logger.warning(f"Il file {file_path} è vuoto o contiene solo spazi.")
                return []
            
            return [Document(
                page_content=content,
                metadata={"source": file_path}
            )]
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Errore nel caricamento del file TXT {file_path} con codifica {encoding}: {str(e)}")
    
    logger.error(f"Impossibile leggere il file {file_path} con nessuna delle codifiche provate.")
    return []

def load_documents_from_folder(folder_path):
    """
    Carica tutti i PDF, DOCX e TXT presenti nella cartella 'folder_path' e restituisce una lista di documenti.
    """
    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if filename.lower().endswith(".pdf"):
                logger.info(f"Caricamento PDF con PyMuPDF: {filename}")
                pdf_docs = load_pdf_with_pymupdf(file_path)
                docs.extend(pdf_docs)
                logger.info(f"PDF caricato con successo: {filename}, pagine: {len(pdf_docs)}")
            elif filename.lower().endswith(".docx"):
                logger.info(f"Caricamento DOCX: {filename}")
                loader = Docx2txtLoader(file_path)
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
                logger.info(f"DOCX caricato con successo: {filename}")
            elif filename.lower().endswith(".txt"):
                logger.info(f"Caricamento TXT: {filename}")
                txt_docs = load_txt_with_encoding_fallback(file_path)
                if txt_docs:
                    docs.extend(txt_docs)
                    logger.info(f"TXT caricato con successo: {filename}")
                else:
                    logger.error(f"Impossibile caricare il file TXT: {filename}")
        except Exception as e:
            logger.error(f"Errore nel caricamento del file {filename}: {str(e)}")
    
    return docs

def ingest_documents_to_vectordb(folder_path, persist_directory):
    """
    Elabora tutti i PDF, DOCX e TXT in 'folder_path' e salva il vector DB nella cartella 'persist_directory'.
    """
    logger.info(f"Caricamento dei documenti (PDF, DOCX e TXT) dalla cartella: {folder_path}")
    docs = load_documents_from_folder(folder_path)
    
    if not docs:
        logger.warning("Nessun documento PDF, DOCX o TXT trovato o caricato correttamente.")
        return
    
    logger.info(f"Trovati {len(docs)} documenti da elaborare.")
    
    logger.info("Esecuzione del text splitting...")
    text_splitter = TokenTextSplitter(
        model_name="gpt-4o-mini", 
        chunk_size=1000, 
        chunk_overlap=100, 
        add_start_index=True,
    )
    docs = text_splitter.split_documents(docs)
    logger.info(f"Creati {len(docs)} chunks di testo.")
    
    logger.info("Creazione delle embeddings e del vector store...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Assicurati che la directory di destinazione esista
    os.makedirs(persist_directory, exist_ok=True)
    
    logger.info(f"Salvataggio del vector DB in locale nella cartella: {persist_directory}")
    vector_store.save_local(persist_directory)
    logger.info("Ingestione completata con successo!")

if __name__ == '__main__':
    folder_path = "data"        # Cartella contenente i PDF, DOCX e TXT
    persist_directory = "vectordb"    # Cartella in cui salvare il vector DB
    ingest_documents_to_vectordb(folder_path, persist_directory)