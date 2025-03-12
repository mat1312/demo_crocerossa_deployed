import os
import requests
import uuid
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carica le variabili d'ambiente dal file .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("La variabile OPENAI_API_KEY non è stata caricata correttamente.")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise Exception("La variabile ELEVENLABS_API_KEY non è stata caricata correttamente.")

# Imposta il percorso del vector DB persistente
persist_directory = "vectordb"
if not os.path.exists(persist_directory):
    raise Exception("Il vector DB non è stato trovato. Esegui prima l'ingestione con 'ingest.py'.")

# Carica il vector DB
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)

# Definisci il system prompt per l'assistente della Croce Rossa Italiana
SYSTEM_PROMPT = """
Sei "Assistente per Procedure e Regolamentazioni" della Croce Rossa Italiana. Il tuo compito è fornire informazioni e orientamenti preliminari su procedure, regolamentazioni e normative inerenti alle attività della Croce Rossa Italiana, mantenendo un tono professionale, chiaro ed empatico e rivolgendoti sempre con il "Lei".

Rispondi sia a domande specifiche relative alle procedure della Croce Rossa Italiana che a domande generali inerenti normative e regolamentazioni. Se necessario, consulta la tua knowledge base per informazioni aggiornate e dettagliate.
Se il caso richiede ulteriori approfondimenti, informa l'utente che un operatore della Croce Rossa Italiana lo contatterà.
Fai una domanda alla volta e guida l'utente in modo naturale.
Chiudi la conversazione solo dopo aver raccolto l'email e il numero di telefono dell'utente.
"""

# Inizializza il modello LLM
llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Inizializza la catena RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# Dizionario per le conversazioni attive
active_conversations = {}

# Variabili globali
transcript_global = []        # Per salvare il transcript da ElevenLabs
conversation_memory = []      # Per memorizzare le coppie domanda-risposta della chat

# Modello per il webhook di ElevenLabs
class ElevenLabsWebhook(BaseModel):
    text: str
    conversation_id: str = None

class CRIAssistant:
    def __init__(self):
        self.vector_store = vector_store
        self.qa_chain = qa_chain
    
    def retrieve_relevant_documents(self, query: str) -> List:
        """
        Recupera i documenti rilevanti per una query.
        
        Args:
            query: La query dell'utente
            
        Returns:
            I documenti rilevanti
        """
        return self.vector_store.similarity_search(query)
    
    def format_retrieved_documents(self, documents: List) -> str:
        """
        Formatta i documenti recuperati in testo leggibile.
        
        Args:
            documents: I documenti recuperati
            
        Returns:
            Il testo formattato
        """
        result = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "Fonte sconosciuta").replace("\\", "/")
            page = doc.metadata.get("page", "N/A")
            source_info = f"Documento: {source}"
            if page != "N/A":
                source_info += f" (Pagina: {page})"
            
            result.append(f"Informazione {i+1}:\n{doc.page_content}\n\nFonte: {source_info}\n")
        
        return "\n".join(result)
    
    async def elevenlabs_webhook_handler(self, request_data: ElevenLabsWebhook) -> Dict[str, Any]:
        """
        Gestisce le richieste webhook da ElevenLabs.
        Restituisce i top 3 chunk più pertinenti invece di passarli all'LLM.
        
        Args:
            request_data: I dati della richiesta webhook
            
        Returns:
            La risposta formattata per ElevenLabs con i chunk più pertinenti
        """
        try:
            # Log della richiesta
            logger.info(f"Ricevuta richiesta webhook da ElevenLabs: {request_data}")
            
            # Estrai i dati dalla richiesta di ElevenLabs
            input_text = request_data.text
            conversation_id = request_data.conversation_id or str(uuid.uuid4())
            
            if not input_text:
                logger.warning("Ricevuta richiesta webhook senza testo")
                return {
                    "text": "Non è stata fornita alcuna domanda.",
                    "conversation_id": conversation_id
                }
            
            # Recupera i documenti rilevanti (limitati a 3)
            documents = self.retrieve_relevant_documents(input_text)[:3]
            
            if not documents:
                logger.warning("Nessun documento recuperato per la query")
                return {
                    "text": "Non ho trovato informazioni pertinenti alla tua domanda.",
                    "conversation_id": conversation_id
                }
            
            # Formatta i documenti in un contesto (utilizza la funzione esistente)
            chunks_text = self.format_retrieved_documents(documents)
            
            # Formatta la risposta per ElevenLabs
            response = {
                "text": chunks_text,
                "conversation_id": conversation_id
            }
            
            logger.info(f"Inviata risposta webhook con i top 3 chunk per conversation_id: {conversation_id}")
            return response
            
        except Exception as e:
            error_msg = f"Errore nel webhook: {str(e)}"
            logger.error(error_msg)
            return {
                "text": "Mi dispiace, si è verificato un errore durante l'elaborazione della richiesta.",
                "conversation_id": conversation_id
            }
    
    def reset_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        Resetta la memoria della conversazione.
        
        Args:
            session_id: ID di sessione
            
        Returns:
            Un messaggio di conferma
        """
        if session_id in active_conversations:
            active_conversations[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            logger.info(f"Conversazione resettata per session_id: {session_id}")
        else:
            logger.warning(f"Tentativo di reset per una sessione inesistente: {session_id}")
        
        return {
            "status": "success",
            "message": "Conversazione resettata con successo",
            "session_id": session_id
        }

# Inizializza l'assistente CRI
cri_assistant = CRIAssistant()

# Funzioni helper per recuperare le conversazioni da ElevenLabs
def get_last_conversation(agent_id: str, api_key: str):
    url = "https://api.elevenlabs.io/v1/convai/conversations"
    headers = {"xi-api-key": api_key}
    params = {"agent_id": agent_id, "page_size": 1}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
         return None
    data = response.json()
    conversations = data.get("conversations", [])
    if not conversations:
         return None
    return conversations[0].get("conversation_id")

def get_conversation_details(conversation_id: str, api_key: str):
    url = f"https://api.elevenlabs.io/v1/convai/conversations/{conversation_id}"
    headers = {"xi-api-key": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
         return None
    return response.json()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Modello Pydantic per la richiesta Q&A
class QARequest(BaseModel):
    question: str

# Modello per il reset della conversazione
class ResetRequest(BaseModel):
    session_id: str

@app.post("/api/qa")
def qa_endpoint(request: QARequest):
    global conversation_memory
    if not request.question:
        raise HTTPException(status_code=400, detail="La domanda è obbligatoria.")
    
    # Costruisci la storia della conversazione con le coppie domanda-risposta salvate
    chat_history_str = ""
    for entry in conversation_memory:
        chat_history_str += f"Domanda: {entry['question']}\nRisposta: {entry['answer']}\n"
    
    # Crea il prompt unendo il system prompt, la storia della conversazione e la nuova domanda
    prompt_text = SYSTEM_PROMPT.strip() + "\n\n" + chat_history_str + "Domanda: " + request.question.strip()
    
    result = qa_chain.invoke(prompt_text)
    answer = result["result"] if isinstance(result, dict) and "result" in result else result

    # Salva la nuova coppia domanda-risposta nella memoria della chat
    conversation_memory.append({
        "question": request.question.strip(),
        "answer": answer
    })

    # Estrai le fonti (se presenti)
    source_docs = result.get("source_documents", [])
    sources = {}
    for doc in source_docs:
        metadata = doc.metadata
        if "source" in metadata:
            source = metadata["source"].replace("\\", "/")
            page = metadata.get("page", None)
            line = metadata.get("start_index", None)
            sources.setdefault(source, []).append((page, line))
    return {"answer": answer, "sources": sources}

@app.get("/api/transcript")
def transcript_endpoint():
    agent_id = "j9lr1Zv4W5s6khw0lxK5"  # ID dell'agente della Croce Rossa Italiana
    conv_id = get_last_conversation(agent_id, ELEVENLABS_API_KEY)
    if not conv_id:
        raise HTTPException(status_code=404, detail="Nessuna conversazione trovata.")
    details = get_conversation_details(conv_id, ELEVENLABS_API_KEY)
    if not details:
        raise HTTPException(status_code=404, detail="Errore nel recuperare i dettagli della conversazione.")
    transcript = details.get("transcript", [])
    global transcript_global
    transcript_global = transcript  # Salva il transcript per l'endpoint di estrazione contatti

    transcript_html = ""
    if transcript:
        for msg in transcript:
            role = msg.get("role", "unknown").capitalize()
            time_in_call_secs = msg.get("time_in_call_secs", "")
            message = msg.get("message", "")
            transcript_html += f"<p><strong>{role} [{time_in_call_secs}s]:</strong> {message}</p>"
    else:
        transcript_html = "<p>Nessun transcript disponibile.</p>"
    return {"transcript_html": transcript_html, "transcript": transcript}

@app.get("/api/extract_contacts")
def extract_contacts():
    global transcript_global
    if not transcript_global:
        raise HTTPException(status_code=404, detail="Nessun transcript disponibile per l'analisi dei contatti.")
    # Estrai solo i messaggi dell'utente
    user_messages = [msg.get("message", "") for msg in transcript_global if msg.get("role", "").lower() == "user"]
    transcript_text = "\n".join(user_messages)
    if not transcript_text.strip():
        raise HTTPException(status_code=404, detail="Nessun messaggio utente trovato per l'analisi dei contatti.")
    
    prompt_template_text = """
Analizza la seguente trascrizione di una conversazione tra un utente e un assistente virtuale.
Estrai, se presenti, l'indirizzo email e il numero di telefono dell'utente e riassumi dettagliatamente in maniera strutturata con tutti i dettagli rilevanti per la richiesta di assistenza riguardante procedure e regolamentazioni.
Rispondi nel seguente formato:
Email: <indirizzo email>
Telefono: <numero di telefono>
Riassunto: <riassunto dettagliato>

Se non trovi alcun dato, indica "Non trovato".
Se vedi qualche termine simile a "chiocciola" si tratta di un'email e cambiala con il carattere "@".

Trascrizione:
{transcript}
"""
    template = PromptTemplate(input_variables=["transcript"], template=prompt_template_text)
    contact_chain = LLMChain(llm=llm, prompt=template)
    contact_info = contact_chain.run(transcript=transcript_text)
    return {"contact_info": contact_info}

@app.post("/api/elevenlabs/webhook")
async def elevenlabs_webhook(webhook_data: ElevenLabsWebhook):
    """
    Endpoint per ricevere i webhook da ElevenLabs.
    
    Args:
        webhook_data: I dati ricevuti dal webhook
        
    Returns:
        La risposta per ElevenLabs
    """
    return await cri_assistant.elevenlabs_webhook_handler(webhook_data)

@app.post("/api/reset_conversation")
def reset_conversation_endpoint(request: ResetRequest):
    """
    Endpoint per resettare una conversazione.
    
    Args:
        request: La richiesta contenente l'ID di sessione
        
    Returns:
        Lo stato dell'operazione
    """
    return cri_assistant.reset_conversation(request.session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)