# Uso come webhook per ElevenLabs:
# Esegui direttamente: python rag.py
# Quindi invia richieste POST a http://tuo-server:5000/elevenlabs-webhook

# Oppure importalo in un altro script:
from rag import query_rag

risposta = query_rag("primo articolo del manuale rfl?", session_id="my-session-123")
print(risposta["answer"])