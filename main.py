from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles
embedding_model = SentenceTransformer("BAAI/bge-m3")
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# ChromaDB
chroma_client = chromadb.PersistentClient(path="./database")
collection = chroma_client.get_collection("matthieu")

# Groq
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Historique par session (simple, en mémoire)
chat_histories = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    question = request.message

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_history = chat_histories[session_id]

    # Enrichissement de la requête avec la question précédente
    search_query = question
    if len(chat_history) >= 1:
        last_q = chat_history[-1]["question"]
        search_query = last_q + " " + question

    # Embedding + retrieval
    question_embedding = embedding_model.encode(search_query)
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=20
    )

    candidates = results["documents"][0]
    metadatas_raw = results["metadatas"][0]

    # Re-ranking
    pairs = [[question, doc] for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates, metadatas_raw),
        key=lambda x: x[0] + (0.5 * x[2].get("importance", 1)),
        reverse=True
    )

    top_ranked = ranked[:4]
    docs_sorted = [item[1] for item in top_ranked]
    metas_sorted = [item[2] for item in top_ranked]

    context = "\n\n".join(docs_sorted)
    sources = list(set(meta["source"] for meta in metas_sorted))

    # Historique
    history_text = ""
    for entry in chat_history[-3:]:
        history_text += f"Question précédente : {entry['question']}\nRéponse précédente : {entry['answer']}\n\n"

    prompt = f"""<|system|>
Tu es Matthieu Musy. Tu parles à la première personne à un recruteur.
RÈGLES ABSOLUES :
- Utilise UNIQUEMENT les informations du CONTEXTE ci-dessous.
- Ne te présente PAS si l'HISTORIQUE montre que la conversation est déjà engagée.
- Aucun chiffre, pourcentage ou métrique absent du CONTEXTE.
- Si l'info est absente du CONTEXTE : "Je n'ai pas cette information précise en tête."
- Réponds en français, ton professionnel et naturel.
</|system|>

<|historique|>
{history_text}
</|historique|>

<|contexte|>
{context}
</|contexte|>

<|question|>
{question}
</|question|>

<|réponse|>"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    chat_histories[session_id].append({
        "question": question,
        "answer": answer
    })

    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
async def health():
    return {"status": "ok"}


# Sert le frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")