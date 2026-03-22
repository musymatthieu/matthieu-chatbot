import os
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder  # ✅ Import CrossEncoder
import ollama

# ✅ Même modèle que create_database.py (était all-MiniLM → buge critique)
embedding_model = SentenceTransformer("BAAI/bge-m3")

# ✅ CrossEncoder correctement importé
reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

client = chromadb.PersistentClient(path="./database")
collection = client.get_collection("matthieu")

print("Chatbot prêt ! Tape 'exit' pour quitter.")

chat_history = []

while True:

    question = input("\nPose ta question : ")

    if question.lower() == "exit":
        break

    # Enrichissement de la requête avec l'historique
    search_query = question
    if len(chat_history) >= 2:
        last_q1 = chat_history[-1][0]
        search_query = last_q2 + " " + last_q1 + " " + question

    question_embedding = embedding_model.encode(search_query)

    question_lower = question.lower()

    filter_metadata = None

    # ✅ Chemins avec os.path.join → compatible Linux/Mac/Windows
    if "projet" in question_lower:
        filter_metadata = {"folder": "projects"}
    elif "stage" in question_lower or "expérience" in question_lower:
        filter_metadata = {"folder": "experience"}
    elif "compétence" in question_lower:
        filter_metadata = {"folder": "skills"}

    # ✅ On récupère 20 candidats pour le reranker (au lieu de 4)
    query_params = dict(
        query_embeddings=[question_embedding.tolist()],
        n_results=40
    )
    if filter_metadata:
        query_params["where"] = filter_metadata

    results = collection.query(**query_params)

    candidates = results["documents"][0]
    metadatas_raw = results["metadatas"][0]

    # ✅ Re-ranking : le CrossEncoder score chaque paire (question, chunk)
    pairs = [[question, doc] for doc in candidates]
    scores = reranker.predict(pairs)

    # ✅ Tri par score reranker décroissant (le plus pertinent en premier)
    ranked = sorted(
        zip(scores, candidates, metadatas_raw),
        key=lambda x: x[0] + (0.5 * x[2].get("importance", 1)),
        reverse=True
    )

    # On prend les 4 meilleurs chunks après reranking
    top_ranked = ranked[:4]
    docs_sorted = [item[1] for item in top_ranked]
    metas_sorted = [item[2] for item in top_ranked]

    context = "\n\n".join(docs_sorted)

    sources = set(meta["source"] for meta in metas_sorted)
    print("\nSources utilisées :")
    for s in sources:
        print("-", s)

    print("\nDOCUMENTS RETRIEVED (après reranking) :")
    for score, _, meta in top_ranked:
        print(f"  [{score:.3f}] {meta['source']} (chunk {meta.get('chunk_index', '?')})")

    # Construction du prompt avec historique
    history_text = ""
    for q, r in chat_history[-3:]:
        history_text += f"Question précédente : {q}\nRéponse précédente : {r}\n\n"

    prompt = f"""
Tu es Matthieu Musy, ingénieur en informatique spécialisé en intelligence artificielle.
Tu réponds comme Matthieu Musy à un recruteur.

Règles :
- Réponds UNIQUEMENT avec les informations présentes dans le contexte ci-dessous.
- Si l'information n'est pas dans le contexte, dis "Je n'ai pas cette information sous la main".
- Ne déduis pas, n'invente pas, n'ajoute pas d'informations.
- Garde un ton professionnel et naturel.
- Ne te réintroduis pas si l'historique montre que la conversation est déjà engagée.
- Ne cite JAMAIS de chiffres, pourcentages ou métriques qui ne sont pas explicitement écrits dans le contexte.
- Si la question fait référence à "ce projet" ou "il", utilise l'HISTORIQUE pour identifier de quel projet il s'agit.

HISTORIQUE :
{history_text}

CONTEXTE :
{context}

QUESTION :
{question}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    print("\nRéponse :\n")
    print(answer)

    chat_history.append((question, answer))