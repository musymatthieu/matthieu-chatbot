# Guide de déploiement — Chatbot Matthieu Musy

## Structure du projet

```
chatbot/
├── main.py               ← Backend FastAPI
├── create_database.py    ← Script d'indexation (inchangé)
├── requirements.txt
├── render.yaml
├── .gitignore
├── data/                 ← Tes fichiers texte (inchangés)
│   ├── competence/
│   ├── experience/
│   ├── faq/
│   ├── projects/
│   └── profile/
├── database/             ← Générée par create_database.py
└── static/
    └── index.html        ← Interface web
```

---

## Étape 1 — Obtenir une clé Groq (gratuit)

1. Va sur https://console.groq.com
2. Crée un compte (gratuit)
3. Dans "API Keys" → "Create API Key"
4. Copie la clé (commence par `gsk_...`)

---

## Étape 2 — Préparer le projet en local

```bash
# Installe les nouvelles dépendances
pip install fastapi uvicorn groq python-multipart

# Régénère la base avec create_database.py si ce n'est pas fait
python create_database.py

# Teste en local
uvicorn main:app --reload
# → ouvre http://localhost:8000
```

---

## Étape 3 — Pousser sur GitHub

```bash
cd ton_dossier_chatbot
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/TON_USERNAME/matthieu-chatbot.git
git push -u origin main
```

> ⚠️ Le dossier `database/` est dans .gitignore — il faut le commit manuellement :
> ```bash
> git add -f database/
> git commit -m "add vector database"
> git push
> ```

---

## Étape 4 — Déployer sur Render

1. Va sur https://render.com → "New Web Service"
2. Connecte ton repo GitHub
3. Render détecte automatiquement `render.yaml`
4. Dans "Environment Variables" → ajoute :
   - Key: `GROQ_API_KEY`
   - Value: ta clé `gsk_...`
5. Clique "Deploy"

Render te donne une URL du type :
`https://matthieu-chatbot.onrender.com`

---

## Notes importantes

- **Free tier Render** : le service "dort" après 15min d'inactivité.
  Le premier chargement peut prendre ~30 secondes.
- **Groq gratuit** : 14 400 requêtes/jour avec llama3-70b — largement suffisant.
- **database/** commitée : la base vectorielle est embarquée dans le repo,
  pas besoin de la régénérer sur le serveur.
