import streamlit as st
import openai
import json
import numpy as np
import faiss

# ========== Configuration de la page ==========
st.set_page_config(page_title="Cross-Selling RAG", page_icon="🛒", layout="centered")
st.title("🛒 Générateur de Cross-Selling (Retrieval + Generation)")
st.caption("Récupération de candidats dans le catalogue, puis génération de recommandations par le LLM.")

# ========== Catalogue de produits simulé (à remplacer par votre base de données réelle) ==========
# Dans un vrai projet, ces données proviendraient de votre base de données ou API produits
CATALOG = [
    {"id": "p001", "name": "Appareil photo reflex", "desc": "Boîtier plein format 24MP, monture EF", "category": "Appareil photo"},
    {"id": "p002", "name": "Carte mémoire SD 128Go", "desc": "UHS-II, lecture 300Mo/s, pour photos 4K", "category": "Accessoire"},
    {"id": "p003", "name": "Trépied aluminium", "desc": "Hauteur max 1.8m, rotule ball, charge 8kg", "category": "Accessoire"},
    {"id": "p004", "name": "Sac à dos photo", "desc": "Compartiments rembourrés, résistant à l'eau, 25L", "category": "Accessoire"},
    {"id": "p005", "name": "Objectif 50mm f/1.8", "desc": "Objectif standard lumineux, monture EF", "category": "Objectif"},
    {"id": "p006", "name": "Ordinateur portable", "desc": "14 pouces, 16Go RAM, SSD 512Go, léger", "category": "Informatique"},
    {"id": "p007", "name": "Souris sans fil", "desc": "Ergonomique, Bluetooth, autonomie 6 mois", "category": "Accessoire"},
    {"id": "p008", "name": "Sacoche ordinateur 14\"", "desc": "Résistante à l'eau, compartiment rembourré", "category": "Accessoire"},
    {"id": "p009", "name": "Hub USB-C 7-en-1", "desc": "HDMI, USB 3.0, lecteur SD, charge 100W", "category": "Accessoire"},
    {"id": "p010", "name": "Écran externe 27\"", "desc": "4K, IPS, USB-C, hauteur réglable", "category": "Périphérique"},
]

# ========== Initialisation des Embeddings du catalogue (mise en cache) ==========
@st.cache_resource(show_spinner="Indexation du catalogue en cours...")
def build_faiss_index(api_key: str):
    """Calcule les embeddings du catalogue et construit l'index FAISS."""
    client = openai.OpenAI(api_key=api_key)
    
    # Concatène nom + description pour un embedding plus riche
    texts = [f"{p['name']}. {p['desc']}" for p in CATALOG]
    
    # Génération par lots des embeddings
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    vectors = np.array([e.embedding for e in response.data]).astype('float32')
    
    # Construction de l'index FAISS (recherche L2)
    dimension = vectors.shape[1]  # 1536 pour text-embedding-3-small
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    
    return index, vectors, client

# ========== Barre latérale ==========
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Clé API OpenAI", type="password")
    top_k = st.slider("Nombre de candidats à récupérer", min_value=3, max_value=10, value=5)
    st.markdown("---")
    st.info("Le système récupère d'abord les produits similaires dans le catalogue, puis le LLM filtre et justifie les compléments pertinents.")

# ========== Formulaire ==========
with st.form("rag_cross_sell"):
    product_name = st.text_input("Nom du produit principal", placeholder="Ex: Appareil photo reflex")
    product_desc = st.text_area("Description (optionnel)", placeholder="Précisez le modèle, l'usage...")
    submitted = st.form_submit_button("Générer les recommandations")

# ========== Traitement ==========
if submitted:
    if not api_key:
        st.error("Veuillez saisir votre clé API OpenAI.")
    elif not product_name:
        st.warning("Veuillez indiquer le nom du produit principal.")
    else:
        try:
            # --- Étape 1 : Construction de l'index (mise en cache) ---
            index, catalog_vectors, client = build_faiss_index(api_key)
            
            # --- Étape 2 : Retrieval (récupération vectorielle) ---
            query_text = f"{product_name}. {product_desc}" if product_desc else product_name
            query_resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=[query_text]
            )
            query_vec = np.array(query_resp.data[0].embedding).astype('float32').reshape(1, -1)
            
            # Recherche des top_k voisins les plus proches
            distances, indices = index.search(query_vec, top_k)
            
            # Construction de la liste des candidats
            candidates = []
            for i in indices[0]:
                if i < len(CATALOG):
                    candidates.append(CATALOG[i])
            
            # --- Étape 3 : Generation (le LLM filtre et justifie) ---
            system_prompt = """Tu es un expert en cross-selling.
On te fournit un produit principal et une liste de produits candidats récupérés dans un catalogue.
Ta mission : sélectionner parmi les candidats UNIQUEMENT ceux qui sont de VRAIS compléments pertinents pour le produit principal.
Ignore les candidats non pertinents. Si aucun candidat n'est pertinent, retourne une liste vide.

Réponds UNIQUEMENT en JSON avec cette structure :
{
  "main_product": "nom du produit principal",
  "recommendations": [
    {
      "product_id": "id du candidat",
      "name": "nom du produit",
      "reason": "pourquoi c'est un bon complément (1 phrase)",
      "category": "catégorie du produit"
    }
  ]
}"""

            user_prompt = f"""Produit principal : {product_name}
{f'Description : {product_desc}' if product_desc else ''}

Produits candidats récupérés :
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Sélectionne les compléments pertinents parmi ces candidats."""

            with st.spinner("Le LLM analyse les candidats et génère les recommandations..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3  # Température basse pour rester factuel
                )
            
            result = json.loads(response.choices[0].message.content)
            
            # --- Affichage ---
            st.success(f"Recommandations pour : **{result.get('main_product', product_name)}**")
            
            # Debug : afficher les candidats récupérés (optionnel)
            with st.expander("🔍 Candidats récupérés (Retrieval)"):
                for c in candidates:
                    st.write(f"- **{c['name']}** ({c['category']})")
            
            recs = result.get("recommendations", [])
            if not recs:
                st.warning("Le LLM n'a trouvé aucun complément pertinent parmi les candidats récupérés.")
            else:
                st.subheader(f"✅ {len(recs)} complément(s) pertinent(s)")
                for i, rec in enumerate(recs, 1):
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{i}. {rec.get('name', 'Sans nom')}**")
                            st.write(rec.get("reason", ""))
                        with col2:
                            st.caption(f"🏷️ {rec.get('category', 'N/A')}")
                            st.caption(f"ID: {rec.get('product_id', 'N/A')}")

        except json.JSONDecodeError:
            st.error("Le LLM n'a pas renvoyé un JSON valide. Réessayez.")
        except Exception as e:
            st.error(f"Erreur : {e}")