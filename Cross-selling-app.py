import streamlit as st
import openai
import json

# --- Configuration de la page ---
st.set_page_config(page_title="Cross-Selling AI", page_icon="🛒", layout="centered")

st.title("🛒 Générateur de Cross-Selling")
st.caption("Trouvez des produits complémentaires à vendre avec votre article principal.")

# --- Barre latérale : Clé API ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Clé API OpenAI", type="password")
    model_choice = st.selectbox("Modèle", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    st.markdown("---")
    st.info("L'IA analyse le produit principal et suggère des accessoires ou services complémentaires.")

# --- Formulaire de saisie ---
with st.form("cross_sell_form"):
    product_name = st.text_input("Nom du produit principal", placeholder="Ex: Appareil photo reflex, Ordinateur portable, TV 4K...")
    product_desc = st.text_area("Description courte (optionnel)", placeholder="Précisez le modèle, la marque, l'usage prévu...")
    
    submitted = st.form_submit_button("Générer les recommandations")

# --- Traitement ---
if submitted:
    if not api_key:
        st.error("Veuillez saisir votre clé API OpenAI dans la barre latérale.")
    elif not product_name:
        st.warning("Veuillez indiquer le nom du produit principal.")
    else:
        # Initialisation du client OpenAI
        client = openai.OpenAI(api_key=api_key)

        # Construction du prompt
        system_prompt = """Tu es un expert en stratégie commerciale et en cross-selling.
À partir du produit principal fourni par l'utilisateur, propose 5 produits ou services complémentaires pertinents.
Chaque suggestion doit être directement utile à l'acheteur du produit principal.
Réponds UNIQUEMENT en JSON avec la structure suivante :
{
  "main_product": "nom du produit",
  "recommendations": [
    {
      "name": "nom du produit complémentaire",
      "reason": "pourquoi c'est pertinent (1 phrase)",
      "category": "catégorie (ex: Accessoire, Consommable, Service, Garantie)"
    }
  ]
}"""

        user_prompt = f"Produit principal : {product_name}"
        if product_desc:
            user_prompt += f"\nDescription : {product_desc}"

        try:
            with st.spinner("Analyse du produit et génération des recommandations..."):
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )

            # Parsing de la réponse
            result = json.loads(response.choices[0].message.content)

            # --- Affichage des résultats ---
            st.success(f"Recommandations pour : **{result.get('main_product', product_name)}**")
            
            st.subheader("Produits complémentaires suggérés")
            
            for i, rec in enumerate(result.get("recommendations", []), 1):
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{i}. {rec.get('name', 'Sans nom')}**")
                        st.write(rec.get("reason", ""))
                    with col2:
                        st.caption(f"🏷️ {rec.get('category', 'N/A')}")

        except json.JSONDecodeError:
            st.error("L'IA n'a pas renvoyé un JSON valide. Essayez à nouveau.")
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")

# --- Pied de page ---
st.markdown("---")
st.caption("Astuce : Utilisez `gpt-4o-mini` pour un bon rapport qualité/prix sur ce type de tâche.")