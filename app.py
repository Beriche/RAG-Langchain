from huggingface_hub import upload_file
import streamlit as st
import os
import time
import pandas as pd
import logging
import json

from dotenv import load_dotenv
from src.backend.rag_main import init_rag_system as init_rag_backend
from src.frontend.styles import CSS_STYLES, FOOTER_HTML
from src.frontend.components import (
    display_dossier_details_enhanced,
    display_dossier_summary,
)

load_dotenv()
# Définition plus robuste de DATA_ROOT pour app.py
# En supposant que app.py est à la racine du projet.
_PROJECT_ROOT_APP = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT_APP_DEFAULT = os.path.join(_PROJECT_ROOT_APP, "data")
DATA_ROOT = os.getenv("DATA_ROOT", DATA_ROOT_APP_DEFAULT)

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streamlit_app")

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Assistant KAP Numérique",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@kap-numerique.fr',
        'Report a bug': "mailto:support@kap-numerique.fr",
        'About': """
        **Assistant KAP Numérique Beta v1.0**
        Cet assistant utilise un système RAG pour répondre aux questions des instructeurs.
        """
    }
)

# --- Variables d'état (Session State) ---
def init_session_state_vars():
    defaults = {
        'app_mode': "general",
        'initialized': False,
        'chat_history': [],
        'is_processing': False,
        'rag_components': None,
        'sources_table_expanded': False, 
        'show_sources': True, 
        'show_db_results': True,
        'last_result': None,
        'error_message': None,
        'system_status_msg': "Système non initialisé.",
        'dossier_search_results': None,
        'filter_statut': "Tous",
        'filter_date_debut': None,
        'filter_instructeur': "Tous",
        'filter_date_fin': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state_vars()

# --- Importation et Initialisation du Module RAG (Backend) ---
rag_import_success = True
try:
    if not st.session_state.initialized:
        placeholder = st.empty()
        with placeholder.container():
            with st.spinner("⏳ Initialisation de l'assistant, veuillez patienter..."):
                try:
                    start_time = time.time()
                    st.session_state.rag_components = init_rag_backend()
                    
                    if isinstance(st.session_state.rag_components, dict) and \
                       st.session_state.rag_components.get("graph_ok") and \
                       st.session_state.rag_components.get("llm_ok") and \
                       st.session_state.rag_components.get("embeddings_ok"):
                        st.session_state.initialized = True
                        st.session_state.system_status_msg = "✅ Système prêt."
                        st.session_state.error_message = None
                        duration = time.time() - start_time
                        st.success(f"Assistant initialisé en {duration:.2f}s !")
                        time.sleep(2)
                    else:
                        error_details = st.session_state.rag_components.get("error_messages", ["Détail inconnu"])
                        logger.error(f"Échec partiel de l'initialisation du backend RAG: {error_details}")
                        st.session_state.initialized = False
                        st.session_state.error_message = f"❌ Erreur initialisation backend: {', '.join(error_details)}. Assistant limité."
                        st.session_state.system_status_msg = "❌ Erreur init. backend."
                        st.error(st.session_state.error_message)

                except Exception as e:
                    logger.error(f"Erreur critique lors de l'appel à init_rag_backend: {e}", exc_info=True)
                    st.session_state.initialized = False
                    st.session_state.error_message = f"❌ Erreur critique init: {e}."
                    st.session_state.system_status_msg = "❌ Erreur critique init."
                    st.error(st.session_state.error_message)
        placeholder.empty()
except Exception as e:
    rag_import_success = False
    st.session_state.initialized = False
    st.session_state.system_status_msg = "❌ Module RAG Orchestrator introuvable."
    st.error(f"Impossible de charger le module backend: {e}")
    logger.error(f"Échec import src.rag_orchestrator: {e}", exc_info=True)

# --- Application des Styles CSS ---
st.markdown(CSS_STYLES, unsafe_allow_html=True)


# --- Fonctions Utilitaires de l'UI 
def save_chat_history_to_file():
    try:
        history_dir = os.path.join(".", "data", "chat_history")
        os.makedirs(history_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(history_dir, f"chat_{timestamp}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
        logger.info(f"Historique sauvegardé: {filename}")
        return True, filename
    except Exception as e:
        logger.error(f"Erreur sauvegarde historique: {e}", exc_info=True)
        return False, None

def load_chat_history_from_file(filename: str) -> bool:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            st.session_state.chat_history = json.load(f)
        logger.info(f"Historique chargé depuis {filename}")
        st.session_state.last_result = None
        return True
    except FileNotFoundError:
        st.session_state.error_message = f"Fichier {filename} non trouvé."
        return False
    except json.JSONDecodeError:
        st.session_state.error_message = f"Fichier {filename} corrompu."
        return False
    except Exception as e:
        st.session_state.error_message = f"Erreur chargement: {e}"
        return False

def process_user_query(query: str) -> str:
    if not st.session_state.initialized or not st.session_state.rag_components:
        return "Système RAG non initialisé. Veuillez réessayer ou vérifier les logs."
    
    st.session_state.is_processing = True
    try:
        graph = st.session_state.rag_components.get("graph")
        if not graph:
            return "Composant Graph du RAG non trouvé. Vérifiez l'initialisation."
        
        initial_state = {
            "question": query,
            "context": [], "db_results": [], "answer": "",
            "history": st.session_state.chat_history[-5:] # récupere les 5 dernier message, pour garder le context
        }
        
        start_time = time.time()
        result = graph.invoke(initial_state)
        processing_time = time.time() - start_time
        logger.info(f"Requête traitée en {processing_time:.2f}s.")
        
        st.session_state.last_result = result
        return result.get("answer", "Pas de réponse claire obtenue.")

    except Exception as e:
        logger.error(f"Erreur traitement requête: {e}", exc_info=True)
        return f"Erreur lors du traitement: {e}"
    finally:
        st.session_state.is_processing = False

# ===== INTERFACE PRINCIPALE =====
if not st.session_state.initialized and rag_import_success:
     st.error("L'initialisation de l'assistant a échoué. Certaines fonctionnalités peuvent être indisponibles.")
elif not rag_import_success:
     st.error("Le module backend RAG n'a pas pu être chargé. L'application ne peut pas démarrer correctement.")
     st.stop()

# --- Barre Latérale ---
with st.sidebar:
    st.image("src/frontend/logo_region_reunion.png", width=150)
    st.header("⚙️ Contrôle Système")
    st.title("Assistant KAP Numérique")

    status_text = st.session_state.system_status_msg
    if "✅" in status_text: status_color = "green"
    elif "⏳" in status_text or "Initialisation" in status_text : status_color = "orange"
    else: status_color = "red"
    st.markdown(f"**Statut:** <span style='color:{status_color};'>{status_text}</span>", unsafe_allow_html=True)

    with st.expander("👁️ Options d'Affichage", expanded=True):
        st.session_state.show_sources = st.toggle(
            "Afficher section sources", 
            value=st.session_state.show_sources, 
            key="toggle_show_src_section"
        )
    
        if st.session_state.show_sources: # On n'affiche ce toggle que si la section des sources est visible
            st.session_state.sources_table_expanded = st.toggle(
                "Développer tableau sources", 
                value=st.session_state.sources_table_expanded, 
                key="toggle_expand_src_table"
            )
        
        st.session_state.show_db_results = st.toggle(
            "Afficher détails dossiers trouvés", 
            value=st.session_state.show_db_results, 
            key="toggle_db"
        )


    with st.expander("🕒 Gestion Historique", expanded=False):
        col_h1, col_h2 = st.columns(2)
        if col_h1.button("💾 Sauvegarder", use_container_width=True, key="save_h"):
            success, fname = save_chat_history_to_file()
            if success: st.success(f"Sauvegardé: `{fname}`")
            else: st.error("Échec sauvegarde.")
        if col_h2.button("🗑️ Effacer", use_container_width=True, key="clear_h"):
            if st.session_state.chat_history:
                st.session_state.chat_history = []
                st.session_state.last_result = None
                st.success("Historique effacé.")
                st.rerun()
            else: st.info("Historique déjà vide.")
    
    with st.expander("❓ Aide", expanded=False):
        st.markdown("""
        **Utilisation :**
        1.  Le système s'initialise automatiquement.
        2.  **Questions Générales**: infos sur le dispositif.
        3.  **Consultation de Dossier**: rechercher et interroger un dossier.
        """)

    if st.session_state.error_message:
        st.error(f"🚨 Erreur: {st.session_state.error_message}")
        if st.button("❌ Effacer Erreur", key="clear_err_btn"):
            st.session_state.error_message = None
            st.rerun()
            
            
    #gestion ajout de nouveaux source de connaissance 
    with st.sidebar.expander("📤 Gestion des Connaissances",expanded=False):
        uploaded_files = st.file_uploader(
            "Ajouter des documents (PDF,TXT)",
            type=["pdf","txt"], 
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        
        
        if uploaded_files:
            if st.button("Traiter les fichiers uploadés",key="process_uploads_btn"):
            
                st.success(f"{len(uploaded_files)} fichier(s) pris en compte. Reconstruction de la base utilisateur en cours...") 
                
                USER_UPLOADS_PATH = os.path.join(DATA_ROOT, "user_uploads") # Ou DATA_ROOT
                os.makedirs(USER_UPLOADS_PATH, exist_ok=True)
                
                saved_files_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(USER_UPLOADS_PATH,uploaded_file.name)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        saved_files_paths.append(file_path)
                        logger.info(f"Fichier sauvegardé: {file_path}")
                
                # Appeler une fonction pour (re)construire le vector store utilisateur
                if st.session_state.rag_components and callable(st.session_state.rag_components.get("update_user_vector_store")):
                    update_user_vs_func = st.session_state.rag_components.get("update_user_vector_store")
                    
                    with st.spinner("Mise à jour de la base de connaissance utilisateur..."):
                        success_update = update_user_vs_func()
                        
                    if success_update:
                        st.success("Base de connaissance utilisateur mise à jour ! ")
                    else:
                        st.error("Erreur lors de la mise à jour ! ")
                else:
                    st.error("Fonction de mise à jour du vector store utilisteur non disponible.")
                    

# --- Zone Principale ---
st.title("🤖 Assistant KAP Numérique")
st.caption("Votre assistant intelligent pour le dispositif KAP Numérique.")

tab_general, tab_dossier_consult = st.tabs(["💬 Questions Générales", "📁 Consultation de Dossier"])

# --- Onglet Questions Générales ---
with tab_general:
    st.header("Posez une question générale")
    chat_container_general = st.container(height=500)
    with chat_container_general:
        if not st.session_state.chat_history:
            welcome_msg = "👋 Bonjour ! Posez-moi une question sur le dispositif KAP Numérique."
            if not rag_import_success: welcome_msg = "⚠️ Module RAG non chargé. Fonctionnalités limitées."
            elif not st.session_state.initialized: welcome_msg = "⏳ Assistant en cours d'initialisation..."
            st.info(welcome_msg)
        else:
            for message in st.session_state.chat_history:
                avatar = "👤" if message["role"] == "user" else "🤖"
                with st.chat_message(message["role"], avatar=avatar):
                    st.write(message["content"])
    
    # Affichage des sources et détails si applicable (après le chat)
    if st.session_state.last_result and isinstance(st.session_state.last_result, dict) :
        
        sources_docs = st.session_state.last_result.get('context', [])
        
        if st.session_state.show_sources and isinstance(sources_docs, list) and sources_docs:
            st.markdown("---") 
           
            expander_label = f"📊 Sources consultées ({len(sources_docs)} utilisée(s))"
            with st.expander(expander_label, expanded=st.session_state.sources_table_expanded):
              
                source_data_for_df = []
                for i, doc in enumerate(sources_docs):
                    try:
                        source_name = doc.metadata.get('source', 'N/A')
                        doc_type = doc.metadata.get('type', doc.metadata.get('document_type', 'N/A'))
                        content_preview = doc.page_content
                        if len(content_preview) > 150:
                            content_preview = content_preview[:150] + "..."
                        
                        source_data_for_df.append({
                            "Source": source_name,
                            "Type": doc_type,
                            "Contenu": content_preview
                        })
                    except AttributeError:
                        logger.warning(f"La source {i+1} (type: {type(doc)}) n'a pas les attributs 'metadata' ou 'page_content' attendus.")
                        source_data_for_df.append({"Source": "Erreur de format", "Type": "N/A", "Contenu": "Impossible d'extraire."})
                    except Exception as e:
                        logger.error(f"Erreur inattendue lors du traitement de la source {i+1}: {e}")
                        source_data_for_df.append({"Source": "Erreur inconnue", "Type": "N/A", "Contenu": f"Erreur: {e}"})

                if source_data_for_df:
                    df_sources = pd.DataFrame(source_data_for_df)
                    st.dataframe(
                        df_sources, 
                        use_container_width=True, 
                        hide_index=False
                    )
        
        db_results = st.session_state.last_result.get('db_results', [])
        if st.session_state.show_db_results and isinstance(db_results, list) and db_results:
            with st.expander("📋 Dossiers Mentionnés/Trouvés", expanded=True): 
                for dossier_data in db_results:
                    if isinstance(dossier_data, dict):
                        display_dossier_summary(dossier_data)
                    else:
                        st.warning(f"Format de résultat dossier BDD inattendu: {type(dossier_data)}")

# --- Onglet Consultation de Dossier ---
with tab_dossier_consult:
    st.header("Rechercher et consulter un dossier")
    
    with st.expander("🔍 Afficher les Filtres", expanded=False):
        get_distinct_values_func = None
        if st.session_state.rag_components:
            get_distinct_values_func = st.session_state.rag_components.get('get_distinct_values_function')

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            statuts_opts = ["Tous"]
            if callable(get_distinct_values_func):
                try: statuts_opts.extend(get_distinct_values_func('statut'))
                except Exception as e: logger.warning(f"Err chargement statuts: {e}")
            st.session_state.filter_statut = st.selectbox("Statut", statuts_opts, key="filt_stat")
            st.session_state.filter_date_debut = st.date_input("Créés après le", key="filt_dd")
        with col_f2:
            instr_opts = ["Tous"]
            if callable(get_distinct_values_func):
                try: instr_opts.extend(get_distinct_values_func('instructeur'))
                except Exception as e: logger.warning(f"Err chargement instructeurs: {e}")
            st.session_state.filter_instructeur = st.selectbox("Instructeur", instr_opts, key="filt_inst")
            st.session_state.filter_date_fin = st.date_input("Créés avant le", key="filt_df")

    s_col, b_col = st.columns([4,1])
    dossier_search_term = s_col.text_input("N° dossier ou Nom usager...", placeholder="XX-YYYY ou terme...", label_visibility="collapsed", key="doss_search_term")
    search_doss_btn = b_col.button("🔎 Rechercher", use_container_width=True, key="search_doss_btn", disabled=not st.session_state.initialized)

    if search_doss_btn:
        term = dossier_search_term.strip() # .strip() pour enlever les espaces en trop
        statut_param = st.session_state.filter_statut if st.session_state.filter_statut != "Tous" else None
        instr_param = st.session_state.filter_instructeur if st.session_state.filter_instructeur != "Tous" else None
        
        # Récupérer les dates de l'expander
        date_debut_expander = st.session_state.filter_date_debut
        date_fin_expander = st.session_state.filter_date_fin

        # Déterminer les dates à envoyer au backend
        final_date_debut = None
        final_date_fin = None

        if not term: # Si la barre de recherche principale est vide, on utilise les filtres de l'expander (y compris les dates)
            final_date_debut = date_debut_expander
            final_date_fin = date_fin_expander
      
        if not any([term, statut_param, instr_param, final_date_debut, final_date_fin]):
            st.warning("Veuillez entrer un terme ou sélectionner au moins un filtre.")
            st.session_state.dossier_search_results = []
        else:
            search_func = st.session_state.rag_components.get("search_function") 
            if callable(search_func):
                with st.spinner("Recherche des dossiers..."):
                    try:
                        logger.info(f"Appel recherche frontend: term='{term}', statut='{statut_param}', instr='{instr_param}', dd='{final_date_debut}', df='{final_date_fin}'")
                        found = search_func(
                            search_term=term if term else None, # Le backend gère si 'term' est un numéro exact
                            statut=statut_param,
                            instructeur=instr_param,
                            date_debut_creation=final_date_debut, # Utiliser les dates conditionnellement définies
                            date_fin_creation=final_date_fin    # Utiliser les dates conditionnellement définies
                        )
                        st.session_state.dossier_search_results = found
                        if found: 
                            st.success(f"{len(found)} dossier(s) trouvé(s).")
                        else: 
                            st.info("Aucun dossier ne correspond à vos critères de recherche.")
                    except Exception as e:
                        logger.error(f"Erreur recherche dossier (UI): {e}", exc_info=True)
                        st.error(f"Erreur lors de la recherche: {str(e)}") 
                        
                        st.session_state.dossier_search_results = None # Réinitialiser en cas d'erreur
            else:
                st.error("Fonction de recherche de dossier non disponible ou non initialisée.")
                st.session_state.dossier_search_results = None
    
    if 'dossier_search_results' in st.session_state and st.session_state.dossier_search_results:
        results = st.session_state.dossier_search_results
        st.markdown(f"--- \n### {len(results)} Dossier(s) Trouvé(s)")
        for idx, dossier_item in enumerate(results):
            if isinstance(dossier_item, dict):
                display_dossier_details_enhanced(dossier_item, idx) 
            else:
                st.warning(f"Format résultat dossier inattendu (idx {idx}): {type(dossier_item)}")
                st.json(dossier_item)
                
# --- Traitement Centralisé des Questions (après chaque interaction) ---
if st.session_state.chat_history and \
   st.session_state.chat_history[-1]["role"] == "user" and \
   not st.session_state.is_processing:
    
    last_user_query = st.session_state.chat_history[-1]["content"]
    with st.spinner("🧠 Traitement de votre question..."):
        assistant_response = process_user_query(last_user_query)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    st.rerun() 


user_prompt = st.chat_input(
    "Posez votre question ici...", 
    key="main_chat_input",
    disabled= not st.session_state.initialized or st.session_state.is_processing
)
if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    st.rerun()
# --- Footer ---
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
