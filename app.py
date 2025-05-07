# app.py

import streamlit as st
import os
import time
import pandas as pd 
import logging
import json
from datetime import date, timedelta # timedelta n'est pas utilisé ici mais gardé
from typing import Dict, Any, List, Optional


from src.backend.rag_orchestrator import init_rag_system as init_rag_backend 
from src.frontend.styles import CSS_STYLES, FOOTER_HTML 
from src.frontend.components import ( 
    display_dossier_timeline,
    display_dossier_details_enhanced,
    display_source, # MODIFIÉ: anciennement display_source_document
    display_dossier_summary,
    display_dossier_details_df
)
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
        'sources_expanded': False, # Ce toggle contrôlera l'expansion par défaut des sources individuelles
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


# --- Fonctions Utilitaires de l'UI (celles qui restent dans app.py) ---
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
            "history": st.session_state.chat_history[-5:] 
        }
        
        start_time = time.time()
        result = graph.invoke(initial_state) 
        processing_time = time.time() - start_time
        logger.info(f"Requête traitée en {processing_time:.2f}s. Résultat: {type(result)}")
        
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
        st.session_state.show_sources = st.toggle("Afficher sources", value=st.session_state.show_sources, key="toggle_src")
        st.session_state.show_db_results = st.toggle("Afficher détails dossiers trouvés", value=st.session_state.show_db_results, key="toggle_db")
        # Ce toggle contrôle maintenant si les sources individuelles sont développées par défaut
        st.session_state.sources_expanded = st.toggle("Développer sources", value=st.session_state.sources_expanded, key="toggle_exp_src")

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
    
    if st.session_state.last_result and isinstance(st.session_state.last_result, dict) :
        sources = st.session_state.last_result.get('context', [])
        if st.session_state.show_sources and isinstance(sources, list) and sources:
            # MODIFIÉ: Suppression de l'expander externe.
            # Le titre est maintenant un simple markdown.
            st.markdown(f"### 📚 {len(sources)} source(s) utilisée(s)")
            for i, src_doc in enumerate(sources):
                # MODIFIÉ: Appel à display_source et passage de st.session_state.sources_expanded
                display_source(src_doc, i, expanded_by_default=st.session_state.sources_expanded)
        
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
        term = dossier_search_term
        statut = st.session_state.filter_statut if st.session_state.filter_statut != "Tous" else None
        instr = st.session_state.filter_instructeur if st.session_state.filter_instructeur != "Tous" else None
        dd = st.session_state.filter_date_debut
        df = st.session_state.filter_date_fin

        if not any([term, statut, instr, dd, df]):
            st.warning("Veuillez entrer un terme ou sélectionner un filtre.")
            st.session_state.dossier_search_results = []
        else:
            search_func = st.session_state.rag_components.get("search_function") 
            if callable(search_func):
                with st.spinner("Recherche des dossiers..."):
                    try:
                        found = search_func(
                            search_term=term if term else None, statut=statut, instructeur=instr,
                            date_debut_creation=dd, date_fin_creation=df
                        )
                        st.session_state.dossier_search_results = found
                        if found: st.success(f"{len(found)} dossier(s) trouvé(s).")
                        else: st.info("Aucun dossier ne correspond à vos critères.")
                    except Exception as e:
                        logger.error(f"Erreur recherche dossier (UI): {e}", exc_info=True)
                        st.error(f"Erreur recherche: {e}")
                        st.session_state.dossier_search_results = None
            else:
                st.error("Fonction de recherche de dossier non disponible.")
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

# --- Zone de Saisie du Chat (toujours en bas) ---
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