import os
import logging
from typing import Dict, Any, Optional, List 

from dotenv import load_dotenv

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings

from .db_manager import DatabaseManager
from .document_processor import load_all_documents, split_documents, load_user_uploaded_documents
from .vector_store_utils import create_vector_store,update_user_vs_and_get_updated_graph
from .rag_pipeline import build_graph_with_deps

logger = logging.getLogger(__name__)
load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT", "../../data") 
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # Racine du projet
DATA_ROOT_DEFAULT = os.path.join(_PROJECT_ROOT, "data")
CACHE_DIR_DEFAULT = os.path.join(_PROJECT_ROOT, "cache_embeddings") # Pour les embeddings

DATA_ROOT = os.getenv("DATA_ROOT", DATA_ROOT_DEFAULT)
ECHANGES_PATH = os.path.join(DATA_ROOT, "echanges")
REGLES_PATH = os.path.join(DATA_ROOT, "regles")
OFFICIAL_DOCS_PATH = os.path.join(DATA_ROOT, "docs_officiels")
USER_UPLOADS_PATH = os.path.join(DATA_ROOT, "user_uploads")
CACHE_DIR = os.getenv("CACHE_DIR", CACHE_DIR_DEFAULT) # Où les FAISS index sont sauvegardés
USER_VS_CACHE_DIR = os.path.join(CACHE_DIR, "user_docs_store")

#Variables globale
rules_vector_store: Optional[VectorStore] = None
official_docs_vector_store: Optional[VectorStore] = None
echanges_vector_store: Optional[VectorStore] = None
user_docs_vector_store: Optional[VectorStore] = None

llm: Optional[Any] = None
embeddings: Optional[Embeddings] = None
db_connected: bool = False
db_manager: Optional[DatabaseManager] = None


def trigger_user_vs_and_graph_update_from_rag_main() -> bool:
    global user_docs_vector_store, llm, embeddings, db_manager, db_connected
    global rules_vector_store, official_docs_vector_store, echanges_vector_store
    # Import Streamlit ici seulement pour la mise à jour de session_state, si on garde cette logique ici
    # Sinon, cette partie devrait être gérée par app.py après le retour de cette fonction.
    # Pour la simplicité de l'appel depuis app.py, on peut le garder ici pour l'instant.
    import streamlit as st # Attention, cela couple un peu rag_main à Streamlit

    if not embeddings: # Vérification essentielle
        logger.error("Embeddings non initialisés. Mise à jour annulée.")
        if 'rag_components' in st.session_state and st.session_state.rag_components:
            st.session_state.rag_components["user_docs_store_ok"] = False
        return False

    new_vs, new_graph, success = update_user_vs_and_get_updated_graph(
        embeddings_instance=embeddings,
        llm_instance=llm,
        db_manager_instance=db_manager,
        db_connection_status=db_connected,
        rules_vs_instance=rules_vector_store,
        official_vs_instance=official_docs_vector_store,
        echanges_vs_instance=echanges_vector_store,
        user_uploads_path=USER_UPLOADS_PATH,
        user_vs_cache_path=USER_VS_CACHE_DIR,
        current_user_vs=user_docs_vector_store
    )

    if success and new_vs is not None and new_graph is not None:
        user_docs_vector_store = new_vs # Mettre à jour la globale
        logger.info("Global user_docs_vector_store mis à jour.")
        
        # Mettre à jour le graphe dans st.session_state
        # Cela suppose que init_rag_system a déjà peuplé rag_components
        if 'rag_components' in st.session_state and st.session_state.rag_components:
            st.session_state.rag_components["graph"] = new_graph
            st.session_state.rag_components["user_docs_store_ok"] = True
            logger.info("Graphe RAG dans st.session_state mis à jour.")
        else:
            logger.warning("st.session_state.rag_components non trouvé pour la mise à jour du graphe.")
        return True
    else:
        logger.error("Échec de la mise à jour du VS utilisateur ou du graphe depuis vector_store_utils.")
        if 'rag_components' in st.session_state and st.session_state.rag_components:
            st.session_state.rag_components["user_docs_store_ok"] = False
        # Potentiellement mettre à jour user_docs_vector_store avec new_vs même si le graphe a échoué
        if new_vs is not None:
            user_docs_vector_store = new_vs
        return False

def init_rag_system() -> Dict[str, Any]:
    """Initialise tous les composants du système RAG (LLM, Embeddings, DB, Vector Stores, Graphe).
    """
    
    # Utiliser les variables globales pour stocker les objets initialisés
    global llm, embeddings, db_manager, db_connected
    global rules_vector_store, official_docs_vector_store, echanges_vector_store , user_docs_vector_store

    logger.info("="*20 + " Initialisation du Système RAG " + "="*20)
    status = {
        "embeddings_ok": False, "llm_ok": False, "db_ok": False,
        "rules_store_ok": False, "official_docs_store_ok": False, "echanges_store_ok": False,
        "user_docs_store_ok": False,
        "graph_ok": False,
        "counts": {"rules": 0, "official": 0, "echanges": 0, "rules_splits": 0, "official_splits": 0, "echanges_splits": 0},
        "search_function": None, 
        "get_distinct_values_function": None, 
        "update_user_vector_store":trigger_user_vs_and_graph_update_from_rag_main,
        "error_messages": []
    }
    graph = None 

    # 1. Initialiser Embeddings
    try:
        embeddings = MistralAIEmbeddings()
        #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        logger.info(f"Embeddings initialisés : {type(embeddings).__name__}")
        status["embeddings_ok"] = True
    except Exception as e:
        logger.error(f"Échec initialisation Embeddings: {e}", exc_info=True)
        status["error_messages"].append(f"Embeddings: {e}")
        embeddings = None

    # 2. Initialiser LLM
    try:
        llm = ChatMistralAI(model="mistral-large-latest")
        #llm = ChatOpenAI(model="gpt-4o")
        logger.info(f"LLM initialisé: {type(llm).__name__} (Model: {getattr(llm, 'model', 'N/A')})")
        status["llm_ok"] = True
    except Exception as e:
        logger.error(f"Échec initialisation LLM: {e}", exc_info=True)
        status["error_messages"].append(f"LLM: {e}")
        llm = None

    if not status["embeddings_ok"] or not status["llm_ok"]:
        logger.critical("Arrêt initialisation: Embeddings ou LLM manquants.")
        # AJOUTER la clé graph (avec la valeur None) au status avant de retourner
        status["graph"] = None
        return status

    # 3. Initialiser et tester la connexion DB
    try:
        db_manager = DatabaseManager()
        if db_manager._is_config_valid(): # Vérifier si la config est là avant de tester
             db_connected = db_manager.tester_connexion()
             status["db_ok"] = db_connected
             if db_connected:
                 status["search_function"] = db_manager.rechercher_dossier
              
                 logger.info("Connexion DB et fonctions DB prêtes.")
             else:
                 logger.warning("Connexion DB échouée, vérifier la config")
                 status["error_messages"].append("DB: Connexion échouée")
        else:
             logger.warning("DB non configurée (variables d'env manquantes).")
             status["db_ok"] = False
             status["error_messages"].append("DB: Configuration manquante")
             db_manager = None # Pas de manager si pas de config
             db_connected = False
    except Exception as e:
        logger.error(f"Erreur initialisation DB Manager: {e}", exc_info=True)
        status["error_messages"].append(f"DB Manager: {e}")
        db_manager = None
        db_connected = False
        status["db_ok"] = False

    # 4. Charger tous les documents
    official_docs, echanges_docs, rules_docs = load_all_documents(
        official_docs_path=OFFICIAL_DOCS_PATH,
        echanges_path=ECHANGES_PATH,
        regles_path=REGLES_PATH
    )
   
    status["counts"]["official"] = len(official_docs)
    status["counts"]["echanges"] = len(echanges_docs)
    status["counts"]["rules"] = len(rules_docs)


    # 5. Découper les documents
    official_splits = split_documents(official_docs, chunk_size=800, chunk_overlap=512)
    echanges_splits = split_documents(echanges_docs, chunk_size=800, chunk_overlap=512)
    rules_splits = split_documents(rules_docs, chunk_size=700, chunk_overlap=512)# Plus petit pour règles
  
    status["counts"]["official_splits"] = len(official_splits)
    status["counts"]["echanges_splits"] = len(echanges_splits)
    status["counts"]["rules_splits"] = len(rules_splits)
    logger.info(f"Documents découpés: {len(official_splits)} officiels, {len(echanges_splits)} échanges, {len(rules_splits)} règles.")


    # 6. Créer/Charger les Vector Stores
    rules_store_cache_dir = os.path.join(CACHE_DIR, "rules_store") # CACHE_DIR est maintenant le dossier des embeddings
    official_docs_store_cache_dir = os.path.join(CACHE_DIR, "official_docs_store")
    echanges_store_cache_dir = os.path.join(CACHE_DIR, "echanges_store")

    rules_vector_store = create_vector_store(rules_splits, embeddings, rules_store_cache_dir)
    status["rules_store_ok"] = rules_vector_store is not None
    if not status["rules_store_ok"]: status["error_messages"].append("VectorStore Règles: Échec création/chargement")

    official_docs_vector_store = create_vector_store(official_splits, embeddings, official_docs_store_cache_dir)
    status["official_docs_store_ok"] = official_docs_vector_store is not None
    if not status["official_docs_store_ok"]: status["error_messages"].append("VectorStore Officiels: Échec création/chargement")

    echanges_vector_store = create_vector_store(echanges_splits, embeddings, echanges_store_cache_dir)
    status["echanges_store_ok"] = echanges_vector_store is not None
    if not status["echanges_store_ok"]: status["error_messages"].append("VectorStore Echanges: Échec création/chargement")
    
    # Initialisation du Vector Store utilisateur au démarrage
    logger.info(f"Chargement initial des documents utilisateur depuis {USER_UPLOADS_PATH}")
    os.makedirs(USER_UPLOADS_PATH, exist_ok=True)
    user_docs = load_user_uploaded_documents(USER_UPLOADS_PATH) # Utilise la fonction de document_processor
    status["counts"]["user_uploaded"] = len(user_docs)
    user_splits = split_documents(user_docs, chunk_size=800, chunk_overlap=512)
    status["counts"]["user_uploaded_splits"] = len(user_splits)

    # Utilise create_vector_store de vector_store_utils.py
    user_docs_vector_store = create_vector_store(user_splits, embeddings, USER_VS_CACHE_DIR)
    status["user_docs_store_ok"] = user_docs_vector_store is not None
    if not status["user_docs_store_ok"]:
        status["error_messages"].append("VectorStore Utilisateur: Échec création/chargement initial")

    # 7. Construire le graphe en passant les dépendances initialisées
    try:
        graph = build_graph_with_deps(
            db_man_dep=db_manager,
            db_conn_status_dep=db_connected,
            llm_dep=llm,
            rules_vs_dep=rules_vector_store,
            official_vs_dep=official_docs_vector_store,
            echanges_vs_dep=echanges_vector_store,
            user_vs_dep=user_docs_vector_store
        )
        status["graph_ok"] = True
        logger.info("Graphe RAG final prêt.")
    except Exception as e:
        logger.error(f"Échec construction graphe RAG: {e}", exc_info=True)
        status["error_messages"].append(f"Graphe: {e}")
        status["graph_ok"] = False
        graph = None # Assurer que graph est None si la construction échoue

    status["graph"] = graph

    logger.info("="*20 + " Fin Initialisation Système RAG " + "="*20)
    if status["error_messages"]:
         logger.warning(f"Erreurs/Avertissements pendant initialisation: {status['error_messages']}")
    return status