import os
import logging
from typing import Dict, Any, Optional, List 

from dotenv import load_dotenv

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from rank_bm25 import BM25Okapi # Pour type hinting

from .db_manager import DatabaseManager
from .document_processor import load_all_documents, split_documents, load_user_uploaded_documents
# Importer les fonctions et types mis à jour
from .vector_store_utils import (
    create_retrieval_components, 
    update_user_retrieval_and_graph, # Renommée et mise à jour
    BM25Components # Type hint pour BM25
)
from .rag_pipeline import build_graph_with_deps, BM25Components as Bm25CompsPipeline # Importer aussi depuis pipeline pour cohérence

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
CACHE_DIR = os.getenv("CACHE_DIR", CACHE_DIR_DEFAULT) # Racine pour les caches
FAISS_CACHE_DIR = os.path.join(CACHE_DIR, "faiss_store") # Sous-dossier pour FAISS
BM25_CACHE_DIR = os.path.join(CACHE_DIR, "bm25_store") # Sous-dossier pour BM25

USER_FAISS_CACHE_DIR = os.path.join(FAISS_CACHE_DIR, "user_docs_store")
USER_BM25_CACHE_DIR = os.path.join(BM25_CACHE_DIR, "user_docs_store")

# Variables globales pour FAISS
rules_vector_store: Optional[VectorStore] = None
official_docs_vector_store: Optional[VectorStore] = None
echanges_vector_store: Optional[VectorStore] = None
user_docs_vector_store: Optional[VectorStore] = None

# Variables globales pour BM25
rules_bm25_components: BM25Components = None
official_bm25_components: BM25Components = None
echanges_bm25_components: BM25Components = None
user_bm25_components: BM25Components = None

llm: Optional[Any] = None
embeddings: Optional[Embeddings] = None
db_connected: bool = False
db_manager: Optional[DatabaseManager] = None

def trigger_user_retrieval_and_graph_update() -> bool: # Renommée
    """Déclenche la mise à jour des composants utilisateur (FAISS & BM25) et la reconstruction du graphe."""
    global user_docs_vector_store, user_bm25_components # Globals pour user
    global llm, embeddings, db_manager, db_connected # Autres globals nécessaires
    global rules_vector_store, official_docs_vector_store, echanges_vector_store # VS fixes
    global rules_bm25_components, official_bm25_components, echanges_bm25_components # BM25 fixes

    # Logique Streamlit (si conservée ici)
    import streamlit as st # Attention au couplage

    if not embeddings:
        logger.error("Embeddings non initialisés. Mise à jour annulée.")
        if 'rag_components' in st.session_state and st.session_state.rag_components:
            st.session_state.rag_components["user_retrieval_ok"] = False # Statut combiné
        return False

    # Appel de la fonction mise à jour de vector_store_utils
    new_vs, new_bm25, new_graph, success = update_user_retrieval_and_graph(
        embeddings_instance=embeddings,
        llm_instance=llm,
        db_manager_instance=db_manager,
        db_connection_status=db_connected,
        # Composants fixes (passés depuis les globales)
        rules_vs=rules_vector_store, rules_bm25_comps=rules_bm25_components,
        official_vs=official_docs_vector_store, official_bm25_comps=official_bm25_components,
        echanges_vs=echanges_vector_store, echanges_bm25_comps=echanges_bm25_components,
        # Infos utilisateur
        user_uploads_path=USER_UPLOADS_PATH,
        user_faiss_cache_path=USER_FAISS_CACHE_DIR, # Chemin cache FAISS user
        user_bm25_cache_path=USER_BM25_CACHE_DIR,   # Chemin cache BM25 user
        current_user_vs=user_docs_vector_store,     # VS FAISS user actuel
        current_user_bm25_comps=user_bm25_components # Composants BM25 user actuels
    )

    # Mise à jour des globales utilisateur même si le graphe échoue
    user_docs_vector_store = new_vs
    user_bm25_components = new_bm25
    logger.info("Globals user_docs_vector_store et user_bm25_components mis à jour.")

    user_retrieval_ok = (new_vs is not None) or (new_bm25 is not None) # Au moins un des deux a réussi

    if success and new_graph is not None:
        logger.info("Mise à jour réussie, nouveau graphe RAG généré.")
        # Mise à jour Streamlit session_state
        if 'rag_components' in st.session_state and st.session_state.rag_components:
            st.session_state.rag_components["graph"] = new_graph
            st.session_state.rag_components["user_retrieval_ok"] = user_retrieval_ok
            logger.info("Graphe RAG et statut user dans st.session_state mis à jour.")
        else:
            logger.warning("st.session_state.rag_components non trouvé pour la mise à jour.")
        return True
    else:
        logger.error("Échec de la mise à jour du graphe RAG après la mise à jour des composants utilisateur.")
        if 'rag_components' in st.session_state and st.session_state.rag_components:
            # Mettre à jour le statut même si le graphe a échoué
            st.session_state.rag_components["user_retrieval_ok"] = user_retrieval_ok
            st.session_state.rag_components["graph"] = None # Indiquer que le graphe n'est pas valide
        return False

def init_rag_system() -> Dict[str, Any]:
    """Initialise tous les composants du système RAG (LLM, Embeddings, DB, Vector Stores, Graphe).
    """
    
    # Utiliser les variables globales pour stocker les objets initialisés
    global llm, embeddings, db_manager, db_connected # Globals existants
    # Globals FAISS
    global rules_vector_store, official_docs_vector_store, echanges_vector_store, user_docs_vector_store
    # Globals BM25
    global rules_bm25_components, official_bm25_components, echanges_bm25_components, user_bm25_components

    logger.info("="*20 + " Initialisation du Système RAG (FAISS + BM25) " + "="*20)
    status = {
        "embeddings_ok": False, "llm_ok": False, "db_ok": False,
        # Statuts FAISS
        "rules_vs_ok": False, "official_vs_ok": False, "echanges_vs_ok": False, "user_vs_ok": False,
        # Statuts BM25
        "rules_bm25_ok": False, "official_bm25_ok": False, "echanges_bm25_ok": False, "user_bm25_ok": False,
        # Statut combiné pour utilisateur
        "user_retrieval_ok": False,
        "graph_ok": False,
        "counts": {"rules": 0, "official": 0, "echanges": 0, "user_uploaded": 0,
                   "rules_splits": 0, "official_splits": 0, "echanges_splits": 0, "user_uploaded_splits": 0},
        "search_function": None,
        "get_distinct_values_function": None,
        "update_user_retrieval": trigger_user_retrieval_and_graph_update, # Fonction de mise à jour renommée
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
    official_splits = split_documents(official_docs, chunk_size=600, chunk_overlap=300)
    echanges_splits = split_documents(echanges_docs, chunk_size=800, chunk_overlap=512)
    rules_splits = split_documents(rules_docs, chunk_size=512, chunk_overlap=200)# Plus petit pour règles
   
  
    status["counts"]["official_splits"] = len(official_splits)
    status["counts"]["echanges_splits"] = len(echanges_splits)
    status["counts"]["rules_splits"] = len(rules_splits)
    logger.info(f"Documents découpés: {len(official_splits)} officiels, {len(echanges_splits)} échanges, {len(rules_splits)} règles.")


    # 6. Créer/Charger les composants de retrieval (FAISS + BM25) pour chaque source fixe
    # Définir les chemins de cache spécifiques pour chaque source
    rules_faiss_cache = os.path.join(FAISS_CACHE_DIR, "rules_store")
    rules_bm25_cache = os.path.join(BM25_CACHE_DIR, "rules_store")
    official_faiss_cache = os.path.join(FAISS_CACHE_DIR, "official_docs_store")
    official_bm25_cache = os.path.join(BM25_CACHE_DIR, "official_docs_store")
    echanges_faiss_cache = os.path.join(FAISS_CACHE_DIR, "echanges_store")
    echanges_bm25_cache = os.path.join(BM25_CACHE_DIR, "echanges_store")

    # Règles
    rules_vector_store, rules_bm25_components = create_retrieval_components(
        rules_splits, embeddings, rules_faiss_cache, rules_bm25_cache
    )
    status["rules_vs_ok"] = rules_vector_store is not None
    status["rules_bm25_ok"] = rules_bm25_components is not None
    if not status["rules_vs_ok"]: status["error_messages"].append("FAISS Règles: Échec")
    if not status["rules_bm25_ok"]: status["error_messages"].append("BM25 Règles: Échec")

    # Documents Officiels
    official_docs_vector_store, official_bm25_components = create_retrieval_components(
        official_splits, embeddings, official_faiss_cache, official_bm25_cache
    )
    status["official_vs_ok"] = official_docs_vector_store is not None
    status["official_bm25_ok"] = official_bm25_components is not None
    if not status["official_vs_ok"]: status["error_messages"].append("FAISS Officiels: Échec")
    if not status["official_bm25_ok"]: status["error_messages"].append("BM25 Officiels: Échec")

    # Échanges
    echanges_vector_store, echanges_bm25_components = create_retrieval_components(
        echanges_splits, embeddings, echanges_faiss_cache, echanges_bm25_cache
    )
    status["echanges_vs_ok"] = echanges_vector_store is not None
    status["echanges_bm25_ok"] = echanges_bm25_components is not None
    if not status["echanges_vs_ok"]: status["error_messages"].append("FAISS Echanges: Échec")
    if not status["echanges_bm25_ok"]: status["error_messages"].append("BM25 Echanges: Échec")

    # Initialisation des composants utilisateur au démarrage
    logger.info(f"Chargement initial des documents utilisateur depuis {USER_UPLOADS_PATH}")
    os.makedirs(USER_UPLOADS_PATH, exist_ok=True)
    user_docs = load_user_uploaded_documents(USER_UPLOADS_PATH)
    status["counts"]["user_uploaded"] = len(user_docs)
    user_splits = split_documents(user_docs, chunk_size=512, chunk_overlap=200)
    status["counts"]["user_uploaded_splits"] = len(user_splits)

    user_docs_vector_store, user_bm25_components = create_retrieval_components(
        user_splits, embeddings, USER_FAISS_CACHE_DIR, USER_BM25_CACHE_DIR
    )
    status["user_vs_ok"] = user_docs_vector_store is not None
    status["user_bm25_ok"] = user_bm25_components is not None
    status["user_retrieval_ok"] = status["user_vs_ok"] or status["user_bm25_ok"] # OK si au moins un des deux est OK
    if not status["user_vs_ok"]: status["error_messages"].append("FAISS Utilisateur: Échec initial")
    if not status["user_bm25_ok"]: status["error_messages"].append("BM25 Utilisateur: Échec initial")


    # 7. Construire le graphe en passant TOUTES les dépendances (FAISS et BM25)
    try:
        # Note: Les paramètres k et weights peuvent être passés ici si on veut les configurer globalement
        graph = build_graph_with_deps(
            db_man_dep=db_manager,
            db_conn_status_dep=db_connected,
            llm_dep=llm,
            # FAISS
            rules_vs_dep=rules_vector_store,
            official_vs_dep=official_docs_vector_store,
            echanges_vs_dep=echanges_vector_store,
            user_vs_dep=user_docs_vector_store,
            # BM25
            rules_bm25_comps_dep=rules_bm25_components,
            official_bm25_comps_dep=official_bm25_components,
            echanges_bm25_comps_dep=echanges_bm25_components,
            user_bm25_comps_dep=user_bm25_components
            # k_faiss_dep=..., k_bm25_dep=..., ensemble_weights_dep=... # Optionnel
        )
        status["graph_ok"] = True
        logger.info("Graphe RAG final (Hybride) prêt.")
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
