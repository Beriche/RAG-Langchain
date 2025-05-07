# src/backend/rag_orchestrator.py

import os
import logging
from typing import Dict, Any, Optional, List 

from dotenv import load_dotenv

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

from .db_manager import DatabaseManager
from .document_processor import load_all_documents, split_documents
from .vector_store_utils import create_vector_store
from .rag_pipeline import build_graph_with_deps # MODIFIÉ: Importer la nouvelle fonction

logger = logging.getLogger(__name__)
load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT", "../../data") # Ajusté pour être relatif à la racine si ce script est dans src/backend
# Si app.py est à la racine et DATA_ROOT dans .env est "./data" ou "../data" relatif à la racine,
# alors os.getenv("DATA_ROOT", "./data") pourrait être mieux si .env est bien lu.
# Pour l'instant, on assume que .env est à la racine et DATA_ROOT pointe vers data/ à la racine.
# Si tu lances depuis la racine et .env est là, DATA_ROOT = os.getenv("DATA_ROOT", "data")
# Ce chemin relatif "../data" est si tu exécutes quelque chose DEPUIS src/backend.
# Pour un projet lancé depuis la racine, DATA_ROOT devrait être "data" (si data est à la racine)
# ou un chemin absolu.

# Pour plus de robustesse, en supposant que .env est à la racine du projet
# et que le script est lancé depuis la racine du projet (ex: streamlit run app.py)
# Et que DATA_ROOT dans .env est par ex. "data" (pour un dossier data à la racine)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # Racine du projet
DATA_ROOT_DEFAULT = os.path.join(_PROJECT_ROOT, "data")
CACHE_DIR_DEFAULT = os.path.join(_PROJECT_ROOT, "cache_embeddings") # Pour les embeddings

DATA_ROOT = os.getenv("DATA_ROOT", DATA_ROOT_DEFAULT)
ECHANGES_PATH = os.path.join(DATA_ROOT, "echanges")
REGLES_PATH = os.path.join(DATA_ROOT, "regles")
OFFICIAL_DOCS_PATH = os.path.join(DATA_ROOT, "docs_officiels")
CACHE_DIR = os.getenv("CACHE_DIR", CACHE_DIR_DEFAULT) # Où les FAISS index sont sauvegardés


# Ces variables globales seront initialisées.
# rag_pipeline.py n'essaiera plus de les importer.
rules_vector_store_global: Optional[VectorStore] = None
official_docs_vector_store_global: Optional[VectorStore] = None
echanges_vector_store_global: Optional[VectorStore] = None
llm_global: Optional[Any] = None
embeddings_global: Optional[Embeddings] = None
db_connected_global: bool = False
db_manager_global: Optional[DatabaseManager] = None

def init_rag_system() -> Dict[str, Any]:
    global llm_global, embeddings_global, db_manager_global, db_connected_global
    global rules_vector_store_global, official_docs_vector_store_global, echanges_vector_store_global

    logger.info("="*20 + " Initialisation du Système RAG " + "="*20)
    # ... (début de la fonction inchangé, initialisation de status) ...
    status = {
        "embeddings_ok": False, "llm_ok": False, "db_ok": False,
        "rules_store_ok": False, "official_docs_store_ok": False, "echanges_store_ok": False,
        "graph_ok": False,
        "counts": {"rules": 0, "official": 0, "echanges": 0, "rules_splits": 0, "official_splits": 0, "echanges_splits": 0},
        "search_function": None, 
        "get_distinct_values_function": None, 
        "error_messages": []
    }
    graph = None 

    # 1. Initialiser Embeddings
    try:
        embeddings_global = OpenAIEmbeddings(model="text-embedding-3-large")
        logger.info(f"Embeddings initialisés: {type(embeddings_global).__name__}")
        status["embeddings_ok"] = True
    except Exception as e:
        logger.error(f"Échec initialisation Embeddings: {e}", exc_info=True)
        status["error_messages"].append(f"Embeddings: {e}")
        embeddings_global = None

    # 2. Initialiser LLM
    try:
        llm_global = ChatOpenAI(model="gpt-4o")
        logger.info(f"LLM initialisé: {type(llm_global).__name__} (Model: {getattr(llm_global, 'model', 'N/A')})")
        status["llm_ok"] = True
    except Exception as e:
        logger.error(f"Échec initialisation LLM: {e}", exc_info=True)
        status["error_messages"].append(f"LLM: {e}")
        llm_global = None

    if not status["embeddings_ok"] or not status["llm_ok"]:
        logger.critical("Arrêt initialisation: Embeddings ou LLM manquants.")
        status["graph"] = None
        return status

    # 3. Initialiser et tester la connexion DB
    try:
        db_manager_global = DatabaseManager()
        if db_manager_global._is_config_valid():
             db_connected_global = db_manager_global.tester_connexion()
             status["db_ok"] = db_connected_global
             if db_connected_global:
                 status["search_function"] = db_manager_global.rechercher_dossier
                 # status["get_distinct_values_function"] = db_manager_global.get_distinct_values
                 logger.info("Connexion DB et fonctions DB prêtes.")
             else:
                 logger.warning("Connexion DB échouée.")
                 status["error_messages"].append("DB: Connexion échouée")
        else:
             logger.warning("DB non configurée.")
             status["db_ok"] = False
             status["error_messages"].append("DB: Configuration manquante")
             db_manager_global = None 
             db_connected_global = False
    except Exception as e:
        logger.error(f"Erreur initialisation DB Manager: {e}", exc_info=True)
        status["error_messages"].append(f"DB Manager: {e}")
        db_manager_global = None
        db_connected_global = False
        status["db_ok"] = False

    # 4. Charger tous les documents
    official_docs, echanges_docs, rules_docs = load_all_documents(
        official_docs_path=OFFICIAL_DOCS_PATH,
        echanges_path=ECHANGES_PATH,
        regles_path=REGLES_PATH
    )
    # ... (status["counts"] inchangé) ...
    status["counts"]["official"] = len(official_docs)
    status["counts"]["echanges"] = len(echanges_docs)
    status["counts"]["rules"] = len(rules_docs)


    # 5. Découper les documents
    official_splits = split_documents(official_docs, chunk_size=800, chunk_overlap=512)
    echanges_splits = split_documents(echanges_docs, chunk_size=800, chunk_overlap=512)
    rules_splits = split_documents(rules_docs, chunk_size=700, chunk_overlap=512)
    # ... (status["counts"] splits inchangé) ...
    status["counts"]["official_splits"] = len(official_splits)
    status["counts"]["echanges_splits"] = len(echanges_splits)
    status["counts"]["rules_splits"] = len(rules_splits)
    logger.info(f"Documents découpés: {len(official_splits)} officiels, {len(echanges_splits)} échanges, {len(rules_splits)} règles.")


    # 6. Créer/Charger les Vector Stores
    rules_store_cache_dir = os.path.join(CACHE_DIR, "rules_store") # CACHE_DIR est maintenant le dossier des embeddings
    official_docs_store_cache_dir = os.path.join(CACHE_DIR, "official_docs_store")
    echanges_store_cache_dir = os.path.join(CACHE_DIR, "echanges_store")

    rules_vector_store_global = create_vector_store(rules_splits, embeddings_global, rules_store_cache_dir)
    status["rules_store_ok"] = rules_vector_store_global is not None
    if not status["rules_store_ok"]: status["error_messages"].append("VS Règles: Échec")

    official_docs_vector_store_global = create_vector_store(official_splits, embeddings_global, official_docs_store_cache_dir)
    status["official_docs_store_ok"] = official_docs_vector_store_global is not None
    if not status["official_docs_store_ok"]: status["error_messages"].append("VS Officiels: Échec")

    echanges_vector_store_global = create_vector_store(echanges_splits, embeddings_global, echanges_store_cache_dir)
    status["echanges_store_ok"] = echanges_vector_store_global is not None
    if not status["echanges_store_ok"]: status["error_messages"].append("VS Echanges: Échec")

    # 7. Construire le graphe en passant les dépendances initialisées
    try:
        graph = build_graph_with_deps(
            db_man_dep=db_manager_global,
            db_conn_status_dep=db_connected_global,
            llm_dep=llm_global,
            rules_vs_dep=rules_vector_store_global,
            official_vs_dep=official_docs_vector_store_global,
            echanges_vs_dep=echanges_vector_store_global
        )
        status["graph_ok"] = True
        logger.info("Graphe RAG final prêt (avec dépendances injectées).")
    except Exception as e:
        logger.error(f"Échec construction graphe RAG: {e}", exc_info=True)
        status["error_messages"].append(f"Graphe: {e}")
        status["graph_ok"] = False
        graph = None

    status["graph"] = graph
    # ... (fin de la fonction inchangée) ...
    logger.info("="*20 + " Fin Initialisation Système RAG " + "="*20)
    if status["error_messages"]:
         logger.warning(f"Erreurs/Avertissements pendant initialisation: {status['error_messages']}")
    return status