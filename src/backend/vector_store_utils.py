import os
import hashlib
import logging
from typing import List, Optional, Any, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS

from .document_processor import load_user_uploaded_documents, split_documents 
from .rag_pipeline import build_graph_with_deps 

logger = logging.getLogger(__name__)

def calculate_documents_hash(documents: List[Document], embeddings: Embeddings) -> str:
    """Calcule un hash SHA256 basé sur le contenu des documents et le type/modèle d'embeddings."""
    hasher = hashlib.sha256()
    hasher.update(str(type(embeddings).__name__).encode('utf-8'))
    if hasattr(embeddings, "model"):
         hasher.update(str(embeddings.model).encode('utf-8'))
    elif hasattr(embeddings, "model_name"):
         hasher.update(str(embeddings.model_name).encode('utf-8'))
    
    sorted_docs = sorted(documents, key=lambda d: (d.page_content, str(sorted(d.metadata.items()))))
    for doc in sorted_docs:
        hasher.update(doc.page_content.encode('utf-8'))
        hasher.update(str(sorted(doc.metadata.items())).encode('utf-8'))
    return hasher.hexdigest()

def load_cached_embeddings_from_dir(embeddings: Embeddings, cache_dir: str, current_hash: str) -> Optional[VectorStore]:
    """Tente de charger un index FAISS depuis un cache si le hash correspond."""
    os.makedirs(cache_dir, exist_ok=True)
    hash_file_path = os.path.join(cache_dir, "documents_hash.txt")
    faiss_index_path = os.path.join(cache_dir, "faiss_index")

    if os.path.exists(hash_file_path) and os.path.isdir(faiss_index_path):
        try:
            with open(hash_file_path, "r", encoding='utf-8') as f:
                cached_hash = f.read().strip()
            if cached_hash == current_hash:
                logger.info(f"Cache HIT: Hash correspondant trouvé dans {cache_dir}. Chargement de l'index FAISS...")
                vector_store = FAISS.load_local(
                    faiss_index_path, embeddings, allow_dangerous_deserialization=True
                )
                logger.info(f"Cache - Index FAISS chargé avec succès depuis {cache_dir}.")
                return vector_store
            else:
                logger.info(f"Cache MISS: Hash différent dans {cache_dir}. Recalcul nécessaire.")
        except FileNotFoundError:
             logger.warning(f"Cache - Fichier hash ou dossier index non trouvé dans {cache_dir}.")
        except Exception as e:
            logger.error(f"Cache - Erreur lors du chargement de l'index FAISS depuis {cache_dir}: {e}", exc_info=True)
    else:
        logger.info(f"Cache - Cache non trouvé ou incomplet dans {cache_dir}.")
    return None

def save_embeddings_cache_to_dir(vector_store: VectorStore, documents_hash: str, cache_dir: str):
    """Sauvegarde l'index FAISS et le hash des documents dans un répertoire de cache."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        hash_file_path = os.path.join(cache_dir, "documents_hash.txt")
        faiss_index_path = os.path.join(cache_dir, "faiss_index")
        vector_store.save_local(faiss_index_path)
        with open(hash_file_path, "w", encoding='utf-8') as f:
            f.write(documents_hash)
        logger.info(f"Cache - Embeddings et hash sauvegardés avec succès dans {cache_dir}.")
    except Exception as e:
         logger.error(f"Cache - Erreur critique lors de la sauvegarde du cache dans {cache_dir}: {e}", exc_info=True)

def create_vector_store(documents: List[Document], embeddings: Embeddings, cache_dir: str) -> Optional[VectorStore]:
    """Crée ou charge un vector store FAISS depuis le cache."""
    if not documents:
        logger.warning(f"Aucun document fourni pour le vector store dans {cache_dir}. Création annulée.")
        return None
    if embeddings is None:
        logger.error(f"Modèle d'embeddings non fourni pour {cache_dir}. Création annulée.")
        return None

    logger.info(f"Préparation du vector store pour {cache_dir} ({len(documents)} documents)...")
    current_hash = calculate_documents_hash(documents, embeddings)
    vector_store = load_cached_embeddings_from_dir(embeddings, cache_dir, current_hash)
    if vector_store is not None:
        return vector_store

    logger.info(f"Création d'un nouvel index FAISS pour {cache_dir}...")
    try:
        vector_store = FAISS.from_documents(documents, embeddings)
        logger.info(f"Nouvel index FAISS créé pour {cache_dir}. Sauvegarde en cache...")
        save_embeddings_cache_to_dir(vector_store, current_hash, cache_dir)
        return vector_store
    except Exception as e:
        logger.error(f"Erreur critique lors de la création de l'index FAISS pour {cache_dir}: {e}", exc_info=True)
        return None
    
def update_user_vs_and_get_updated_graph(
    embeddings_instance: Embeddings,
    llm_instance: Any,
    db_manager_instance: Optional[Any],
    db_connection_status: bool,
    rules_vs_instance: Optional[VectorStore],
    official_vs_instance: Optional[VectorStore],
    echanges_vs_instance: Optional[VectorStore],
    user_uploads_path: str,
    user_vs_cache_path: str,
    current_user_vs: Optional[VectorStore] 
) -> Tuple[Optional[VectorStore], Optional[Any], bool]:
    """
    Met à jour le VectorStore des documents utilisateur et reconstruit le graphe RAG.
    Retourne le nouveau VectorStore utilisateur, le nouveau graphe, et un booléen de succès.
    """
    logger.info(f"Début de la mise à jour du VectorStore utilisateur et du graphe RAG via {user_uploads_path}")

    if not embeddings_instance:
        logger.error("Embeddings non fournis. Impossible de mettre à jour le VectorStore utilisateur.")
        return current_user_vs, None, False # Retourne le VS actuel et pas de nouveau graphe

    new_user_vector_store: Optional[VectorStore] = None
    new_graph: Optional[Any] = None
    success_flag = False

    try:
        # 1. Charger et splitter les documents utilisateur
        os.makedirs(user_uploads_path, exist_ok=True) # S'assurer que le dossier existe
        user_docs = load_user_uploaded_documents(user_uploads_path)
        user_splits = split_documents(user_docs, chunk_size=512, chunk_overlap=200) 

        # 2. Créer/Mettre à jour le VectorStore utilisateur
        # create_vector_store gère déjà le cache
        new_user_vector_store = create_vector_store(user_splits, embeddings_instance, user_vs_cache_path)

        if new_user_vector_store is None:
            logger.error("Échec de la création/mise à jour du user_docs_vector_store.")
            return current_user_vs, None, False

        logger.info(f"VectorStore utilisateur mis à jour avec {len(user_splits)} chunks.")

        # 3. Reconstruire le graphe RAG avec le nouveau VectorStore utilisateur
        # Note: Tous les autres VectorStores et composants LLM/DB doivent être passés
        if not llm_instance:
            logger.error("Instance LLM non fournie. Impossible de reconstruire le graphe.")
            return new_user_vector_store, None, False # Retourne le VS mis à jour mais pas de graphe

        logger.info("Reconstruction du graphe RAG avec le VectorStore utilisateur mis à jour...")
        new_graph = build_graph_with_deps(
            db_man_dep=db_manager_instance,
            db_conn_status_dep=db_connection_status,
            llm_dep=llm_instance,
            rules_vs_dep=rules_vs_instance,
            official_vs_dep=official_vs_instance,
            echanges_vs_dep=echanges_vs_instance,
            user_vs_dep=new_user_vector_store  # Utilise le VS utilisateur fraîchement créé/mis à jour
        )
        logger.info("Graphe RAG reconstruit avec succès.")
        success_flag = True

    except Exception as e:
        logger.error(f"Erreur critique lors de la mise à jour du VS utilisateur et/ou du graphe : {e}", exc_info=True)
        # Retourne le nouveau VS s'il a été créé, sinon l'ancien, et pas de nouveau graphe
        return new_user_vector_store if new_user_vector_store else current_user_vs, None, False

    return new_user_vector_store, new_graph, success_flag