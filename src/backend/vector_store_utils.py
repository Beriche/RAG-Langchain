# src/vector_store_utils.py

import os
import hashlib
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS

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