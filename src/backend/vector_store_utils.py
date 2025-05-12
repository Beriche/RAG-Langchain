import os
import hashlib
import logging
from typing import List, Optional, Any, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi # Import BM25Okapi

from .document_processor import load_user_uploaded_documents, split_documents
from .rag_pipeline import build_graph_with_deps
# Import BM25 utilities
from .bm25_utils import create_bm25_index, BM25Retriever

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

# --- Type Hint pour les composants BM25 ---
BM25Components = Optional[Tuple[BM25Okapi, List[Document]]]

def create_retrieval_components(
    documents: List[Document],
    embeddings: Embeddings,
    faiss_cache_dir: str,
    bm25_cache_dir: str # Répertoire distinct pour le cache BM25
) -> Tuple[Optional[VectorStore], BM25Components]:
    """
    Crée ou charge les composants de retrieval (Vector store FAISS et Index BM25) depuis le cache.
    Retourne un tuple: (VectorStore FAISS, (BM25Okapi, List[Document]))
    """
    # --- Validation initiale ---
    if not documents:
        logger.warning(f"Aucun document fourni pour les composants de retrieval ({faiss_cache_dir}, {bm25_cache_dir}). Création annulée.")
        return None, None
    if embeddings is None:
        logger.error(f"Modèle d'embeddings non fourni pour FAISS ({faiss_cache_dir}). Création annulée.")
        # On pourrait potentiellement créer BM25 même sans embeddings, mais gardons la logique couplée pour l'instant
        return None, None

    # --- Création/Chargement FAISS ---
    logger.info(f"FAISS - Préparation du vector store pour {faiss_cache_dir} ({len(documents)} documents)...")
    faiss_vector_store: Optional[VectorStore] = None
    faiss_hash = calculate_documents_hash(documents, embeddings) # Utilise la fonction existante pour FAISS
    cached_vs = load_cached_embeddings_from_dir(embeddings, faiss_cache_dir, faiss_hash)

    if cached_vs is not None:
        faiss_vector_store = cached_vs
    else:
        logger.info(f"FAISS - Création d'un nouvel index FAISS pour {faiss_cache_dir}...")
        try:
            faiss_vector_store = FAISS.from_documents(documents, embeddings)
            logger.info(f"FAISS - Nouvel index FAISS créé pour {faiss_cache_dir}. Sauvegarde en cache...")
            save_embeddings_cache_to_dir(faiss_vector_store, faiss_hash, faiss_cache_dir)
        except Exception as e:
            logger.error(f"FAISS - Erreur critique lors de la création de l'index FAISS pour {faiss_cache_dir}: {e}", exc_info=True)
            # Ne pas retourner ici, essayer de créer BM25 quand même si possible

    # --- Création/Chargement BM25 ---
    # Note: BM25 n'a pas besoin des embeddings, seulement des documents.
    # create_bm25_index gère son propre cache et hash basé sur les documents.
    bm25_components: BM25Components = create_bm25_index(documents, bm25_cache_dir)

    if faiss_vector_store is None and bm25_components is None:
         logger.error("Échec de la création des composants FAISS et BM25.")
         return None, None
    elif faiss_vector_store is None:
         logger.warning("Échec de la création du composant FAISS, mais BM25 a réussi.")
    elif bm25_components is None:
         logger.warning("Échec de la création du composant BM25, mais FAISS a réussi.")
    else:
         logger.info("Composants FAISS et BM25 créés/chargés avec succès.")

    return faiss_vector_store, bm25_components

def update_user_retrieval_and_graph(
    embeddings_instance: Embeddings,
    llm_instance: Any,
    db_manager_instance: Optional[Any],
    db_connection_status: bool,
    # Composants pour les sources fixes (passés en argument)
    rules_vs: Optional[VectorStore], rules_bm25_comps: BM25Components,
    official_vs: Optional[VectorStore], official_bm25_comps: BM25Components,
    echanges_vs: Optional[VectorStore], echanges_bm25_comps: BM25Components,
    # Chemins et état actuel pour les docs utilisateur
    user_uploads_path: str,
    user_faiss_cache_path: str,
    user_bm25_cache_path: str, # Nouveau chemin pour cache BM25 utilisateur
    current_user_vs: Optional[VectorStore],
    current_user_bm25_comps: BM25Components
) -> Tuple[Optional[VectorStore], BM25Components, Optional[Any], bool]:
    """
    Met à jour les composants de retrieval (FAISS & BM25) des documents utilisateur
    et reconstruit le graphe RAG avec tous les composants à jour.
    Retourne: (nouveau VS user FAISS, nouveaux composants user BM25, nouveau graphe, succès)
    """
    logger.info(f"Début de la mise à jour des composants utilisateur (FAISS/BM25) et du graphe RAG via {user_uploads_path}")

    # Initialisation des valeurs de retour par défaut (état actuel)
    final_user_vs = current_user_vs
    final_user_bm25_comps = current_user_bm25_comps
    new_graph: Optional[Any] = None
    success_flag = False

    if not embeddings_instance:
        logger.error("Embeddings non fournis. Impossible de mettre à jour les composants FAISS utilisateur.")
        # On pourrait potentiellement mettre à jour BM25, mais pour l'instant on arrête.
        return final_user_vs, final_user_bm25_comps, None, False

    try:
        # 1. Charger et splitter les documents utilisateur
        os.makedirs(user_uploads_path, exist_ok=True)
        user_docs = load_user_uploaded_documents(user_uploads_path)
        user_splits = split_documents(user_docs, chunk_size=512, chunk_overlap=200)

        # 2. Créer/Mettre à jour les composants de retrieval (FAISS et BM25) pour les docs utilisateur
        # create_retrieval_components gère le cache pour les deux
        updated_user_vs, updated_user_bm25_comps = create_retrieval_components(
            documents=user_splits,
            embeddings=embeddings_instance,
            faiss_cache_dir=user_faiss_cache_path,
            bm25_cache_dir=user_bm25_cache_path # Utilise le nouveau chemin de cache
        )

        # Mettre à jour l'état final même en cas d'échec partiel
        final_user_vs = updated_user_vs if updated_user_vs is not None else current_user_vs
        final_user_bm25_comps = updated_user_bm25_comps if updated_user_bm25_comps is not None else current_user_bm25_comps

        if updated_user_vs is None and updated_user_bm25_comps is None:
            logger.error("Échec complet de la création/mise à jour des composants de retrieval utilisateur.")
            # Retourne l'état précédent, pas de nouveau graphe
            return current_user_vs, current_user_bm25_comps, None, False
        elif not user_splits:
             logger.info("Aucun document utilisateur trouvé ou chargé. Utilisation des composants vides/précédents.")
             # Pas d'erreur, mais on utilise les composants potentiellement vides/mis à jour
        else:
             logger.info(f"Composants utilisateur mis à jour avec {len(user_splits)} chunks.")


        # 3. Reconstruire le graphe RAG avec TOUS les composants (fixes et utilisateur mis à jour)
        if not llm_instance:
            logger.error("Instance LLM non fournie. Impossible de reconstruire le graphe.")
            # Retourne les composants mis à jour mais pas de nouveau graphe
            return final_user_vs, final_user_bm25_comps, None, False

        logger.info("Reconstruction du graphe RAG avec tous les composants de retrieval à jour...")
        # !! NOTE : build_graph_with_deps devra être modifié pour accepter et utiliser les composants BM25 !!
        # Pour l'instant, on passe les composants mis à jour, mais build_graph_with_deps ne les utilisera pas encore.
        new_graph = build_graph_with_deps(
            db_man_dep=db_manager_instance,
            db_conn_status_dep=db_connection_status,
            llm_dep=llm_instance,
            # Passage des composants FAISS et BM25 pour chaque source
            rules_vs_dep=rules_vs, rules_bm25_comps_dep=rules_bm25_comps,
            official_vs_dep=official_vs, official_bm25_comps_dep=official_bm25_comps,
            echanges_vs_dep=echanges_vs, echanges_bm25_comps_dep=echanges_bm25_comps,
            user_vs_dep=final_user_vs, user_bm25_comps_dep=final_user_bm25_comps # Utilise les composants utilisateur finaux
        )
        logger.info("Graphe RAG reconstruit (potentiellement avec les nouvelles dépendances BM25 si build_graph_with_deps est mis à jour).")
        success_flag = True

    except Exception as e:
        logger.error(f"Erreur critique lors de la mise à jour des composants utilisateur et/ou du graphe : {e}", exc_info=True)
        # Retourne l'état le plus récent possible des composants, mais pas de nouveau graphe
        return final_user_vs, final_user_bm25_comps, None, False

    return final_user_vs, final_user_bm25_comps, new_graph, success_flag
