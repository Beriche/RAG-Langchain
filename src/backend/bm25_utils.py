import os
import pickle
import hashlib
import logging
import re
from typing import List, Optional, Tuple, Any

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- Tokenizer Simple ---
# Vous pourriez améliorer ce tokenizer (stopwords, stemming/lemmatization) si nécessaire
def simple_tokenizer(text: str) -> List[str]:
    """Tokenize simple: minuscules, caractères alphanumériques et espaces."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Garde seulement lettres, chiffres, espaces
    return text.split()

# --- Gestion du Cache ---

def calculate_bm25_data_hash(documents: List[Document]) -> str:
    """Calcule un hash SHA256 basé uniquement sur le contenu des documents."""
    hasher = hashlib.sha256()
    # Trier pour assurer la cohérence du hash indépendamment de l'ordre initial
    sorted_docs = sorted(documents, key=lambda d: (d.page_content, str(sorted(d.metadata.items()))))
    for doc in sorted_docs:
        hasher.update(doc.page_content.encode('utf-8'))
        # Inclure les métadonnées peut être pertinent si elles influencent le contenu utilisé
        hasher.update(str(sorted(doc.metadata.items())).encode('utf-8'))
    return hasher.hexdigest()

def load_bm25_index_from_cache(cache_dir: str, current_hash: str) -> Optional[Tuple[BM25Okapi, List[Document]]]:
    """Tente de charger un index BM25 et les documents associés depuis un cache."""
    os.makedirs(cache_dir, exist_ok=True)
    hash_file_path = os.path.join(cache_dir, "bm25_hash.txt")
    index_file_path = os.path.join(cache_dir, "bm25_index.pkl")
    docs_file_path = os.path.join(cache_dir, "bm25_docs.pkl")

    if os.path.exists(hash_file_path) and os.path.exists(index_file_path) and os.path.exists(docs_file_path):
        try:
            with open(hash_file_path, "r", encoding='utf-8') as f:
                cached_hash = f.read().strip()

            if cached_hash == current_hash:
                logger.info(f"BM25 Cache HIT: Hash correspondant trouvé dans {cache_dir}. Chargement...")
                with open(index_file_path, "rb") as f_idx, open(docs_file_path, "rb") as f_docs:
                    bm25_index = pickle.load(f_idx)
                    documents = pickle.load(f_docs)
                logger.info(f"BM25 Cache - Index et documents chargés avec succès depuis {cache_dir}.")
                return bm25_index, documents
            else:
                logger.info(f"BM25 Cache MISS: Hash différent dans {cache_dir}. Recalcul nécessaire.")
        except FileNotFoundError:
             logger.warning(f"BM25 Cache - Fichier hash, index ou docs non trouvé dans {cache_dir}.")
        except Exception as e:
            logger.error(f"BM25 Cache - Erreur lors du chargement depuis {cache_dir}: {e}", exc_info=True)
    else:
        logger.info(f"BM25 Cache - Cache non trouvé ou incomplet dans {cache_dir}.")
    return None

def save_bm25_index_to_cache(bm25_index: BM25Okapi, documents: List[Document], documents_hash: str, cache_dir: str):
    """Sauvegarde l'index BM25, les documents associés et le hash dans le cache."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        hash_file_path = os.path.join(cache_dir, "bm25_hash.txt")
        index_file_path = os.path.join(cache_dir, "bm25_index.pkl")
        docs_file_path = os.path.join(cache_dir, "bm25_docs.pkl")

        with open(index_file_path, "wb") as f_idx, open(docs_file_path, "wb") as f_docs:
            pickle.dump(bm25_index, f_idx)
            pickle.dump(documents, f_docs)

        with open(hash_file_path, "w", encoding='utf-8') as f_hash:
            f_hash.write(documents_hash)

        logger.info(f"BM25 Cache - Index, documents et hash sauvegardés avec succès dans {cache_dir}.")
    except Exception as e:
         logger.error(f"BM25 Cache - Erreur critique lors de la sauvegarde du cache dans {cache_dir}: {e}", exc_info=True)

# --- Création et Recherche ---

def create_bm25_index(documents: List[Document], cache_dir: str) -> Optional[Tuple[BM25Okapi, List[Document]]]:
    """
    Crée ou charge un index BM25 depuis le cache pour une liste de documents.
    Retourne l'index BM25 et la liste des documents utilisés pour le créer.
    """
    if not documents:
        logger.warning(f"BM25 - Aucun document fourni pour l'index dans {cache_dir}. Création annulée.")
        return None

    logger.info(f"BM25 - Préparation de l'index pour {cache_dir} ({len(documents)} documents)...")
    current_hash = calculate_bm25_data_hash(documents)

    # Tenter de charger depuis le cache
    cached_data = load_bm25_index_from_cache(cache_dir, current_hash)
    if cached_data:
        return cached_data # Retourne (bm25_index, documents) depuis le cache

    logger.info(f"BM25 - Création d'un nouvel index BM25 pour {cache_dir}...")
    try:
        # 1. Extraire le contenu textuel
        corpus = [doc.page_content for doc in documents]

        # 2. Tokeniser le corpus
        tokenized_corpus = [simple_tokenizer(text) for text in corpus]

        # 3. Créer l'index BM25Okapi
        bm25_index = BM25Okapi(tokenized_corpus)

        logger.info(f"BM25 - Nouvel index créé pour {cache_dir}. Sauvegarde en cache...")
        save_bm25_index_to_cache(bm25_index, documents, current_hash, cache_dir)

        # Retourne le nouvel index et la liste de documents originale
        return bm25_index, documents

    except Exception as e:
        logger.error(f"BM25 - Erreur critique lors de la création de l'index BM25 pour {cache_dir}: {e}", exc_info=True)
        return None

def bm25_search(
    query: str,
    bm25_index: BM25Okapi,
    documents: List[Document], # La liste des docs correspondant à l'index
    k: int = 5
) -> List[Document]:
    """
    Effectue une recherche BM25 et retourne les k meilleurs Documents Langchain.
    """
    if not query or bm25_index is None or not documents:
        return []

    try:
        tokenized_query = simple_tokenizer(query)
        # get_top_n retourne les documents (textes tokenisés) du corpus original,
        # mais nous avons besoin des indices pour retrouver les objets Document.
        # Utilisons get_scores pour obtenir les scores de tous les documents,
        # puis trions pour obtenir les indices des k meilleurs.

        doc_scores = bm25_index.get_scores(tokenized_query)

        # Obtenir les indices des k meilleurs scores
        # argsort retourne les indices qui trieraient le tableau.
        # [-k:] prend les k derniers (les plus grands scores)
        # [::-1] inverse pour avoir le meilleur score en premier
        top_k_indices = doc_scores.argsort()[-k:][::-1]

        # Récupérer les Documents correspondants
        results = [documents[i] for i in top_k_indices if i < len(documents)]

        logger.debug(f"BM25 Search - Query: '{query[:50]}...', k={k}. Found {len(results)} results.")
        return results

    except Exception as e:
        logger.error(f"BM25 Search - Erreur lors de la recherche BM25: {e}", exc_info=True)
        return []

# --- Retriever Langchain Personnalisé (Optionnel mais recommandé) ---

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class BM25Retriever(BaseRetriever):
    """Retriever Langchain utilisant un index BM25 pré-calculé."""
    vectorstore: Optional[Any] = None # Rend ce champ optionnel
    bm25_index: BM25Okapi
    docs: List[Document]
    k: int = 4 # Nombre de documents à retourner par défaut

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Effectue la recherche BM25."""
        tokenized_query = simple_tokenizer(query)
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        top_k_indices = doc_scores.argsort()[-self.k:][::-1]
        
        # Ajouter les scores aux métadonnées pour un éventuel reranking/fusion
        relevant_docs = []
        for i in top_k_indices:
             if i < len(self.docs):
                 doc = self.docs[i]
                 # Créer une copie pour ne pas modifier l'original dans self.docs
                 doc_copy = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
                 doc_copy.metadata["bm25_score"] = doc_scores[i]
                 doc_copy.metadata["retrieval_source"] = "bm25"
                 relevant_docs.append(doc_copy)
                 
        return relevant_docs
