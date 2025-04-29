# -*- coding: utf-8 -*-
import os
import re
import hashlib
import mysql.connector
import logging
import json
import glob
from datetime import date
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Langchain Imports - Mettre à jour selon les versions récentes
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# FAISS est maintenant dans langchain_community
from langchain_community.vectorstores import FAISS
# Importer les modèles spécifiques utilisés
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
# Décommenter si OpenAI est utilisé
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import StateGraph, END, START

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_module")

# Chargement des variables d'environnement
load_dotenv()

# Chemins vers les répertoires de données
DATA_ROOT = os.getenv("DATA_ROOT", "../data") # S'assurer que ce chemin est correct par rapport à l'exécution
ECHANGES_PATH = os.path.join(DATA_ROOT, "echanges")
REGLES_PATH = os.path.join(DATA_ROOT, "regles")
OFFICIAL_DOCS_PATH = os.path.join(DATA_ROOT, "docs_officiels")
CACHE_DIR = "./cache"  # Répertoire pour le cache (sera créé s'il n'existe pas)

# --- VARIABLES GLOBALES ---
# Séparation des vector stores
rules_vector_store: Optional[VectorStore] = None
official_docs_vector_store: Optional[VectorStore] = None
echanges_vector_store: Optional[VectorStore] = None
# Modèles et état de la connexion BDD
llm: Optional[Any] = None # Type générique, sera ChatMistralAI ou ChatOpenAI
embeddings: Optional[Embeddings] = None
db_connected: bool = False
db_manager: Optional['DatabaseManager'] = None # Pour garder une instance


# Définition de l'état du graphe LangGraph
class State(Dict):
    """Structure pour représenter l'état du système RAG."""
    question: str # Question posée par l'utilisateur
    context: List[Document] # Contexte récupéré pour la question
    db_results: List[Dict[str, Any]]  # Résultats bruts de la base de données
    answer: str # Réponse générée


#================= GESTION DE LA BASE DE DONNÉES =================
class DatabaseManager:
    """Gestionnaire de connexion et d'interrogation de la base de données."""

    def __init__(self):
        """Initialise la configuration de la base de données."""
        self.config = {
            'user': os.getenv('SQL_USER'),
            'password': os.getenv('SQL_PASSWORD', ''),
            'host': os.getenv('SQL_HOST', 'localhost'),
            'database': os.getenv('SQL_DB'),
            'port': int(os.getenv('SQL_PORT', '3306'))
        }
        # Vérification initiale des variables essentielles
        if not all([self.config['user'], self.config['host'], self.config['database']]):
            logger.error("Variables d'environnement manquantes pour la connexion DB: SQL_USER, SQL_HOST ou SQL_DB ne sont pas définies.")
            # L'initialisation continue, mais tester_connexion échouera probablement.

    def _is_config_valid(self) -> bool:
        """Vérifie si la configuration essentielle est présente."""
        valid = all([self.config['user'], self.config['host'], self.config['database']])
        if not valid:
             logger.warning("Configuration DB incomplète (SQL_USER, SQL_HOST, SQL_DB).")
        return valid

    def tester_connexion(self) -> bool:
        """Teste la connexion à la base de données."""
        if not self._is_config_valid():
            return False

        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(**self.config, connect_timeout=5) # Timeout court
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            logger.info("Connexion réussie à la base de données.")
            return True
        except mysql.connector.Error as erreur:
            logger.error(f"Échec de la connexion à la base de données: {erreur}")
            return False
        except Exception as e:
            logger.error(f"Erreur inattendue lors du test de connexion DB: {e}", exc_info=True)
            return False
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

    def rechercher_dossier(self,
                      search_term: Optional[str] = None,
                      numero_dossier: Optional[str] = None,
                      statut: Optional[str] = None,
                      instructeur: Optional[str] = None,
                      date_debut_creation: Optional[date] = None,
                      date_fin_creation: Optional[date] = None,
                      limit: Optional[int] = 50,
                      **kwargs) -> List[Dict[str, Any]]:
        """
        Recherche des dossiers dans la base de données avec gestion des erreurs et fermeture de connexion.
        """
        if not self._is_config_valid():
            return []

        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(**self.config, connect_timeout=10) # Timeout un peu plus long pour query
            cursor = conn.cursor(dictionary=True) # Résultats sous forme de dict

            base_query = "SELECT * FROM dossiers"
            conditions = []
            parametres = []

            # Priorité au numéro de dossier s'il est fourni directement
            if numero_dossier:
                conditions.append("Numero = %s")
                parametres.append(numero_dossier.strip())
                logger.info(f"Recherche BDD par numéro direct: {numero_dossier.strip()}")
            # Sinon, analyser search_term
            elif search_term:
                cleaned_term = search_term.strip()
                # Format exact XX-YYYY ou XX YYYY
                is_exact_numero = re.fullmatch(r'\d{2}[-\s]?\d{4}', cleaned_term)
                if is_exact_numero:
                    # Normaliser au format XX-YYYY pour la recherche
                    normalized_numero = re.sub(r'\s', '-', cleaned_term)
                    conditions.append("Numero = %s")
                    parametres.append(normalized_numero)
                    logger.info(f"Recherche BDD par numéro exact détecté dans search_term: {normalized_numero}")
                else:
                    conditions.append("(Numero LIKE %s OR nom_usager LIKE %s)")
                    fuzzy_term = f"%{cleaned_term}%"
                    parametres.extend([fuzzy_term, fuzzy_term])
                    logger.info(f"Recherche BDD floue (numéro/nom) pour: {cleaned_term}")

            # Autres filtres
            if statut and statut.lower() != "tous":
                conditions.append("statut = %s")
                parametres.append(statut)
            if instructeur and instructeur.lower() != "tous":
                conditions.append("instructeur = %s")
                parametres.append(instructeur)
            if date_debut_creation:
                conditions.append("date_creation >= %s")
                parametres.append(date_debut_creation)
            if date_fin_creation:
                 conditions.append("date_creation <= %s")
                 parametres.append(date_fin_creation)
                 # Vérification simple de cohérence
                 if date_debut_creation and date_debut_creation > date_fin_creation:
                      logger.warning("Date de début postérieure à la date de fin dans la recherche BDD.")

            # Critères kwargs (utiliser avec prudence si les clés viennent de l'extérieur)
            for cle, valeur in kwargs.items():
                if valeur is not None:
                    # Exemple simple, pourrait nécessiter une validation des clés
                    conditions.append(f"`{cle}` = %s") # Backticks pour noms de colonnes
                    parametres.append(valeur)

            # Construction de la requête finale
            requete = base_query
            if conditions:
                requete += " WHERE " + " AND ".join(conditions)
            requete += " ORDER BY derniere_modification DESC" # Trier par défaut

            # Appliquer la limite seulement si ce n'est pas une recherche par numéro exact
            apply_limit = True
            if numero_dossier or (search_term and re.fullmatch(r'\d{2}[-\s]?\d{4}', search_term.strip())):
                apply_limit = False

            if apply_limit and limit is not None and limit > 0:
                requete += " LIMIT %s"
                parametres.append(limit)

            logger.info(f"Exécution requête BDD: {requete} | Params: {parametres}")
            cursor.execute(requete, tuple(parametres)) # Exécuter avec un tuple de paramètres
            resultats = cursor.fetchall()
            logger.info(f"{len(resultats)} dossiers trouvés dans la BDD.")
            return resultats

        except mysql.connector.Error as erreur:
            logger.error(f"Erreur lors de la recherche BDD: {erreur}")
            return []
        except Exception as e:
            logger.error(f"Erreur inattendue dans rechercher_dossier: {e}", exc_info=True)
            return []
        finally:
            # Assurer la fermeture du curseur et de la connexion
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

# === GESTION DES DOCUMENTS ET EMBEDDINGS ====

def load_official_docs_from_json(official_docs_path: str) -> List[Document]:
    """
    Charge les documents officiels depuis des fichiers JSON dans un répertoire,
    en gérant les types de blocs 'text' et 'qa' (y compris lorsque
    question/réponse sont des listes).
    """
    docs: List[Document] = []
    if not os.path.isdir(official_docs_path):
        logger.warning(f"Répertoire des documents officiels non trouvé: {official_docs_path}")
        return docs

    json_files = glob.glob(os.path.join(official_docs_path, "*.json"))
    logger.info(f"Chargement docs officiels: {len(json_files)} fichiers JSON trouvés dans {official_docs_path}.")

    for json_file in json_files:
        logger.debug(f"Traitement du fichier: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            doc_meta = data.get("document_metadata", {})
            global_meta = data.get("global_block_metadata", {})
            blocks_data = [] # Pour stocker le texte formaté de ce document

            # Extraire et concaténer le contenu textuel avec contexte structurel
            for page in data.get("pages", []):
                page_num = page.get("page_number", "")
                for section in page.get("sections", []):
                    title = section.get("section_title", "")
                    for block_index, block in enumerate(section.get("content_blocks", [])): # Ajouter un index pour le debug
                        block_type = block.get("type")
                        block_content = block.get("content")
                        extracted_text = "" # Texte à extraire de ce bloc

                        # --- Logique d'extraction adaptée ---
                        if block_type == "text" and isinstance(block_content, str):
                            cleaned_text = block_content.strip()
                            if cleaned_text:
                                extracted_text = cleaned_text
                                logger.debug(f"  [Fichier: {os.path.basename(json_file)}] Bloc texte trouvé (Page {page_num}, Section '{title}', Bloc {block_index})")

                        elif block_type == "qa" and isinstance(block_content, dict):
                            logger.debug(f"  [Fichier: {os.path.basename(json_file)}] Bloc QA trouvé (Page {page_num}, Section '{title}', Bloc {block_index})")
                            # Récupérer question et réponse brutes
                            question_raw = block_content.get("question", "")
                            answer_raw = block_content.get("answer", "")

                            # --- Traitement robuste pour question ---
                            question_str = ""
                            if isinstance(question_raw, str):
                                question_str = question_raw.strip()
                            elif isinstance(question_raw, list):
                                # Joindre les éléments de la liste (non vides après strip) avec un saut de ligne
                                question_str = "\n".join(str(item).strip() for item in question_raw if str(item).strip())
                                if question_str: logger.debug("    Question (liste) jointe.")
                            else:
                                logger.warning(f"    Type inattendu pour 'question' dans bloc QA ({type(question_raw)}), ignoré. Contenu: {question_raw!r}")

                            # --- Traitement robuste pour réponse ---
                            answer_str = ""
                            if isinstance(answer_raw, str):
                                answer_str = answer_raw.strip() # C'est ici que l'erreur se produisait potentiellement
                            elif isinstance(answer_raw, list):
                                # Joindre les éléments de la liste (non vides après strip) avec un saut de ligne
                                answer_str = "\n".join(str(item).strip() for item in answer_raw if str(item).strip())
                                if answer_str: logger.debug("    Réponse (liste) jointe.")
                            else:
                                logger.warning(f"    Type inattendu pour 'answer' dans bloc QA ({type(answer_raw)}), ignoré. Contenu: {answer_raw!r}")

                            # Construire le texte extrait seulement si on a une question ou une réponse non vide
                            if question_str or answer_str:
                                extracted_text = f"Question : {question_str}\nRéponse : {answer_str}"
                            else:
                                logger.debug("    Bloc QA ignoré (question et réponse vides après traitement).")


                        # --- Fin de la logique adaptée ---

                        # Ajouter le texte extrait (si existant) avec son en-tête structurel
                        if extracted_text:
                            block_header = f"Page {page_num}"
                            if title: block_header += f" - Section: {title}"
                            # Inclure le type de bloc peut être utile pour le contexte LLM
                            blocks_data.append(f"### {block_header} (Type: {block_type})\n\n{extracted_text}")

            # Si aucun bloc pertinent n'a été trouvé dans tout le fichier
            if not blocks_data:
                logger.warning(f"Aucun contenu texte pertinent (type 'text' ou 'qa' avec contenu) n'a été extrait de {json_file}, fichier ignoré.")
                continue # Passer au fichier JSON suivant

            # Concaténer tous les blocs extraits du fichier
            full_content = "\n\n---\n\n".join(blocks_data)

            # Créer les métadonnées (ajuster selon les besoins)
            metadata = {
                "source": json_file, # Chemin complet utile pour le debug
                "source_file": os.path.basename(json_file),
                "document_title": doc_meta.get("document_title"),
                "program_name": doc_meta.get("program_name"),
                "category": "docs_officiels", # Catégorie fixe pour cette fonction
                "type": "knowledge_document", # Type fixe pour cette fonction
                "date_update": global_meta.get("date_update") or doc_meta.get("date_update"),
                "tags": doc_meta.get("tags", []),
                "priority": doc_meta.get("priority", 100), # Priorité par défaut
            }
            # Nettoyer les métadonnées (supprimer clés avec valeur None ou vide)
            metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}

            # Créer et ajouter le Document Langchain
            docs.append(Document(page_content=full_content, metadata=metadata))
            logger.info(f"Document créé avec succès pour {os.path.basename(json_file)} avec {len(blocks_data)} bloc(s) extrait(s).")

        except json.JSONDecodeError as je:
            logger.error(f"Erreur de décodage JSON dans le fichier {json_file}: {je}")
        except Exception as e:
            # Capturer d'autres erreurs potentielles pendant le traitement du fichier
            logger.error(f"Erreur inattendue lors du traitement du fichier JSON {json_file}: {e}", exc_info=True) # exc_info=True ajoute le traceback au log

    logger.info(f"Chargement des documents officiels terminé. {len(docs)} documents ont été créés au total.")
    return docs

def load_rules_from_json(rules_path: str) -> List[Document]:
    """Charge les règles depuis des fichiers JSON."""
    rules_docs = []
    if not os.path.isdir(rules_path):
        logger.warning(f"Répertoire des règles non trouvé: {rules_path}")
        return rules_docs

    json_files = glob.glob(os.path.join(rules_path, "*.json"))
    logger.info(f"Chargement règles: {len(json_files)} fichiers JSON trouvés.")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)

            global_metadata = rules_data.get("global_rule_metadata", {})
            file_rules = rules_data.get("rules", [])

            if not file_rules:
                 logger.warning(f"Aucune règle trouvée dans {json_file}.")
                 continue

            for rule in file_rules:
                # Formatage du contenu de la règle
                content = f"""**Règle N°{rule.get('rule_number', 'N/A')} : {rule.get('title', 'Sans titre')}**
*Contexte d'application* : {rule.get('context', 'Non spécifié')}
*Action/Directive* : {rule.get('action', 'Non spécifiée')}"""

                # Création des métadonnées
                metadata = {
                    "source": json_file,
                    "source_file": os.path.basename(json_file),
                    "category": "regles", # Catégorie fixe
                    "type": "rule_document", # Type fixe
                    "rule_number": rule.get("rule_number", "N/A"),
                    "title": rule.get("title", "Sans titre"),
                    "tags": rule.get("metadata", {}).get("tags", []),
                    "priority": global_metadata.get("priority", 50), # Priorité plus élevée pour règles
                    "keywords": rule.get("metadata", {}).get("keywords", []),
                }
                metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}

                rules_docs.append(Document(page_content=content.strip(), metadata=metadata))

        except json.JSONDecodeError as je:
            logger.error(f"Erreur de décodage JSON dans {json_file}: {je}")
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier de règles {json_file}: {e}", exc_info=True)

    logger.info(f"Chargement règles terminé. {len(rules_docs)} règles créées.")
    return rules_docs

def load_all_documents() -> Tuple[List[Document], List[Document], List[Document]]:
    """
    Charge tous les documents (officiels, échanges, règles) et les retourne en listes séparées.
    """
    logger.info("Début du chargement de tous les types de documents...")

    # Charger les règles
    rules_docs = load_rules_from_json(REGLES_PATH)

    # Charger les documents officiels
    official_docs = load_official_docs_from_json(OFFICIAL_DOCS_PATH)

    # Charger les documents d'échanges (TXT)
    echanges_docs: List[Document] = []
    if os.path.isdir(ECHANGES_PATH):
        try:
            echanges_loader = DirectoryLoader(
                ECHANGES_PATH,
                glob="**/*.txt", # Recherche des .txt dans tous les sous-dossiers
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                recursive=True,
                show_progress=True,
                use_multithreading=True, # Peut accélérer si beaucoup de petits fichiers
                # silent_errors=True # Pourrait masquer des problèmes de chargement
            )
            loaded_echanges = echanges_loader.load() # Charge les documents
            logger.info(f"Chargement échanges: {len(loaded_echanges)} fichiers TXT trouvés et chargés.")

            # Ajouter/Vérifier les métadonnées pour chaque document d'échange
            for doc in loaded_echanges:
                if not hasattr(doc, 'metadata'): # Sécurité si un loader retourne un objet sans metadata
                    doc.metadata = {}
                doc.metadata["category"] = "echanges"
                doc.metadata["type"] = "echange_document"
                # Essayer d'extraire le nom de fichier depuis la source
                source_path = doc.metadata.get("source", "inconnu.txt")
                doc.metadata["source_file"] = os.path.basename(source_path)
                # Garder le chemin complet aussi
                doc.metadata["source"] = source_path
                # Nettoyer
                doc.metadata = {k: v for k, v in doc.metadata.items() if v is not None and v != ""}
            echanges_docs.extend(loaded_echanges)

        except Exception as e:
            logger.error(f"Erreur lors du chargement des documents d'échanges depuis {ECHANGES_PATH}: {e}", exc_info=True)
    else:
        logger.warning(f"Répertoire des échanges non trouvé: {ECHANGES_PATH}")

    logger.info(f"Chargement complet: {len(official_docs)} officiels, {len(echanges_docs)} échanges, {len(rules_docs)} règles.")
    return official_docs, echanges_docs, rules_docs

# --- Fonctions de gestion du cache et des Vector Stores ---

def calculate_documents_hash(documents: List[Document], embeddings: Embeddings) -> str:
    """Calcule un hash SHA256 basé sur le contenu des documents et le type/modèle d'embeddings."""
    hasher = hashlib.sha256()

    # Inclure des informations sur le modèle d'embedding
    hasher.update(str(type(embeddings).__name__).encode('utf-8'))
    if hasattr(embeddings, "model"): # Pour OpenAIEmbeddings et certains autres
         hasher.update(str(embeddings.model).encode('utf-8'))
    elif hasattr(embeddings, "model_name"): # Pour MistralAIEmbeddings
         hasher.update(str(embeddings.model_name).encode('utf-8'))

    # Trier les documents par contenu et métadonnées pour un hash cohérent
    # Convertir les métadonnées en une chaîne triée par clé pour la stabilité
    sorted_docs = sorted(documents, key=lambda d: (d.page_content, str(sorted(d.metadata.items()))))

    # Ajouter le contenu et les métadonnées triées de chaque document au hash
    for doc in sorted_docs:
        hasher.update(doc.page_content.encode('utf-8'))
        hasher.update(str(sorted(doc.metadata.items())).encode('utf-8'))

    return hasher.hexdigest()

def load_cached_embeddings_from_dir(embeddings: Embeddings, cache_dir: str, current_hash: str) -> Optional[VectorStore]:
    """Tente de charger un index FAISS depuis un cache si le hash correspond."""
    os.makedirs(cache_dir, exist_ok=True) # S'assurer que le dossier cache existe
    hash_file_path = os.path.join(cache_dir, "documents_hash.txt")
    faiss_index_path = os.path.join(cache_dir, "faiss_index") # Dossier pour l'index FAISS

    if os.path.exists(hash_file_path) and os.path.isdir(faiss_index_path):
        try:
            with open(hash_file_path, "r", encoding='utf-8') as f:
                cached_hash = f.read().strip()

            if cached_hash == current_hash:
                logger.info(f"Cache HIT: Hash correspondant trouvé dans {cache_dir}. Chargement de l'index FAISS...")
                # allow_dangerous_deserialization est nécessaire pour FAISS
                vector_store = FAISS.load_local(
                    faiss_index_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Cache - Index FAISS chargé avec succès depuis {cache_dir}.")
                return vector_store
            else:
                logger.info(f"Cache MISS: Hash différent dans {cache_dir}. Recalcul nécessaire.")
        except FileNotFoundError:
             logger.warning(f"Cache - Fichier hash ou dossier index non trouvé dans {cache_dir} (peut être normal lors du premier lancement).")
        except Exception as e:
            # Erreur de chargement (ex: version incompatible, fichier corrompu)
            logger.error(f"Cache - Erreur lors du chargement de l'index FAISS depuis {cache_dir}: {e}", exc_info=True)
            # Optionnel: Supprimer le cache potentiellement corrompu pour forcer la recréation
            # import shutil
            # try:
            #     shutil.rmtree(faiss_index_path, ignore_errors=True)
            #     if os.path.exists(hash_file_path): os.remove(hash_file_path)
            #     logger.info(f"Cache corrompu supprimé pour {cache_dir}.")
            # except OSError as oe:
            #      logger.error(f"Erreur lors de la suppression du cache corrompu: {oe}")
    else:
        logger.info(f"Cache - Cache non trouvé ou incomplet dans {cache_dir}.")

    return None # Retourner None si le cache n'est pas valide ou n'existe pas

def save_embeddings_cache_to_dir(vector_store: VectorStore, documents_hash: str, cache_dir: str):
    """Sauvegarde l'index FAISS et le hash des documents dans un répertoire de cache."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        hash_file_path = os.path.join(cache_dir, "documents_hash.txt")
        faiss_index_path = os.path.join(cache_dir, "faiss_index")

        # Sauvegarder l'index FAISS
        vector_store.save_local(faiss_index_path)

        # Sauvegarder le hash (seulement après la sauvegarde réussie de l'index)
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
    # 1. Calculer le hash actuel
    current_hash = calculate_documents_hash(documents, embeddings)

    # 2. Essayer de charger depuis le cache
    vector_store = load_cached_embeddings_from_dir(embeddings, cache_dir, current_hash)

    # 3. Si le cache est trouvé et valide, le retourner
    if vector_store is not None:
        return vector_store

    # 4. Sinon, créer un nouveau vector store
    logger.info(f"Création d'un nouvel index FAISS pour {cache_dir}...")
    try:
        vector_store = FAISS.from_documents(documents, embeddings)
        logger.info(f"Nouvel index FAISS créé pour {cache_dir}. Sauvegarde en cache...")
        # 5. Sauvegarder le nouveau store dans le cache
        save_embeddings_cache_to_dir(vector_store, current_hash, cache_dir)
        return vector_store
    except Exception as e:
        logger.error(f"Erreur critique lors de la création de l'index FAISS pour {cache_dir}: {e}", exc_info=True)
        return None # Échec de la création


# ================= EXTRACTION ET TRAITEMENT DES DONNÉES (pour le graphe) =================

def extract_dossier_number(question: str) -> List[str]:
    """Extrait les numéros de dossier (format XX-YYYY ou XX YYYY) de la question."""
    # Pattern pour XX-YYYY ou XX YYYY, potentiellement précédé de "dossier", "n°", etc.
    patterns = [
        r'\b(\d{2}[-\s]\d{4})\b',  # Format isolé
        r'(?:dossier|cas|référence)\s*(?:n°|numéro|numero|n|°|:)?\s*(\d{2}[-\s]\d{4})\b' # Précédé d'un mot clé
    ]
    results = []
    for pattern in patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        for match in matches:
            # Normaliser au format XX-YYYY et supprimer les doublons
            normalized_match = re.sub(r'\s', '-', match).strip()
            if normalized_match not in results:
                results.append(normalized_match)

    if results:
         logger.info(f"Numéros de dossier potentiels extraits de la question: {results}")
    return results

def db_resultats_to_documents(resultats: List[Dict[str, Any]]) -> List[Document]:
    """Convertit les résultats de la base de données (liste de dictionnaires) en Documents Langchain."""
    documents = []
    if not resultats:
        return documents

    logger.info(f"Conversion de {len(resultats)} résultat(s) BDD en Documents Langchain.")
    for i, resultat in enumerate(resultats):
        # Formatage du contenu pour lisibilité
        content_parts = [f"**Informations Dossier BDD {resultat.get('Numero', 'N/A')} (Résultat {i+1})**"]
        for key, value in resultat.items():
             # Formater les dates si présentes
             if isinstance(value, date):
                 value_str = value.strftime('%d/%m/%Y')
             else:
                 value_str = str(value) if value is not None else 'N/A'
             # Formater la ligne (ex: Nom Usager: Dupont)
             formatted_key = key.replace('_', ' ').capitalize()
             content_parts.append(f"- {formatted_key}: {value_str}")

        content = "\n".join(content_parts)

        # Création des métadonnées spécifiques à la BDD
        metadata = {
            "source": "base_de_donnees",
            "category": "dossier_bdd", # Catégorie claire
            "type": "db_data",       # Type clair
            "numero_dossier": resultat.get('Numero', 'N/A'),
            "db_result_index": i + 1, # Index du résultat dans la liste BDD
            "update_date": resultat.get('derniere_modification', None) # Date objet si possible
        }
        metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}

        doc = Document(page_content=content.strip(), metadata=metadata)
        documents.append(doc)

    return documents


# ================= FONCTIONS DU GRAPHE RAG =================

def search_database(state: State) -> Dict[str, Any]:
    """Noeud du graphe: Recherche dans la BDD basé sur les numéros extraits."""
    global db_manager, db_connected
    logger.info("Noeud: search_database")

    question = state["question"]
    dossier_numbers = extract_dossier_number(question)

    db_results = []
    if not db_connected or not db_manager:
        logger.warning("Recherche BDD annulée: connexion non disponible ou manager non initialisé.")
        return {"db_results": []}

    if dossier_numbers:
        # Stratégie actuelle: rechercher le premier numéro trouvé.
        # Pourrait être modifié pour rechercher tous les numéros ou utiliser d'autres termes.
        num_to_search = dossier_numbers[0]
        logger.info(f"Tentative de recherche BDD pour le numéro: {num_to_search}")
        try:
            db_results = db_manager.rechercher_dossier(numero_dossier=num_to_search)
            logger.info(f"{len(db_results)} résultats BDD trouvés pour {num_to_search}.")
        except Exception as e:
            logger.error(f"Erreur pendant l'appel à rechercher_dossier: {e}", exc_info=True)
            db_results = [] # Assurer une liste vide en cas d'erreur
    else:
         logger.info("Aucun numéro de dossier détecté dans la question pour la recherche BDD.")

    return {"db_results": db_results}

def retrieve(state: State) -> Dict[str, Any]:
    """Noeud du graphe: Récupère les documents depuis BDD et les 3 vector stores."""
    global rules_vector_store, official_docs_vector_store, echanges_vector_store
    logger.info("Noeud: retrieve")

    question = state["question"]
    # 1. Convertir les résultats BDD (déjà dans l'état) en Documents
    db_docs = db_resultats_to_documents(state.get("db_results", []))

    # Initialiser les listes de résultats des recherches vectorielles
    relevant_rules = []
    relevant_official_docs = []
    relevant_echanges = []

    # 2. Interroger le Vector Store des Règles (k faible, haute pertinence attendue)
    if rules_vector_store:
        try:
            logger.debug(f"Recherche Règles pour: '{question[:50]}...'")
            relevant_rules = rules_vector_store.similarity_search(question, k=3)
            logger.info(f"Retrieve - {len(relevant_rules)} règles récupérées.")
            # Vérifier/forcer les métadonnées si nécessaire (normalement déjà fait au chargement)
            # for doc in relevant_rules: doc.metadata['category'] = 'regles'; doc.metadata['type'] = 'rule_document'
        except Exception as e:
            logger.error(f"Retrieve - Erreur recherche rules_vector_store: {e}", exc_info=True)
    else:
        logger.warning("Retrieve - rules_vector_store non disponible.")

    # 3. Interroger le Vector Store des Documents Officiels (k moyen, base de connaissance)
    if official_docs_vector_store:
        try:
            logger.debug(f"Recherche Docs Officiels pour: '{question[:50]}...'")
            relevant_official_docs = official_docs_vector_store.similarity_search(question, k=5)
            logger.info(f"Retrieve - {len(relevant_official_docs)} documents officiels récupérés.")
        except Exception as e:
            logger.error(f"Retrieve - Erreur recherche official_docs_vector_store: {e}", exc_info=True)
    else:
        logger.warning("Retrieve - official_docs_vector_store non disponible.")

    # 4. Interroger le Vector Store des Échanges (k faible/moyen, pour style/exemples)
    if echanges_vector_store:
        try:
            logger.debug(f"Recherche Echanges pour: '{question[:50]}...'")
            relevant_echanges = echanges_vector_store.similarity_search(question, k=3)
            logger.info(f"Retrieve - {len(relevant_echanges)} documents d'échanges récupérés.")
        except Exception as e:
            logger.error(f"Retrieve - Erreur recherche echanges_vector_store: {e}", exc_info=True)
    else:
        logger.warning("Retrieve - echanges_vector_store non disponible.")

    # 5. Combiner les résultats dans l'ordre de priorité pour le LLM: BDD > Règles > Officiels > Échanges
    combined_docs = db_docs + relevant_rules + relevant_official_docs + relevant_echanges
    logger.info(f"Retrieve - Documents combinés avant déduplication: {len(combined_docs)} "
                f"(BDD: {len(db_docs)}, Règles: {len(relevant_rules)}, "
                f"Officiels: {len(relevant_official_docs)}, Echanges: {len(relevant_echanges)})")

    # 6. Déduplication simple basée sur le contenu exact (peut être affinée si nécessaire)
    seen_content = set()
    unique_docs = []
    for doc in combined_docs:
        # Clé de déduplication: contenu + source principale (pour différencier légèrement)
        dedup_key = (doc.page_content, doc.metadata.get("source_file", doc.metadata.get("source")))
        if dedup_key not in seen_content:
            unique_docs.append(doc)
            seen_content.add(dedup_key)

    if len(unique_docs) < len(combined_docs):
         logger.info(f"Retrieve - Documents après déduplication: {len(unique_docs)}")

    return {"context": unique_docs}

def generate(state: State) -> Dict[str, Any]:
    """Noeud du graphe: Génère la réponse en utilisant le LLM, le contexte et un prompt structuré."""
    global llm
    logger.info("Noeud: generate")

    question = state["question"]
    context_docs = state.get("context", [])

    if llm is None:
        logger.error("Génération impossible: Le modèle LLM n'est pas initialisé.")
        return {"answer": "Erreur: Le service de génération de réponse n'est pas disponible."}

    if not context_docs:
        logger.warning("Génération: Aucun document de contexte fourni après l'étape retrieve.")
        # Vérifier si la BDD avait des résultats même si le reste est vide
        db_results_in_state = state.get("db_results", [])
        if db_results_in_state:
             db_docs_only = db_resultats_to_documents(db_results_in_state)
             db_content_only = "\n\n".join([doc.page_content for doc in db_docs_only])
             answer_only_db = (f"J'ai trouvé les informations suivantes dans la base de données concernant votre demande :\n\n{db_content_only}\n\n"
                               "Cependant, je n'ai pas trouvé d'informations complémentaires ou de règles spécifiques dans ma base de connaissances pour élaborer davantage.")
             return {"answer": answer_only_db}
        else:
            return {"answer": "Je suis désolé, mais je n'ai trouvé aucune information pertinente (ni dans la base de données, ni dans les documents de référence) pour répondre à votre question."}

    # Préparer le contexte structuré pour le prompt
    rules_content, db_content, official_content, echanges_content, other_content = [], [], [], [], []
    docs_details_list = []
    source_counter = 1

    for doc in context_docs:
        category = doc.metadata.get("category", "inconnu")
        doc_type = doc.metadata.get("type", "inconnu")
        source_file = doc.metadata.get("source_file", os.path.basename(doc.metadata.get("source", "Source inconnue")))
        source_id = f"SOURCE {source_counter}"
        content_with_id = f"[{source_id}]\n{doc.page_content.strip()}"

        doc_info = {
            "id": source_id,
            "file": source_file,
            "category": category,
            "type": doc_type,
        }
        docs_details_list.append(doc_info)

        # Classifier pour le prompt basé sur la catégorie/type
        if category == "regles" or doc_type == "rule_document":
            rules_content.append(content_with_id)
        elif category == "dossier_bdd" or doc_type == "db_data":
            db_content.append(content_with_id)
        elif category == "docs_officiels" or doc_type == "knowledge_document":
            official_content.append(content_with_id)
        elif category == "echanges" or doc_type == "echange_document":
            echanges_content.append(content_with_id)
        else:
            other_content.append(f"[{source_id} - Catégorie: {category}]\n{doc.page_content.strip()}") # Préciser la catégorie si inconnue

        source_counter += 1

    # Construire les sections du contexte pour le prompt
    context_sections = []
    if db_content: context_sections.append("**INFORMATIONS SPÉCIFIQUES AU DOSSIER (Base de Données - Priorité 1 - Absolue):**\n" + "\n\n---\n\n".join(db_content))
    if rules_content: context_sections.append("**RÈGLES APPLICABLES (Priorité 2 - Très Haute):**\n" + "\n\n---\n\n".join(rules_content))
    if official_content: context_sections.append("**DOCUMENTS OFFICIELS / PROCÉDURES (Priorité 3 - Standard):**\n" + "\n\n---\n\n".join(official_content))
    if echanges_content: context_sections.append("**EXEMPLES D'ÉCHANGES (Utiliser pour style/ton SEULEMENT - Priorité 4 - Basse):**\n" + "\n\n---\n\n".join(echanges_content))
    if other_content: context_sections.append("**AUTRES DOCUMENTS (Vérifier pertinence - Priorité 5 - Très Basse):**\n" + "\n\n---\n\n".join(other_content))

    context_string_for_prompt = "\n\n".join(context_sections)

    # Formater la liste des sources pour référence
    formatted_sources_list = "\n".join([f"- [{d['id']}] {d['file']} (Cat: {d['category']}, Type: {d['type']})" for d in docs_details_list])

    # Instructions Système (adaptées de votre version précédente)
    system_instructions = (
        "Tu es un assistant expert spécialisé dans le dispositif KAP Numérique, conçu pour aider les agents instructeurs.\n"
        "Ta mission est de fournir des réponses précises, structurées et professionnelles basées **exclusivement** sur les informations fournies dans le contexte.\n\n"
        "**RÈGLES IMPÉRATIVES POUR LA GÉNÉRATION DE RÉPONSE:**\n"
        "1.  **HIÉRARCHIE STRICTE DES SOURCES:** Analyse le contexte en respectant l'ordre de priorité suivant:\n"
        "    1.  Base de Données (`dossier_bdd`, `db_data`): **Priorité absolue**. Fais confiance à ces données pour les informations spécifiques au dossier.\n"
        "    2.  Règles (`regles`, `rule_document`): **Priorité très haute**. Applique ces directives internes.\n"
        "    3.  Documents Officiels (`docs_officiels`, `knowledge_document`): **Priorité standard**. Utilise pour les procédures générales et faits.\n"
        "    4.  Échanges (`echanges`, `echange_document`): **Priorité basse**. Utilise **UNIQUEMENT** comme inspiration pour le ton et la formulation. **NE JAMAIS** citer comme source factuelle si cela contredit les sources 1, 2 ou 3.\n"
        "    5.  Autres documents: **Priorité très basse**. Utilise avec extrême prudence, si et seulement si aucune autre source ne répond.\n"
        "2.  **EXCLUSIVITÉ DU CONTEXTE:** Base **toute** ta réponse sur les informations présentes dans le contexte fourni (`[SOURCE X]`). N'ajoute aucune information externe, même si tu penses la connaître.\n"
        "3.  **MANQUE D'INFORMATION:** Si l'information nécessaire pour répondre n'est pas dans le contexte, indique-le clairement (ex: \"Le contexte fourni ne contient pas d'information sur [sujet].\"). **NE JAMAIS INVENTER.**\n"
        "4.  **FORMAT ET STYLE:**\n"
        "    - Adresse-toi à l'agent instructeur.\n"
        "    - Style: Formel, institutionnel, clair, concis.\n"
        "    - Formatage Markdown: Utilise **gras**, *italique*, listes à puces/numérotées, `### Titres` pour structurer.\n"
        "    - Citations: Justifie les points clés en citant la source exacte `[SOURCE X]`. Si multiple: `[SOURCE 1, SOURCE 3]`.\n"
        "5.  **INTERDICTIONS:**\n"
        "    - Ne mentionne jamais que tu es une IA ou un chatbot.\n"
        "    - N'invite pas à contacter un autre agent.\n"
        "    - N'utilise pas les échanges comme source factuelle.\n\n"
        "**Processus:**\n"
        "1. Lire attentivement la question de l'agent.\n"
        "2. Analyser le contexte fourni en respectant la hiérarchie des sources.\n"
        "3. Formuler une réponse directe à la question.\n"
        "4. Détailler et justifier avec les sources `[SOURCE X]`.\n"
        "5. Utiliser le formatage Markdown pour la clarté.\n"
        "6. Si l'information manque, le signaler."
    )

    # Construction de l'invite utilisateur
    user_prompt = (
        f"**Question de l'Agent Instructeur:**\n{question}\n\n"
        f"**Contexte Fourni (analyser en respectant la hiérarchie indiquée dans les instructions système):**\n"
        f"---\n{context_string_for_prompt}\n---\n\n"
        f"**Liste des Sources du Contexte:**\n{formatted_sources_list}\n\n"
        f"**Réponse pour l'Agent (basée EXCLUSIVEMENT sur le contexte, justifiée avec [SOURCE X], format Markdown):**"
    )

    logger.info(f"Génération - Prompt final préparé (longueur approx: {len(user_prompt)} chars). Appel du LLM...")
    # logger.debug(f"--- SYSTEM PROMPT ---\n{system_instructions}\n--- USER PROMPT ---\n{user_prompt[:2000]}...") # Tronquer pour le debug

    # Appel du modèle LLM
    try:
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ]
        response = llm.invoke(
            messages,
            # Options de génération peuvent être ajoutées ici si nécessaire
            # temperature=0.1,
            # max_tokens=2000
            )
        final_answer = response.content
        logger.info("Génération - Réponse reçue du LLM.")
        # logger.debug(f"Réponse brute du LLM: {final_answer}")
        return {"answer": final_answer}

    except Exception as e:
        logger.error(f"Génération - Erreur lors de l'appel LLM: {e}", exc_info=True)
        return {"answer": f"Erreur technique lors de la génération de la réponse par le modèle linguistique. Détails: {e}"}

# ================= CREATION DU GRAPHE RAG =================

def build_graph() -> StateGraph:
    """Construit et compile le graphe LangGraph."""
    logger.info("Construction du graphe RAG...")
    workflow = StateGraph(State)

    # Ajout des noeuds (fonctions définies précédemment)
    workflow.add_node("search_database", search_database)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # Définition des transitions (flux de travail)
    workflow.add_edge(START, "search_database") # Commence par la recherche BDD
    workflow.add_edge("search_database", "retrieve") # Puis récupération vectorielle
    workflow.add_edge("retrieve", "generate")      # Puis génération
    workflow.add_edge("generate", END)             # Termine après la génération

    # Compilation du graphe
    try:
        graph = workflow.compile()
        logger.info("Graphe RAG compilé avec succès.")
        return graph
    except Exception as e:
        logger.error(f"Erreur critique lors de la compilation du graphe: {e}", exc_info=True)
        raise # Renvoyer l'erreur car le système ne peut pas fonctionner sans graphe

# ================= INITIALISATION DU SYSTÈME RAG =================

def init_rag_system() -> Dict[str, Any]:
    """Initialise tous les composants du système RAG (LLM, Embeddings, DB, Vector Stores, Graphe)."""
    # Utiliser les variables globales pour stocker les objets initialisés
    global llm, embeddings, db_manager, db_connected
    global rules_vector_store, official_docs_vector_store, echanges_vector_store

    logger.info("="*20 + " Initialisation du Système RAG " + "="*20)

    # Initialisation du dictionnaire status SANS la clé "graph" pour l'instant
    status = {
        "embeddings_ok": False, "llm_ok": False, "db_ok": False,
        "rules_store_ok": False, "official_docs_store_ok": False, "echanges_store_ok": False,
        "graph_ok": False,
        "counts": {"rules": 0, "official": 0, "echanges": 0, "rules_splits": 0, "official_splits": 0, "echanges_splits": 0},
        # "graph": graph, # <--- SUPPRIMER CETTE LIGNE ICI
        "search_function": None,
        "error_messages": []
    }
    # Initialiser la variable locale graph à None
    graph = None # <--- AJOUTER CECI

    # 1. Initialiser Embeddings
    try:
        embeddings = MistralAIEmbeddings()
        logger.info(f"Embeddings initialisés: {type(embeddings).__name__}")
        status["embeddings_ok"] = True
    except Exception as e:
        logger.error(f"Échec initialisation Embeddings: {e}", exc_info=True)
        status["error_messages"].append(f"Embeddings: {e}")
        embeddings = None

    # 2. Initialiser LLM
    try:
        llm = ChatMistralAI(model="mistral-large-latest", temperature=0.1)
        logger.info(f"LLM initialisé: {type(llm).__name__} (Model: {getattr(llm, 'model', 'N/A')})")
        status["llm_ok"] = True
    except Exception as e:
        logger.error(f"Échec initialisation LLM: {e}", exc_info=True)
        status["error_messages"].append(f"LLM: {e}")
        llm = None

    if not status["embeddings_ok"] or not status["llm_ok"]:
        logger.critical("Arrêt de l'initialisation: Embeddings ou LLM manquants.")
        # AJOUTER la clé graph (avec la valeur None) au status avant de retourner
        status["graph"] = graph # <--- Assurer que la clé existe même en cas d'échec précoce
        return status

    # 3. Initialiser et tester la connexion DB
    # (Code DB inchangé...)
    try:
        db_manager = DatabaseManager()
        if db_manager._is_config_valid(): # Vérifier si la config est là avant de tester
             db_connected = db_manager.tester_connexion()
             status["db_ok"] = db_connected
             if db_connected:
                 status["search_function"] = db_manager.rechercher_dossier
                 logger.info("Connexion DB réussie.")
             else:
                 logger.warning("Connexion DB échouée (vérifier config et accessibilité).")
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
    # (Code load_all_documents inchangé...)
    official_docs, echanges_docs, rules_docs = load_all_documents()
    status["counts"]["official"] = len(official_docs)
    status["counts"]["echanges"] = len(echanges_docs)
    status["counts"]["rules"] = len(rules_docs)


    # 5. Découper les documents
    # (Code splitters inchangé...)
    general_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    rules_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, add_start_index=True) # Plus petit pour règles

    official_splits = general_splitter.split_documents(official_docs) if official_docs else []
    echanges_splits = general_splitter.split_documents(echanges_docs) if echanges_docs else []
    rules_splits = rules_splitter.split_documents(rules_docs) if rules_docs else []
    status["counts"]["official_splits"] = len(official_splits)
    status["counts"]["echanges_splits"] = len(echanges_splits)
    status["counts"]["rules_splits"] = len(rules_splits)
    logger.info(f"Documents découpés: {len(official_splits)} officiels, {len(echanges_splits)} échanges, {len(rules_splits)} règles.")


    # 6. Créer/Charger les Vector Stores
    # (Code create_vector_store inchangé...)
    rules_store_cache_dir = os.path.join(CACHE_DIR, "rules_store")
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

    # 7. Construire le graphe (assignation à la variable locale 'graph')
    try:
        # Assigner à la variable locale 'graph' initialisée plus haut
        graph = build_graph()
        # La clé "graph" n'est plus dans status ici, on l'ajoutera après
        status["graph_ok"] = True
        logger.info("Graphe RAG final prêt.")
    except Exception as e:
        logger.error(f"Échec construction graphe RAG: {e}", exc_info=True)
        status["error_messages"].append(f"Graphe: {e}")
        status["graph_ok"] = False
        graph = None # Assurer que graph est None si la construction échoue

    # AJOUTER la clé "graph" au dictionnaire status MAINTENANT,
    # après la tentative de construction. Sa valeur sera soit l'objet graphe, soit None.
    status["graph"] = graph # <--- AJOUTER LA CLÉ ICI

    logger.info("="*20 + " Fin Initialisation Système RAG " + "="*20)

    if status["error_messages"]:
         logger.warning(f"Erreurs/Avertissements pendant l'initialisation: {status['error_messages']}")

    return status

# --- Fin de la fonction init_rag_system ---

# Initialiser le système RAG
rag_system_status = init_rag_system()

# Mettre à jour la variable globale graph pour qu'elle soit visible partout
# ATTENTION: Renommer la variable globale pour éviter confusion, ou ne pas utiliser de globale ici
# Option 1: Ne pas utiliser de variable globale 'graph' en dehors de tests simples
app_graph = rag_system_status.get("graph") # Récupère le graphe depuis le statut retourné