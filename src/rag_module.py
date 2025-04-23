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
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langgraph.graph import StateGraph, END, START


# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_module")

# Chargement des variables d'environnement
load_dotenv()

#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Chemins vers les répertoires de données
DATA_ROOT = os.getenv("DATA_ROOT", "../data")
ECHANGES_PATH = os.path.join(DATA_ROOT, "echanges")
REGLES_PATH = os.path.join(DATA_ROOT, "regles") 
OFFICIAL_DOCS_PATH = os.path.join(DATA_ROOT, "docs_officiels")
CACHE_DIR = "./cache"  # Répertoire pour le cache

# Variables globales
knowledge_vector_store = None
rules_vector_store = None
llm = None
embeddings = None
db_connected = False
rechercher_dossier_func = None

# Définition de l'état du système
class State(Dict):
    """Structure pour représenter l'état du système RAG."""
    question: str #Question posée par l'utilisateur
    context: List[Document] #Contexte récupéré pour la question
    db_results: List[Dict[str, Any]]  # Résultats de la base de données
    answer: str # Réponse générée

#================= GESTION DE LA BASE DE DONNÉES =================

class DatabaseManager:
    """Gestionnaire de connexion et d'interrogation de la base de données."""
    
    def __init__(self):
        """Initialise la connexion à la base de données."""
     
        self.config = {
            'user': os.getenv('SQL_USER'),
            'password': os.getenv('SQL_PASSWORD', ''),
            'host': os.getenv('SQL_HOST', 'localhost'),
            'database': os.getenv('SQL_DB'),
            'port': int(os.getenv('SQL_PORT', '3306'))
        }
        
        # Vérification des variables essentielles
        if not all([self.config['user'], self.config['host'], self.config['database']]):
            logger.error("Variables essentielles manquantes pour la DB : SQL_USER, SQL_HOST ou SQL_DB.")
            pass
    
    def tester_connexion(self) -> bool:
        """Teste la connexion à la base de données."""
        try:
            # S'assurer que la config est valide avant de tenter la connexion
            if not all([self.config['user'], self.config['host'], self.config['database']]):
                 logger.warning("Impossible de tester la connexion DB, configuration incomplète.")
                 return False
                 
            conn = mysql.connector.connect(**self.config)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            logger.info("Connexion réussie à la base de données.")
            return True
        except mysql.connector.Error as erreur:
            logger.error(f"Échec de la connexion à la base de données: {erreur}")
            return False
        except Exception as e:
            logger.error(f"Erreur inattendue lors du test de connexion DB: {e}", exc_info=True)
            return False
    
    
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
        Recherche des dossiers dans la base de données 
        """
        try:
            # On  Vérifie la connexion avant d'essayer d'interroger
            conn = mysql.connector.connect(**self.config)
            cursor = conn.cursor(dictionary=True)
            
            # Construction de la requête SQL
            base_query = "SELECT * FROM dossiers"
            conditions = []
            parametres = []
            
            # Si numero_dossier est fourni, c'est la priorité absolue
            if numero_dossier:
                conditions.append("Numero = %s")
                parametres.append(numero_dossier.strip())
            # Sinon, on regarde si search_term correspond à un numéro de dossier
            elif search_term:
                cleaned_term = search_term.strip()
                # Format exact de numéro de dossier (ex: 82-4585)
                is_exact_numero = re.fullmatch(r'\d{2}-\d{4}', cleaned_term)
                
                if is_exact_numero:
                    # Recherche par numéro exact
                    conditions.append("Numero = %s")
                    parametres.append(cleaned_term)
                    logger.info(f"Recherche exacte par numéro: {cleaned_term}")
                else:
                    # Recherche floue
                    conditions.append("(Numero LIKE %s OR nom_usager LIKE %s)")
                    fuzzy_term = f"%{cleaned_term}%"
                    parametres.extend([fuzzy_term, fuzzy_term])
                    logger.info(f"Recherche floue: {cleaned_term}")
            
            # Application des autres filtres
            if statut and statut.lower() != "tous":
                conditions.append("statut = %s")
                parametres.append(statut)
            
            if instructeur and instructeur.lower() != "tous":
                conditions.append("instructeur = %s")
                parametres.append(instructeur)
            
            # Gestion des dates
            if date_debut_creation and date_fin_creation:
                if date_debut_creation <= date_fin_creation:
                    conditions.append("date_creation BETWEEN %s AND %s")
                    parametres.extend([date_debut_creation, date_fin_creation])
                else:
                    logger.warning("Date de début postérieure à la date de fin.")
            elif date_debut_creation:
                conditions.append("date_creation >= %s")
                parametres.append(date_debut_creation)
            elif date_fin_creation:
                conditions.append("date_creation <= %s")
                parametres.append(date_fin_creation)
            
            # Critères supplémentaires (kwargs)
            for cle, valeur in kwargs.items():
                if valeur is not None:
                    conditions.append(f"{cle} = %s")
                    parametres.append(valeur)
            
            # Assemblage de la requête
            if conditions:
                requete = f"{base_query} WHERE {' AND '.join(conditions)}"
            else:
                requete = base_query
            
            # Ajout du tri et de la limite
            requete += " ORDER BY derniere_modification DESC"
            
            # Ajoutez la limite que si c'est une recherche floue ou si aucun numéro exact n'est fourni
            if not numero_dossier and not (search_term and re.fullmatch(r'\d{2}-\d{4}', search_term.strip())):
                if limit:
                    requete += " LIMIT %s"
                    parametres.append(limit)
            
            logger.info(f"Exécution de la requête: {requete} avec params: {parametres}")
            cursor.execute(requete, parametres)
            resultats = cursor.fetchall()
            
            logger.info(f"{len(resultats)} dossiers trouvés pour les critères.")
            return resultats
        
        except mysql.connector.Error as erreur:
            logger.error(f"Erreur de recherche de dossier: {erreur}")
            return []
        except Exception as e:
            logger.error(f"Erreur inattendue dans rechercher_dossier: {e}", exc_info=True)
            return []
        finally:
            # Fermer le curseur et la connexion
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn and conn.is_connected():
                conn.close()

# ===GESTION DES DOCUMENTS ET EMBEDDINGS ====

#Chargement des regles
def load_rules_from_json(rules_path: str) -> List[Document]:
    """Charge les règles depuis les fichiers JSON dans un répertoire et les convertit en documents Langchain."""
    
    try:
        if not os.path.exists(rules_path):
             logger.warning(f"Chargement règles - Répertoire des règles n'existe pas: {rules_path}")
             return []
         
        # Récuperer tous les fichiers JSON dans le répertoire
        json_files = glob.glob(os.path.join(rules_path, "*.json"))
        logger.info(f"Chargement règles - Fichiers trouvés:  {json_files}")
        
        if not json_files:
            logger.warning(f"Chargement règles - Aucun fichier JSON trouvé dans{rules_path}")
            return []
        
        rules_docs = []
        
        # Traiter chaque fichier JSON
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                    
                    global_metadata = rules_data.get("global_rule_metadata", {})
                    
                    for rule in rules_data.get("rules", []):
                        content = f"""
                        [Règle n°{rule.get('rule_number', 'N/A')}] Titre: {rule.get('title', 'Sans titre')}
                        Contexte: {rule.get('context', 'Non spécifié')}
                        Action: {rule.get('action', 'Non spécifiée')}
                        """
                        
                        # Fusionner les métadonnées globales et spécifiques
                        metadata = {
                            "source": os.path.basename(json_file),
                            "category": "regles",
                            "rule_number": rule.get("rule_number", "N/A"),
                            "title": rule.get("title", "Sans titre"),
                            "tags": rule.get("metadata", {}).get("tags", []),
                            "priority": global_metadata.get("priority", 100), 
                            "type_usage": global_metadata.get("type_usage", "regle"),
                            "keywords": rule.get("metadata", {}).get("keywords", []),
                            "related_rules": rule.get("metadata", {}).get("related_rules", []),
                             "type": "rule_document"
                        }
                        
                        # Créer le document
                        doc = Document(
                            page_content=content,
                            metadata=metadata
                        )
                    
                        rules_docs.append(doc)
            except json.JSONDecodeError as je:
                    logger.error(f"Chargement règles - Erreur décodage JSON pour {json_file}: {je}")
            except Exception as e:
                logger.error(f"Chargement règles - Erreur traitement fichier {json_file}: {e}", exc_info=True)
        
        logger.info(f"Chargement règles - {len(rules_docs)} règles chargées.")
        return rules_docs
        
    except Exception as e:
        logger.error(f"Chargement règles - Erreur globale: {e}", exc_info=True)
        return []

def load_all_documents() -> Tuple[List[Document], List[Document]]:
    """Charge tous les documents des différentes sources."""
    knowledge_docs = []
    rules_docs = []
    
    # 1. Charger les documents des règles depuis des fichiers JSON dans un dossier
    try:
        rules_docs = load_rules_from_json(REGLES_PATH)
        logger.info(f"Load all docs - {len(rules_docs)} documents de règles chargés.")
    except Exception as e:
        logger.error(f"Load all docs - Erreur lors du chargement des documents de règles: {e}")
  
    # 2. Charger les documents officiels
    try:
        # S'assurer que le répertoire existe avant de charger
        if not os.path.exists(OFFICIAL_DOCS_PATH):
            logger.warning(f"Load all docs - Répertoire des docs officiels n'existe pas: {OFFICIAL_DOCS_PATH}")
        else:
            official_docs_loader = DirectoryLoader(
                OFFICIAL_DOCS_PATH,
                glob="*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            ) 
            official_docs = official_docs_loader.load()
            for doc in official_docs:
                doc.metadata["category"] = "docs_officiels"
                doc.metadata["type"] = "knowledge_document" 
            logger.info(f"Document générale - {len(official_docs)} documents de docs_officiels chargés.")
            knowledge_docs.extend(official_docs)
    except Exception as e:
        logger.error(f"Document générale - Erreur lors du chargement des documents officiels: {e}")
        

    # 3. Charger les documents des échanges
    try:
        # S'assurer que le répertoire existe avant de charger
        if not os.path.exists(ECHANGES_PATH):
            logger.warning(f"Document générale - Répertoire des échanges n'existe pas: {ECHANGES_PATH}")
        else:
            echanges_loader = DirectoryLoader(
                ECHANGES_PATH,
                glob="**/*txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                recursive=True
            )
            echanges_docs = echanges_loader.load()
            for doc in echanges_docs:
                doc.metadata["category"] = "echanges"
                doc.metadata["type"] = "knowledge_document" 
                
            logger.info(f"Document générale - {len(echanges_docs)} documents d'échanges chargés.")
            knowledge_docs.extend(echanges_docs)
    except Exception as e:
        logger.error(f"Document générale - Erreur lors du chargement des dossiers d'échanges: {e}")
    
    logger.info(f"Document générale - Total : {len(knowledge_docs)} documents de connaissances et {len(rules_docs)} règles chargés")
    return knowledge_docs, rules_docs

#Calcul du hash des documents
def calculate_documents_hash(documents: List[Document], embeddings: Embeddings) -> str:
    """Calcule un hash unique pour l'ensemble des documents et le modèle d'embedding."""
    content = str(type(embeddings).__name__)
    if hasattr(embeddings, "model"): # Pour OpenAIEmbeddings
         content += str(embeddings.model)
    elif hasattr(embeddings, "model_name"): # Pour MistralAIEmbeddings
         content += str(embeddings.model_name)
    
    # Trier les documents pour assurer la consistance du hash
    # Utiliser la page_content et les métadonnées pour le tri
    sorted_docs = sorted(documents, key=lambda x: (x.page_content, str(sorted(x.metadata.items()))))
    
    for doc in sorted_docs:
        content += doc.page_content
        if doc.metadata:
            content += str(sorted(doc.metadata.items()))
    
    return hashlib.sha256(content.encode('utf-8')).hexdigest() # Utiliser utf-8 pour l'encodage

def load_cached_embeddings_from_dir(embeddings: Embeddings, cache_dir: str, current_hash: str) -> Optional[VectorStore]:
    """Charge les embeddings depuis un répertoire de cache spécifique si le hash correspond."""
    os.makedirs(cache_dir, exist_ok=True)
    
    hash_file_path = os.path.join(cache_dir, "documents_hash.txt")
    faiss_index_path = os.path.join(cache_dir, "faiss_index")
    
    if os.path.exists(hash_file_path) and os.path.exists(faiss_index_path):
        with open(hash_file_path, "r") as f:
            cached_hash = f.read().strip()
            
        if cached_hash == current_hash:
            try:
                # IMPORTANT: allow_dangerous_deserialization est nécessaire pour charger des indices FAISS créés avec des versions plus anciennes de libraries
                vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
                logger.info(f"Cache - Index FAISS chargé depuis {cache_dir} avec succès.")
                return vector_store
            except Exception as e:
                logger.error(f"Cache - Erreur lors du chargement de l'index FAISS depuis {cache_dir}: {e}")
    
    return None

def identify_relevant_rules(question: str, rules_vector_store: VectorStore) -> List[Document]:
    """Identifie les règles pertinentes pour une question donnée en utilisant la recherche top-k."""
    try:
        if rules_vector_store is None:
            logger.warning("Rules vector store non initialisé. Impossible d'identifier les règles pertinentes.")
            return []

        # Utilisation de la recherche par similarité (Top-K)
        max_rules = 5  # On cherche les 5 règles les plus similaires
        
        relevant_rules = rules_vector_store.similarity_search(question, k=max_rules)
        
        logger.info(f"{len(relevant_rules)} règles potentiellement pertinentes identifiées (top {max_rules}).")
        
        # On s'assure que chaque document a bien la catégorie 'regles' et le type 'rule_document'
        for rule in relevant_rules:
             rule.metadata['category'] = 'regles'
             rule.metadata['type'] = 'rule_document'
             
        return relevant_rules
    except Exception as e:
        logger.error(f"Erreur lors de l'identification des règles pertinentes: {e}", exc_info=True)
        return []
    
    


def load_cached_embeddings(documents: List[Document], embeddings: Embeddings) -> Tuple[Optional[VectorStore], str]:
    """Charge les embeddings depuis le cache s'ils existent et sont valides."""
    # Créer le répertoire de cache s'il n'existe pas
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Calculer le hash des documents courants et du modèle d'embedding
    current_hash = calculate_documents_hash(documents, embeddings)
    hash_file_path = os.path.join(CACHE_DIR, "documents_hash.txt")
    faiss_index_path = os.path.join(CACHE_DIR, "faiss_index")
    
    # Vérifier si le cache est valide
    if os.path.exists(hash_file_path) and os.path.exists(faiss_index_path):
        with open(hash_file_path, "r") as f:
            cached_hash = f.read().strip()
            
        # Si le hash correspond, charger l'index FAISS
        if cached_hash == current_hash:
            try:
                vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
                logger.info("Index FAISS chargé depuis le cache avec succès")
                return vector_store, current_hash
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'index FAISS: {e}")
    
    return None, current_hash

def save_embeddings_cache_to_dir(vector_store: VectorStore, documents_hash: str, cache_dir: str):
    """Sauvegarde les embeddings et le hash des documents dans un répertoire de cache spécifique."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Sauvegarder le hash des documents
    hash_file_path = os.path.join(cache_dir, "documents_hash.txt")
    with open(hash_file_path, "w") as f:
        f.write(documents_hash)
    
    # Sauvegarder l'index FAISS
    faiss_index_path = os.path.join(cache_dir, "faiss_index")
    vector_store.save_local(faiss_index_path)
    logger.info(f"Cache - Embeddings sauvegardés dans {cache_dir}.")



def create_vector_store(documents: List[Document], embeddings: Embeddings, cache_dir: str) -> Optional[VectorStore]:
    """Crée un vectorstore optimisé avec cache pour les embeddings dans un répertoire spécifique."""
    
    # Si la liste de documents est vide, on ne peut pas créer de vector store
    if not documents:
        logger.warning(f"Cache - Aucuns documents fournis pour créer le vector store dans {cache_dir}.")
        return None
        
    current_hash = calculate_documents_hash(documents, embeddings)
    
    # Essayer de charger depuis le cache
    vector_store = load_cached_embeddings_from_dir(embeddings, cache_dir, current_hash)
    
    # Si le cache est valide, utiliser le vectorstore chargé
    if vector_store is not None:
        logger.info(f"Cache - Utilisation des embeddings en cache depuis {cache_dir}.")
        return vector_store
    
    # Sinon, créer un nouveau vectorstore
    logger.info(f"Cache - Calcul des nouveaux embeddings pour {cache_dir}...")
    try:
        vector_store = FAISS.from_documents(documents, embeddings)
        # Sauvegarder les embeddings et le hash des documents dans le cache
        save_embeddings_cache_to_dir(vector_store, current_hash, cache_dir)
        return vector_store
    except Exception as e:
        logger.error(f"Cache - Erreur lors de la création de l'index FAISS pour {cache_dir}: {e}", exc_info=True)
        return None

# ================= EXTRACTION ET TRAITEMENT DES DONNÉES =================

def extract_dossier_number(question: str) -> List[str]:
    """Extrait les numéros de dossier de la question."""
    patterns = [
        r'\b\d{2}-\d{4}\b',  # Format 82-2069
        r'\bdossier\s+(?:n°|numéro|numero|n|°|)?\s*(\w+-\w+)\b',  # Format avec "dossier n° XXX-XXX"
        r'\bdossier\s+(?:n°|numéro|numero|n|°|)?\s*(\d+)\b'  # Format avec "dossier n° XXX"
    ]
    
    results = []
    for pattern in patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        if matches:
            results.extend(matches)
    
    return list(set(results))  # Éliminer les doublons

def db_resultats_to_documents(resultats: List[Dict[str, Any]]) -> List[Document]:
    """Convertit les résultats de la base de données en documents Langchain."""
    documents = [] 
    for resultat in resultats:
        
        # Formater les dates si elles sont des objets date/datetime
        date_creation_str = str(resultat.get('date_creation', 'N/A')) if resultat.get('date_creation') else 'N/A'
        derniere_modification_str = str(resultat.get('derniere_modification', 'N/A')) if resultat.get('derniere_modification') else 'N/A'

        # Contenu formaté à partir des données du dossier
        content = f"""
        - Informations sur le dossier {resultat.get('Numero', 'N/A')}:
        - Nom de l'usager: {resultat.get('nom_usager', 'N/A')}
        - Date de création: {date_creation_str}
        - Dernière modification: {derniere_modification_str}
        - Agent affecté: {resultat.get('agent_affecter', 'N/A')}
        - Instructeur: {resultat.get('instructeur', 'N/A')}
        - Statut actuel: {resultat.get('statut', 'N/A')}
        - Statut visible par l'usager: {resultat.get('statut_visible_usager', 'N/A')}
        - Montant: {resultat.get('montant', 'N/A')} €
        """
        
        # Création d'un document Langchain avec les métadonnées
        doc = Document(
            page_content=content.strip(), # Enlever les espaces au début/fin
            metadata={
                "source": "base_de_donnees",
                "type": "dossier",
                "numéro": resultat.get('Numero', 'N/A'),
                "section": "Informations dossier",
                "page": "N/A", 
                "update_date": derniere_modification_str
            }
        )
        
        documents.append(doc)
        
    return documents

# ================= FONCTIONS DU GRAPHE RAG =================

def search_database(state: State) -> Dict[str, Any]:
    """Extrait les numéros de dossier de la question et cherche dans la base de données."""
    global db_connected, rechercher_dossier_func # Utiliser les globales initialisées

    logger.info("Début de l'étape de recherche dans la database.")
    
    dossier_numbers = extract_dossier_number(state["question"])
    logger.info(f"Recherche DB - Numéros de dossier extraits: {dossier_numbers}")
    
    db_results = []
    # Utiliser l'état de connexion et la fonction initialisées globalement
    if db_connected and rechercher_dossier_func and dossier_numbers:
        try:
            for num in dossier_numbers:
                # Appeler la fonction stockée globalement
                dossier_results = rechercher_dossier_func(numero_dossier=num)
                db_results.extend(dossier_results)
            logger.info(f"Recherche DB - Résultats de la base de données: {len(db_results)} entrées trouvées.")
        except Exception as e:
             logger.error(f"Recherche DB - Erreur lors de la recherche DB via fonction globale: {e}", exc_info=True)
    elif not db_connected:
        logger.warning("Recherche DB - Connexion à la base de données non établie. Recherche DB ignorée.")
    elif not dossier_numbers:
         logger.info("Recherche DB - Aucuns numéros de dossier extraits. Recherche DB ignorée.")
         
    return {"db_results": db_results}

def retrieve(state: State) -> Dict[str, Any]:
    """Récupère les documents pertinents basés sur la question, en priorisant les règles."""
    global knowledge_vector_store, rules_vector_store
    
    logger.info("Début de l'étape de Récuperation.")
    
    try:
        # Initialisation du contexte
        context_docs_from_vectors = [] # Pour stocker les documents des vector stores
        
        
        # Étape 1: Identifier les règles pertinentes d'abord
        if rules_vector_store is not None:
            relevant_rules = identify_relevant_rules(state["question"], rules_vector_store)
            
            # Ajouter les règles au contexte provenant des vectorstores
            context_docs_from_vectors.extend(relevant_rules)
            
            logger.info(f"Récuperation - {len(relevant_rules)} règles pertinentes identifiées et ajoutées au contexte Vector Store.")
        else:
            logger.warning("Récuperation - Le vector store des règles n'est pas initialisé")
        
        # Étape 2: Récupérer les documents généraux pertinents (docs officiels, échanges)
        if knowledge_vector_store is not None:
            retrieved_knowledge_docs = knowledge_vector_store.similarity_search(state["question"], k=7)
            
            # Ajouter les documents de connaissance au contexte provenant des vector stores
            context_docs_from_vectors.extend(retrieved_knowledge_docs)
            
            logger.info(f"Récuperation - {len(retrieved_knowledge_docs)} documents de connaissances récupérés et ajoutés au contexte Vector Store.")
        else:
            logger.error("Récuperation - Le vector store de connaissances n'est pas initialisé")
            
        # Étape 3: Convertir les résultats de la base de données en documents
        db_docs = db_resultats_to_documents(state["db_results"])
        
        logger.info(f"Récuperation - {len(db_docs)} documents créés à partir des résultats de la base de données.")
        
        # Étape 4: Combiner les documents, en plaçant les règles et données BDD en tête dans la liste du contexte passée à generate.
        combined_docs = db_docs + context_docs_from_vectors
        
        # S'assurer que tous les documents ont un type et une catégorie pour la génération
        for doc in combined_docs:
            if "type" not in doc.metadata:
                 doc.metadata["type"] = "Document inconnue" 
            if "category" not in doc.metadata:
                 doc.metadata["category"] = "Categorie inconnue"
                 
        logger.info(f"Récuperation - Total de {len(combined_docs)} documents dans le contexte combiné.")
        
        return {"context": combined_docs}
    except Exception as e:
        logger.error(f"Récuperation - Erreur dans la fonction retrieve: {e}", exc_info=True)
        
        #On Retourne un contexte vide en cas d'erreur pour ne pas bloquer le graphe,
        # la fonction generate gérera le cas du contexte vide.
        return {"context": []}

def generate(state: State) -> Dict[str, Any]:
    """Génère une réponse basée sur la question et le contexte, avec un prompt amélioré."""
    global llm
    
    logger.info("Début de l'étape la géneration de réponse .")
    try:
        # Étape 0: Vérifier si le modèle LLM est initialisé
        if llm is None:
            logger.error("Le modèle LLM n'est pas initialisé")
            return {"answer": "Le système n'est pas correctement initialisé (LLM manquant). Veuillez contacter l'administrateur."}
        
        # Étape 1: Séparer les documents par type pour la structure du prompt
        rules_docs = []
        db_docs = []
        knowledge_docs = []
        docs_details = []
        
        
        for doc in state["context"]:
            doc_type = doc.metadata.get("type")
            category = doc.metadata.get("category", "non classifié")
            source = doc.metadata.get("source", "Source inconnue")
            
            # Classification des documents
            if doc_type == "rule_document" or category == "regles":
                    rules_docs.append(doc)
            elif doc_type == "db_data" or source == "base_de_donnees":
                    db_docs.append(doc)
            else: # docs_officiels, echanges, ou non classifié
                    knowledge_docs.append(doc)
                    
            # Préparation des détails pour le formatage des sources
            if source == "base_de_donnees":
                file_name = f"Base de données - Dossier {doc.metadata.get('numéro', 'inconnu')}"
                section = doc.metadata.get("section", "Section non spécifiée")
                page = "N/A"
                update_date = doc.metadata.get("update_date", "Date non disponible")
            else:
                file_name = os.path.basename(source)
                section = doc.metadata.get("section", "Section non spécifiée")
                page = doc.metadata.get("page", "Page non spécifiée")
                update_date = doc.metadata.get("update_date", "Date non disponible")

            docs_details.append({
                "content": doc.page_content,
                "file_name": file_name,
                "section": section,
                "page": page,
                "update_date": update_date,
                "category": category,
                "type": doc_type # type pour reference
            })
        
        # Étape 2: Formater les règles spécifiques (si présentes)
        rules_content = ""
        if rules_docs:
            rules_content = "RÈGLES APPLICABLES À CETTE SITUATION:\n"
            for i, rule in enumerate(rules_docs, 1):
                # Utiliser directement le page_content qui est déjà formaté par load_rules_from_json
                rules_content += f"{rule.page_content.strip()}\n\n"
                title = rule.metadata.get("title", "Sans titre")
                #rules_content += f"RÈGLE {rule_num}: {title}\n{rule.page_content}\n\n"
                logger.info(f"generation - Ajout de {len(rules_docs)} règles au prompt.")
        
        # Étape 3: Formater les données de la base de données
        db_content = ""
        if db_docs:
            db_content = "INFORMATIONS DU DOSSIER DANS LA BASE DE DONNÉES:\n"
            for doc in db_docs:
                db_content += f"{doc.page_content.strip}\n\n"
                logger.info(f"Génération - Ajout de {len(db_docs)} documents BDD au prompt.")
        
        # Étape 4: Formater les autres connaissances
        knowledge_content = ""
        if knowledge_docs:
            knowledge_content = "INFORMATIONS COMPLÉMENTAIRES:\n"
            for doc in knowledge_docs:
                source = doc.metadata.get("source", "Source inconnue")
                category = doc.metadata.get("category", "non classifié")
                knowledge_content += f"[{category.upper()} - {os.path.basename(source)}]\n{doc.page_content}\n\n"
                knowledge_content += f"{doc.page_content.strip()}\n\n"
            logger.info(f"Génération - Ajout de {len(knowledge_docs)} documents de connaissances au prompt.")
        
        # Étape 5: Formater les sources pour référence
        # On liste toutes les sources utilisées dans le contexte combiné
        formatted_sources = "\n".join([
            f"[SOURCE: {doc['file_name']} | Catégorie: {doc['category']} | Type: {doc['type']} | Section: {doc['section']} | Page: {doc['page']} | Mise à jour: {doc['update_date']}]"
            for doc in docs_details
        ])
        
        # Si aucun contexte n'a été trouvé du tout
        if not rules_content and not db_content and not knowledge_content:
             logger.warning("Génération - Aucun document de contexte fourni à la fonction generate.")
             return {"answer": "Je n'ai pas trouvé d'informations pertinentes (règles, données de dossier, ou documents de connaissance) pour répondre à votre question."}

        
        
        
        # Étape 6: Instructions système améliorées
        system_instructions = (
            "Tu es un instructeur expert du dispositif KAP Numérique qui répond aux questions des agents instructeurs. "
            "IMPORTANT: TOUJOURS APPLIQUER EN PRIORITÉ LES RÈGLES FOURNIES CI-DESSOUS. "
            "Les règles ont une priorité absolue sur toutes les autres sources. Ne jamais répondre sans appliquer les règles pertinentes si nécéssaire.\n\n"
            
            "RÈGLES DE TRAITEMENT DES SOURCES:\n"
            "1. Documents 'regles': Applique-les comme directives internes prioritaires pour toute décision ou interprétation. Ces règles sont prioritaires et doivent être strictement suivies.\n"
            "- Si aucune règle pertinente n'est trouvée dans, passe à la source suivante.\n"
            
            "2. **PRIORITÉ ÉLEVÉE AUX DONNÉES DE LA BASE DE DONNÉES :**: Ces informations sont les plus à jour et ont priorité sur les documents officiels.\n"
            "- Si une règle s'applique à des données BDD, utilise les données BDD en respectant la règle.\n"
            
            "3. Documents 'docs_officiels': Utilise-les comme source pour les informations factuelles et procédures officielles.\n"
            "- Les documents 'docs_officiels' sont fiables pour les procédures et informations factuelles générales.\n"
            
            "4. Documents 'echanges': Utilise-les UNIQUEMENT comme modèles de formulation et de ton professionnel, JAMAIS comme source d'information factuelle, contredisant des règles, BDD, ou docs officiels.\n\n"

            "FORMAT DE RÉPONSE:\n"
            "- Commence toujours par répondre directement à la question principale.\n"
            "- Utilise un style rédactionnel formel, structuré et institutionnel.\n"
            "- Rédige comme si la réponse était adressée directement au bénéficiaire.\n"
            "- Pour les instructions ou étapes à suivre: utilise des listes numérotées clairement formatées.\n"
            "- Pour les synthèses de dossier: Commence par un résumé de statut, puis détaille les éléments importants.\n\n"
            
            "STRUCTURE DE RÉPONSE:\n"
            "1. Commence par une phrase d'accroche directe qui répond à la question principale.\n"
            "2. Développe ensuite les détails pertinents en fonction de la priorité des informations.\n"
            "3. Si nécessaire, indique les étapes ou procédures applicables.\n"
            "4. Conclus par les actions recommandées ou les délais à respecter.\n"
            
            "Si tu ne trouves pas d'information pertinente dans le contexte fourni, indique clairement: \"Je ne dispose pas des informations nécessaires pour répondre précisément à cette question.\"\n"
            
            "CONSIGNES DE RÉDACTION:\n"
            "- Adopte systématiquement un ton professionnel et institutionnel.\n"
            "- Évite les formulations trop familières ou personnelles.\n"
            "- Sois précis et factuel, sans ambiguïté.\n"
            "- Respecte le vocabulaire technique spécifique au dispositif KAP Numérique.\n"
            "- N'invente JAMAIS d'informations qui ne seraient pas présentes dans les sources.\n\n"
            
            "--- CONSIGNES À NE JAMAIS FAIRE --- \n"
            "- N'invite JAMAIS à contacter un agent puisque c'est déjà un agent qui te consulte.\n"
            "- NE JAMAIS inventer d'informations. Si l'information n'est pas dans les sources, indique-le clairement.\n"
            "- Ne mentionne jamais que tu es une IA ou un chatbot dans ta réponse au bénéficiaire.\n"
            
            "Analyse la question, consulte le contexte en respectant scrupuleusement l'ordre de priorité indiqué par les balises, et génère la réponse la plus précise et conforme possible pour le bénéficiaire."
           
        )

        
        # Étape 7: Construire l'invite utilisateur structurée
        user_prompt = (
            f"QUESTION DE L'AGENT INSTRUCTEUR: {state['question']}\n\n"
            f"{rules_content}\n"
            "Veuillez appliquer les règles ci-dessus en priorité pour répondre à la question.\n\n"
            f"{db_content}\n"
           
            f"{knowledge_content}\n\n"
            "Rédige une réponse professionnelle que l'agent pourra transmettre directement au bénéficiaire.\n\n"
            f"SOURCES DE RÉFÉRENCE:\n{formatted_sources}"
        )
        
        logger.info(f"Génération - Prompt utilisateur construit. Longueur: {len(user_prompt)} caractères.")
         
        # Étape 8: Appeler le modèle LLM
        try:
            messages = [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt}
            ]
            
            # Configuration pour l'appel, ajuster temperature si besoin (0.0 pour plus de détermisme, >0 pour créativité)
            # max_tokens peut être utile si les réponses sont trop longues ou coupées.
            
            response = llm.invoke(messages,temperature=0.1, max_tokens=1500)
            
            logger.info("Génération - Appel LLM réussi.")
            
            return {"answer": response.content}
        
        except Exception as e:
            logger.error(f"Génération - Erreur lors de l'appel au LLM: {e}")
            return {"answer": f"Erreur lors de la génération de la réponse : {e}"}
        
    except Exception as e:
        logger.error(f"Génération - Erreur dans la fonction generate: {e}")
        return {"answer": f"Une  erreur s'est produite lors du traitement de votre demande: {e}"}


# ================= CREATION DU GRAPHE RAG =================

def build_graph():
    """Construit et retourne le graphe LangGraph pour le RAG."""
    
    logger.info("Construction du graphe RAG.")
    try:
        # Définir le graphe
        workflow = StateGraph(State)
        
        # Ajouter les nœuds
        workflow.add_node("search_database", search_database)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        
        # Définir les transitions
        workflow.add_edge(START, "search_database")
        workflow.add_edge("search_database", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Compiler le graphe
        graph = workflow.compile()
        
        logger.info("Graphe RAG construit avec succès.")
        
        return graph
        
    except Exception as e:
        logger.error(f"Erreur lors de la construction du graphe: {e}",exc_info=True)
        raise

# ================= INITIALISATION DU SYSTÈME RAG =================

def init_rag_system():
    """Initialise le système RAG complet avec deux vector stores séparés."""
    global knowledge_vector_store, rules_vector_store, llm, embeddings, db_connected, rechercher_dossier_func
    
    logger.info("Début début del'initialisation du système RAG...")
    
    # 1. Initialiser les embeddings
    try:
        # 2. Initialiser les embeddings  
        embeddings = MistralAIEmbeddings()
        # Utiliser OpenAIEmbeddings comme spécifié par l'utilisateur
        #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        logger.info(f"Embeddings initialisés avec succées")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des embeddings: {e}", exc_info=True)
        # Continuer l'initialisation mais noter l'erreur
        embeddings = None # S'assurer que embeddings est None si l'initialisation échoue
        
        
     # 2. Initialiser le modèle LLM
    try:
        # 1. Initialiser le modèle LLM avec Mistral 
        llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
        # Utiliser GPT-4o-mini comme spécifié par l'utilisateur
        #llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.1) # temperature basse pour un comportement plus déterministe
        logger.info(f"Modèle LLM initialisé avec succées")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du LLM: {e}", exc_info=True)
        llm = None # S'assurer que llm est None si l'initialisation échoue
        
     # Vérifier si embeddings ou llm ont échoué
    if embeddings is None or llm is None:
         logger.critical("Initialisation échouée : Embeddings ou LLM n'ont pas pu être initialisés.")
         # On peut choisir de quitter ici ou de retourner des objets None et gérer les erreurs plus tard.
         # Pour ce cas, on va retourner et les étapes suivantes échoueront si elles dépendent de ces objets.
         return {
            "knowledge_docs": [],
            "rules_docs": [],
            "knowledge_vector_store": None,
            "rules_vector_store": None,
            "llm": llm,
            "graph": None, # Le graphe ne peut pas être construit sans LLM/Embeddings
            "db_connected": False,
            "rechercher_dossier": None
         }
    
    # 3. Charger les documents, avec séparation des règles
    knowledge_docs, rules_docs = load_all_documents()
    
    # 4. Découper les documents
    # S'assurer qu'il y a des documents avant de découper
    knowledge_splits = []
    if knowledge_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        knowledge_splits = text_splitter.split_documents(knowledge_docs)
        logger.info(f"{len(knowledge_splits)} chunks créés à partir des documents de connaissances.")
    else:
         logger.warning("Aucun document de connaissance chargé à découper.")
    
    rules_splits = []
    if rules_docs:
        # Pour les règles, on utilise un chunk_size plus petit pour préserver l'intégrité des règles
        rules_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        rules_splits = rules_splitter.split_documents(rules_docs)
        logger.info(f"{len(rules_splits)} chunks créés à partir des documents de règles.")
    else:
         logger.warning("Aucun document de règles chargé à découper.")
    
    # 5. Créer les vector stores séparés
    knowledge_store_cache_dir = os.path.join(CACHE_DIR, "knowledge_store")
    rules_store_cache_dir = os.path.join(CACHE_DIR, "rules_store")

    # Créer le vector store de connaissances seulement s'il y a des splits
    knowledge_vector_store = create_vector_store(knowledge_splits, embeddings, knowledge_store_cache_dir)
    if knowledge_vector_store:
         logger.info("Vector store pour les connaissances créé/chargé.")
    else:
         logger.warning("Vector store pour les connaissances non créé/chargé.")

    
    # Créer le vector store de règles seulement s'il y a des splits
    rules_vector_store = create_vector_store(rules_splits, embeddings, rules_store_cache_dir)
    if rules_vector_store:
        logger.info("Vector store pour les règles créé/chargé.")
    else:
        logger.warning("Vector store pour les règles non créé/chargé.")
    
    # 6. Tester la connexion à la base de données
    try:
        db_manager = DatabaseManager()
        db_connected = db_manager.tester_connexion()
        if db_connected:
            # Stocker la méthode de recherche si la connexion réussit
            rechercher_dossier_func = db_manager.rechercher_dossier
        else:
            rechercher_dossier_func = None
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du DatabaseManager: {e}", exc_info=True)
        db_connected = False
        rechercher_dossier_func = None
    
   # 7. Construire le graphe (peut échouer si LLM est None)
    graph = None
    if llm: # Construire le graphe uniquement si l'LLM a été initialisé avec succès
        try:
            graph = build_graph()
            logger.info("Graphe RAG construit.")
        except Exception as e:
            logger.error(f"Erreur critique lors de la construction du graphe: {e}", exc_info=True)
            graph = None
    else:
        logger.critical("Graphe RAG non construit car le LLM n'a pas pu être initialisé.")

    logger.info("Initialisation du système RAG terminée.")

    # Retourner les objets initialisés
    return {
        "knowledge_docs": knowledge_docs, # Peut être vide si chargement échoue
        "rules_docs": rules_docs, # Peut être vide si chargement échoue
        "knowledge_vector_store": knowledge_vector_store, # Peut être None si création/chargement échoue
        "rules_vector_store": rules_vector_store, # Peut être None si création/chargement échoue
        "llm": llm, # Peut être None si initialisation échoue
        "graph": graph, # Peut être None si construction échoue
        "db_connected": db_connected,
        "rechercher_dossier": rechercher_dossier_func # Peut être None si DB non connectée
    }





# Initialisation du graphe
graph = build_graph()
rag_system_state = init_rag_system()

# ================= POINT D'ENTRÉE DU MODULE =================

if __name__ == "__main__":
    print("Initialisation du système RAG...")
    try:
        rag_system_state = init_rag_system()
        
        # Récupérer les composants essentiels de l'état initialisé
        graph = rag_system_state.get("graph")
        db_connected = rag_system_state.get("db_connected", False)
        # rechercher_dossier_func est stocké globalement lors de l'initialisation de DatabaseManager dans search_database
        # llm, knowledge_vector_store, rules_vector_store sont stockés globalement et utilisés par retrieve/generate

        # Afficher les messages d'état du système
        if rag_system_state.get("llm") is None:
             print("\n-----------------------------------------------------")
             print("ERREUR : Le modèle LLM n'a pas pu être initialisé.")
             print("Vérifiez OPENAI_API_KEY dans votre .env et votre connexion internet.")
             print("-----------------------------------------------------")

        if graph is None:
            print("\n-----------------------------------------------------")
            print("ERREUR CRITIQUE : Le graphe RAG n'a pas pu être construit.")
            print("Cela peut être dû à une erreur d'initialisation de LLM ou à une erreur dans la définition du graphe.")
            print("Le programme va s'arrêter.")
            print("-----------------------------------------------------")
            exit() # Quitter si le graphe n'est pas initialisé

        if not db_connected:
            print("\n-----------------------------------------------------")
            print("AVERTISSEMENT : Impossible de se connecter à la base de données.")
            print("Les recherches de dossiers ne fonctionneront pas.")
            print("Vérifiez vos variables SQL dans le fichier .env et l'état du serveur DB.")
            print("-----------------------------------------------------")
            
        if rag_system_state.get("knowledge_vector_store") is None:
             print("\n-----------------------------------------------------")
             print("AVERTISSEMENT : Le vector store de connaissances n'a pas été créé/chargé.")
             print("Aucune information générale ne sera utilisée (docs officiels, échanges).")
             print("Vérifiez vos chemins DATA_ROOT et les fichiers dans les répertoires.")
             print("-----------------------------------------------------")

        if rag_system_state.get("rules_vector_store") is None:
             print("\n-----------------------------------------------------")
             print("AVERTISSEMENT : Le vector store des règles n'a pas été créé/chargé.")
             print("Aucune règle ne sera utilisée.")
             print("Vérifiez vos chemins DATA_ROOT et les fichiers JSON dans le répertoire des règles.")
             print("-----------------------------------------------------")

        print("\nSystème RAG prêt.")
        print("Tapez 'exit' pour quitter.")

        # Boucle principale
        while True:
            try:
                user_query = input("\nPosez votre question : ")
                if user_query.lower() in ["exit", "quit", "quitter"]:
                    break
                    
                # Initialiser l'état pour cette question
                initial_state = {
                    "question": user_query,
                    "context": [],
                    "db_results": [],
                    "answer": ""
                }
                
                # Invoquer le graphe
                print("Traitement en cours...")
                try:
                    # Le graphe utilise les variables globales qui ont été peuplées par init_rag_system
                    # Pas besoin de passer knowledge_vector_store, rules_vector_store, llm car ils sont globaux
                    final_state = graph.invoke(initial_state)
                    print("\nRéponse :", final_state.get("answer", "Aucune réponse générée ou une erreur est survenue."))
                    
                    # Optionnel: Afficher les sources utilisées pour debug
                    # print("\n--- Sources ---")
                    # if final_state.get("context"):
                    #     for doc in final_state["context"]:
                    #          source = doc.metadata.get("source", "Inconnue")
                    #          category = doc.metadata.get("category", "N/A")
                    #          doc_type = doc.metadata.get("type", "N/A")
                    #          rule_num = doc.metadata.get("rule_number", "")
                    #          title = doc.metadata.get("title", "")
                    #          print(f"- [Cat: {category}, Type: {doc_type}, Source: {source}{f' Rule: {rule_num} ({title})' if rule_num != 'N/A' else ''}]")
                    # else:
                    #     print("Aucune source utilisée.")
                    # print("---------------")

                except Exception as e:
                    logger.error(f"Erreur lors de l'exécution du graphe: {e}", exc_info=True)
                    print(f"\nUne erreur est survenue lors du traitement de votre question par le graphe: {e}")
                    
            except KeyboardInterrupt:
                print("\nProgramme interrompu par l'utilisateur.")
                break
            except Exception as e:
                logger.error(f"\nErreur inattendue dans la boucle principale: {e}", exc_info=True)
                print(f"\nUne erreur inattendue s'est produite: {e}")

    except Exception as e:
        logger.critical(f"Erreur fatale lors de l'initialisation principale du système RAG: {e}", exc_info=True)
        print(f"\nUne erreur fatale est survenue au démarrage du système RAG: {e}")