import os 
import re
import hashlib
import mysql.connector  
import logging
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
from openai import OpenAI

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_module")

# Chargement des variables d'environnement
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Chemins vers les répertoires de données
DATA_ROOT = os.getenv("DATA_ROOT", "../data")
ECHANGES_PATH = os.path.join(DATA_ROOT, "echanges")
REGLES_PATH = os.path.join(DATA_ROOT, "regles") 
OFFICIAL_DOCS_PATH = os.path.join(DATA_ROOT, "docs_officiels")
CACHE_DIR = "./cache"  # Répertoire pour le cache

# Variables globales
vector_store = None
llm = None
embeddings = None

# Définition de l'état du système
class State(Dict):
    """Structure pour représenter l'état du système RAG."""
    question: str
    context: List[Document]
    db_results: List[Dict[str, Any]]
    answer: str

# ================= GESTION DE LA BASE DE DONNÉES =================

class DatabaseManager:
    """Gestionnaire de connexion et d'interrogation de la base de données."""
    
    def __init__(self):
        """Initialise la connexion à la base de données."""
        load_dotenv()
        self.config = {
            'user': os.getenv('SQL_USER'),
            'password': os.getenv('SQL_PASSWORD', ''),
            'host': os.getenv('SQL_HOST', 'localhost'),
            'database': os.getenv('SQL_DB'),
            'port': int(os.getenv('SQL_PORT', '3306'))
        }
        
        # Vérification des variables essentielles
        if not all([self.config['user'], self.config['host'], self.config['database']]):
            raise ValueError("Variables essentielles manquantes : SQL_USER, SQL_HOST ou SQL_DB.")
    
    def tester_connexion(self) -> bool:
        """Teste la connexion à la base de données."""
        try:
            conn = mysql.connector.connect(**self.config)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            logger.info("Connexion réussie à la base de données.")
            return True
        except mysql.connector.Error as erreur:
            logger.error(f"Échec de la connexion : {erreur}")
            return False
    
    
    def rechercher_dossier(self,
                      search_term: Optional[str] = None,
                      numero_dossier: Optional[str] = None,  # Pour la rétrocompatibilité
                      statut: Optional[str] = None,
                      instructeur: Optional[str] = None,
                      date_debut_creation: Optional[date] = None,
                      date_fin_creation: Optional[date] = None,
                      limit: Optional[int] = 50,
                      **kwargs) -> List[Dict[str, Any]]:
        """
        Recherche des dossiers dans la base de données avec priorité au numéro exact.
        
        Args:
            search_term: Terme pour recherche (floue ou exacte selon le format).
            numero_dossier: Recherche exacte par numéro (prioritaire sur search_term).
            statut: Filtre sur le statut.
            instructeur: Filtre sur l'instructeur.
            date_debut_creation: Date de début pour filtrer sur date_creation.
            date_fin_creation: Date de fin pour filtrer sur date_creation.
            limit: Nombre maximum de résultats à retourner.
            **kwargs: Critères supplémentaires pour rétrocompatibilité.
            
        Returns:
            Liste des dossiers correspondants aux critères.
        """
        try:
            conn = mysql.connector.connect(**self.config)
            cursor = conn.cursor(dictionary=True)
            
            # Construction de la requête SQL
            base_query = "SELECT * FROM dossiers"
            conditions = []
            parametres = []
            
            # *** PRIORITÉ À LA RECHERCHE PAR NUMÉRO EXACT ***
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
            
            # N'ajoutez la limite que si c'est une recherche floue ou si aucun numéro exact n'est fourni
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
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn and conn.is_connected():
                conn.close()

# ================= GESTION DES DOCUMENTS ET EMBEDDINGS =================

def load_all_documents() -> List[Document]:
    """Charge tous les documents des différentes sources."""
    all_docs = []
    
    # 1. Charger les documents des échanges
    try:
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
        logger.info(f"{len(echanges_docs)} documents d'échanges chargés.")
        all_docs.extend(echanges_docs)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des dossiers d'échanges: {e}")
    
    # 2. Charger les documents des règles
    try:
        regles_loader = DirectoryLoader(
            REGLES_PATH,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        regles_docs = regles_loader.load()
        for doc in regles_docs:
            doc.metadata["category"] = "regles"
        logger.info(f"{len(regles_docs)} documents de règles chargés.")
        all_docs.extend(regles_docs)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des documents de règles: {e}")
    
    # 3. Charger les documents officiels
    try:
        official_docs_loader = DirectoryLoader(
            OFFICIAL_DOCS_PATH,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        ) 
        official_docs = official_docs_loader.load()
        for doc in official_docs:
            doc.metadata["category"] = "docs_officiels"
        logger.info(f"{len(official_docs)} documents de docs_officiels chargés.")
        all_docs.extend(official_docs)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des documents officiels: {e}")
    
    logger.info(f"Total : {len(all_docs)} documents chargés dans la base de connaissances")
    return all_docs

def calculate_documents_hash(documents: List[Document], embeddings: Embeddings) -> str:
    """Calcule un hash unique pour l'ensemble des documents et le modèle d'embedding."""
    content = str(type(embeddings).__name__)  # Ajouter le type d'embedding
    
    # Pour OpenAI, inclure aussi le nom du modèle
    if hasattr(embeddings, "model"):
        content += embeddings.model
    
    # Trier les documents pour assurer la consistance du hash
    for doc in sorted(documents, key=lambda x: x.page_content):
        content += doc.page_content
        if doc.metadata:
            content += str(sorted(doc.metadata.items()))
    
    return hashlib.sha256(content.encode()).hexdigest()

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

def save_embeddings_cache(vector_store: VectorStore, documents_hash: str):
    """Sauvegarde les embeddings et le hash des documents dans le cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Sauvegarder le hash des documents
    hash_file_path = os.path.join(CACHE_DIR, "documents_hash.txt")
    with open(hash_file_path, "w") as f:
        f.write(documents_hash)
    
    # Sauvegarder l'index FAISS
    faiss_index_path = os.path.join(CACHE_DIR, "faiss_index")
    vector_store.save_local(faiss_index_path)
    logger.info("Embeddings sauvegardés dans le cache.")

def create_vector_store(documents: List[Document], embeddings: Embeddings) -> VectorStore:
    """Crée un vectorstore optimisé avec cache pour les embeddings."""
    # Essayer de charger depuis le cache
    vector_store, documents_hash = load_cached_embeddings(documents, embeddings)
    
    # Si le cache est valide, utiliser le vectorstore chargé
    if vector_store is not None:
        logger.info("Utilisation des embeddings en cache.")
        return vector_store
    
    # Sinon, créer un nouveau vectorstore
    logger.info("Calcul des nouveaux embeddings...")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Sauvegarder les embeddings et le hash des documents dans le cache
    save_embeddings_cache(vector_store, documents_hash)
    
    return vector_store

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
        # Contenu formaté à partir des données du dossier
        content = f"""
        - Informations sur le dossier {resultat.get('Numero', 'N/A')}:
        - Nom de l'usager: {resultat.get('nom_usager', 'N/A')}
        - Date de création: {resultat.get('date_creation', 'N/A')}
        - Dernière modification: {resultat.get('derniere_modification', 'N/A')}
        - Agent affecté: {resultat.get('agent_affecter', 'N/A')}
        - Instructeur: {resultat.get('instructeur', 'N/A')}
        - Statut actuel: {resultat.get('statut', 'N/A')}
        - Statut visible par l'usager: {resultat.get('statut_visible_usager', 'N/A')}
        - Montant: {resultat.get('montant', 'N/A')} €
        """
        
        # Création d'un document Langchain avec les métadonnées
        doc = Document(
            page_content=content,
            metadata={
                "source": "base_de_donnees",
                "type": "dossier",
                "numéro": resultat.get('Numero', 'N/A'),
                "section": "Informations dossier",
                "page": "1", 
                "update_date": resultat.get('derniere_modification', 'N/A')
            }
        )
        
        documents.append(doc)
        
    return documents

# ================= FONCTIONS DU GRAPHE RAG =================

def search_database(state: State) -> Dict[str, Any]:
    """Extrait les numéros de dossier de la question et cherche dans la base de données."""
    # Étape 1: Extraire les numéros de dossier de la question
    dossier_numbers = extract_dossier_number(state["question"])
    logger.info(f"Numéros de dossier extraits: {dossier_numbers}")
    
    # Étape 2: Rechercher dans la base de données si des numéros sont trouvés
    db_results = []
    if dossier_numbers:
        db_manager = DatabaseManager()
        # Vérifier la connexion
        if db_manager.tester_connexion():
            for num in dossier_numbers:
                dossier_results = db_manager.rechercher_dossier(numero_dossier=num)
                db_results.extend(dossier_results)
            logger.info(f"Résultats de la base de données: {len(db_results)} entrées trouvées")
    
    return {"db_results": db_results}

def retrieve(state: State) -> Dict[str, Any]:
    """Récupère les documents pertinents basés sur la question."""
    global vector_store
    
    try:
        # Étape 1: Vérifier que le vectorstore est initialisé
        if vector_store is None:
            logger.error("Le vectorstore n'est pas initialisé")
            return {"context": []}
        
        # Étape 2: Récupérer les documents les plus pertinents depuis le vectorstore
        retrieved_docs = vector_store.similarity_search(state["question"], k=10)
        logger.info(f"{len(retrieved_docs)} documents récupérés du vectorstore")
        
        # Étape 3: Convertir les résultats de la base de données en documents
        db_docs = db_resultats_to_documents(state["db_results"])
        logger.info(f"{len(db_docs)} documents créés à partir des résultats de la base de données")
        
        # Étape 4: Combiner les documents
        combined_docs = db_docs + retrieved_docs
    
        return {"context": combined_docs}
    except Exception as e:
        logger.error(f"Erreur dans la fonction retrieve: {e}", exc_info=True)
        return {"context": []}

def generate(state: State) -> Dict[str, Any]:
    """Génère une réponse basée sur la question et le contexte."""
    global llm
    
    try:
        # Étape 1: Vérifier si le modèle LLM est initialisé
        if llm is None:
            logger.error("Le modèle LLM n'est pas initialisé")
            return {"answer": "Le système n'est pas correctement initialisé. Veuillez contacter l'administrateur."}
        
        # Étape 2: Vérifier si le contexte est vide
        if not state["context"]:
            return {"answer": "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question."}
        
      
        # Étape 3: Préparer les détails des documents pour le contexte
        docs_details = []
        for doc in state["context"]:  
            source = doc.metadata.get("source", "Source inconnue")
            category = doc.metadata.get("category", "non classifié")
            
            # Traitement spécial pour les documents de la base de données
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
                "category": category
            })

        # Étape 4: Agréger le contenu des documents
        docs_content = "\n\n".join(doc["content"] for doc in docs_details)

        # Étape 5: Formater les sources
        formatted_sources = "\n".join([
            f"[Document: {doc['file_name']}, Catégorie: {doc['category']}, Section: {doc['section']}, Page: {doc['page']}, Mise à jour: {doc['update_date']}]"
            for doc in docs_details
        ])
        
        # Étape 6: Définir les instructions pour le LLM
        system_instructions = (
          
            "Tu es un instructeur expert du dispositif KAP Numérique. Tu réponds aux questions en te basant UNIQUEMENT sur les informations fournies dans le contexte.\n\n"
            
            "RÈGLES DE TRAITEMENT DES SOURCES:\n"
            "1. Documents 'docs_officiels': Utilise-les comme source primaire pour les informations factuelles et procédures officielles.\n"
            "2. Documents 'regles': Applique-les comme directives internes prioritaires pour toute décision ou interprétation.\n"
            "3. Documents 'echanges': Utilise-les UNIQUEMENT comme modèles de formulation et de ton professionnel, JAMAIS comme source d'information factuelle.\n"
            "4. Informations de la base de données: Ces informations sont les plus à jour et ont priorité sur toutes les autres sources.\n\n"
            
            "FORMAT DE RÉPONSE:\n"
            "- Pour les questions techniques ou procédurales: Utilise un style rédactionnel avec des paragraphes structurés et concis.\n"
            "- Pour les instructions ou étapes à suivre: Utilise des listes numérotées clairement formatées.\n"
            "- Pour les synthèses de dossier: Commence par un résumé de statut, puis détaille les éléments importants.\n\n"
            
            "CONSIGNES DE RÉDACTION:\n"
            "- Adopte systématiquement un ton professionnel et institutionnel.\n"
            "- Évite les formulations trop familières ou personnelles.\n"
            "- Sois précis et factuel, sans ambiguïté.\n"
            "- Respecte le vocabulaire technique spécifique au dispositif KAP Numérique.\n"
            "- N'invente JAMAIS d'informations qui ne seraient pas présentes dans les sources.\n\n"
            
            "STRUCTURE DE RÉPONSE:\n"
            "1. Commence par une phrase d'accroche directe qui répond à la question principale.\n"
            "2. Développe ensuite les détails pertinents en fonction de la priorité des informations.\n"
            "3. Si nécessaire, indique les étapes ou procédures applicables.\n"
            "4. Conclus par les actions recommandées ou les délais à respecter.\n"
            "5. Cite systématiquement tes sources en fin de réponse.\n\n"
            
            "Si tu ne trouves pas d'information pertinente dans le contexte fourni, indique clairement: \"Je ne dispose pas des informations nécessaires pour répondre précisément à cette question.\"\n"
        
        )
        
        # Étape 7: Construire l'invite utilisateur
        user_prompt = f"Question: {state['question']}\n\nContexte extrait des documents et de la base de données:\n{docs_content}"
        
        # Étape 8: Préparer les messages pour le LLM
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ]
        
        # Étape 9: Appeler le modèle LLM
        try:
            response = llm.invoke(messages)
            return {"answer": response.content}
        except Exception as e:
            logger.error(f"Erreur lors de l'appel au LLM: {e}")
            return {"answer": f"Erreur lors de la génération de la réponse : {e}"}
    except Exception as e:
        logger.error(f"Erreur dans la fonction generate: {e}")
        return {"answer": f"Une erreur s'est produite lors du traitement de votre demande: {e}"}

# ================= CREATION DU GRAPHE RAG =================

def build_graph():
    """Construit et retourne le graphe LangGraph pour le RAG."""
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
        return workflow.compile()
    except Exception as e:
        logger.error(f"Erreur lors de la construction du graphe: {e}")
        raise

# ================= INITIALISATION DU SYSTÈME RAG =================

def init_rag_system():
    """Initialise le système RAG complet et retourne les composants nécessaires."""
    global vector_store, llm, embeddings
    
    logger.info("Initialisation du système RAG...")
    
    # 1. Initialiser le modèle LLM avec Mistral 
    # llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
    # 1. Initialiser le modèle LLM avec ChatGPT 
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    logger.info("Modèle LLM initialisé.")
    
    # 2. Initialiser les embeddings via mistral
    #embeddings = MistralAIEmbeddings()
    
    # 2. Initialiser les embeddings via chatgpt
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    logger.info("Embeddings initialisés.")
    
    # 3. Charger les documents
    docs = load_all_documents()
    
    # 4. Découper les documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    logger.info(f"{len(all_splits)} chunks créés à partir des documents.")
    
    # 5. Créer le vectorstore
    vector_store = create_vector_store(all_splits, embeddings)
    logger.info("Vectorstore créé et documents ajoutés.")
    
    # 6. Tester la connexion à la base de données
    db_manager = DatabaseManager()
    db_connected = db_manager.tester_connexion()
    
    # 7. Construire le graphe
    graph = build_graph()
    logger.info("Graphe RAG construit avec succès.")
    
    return {
        "docs": docs,
        "vector_store": vector_store,
        "llm": llm,
        "graph": graph,
        "db_connected": db_connected,
        "rechercher_dossier": db_manager.rechercher_dossier
    }

# Initialisation du graphe
graph = build_graph()

# ================= POINT D'ENTRÉE DU MODULE =================

if __name__ == "__main__":
    print("Tapez 'exit' pour quitter.")
    
    # Tester la connexion à la base de données
    try:
        db_manager = DatabaseManager()
        if not db_manager.tester_connexion():
            print("Avertissement: Impossible de se connecter à la base de données. Les requêtes sur les dossiers ne fonctionneront pas.")
    except Exception as e:
        print(f"Erreur lors de l'initialisation de la connexion à la base de données: {e}")
    
    # Boucle principale
    while True:
        try:
            user_query = input("\nPosez votre question : ")
            if user_query.lower() in ["exit", "quit"]:
                break
                
            # Initialiser l'état
            initial_state = {
                "question": user_query,
                "context": [],
                "db_results": [],
                "answer": ""
            }
            
            # Invoquer le graphe
            try:
                state = graph.invoke(initial_state)
                print("\nRéponse :", state["answer"])
            except Exception as e:
                print(f"\nErreur lors de l'exécution du graphe: {e}")
                
        except KeyboardInterrupt:
            print("\nProgramme interrompu par l'utilisateur.")
            break
        except Exception as e:
            print(f"\nErreur inattendue: {e}")