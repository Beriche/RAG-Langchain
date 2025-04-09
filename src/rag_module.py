import os 
import pymysql
import re
import hashlib
import pickle
from langgraph.graph import END, START
from typing import List, TypedDict, Dict, Any, Optional,Tuple
from mistralai import Mistral
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain import hub
from langgraph.graph import StateGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_module")

# Chargement des variables d'environnement
load_dotenv()

# Chemins vers les répertoires de données
DATA_ROOT = os.getenv("DATA_ROOT", "../data")
ECHANGES_PATH = os.path.join(DATA_ROOT, "echanges")
REGLES_PATH = os.path.join(DATA_ROOT, "regles") 
OFFICIAL_DOCS_PATH = os.path.join(DATA_ROOT, "docs_officiels")

# Initialisation des variables globales qui seront remplies lors de l'initialisation
vector_store = None
llm = None
embeddings = None


# Définition de l'état de l'application
class State(TypedDict):
    question: str
    context: List[Document]  # Liste d'objets Document
    db_results: List[Dict[str, Any]]  # Résultats de la base de données
    answer: str

class DatabaseManager:
    def __init__(self):
        load_dotenv()
        self.config = {
            'user': os.getenv('SQL_USER'),
            'password': os.getenv('SQL_PASSWORD', ''),
            'host': os.getenv('SQL_HOST', 'localhost'),
            'database': os.getenv('SQL_DB'),
            'port': int(os.getenv('SQL_PORT', '3306'))
        }
        if not all([self.config['user'], self.config['host'], self.config['database']]):
            raise ValueError("Variables essentielles manquantes : SQL_USER, SQL_HOST ou SQL_DB.")
        
    def tester_connexion(self) -> bool:
        try:
            conn = pymysql.connect(**self.config)
            curseur = conn.cursor()
            curseur.execute("SELECT 1")
            curseur.fetchone()
            curseur.close()
            conn.close()
            logger.info("Connexion réussie avec pymysql.")
            return True
        except pymysql.Error as erreur:
            logger.error(f"Échec de la connexion : {erreur}")
            return False
    
    def rechercher_dossier(self, numero_dossier: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        try:
            conn = pymysql.connect(**self.config)
            curseur = conn.cursor(pymysql.cursors.DictCursor)  # Pour des résultats sous forme de dictionnaires
            conditions = []
            parametres = []

            if numero_dossier:
                conditions.append("Numero = %s")
                parametres.append(numero_dossier)

            for cle, valeur in kwargs.items():
                if valeur is not None:
                    conditions.append(f"{cle} = %s")
                    parametres.append(valeur)

            requete = f"SELECT * FROM dossiers WHERE {' AND '.join(conditions) if conditions else '1=1'}"
            curseur.execute(requete, parametres)
            resultats = curseur.fetchall()
            curseur.close()
            conn.close()
            return resultats
        except pymysql.Error as erreur:
            logger.error(f"Erreur de recherche : {erreur}")
            return []


#Optimisation du système en stockant les embedding dans mon disque dur avec FAISS au lieu de la RAM avec inMemoryVectoreStore dans la RAM qui ralentit considérablement mon système RAG

def calculate_documents_hash(documents: List[Document]) -> str:
    """ 
    Calcul un hash unique pour l'ensemble des documents.
    
    Args:
        documents: Liste des documents à hacher
         
    Returns:
        str: Hash SHA-256 représentant l'état actuel des documents
        
    """
    
    content = ""
    
    #On trie d'abord les documents par contenu pour assurer la consistance du hash 
    for doc in sorted(documents,key=lambda x: x.page_content):
        content += doc.page_content
        #Ajouter les métadonnées au contenu à hacher
        if doc.metadata:
            content += str(sorted(doc.metadata.items()))
    
    return hashlib.sha256(content.encode()).hexdigest()

def load_cached_embeddings( documents: List[Document],embeddings:Embeddings,cache_dir: str = "./cache") -> Tuple[Optional[VectorStore],str]:
    """
    Charge les embeddings depuis le cache s'ils existent et sont valides.
    
    Args:
        documents: Liste des documents
        embeddings: Modèle d'embeddings à utiliser
        cache_dir: Répertoire où stocker le cache
        
    Returns:
        Tuple[Optional[VectorStore], str]: Le vectorstore chargé et le hash des documents, ou (None, hash) si pas de cache valide
    """
    
    # On créer le répertoire de cache s'il n'existe pas 
    os.makedirs(cache_dir, exist_ok=True)
    
    #calcule le hash des documents courants
    current_hash = calculate_documents_hash(documents)
    hash_file_path = os.path.join(cache_dir, "documents_hash.txt")
    faiss_index_path = os.path.join(cache_dir, "faiss_index")
    
    # Puis on verifie si le hash correspond , charger l'index FAISS
    
    if os.path.exists(hash_file_path) and os.path.exists(faiss_index_path):
        #Charger le hash precedent
        
        with open(hash_file_path, "r" ) as f:
            cached_hash = f.read().strip() 
            
        # si le hash correspond , charger l'index FAISS
        if cached_hash == current_hash:
            try:
                vector_store = FAISS.load_local(faiss_index_path,embeddings,  allow_dangerous_deserialization=True)
                logger.info("Index FAISS chargé depuis le cache avec succès")
                return vector_store,current_hash
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'index FAISS: {e}")
                return None,current_hash
            
    return None, current_hash

def save_embeddings_cache(vector_store: VectorStore,documents_hash:str,cache_dir:str = "./cache"):
    """
    Sauvegarde les embeddings et le hash des documents dans le cache.
    
    Args:
        vector_store: Le vectorstore à sauvegarder
        documents_hash: Le hash des documents
        cache_dir: Répertoire où stocker le cache
    """
    
    # Créer le répertoire de cache s'il n'existe pas 
    os.makedirs(cache_dir, exist_ok=True)
    
    # Sauvegarder le hash des documents
    hash_file_path = os.path.join(cache_dir, "documents_hash.txt")
    with open(hash_file_path, "w") as f:
        f.write(documents_hash)
    
    # Sauvegarder l'index FAISS
    faiss_index_path = os.path.join(cache_dir, "faiss_index")
    if hasattr(vector_store,"save_local"):
        vector_store.save_local(faiss_index_path)
    else: 
        # Fallback pour InMemoryVectorStore qui ne supporte pas save_local
        with open(os.path.join(cache_dir, "vector_store.pkl"), 'wb') as f:
            pickle.dump(vector_store, f)

def create_optimized_vector_store(documents: List[Document], embeddings: Embeddings, cache_dir: str = "./cache") -> VectorStore:
        """
            Crée un vectorstore optimisé avec cache pour les embeddings.
            
            Args:
                documents: Liste des documents à indexer
                embeddings: Modèle d'embeddings à utiliser
                cache_dir: Répertoire où stocker le cache
                
            Returns:
                VectorStore: Le vectorstore créé ou chargé depuis le cache
         """
    
        # Essayer de charger depuis le cache 
        vector_store, documents_hash = load_cached_embeddings(documents, embeddings, cache_dir)
        
        #si le cache est valide, utiliser le vectorestore chargé
        if vector_store is not None:
            print("Utilisation des embeddings en cache.")
            return vector_store
        
        #Sinon, créer un nouveau vectorstore3
        print("Calcul des nouveaux embeddings...")
           # On utilise FAISS au lieu de InMemoryVectorStore pour permettre la sauvegarde sur disque
        vector_store = FAISS.from_documents(documents, embeddings)
        
        # Sauvegarder les embeddings et le hash des documents dans le cache
        save_embeddings_cache(vector_store, documents_hash, cache_dir)
        print("Embeddings sauvegardés dans le cache.")
        
        return vector_store
           
#Initialisation de mon RAG
def init_rag_system():
    """Initialise le système RAG complet et retourne les composants nécessaires."""
    global vector_store, llm, embeddings
    
    logger.info("Initialisation du système RAG...")
    
    # Initialiser le modèle LLM
    llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
    logger.info("Modèle LLM initialisé.")
    
    
    
    # Initialiser les embeddings
    embeddings = MistralAIEmbeddings()
    logger.info("Embeddings initialisés.")
    
    # Charger les documents
    docs = load_all_documents()
    
    # Découper les documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    logger.info(f"{len(all_splits)} chunks créés à partir des documents.")
    
    # Créer le vectorstore
    vector_store = create_optimized_vector_store(all_splits, embeddings)
    logger.info("Vectorstore créé et documents ajoutés.")
    
    # Tester la connexion à la base de données
    db_manager = DatabaseManager()
    db_connected = db_manager.tester_connexion()
    
    # Construire le graphe
    graph = build_graph()
    logger.info("Graphe RAG construit avec succès.")
    
    return {
        "docs": docs,
        "vector_store": vector_store,
        "llm": llm,
        "graph": graph,
        "db_connected": db_connected,
        "rechercher_dossier": db_manager.rechercher_dossier  # va etre exloiter dans l'applicaiton app.py pour la rechereche de dossier 
    }

def extract_dossier_number(question: str) -> List[str]:
    """Extrait les numéros de dossier de la question."""
    # Pattern pour rechercher les numéros de dossier (format XX-XXXX ou similaire)
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

def search_dossier_in_db(dossier_numbers: List[str]) -> List[Dict[str, Any]]:
    """Recherche les informations d'un dossier dans la base de données."""
    try:
        db_manager = DatabaseManager()
        
        # Tester la connexion avant de procéder
        if not db_manager.tester_connexion():
            logger.warning("Impossible de se connecter à la base de données")
            return []
        
        results = []
        for num in dossier_numbers:
            dossier_results = db_manager.rechercher_dossier(numero_dossier=num)
            results.extend(dossier_results)
        
        return results
    except Exception as e:
        logger.error(f"Erreur lors de la recherche dans la base de données: {e}")
        return []

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

def load_all_documents():
    """Charge tous les documents des différentes sources."""
    all_docs = []
    
    # Charger les documents des échanges 
    try:
        echanges_loader = DirectoryLoader(
            ECHANGES_PATH,
            glob="**/*txt",  # Charger tous les .txt, y compris dans les sous-dossiers
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            recursive=True  # Charger récursivement dans les sous-dossiers
        )
        echanges_docs = echanges_loader.load()
        for doc in echanges_docs:
            doc.metadata["category"] = "echanges"
        logger.info(f"{len(echanges_docs)} documents d'échanges chargés.")
        all_docs.extend(echanges_docs)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des dossiers d'échanges: {e}")
        
    # Charger les documents des règles
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
        
    # Charger les documents des docs_officiels
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

# Fonctions pour le graphe LangGraph
def search_database(state: State) -> Dict[str, Any]:
    """Extrait les numéros de dossier de la question et cherche dans la base de données."""
    # Extraire les numéros de dossier de la question
    dossier_numbers = extract_dossier_number(state["question"])
    logger.info(f"Numéros de dossier extraits: {dossier_numbers}")
    
    # Si des numéros de dossier sont trouvés, rechercher dans la base de données
    db_results = []
    if dossier_numbers:
        db_results = search_dossier_in_db(dossier_numbers)
        logger.info(f"Résultats de la base de données: {len(db_results)} entrées trouvées")
    
    return {"db_results": db_results}

def retrieve(state: State) -> Dict[str, Any]:
    """Récupère les documents pertinents basés sur la question."""
    global vector_store
    
    try:
        # Vérifier que le vectorstore est initialisé
        if vector_store is None:
            logger.error("Le vectorstore n'est pas initialisé")
            return {"context": []}
        
        # Récupérer les documents les plus pertinents depuis le vectorstore
        retrieved_docs = vector_store.similarity_search(state["question"], k=5)
        logger.info(f"{len(retrieved_docs)} documents récupérés du vectorstore")
        
        # Convertir les résultats de la base de données en documents
        db_docs = db_resultats_to_documents(state["db_results"])
        logger.info(f"{len(db_docs)} documents créés à partir des résultats de la base de données")
        
        # Combiner les documents de la base de données et les documents du vectorstore
        combined_docs = db_docs + retrieved_docs
        
        # Prioritisation des documents liés au dossier actif si un dossier est forcé
        if state.get("force_dossier_id"):
            dossier_id = state["force_dossier_id"]
            logger.info(f"Priorisation des documents pour le dossier {dossier_id}")
            
            # Filtrer les documents pour mettre en avant ceux du dossier actif
            prioritized_docs = []
            other_docs = []
            
            for doc in combined_docs:
                # Vérifier si le document est lié au dossier actif
                is_target_dossier = (
                    doc.metadata.get("numéro") == dossier_id or 
                    doc.metadata.get("Numero") == dossier_id or
                    doc.metadata.get("dossier_id") == dossier_id or
                    ("content" in dir(doc) and dossier_id in doc.page_content)
                )
                
                if is_target_dossier:
                    # Ajouter un indicateur dans les métadonnées pour marquer ce document comme prioritaire
                    doc.metadata["is_primary_context"] = True
                    prioritized_docs.append(doc)
                else:
                    other_docs.append(doc)
            
            # Recombiner les documents avec ceux du dossier actif en premier
            combined_docs = prioritized_docs + other_docs
            
            logger.info(f"{len(prioritized_docs)} documents spécifiques au dossier {dossier_id} priorisés")
        
        # Limiter le nombre total de documents pour éviter des problèmes de contexte trop volumineux
        max_docs = 10  # Ajustez selon vos besoins
        if len(combined_docs) > max_docs:
            combined_docs = combined_docs[:max_docs]
            
        return {"context": combined_docs}
    except Exception as e:
        logger.error(f"Erreur dans la fonction retrieve: {e}", exc_info=True)
        return {"context": []}

def generate(state: State) -> Dict[str, Any]:
    """Génère une réponse basée sur la question et le contexte."""
    global llm
    
    try:
        # Vérifier si le modèle LLM est initialisé
        if llm is None:
            logger.error("Le modèle LLM n'est pas initialisé")
            return {"answer": "Le système n'est pas correctement initialisé. Veuillez contacter l'administrateur."}
        
        # Vérifier si le contexte est vide
        if not state["context"]:
            return {"answer": "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question."}
        
        # filtrer et prioriser spécifiquement le dossier actif
        if state.get("force_dossier_id"):
            # Filtrer le contexte pour ne garder que les documents liés au dossier actif
            dossier_id = state["force_dossier_id"]
            filtered_context = []
            
            # D'abord ajouter les documents qui correspondent exactement au dossier actif
            
            for doc in state["context"]:
                if doc.metadata.get("numéro") == dossier_id or (
                    "dossier_id" in doc.metadata and doc.metadata["dossier_id"] == dossier_id
                ):
                    filtered_context.append(doc)
    
            # Si aucun document spécifique n'est trouvé, utiliser tout le contexte
            if filtered_context:
                state["context"] = filtered_context
            
                        
        
        # Agrégation du contenu des documents récupérés avec détails sur les sources
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

        # Agrégation du contenu des documents
        docs_content = "\n\n".join(doc["content"] for doc in docs_details)

        # Formatage des sources avec catégorie, section, page et date
        formatted_sources = "\n".join([
            f"[Document: {doc['file_name']}, Catégorie: {doc['category']}, Section: {doc['section']}, Page: {doc['page']}, Mise à jour: {doc['update_date']}]"
            for doc in docs_details
        ])
        
        # Instructions système pour le LLM
        system_instructions = (
            
            "Tu es un instructeur expert du dispositif KAP Numérique. Tu réponds à des questions en te basant uniquement sur les informations fournies dans le contexte.\n\n"
            
            "Consignes de réponse :\n"
            "1. Commence ta réponse en répétant la question posée, par exemple : 'En réponse à votre question : \"[question]\", voici les informations demandées :'\n"
            "2. Fournis une réponse concise et structurée.\n"
            "3. Utilise des phrases et des listes à puces pour organiser les informations.\n"
            
            "4. Traitement des questions sur un dossier spécifique :\n"
            "   - Vérifie d'abord les informations de la base de données.\n"
            "   - Indique clairement le statut actuel du dossier, la date de dernière modification, et les informations pertinentes du demandeur.\n"
            "   - Consulte ensuite les documents officiels et les règles pour expliquer les procédures.\n"
            "   - Examine les exemples d'échanges similaires pour adapter le style et le contenu de ta réponse.\n"
            
            "5. Traitement des questions générales sur le dispositif KAP Numérique :\n"
            "   - Consulte en priorité les documents officiels puis les règles.\n"
            "   - Utilise les exemples d'échanges pour adapter le format de ta réponse et son niveau de détail.\n"
            
            "6. Limitations :\n"
            "   - Si la question ne concerne ni le dispositif KAP Numérique, ni un bénéficiaire du programme, indique clairement que tu ne peux pas traiter ce type de demande.\n"
            "   - Exemple: 'Votre demande ne semble pas concerner le dispositif KAP Numérique ou l'un de ses bénéficiaires. Je ne peux malheureusement pas traiter ce type de requête.'\n"
            
            "7. Priorisation des sources :\n"
            "   - Documents officiels ('officiel') > Règles ('regles') > Échanges ('echanges')\n"
            "   - Les informations issues de la base de données sont prioritaires pour les questions sur un dossier spécifique.\n"
            
            "8. Cites systématiquement les sources avec le format suivant : [Document: Nom du document, Catégorie: Type de document, Section: Nom de la section, Page: Numéro de page, Mise à jour: Date].\n"
        )
        
        #dossier spécifique
        if state.get("force_dossier_id"):
            system_instructions += (
                f"\nATTENTION : Cette question concerne SPÉCIFIQUEMENT le dossier numéro {state['force_dossier_id']}. "
                f"Concentre-toi UNIQUEMENT sur les informations concernant ce dossier et ignore les informations "
                f"relatives à d'autres dossiers, même si elles semblent pertinentes."
     )
        
        # Construction de l'invite utilisateur
        user_prompt = f"Question: {state['question']}\n\nContexte extrait des documents et de la base de données:\n{docs_content}"
        
        # Messages combinant instructions système et question de l'utilisateur
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ]
        
        # Appel au modèle LLM avec gestion des erreurs
        try:
            response = llm.invoke(messages)
            return {"answer": response.content}
        except Exception as e:
            logger.error(f"Erreur lors de l'appel au LLM: {e}")
            return {"answer": f"Erreur lors de la génération de la réponse : {e}"}
    except Exception as e:
        logger.error(f"Erreur dans la fonction generate: {e}")
        return {"answer": f"Une erreur s'est produite lors du traitement de votre demande: {e}"}

def build_graph():
    """Construit et retourne le graphe LangGraph pour le RAG."""
    try:
        # Définir les nœuds du graphe
        workflow = StateGraph(State)
        
        # Ajouter les nœuds individuellement
        workflow.add_node("search_database", search_database)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        
        # Définir les transitions entre les nœuds
        workflow.add_edge(START, "search_database")
        workflow.add_edge("search_database", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Compiler le graphe
        return workflow.compile()
    except Exception as e:
        logger.error(f"Erreur lors de la construction du graphe: {e}")
        raise

# Point d'entrée pour l'utilisation directe du module

# On initialise le graphe
graph = build_graph()

if __name__ == "__main__":
    print("Tapez 'exit' pour quitter.")
    
    # Tester d'abord la connexion à la base de données
    try:
        db_manager = DatabaseManager()
        if not db_manager.tester_connexion():
            print("Avertissement: Impossible de se connecter à la base de données. Les requêtes sur les dossiers ne fonctionneront pas.")
    except Exception as e:
        print(f"Erreur lors de l'initialisation de la connexion à la base de données: {e}")
    
    while True:
        try:
            user_query = input("\nPosez votre question : ")
            if user_query.lower() in ["exit", "quit"]:
                break
                
            # Initialiser l'état avec des listes vides pour context et db_results
            initial_state = {
                "question": user_query,
                "context": [],
                "db_results": [],
                "answer": ""
            }
            
            # Invoquer le graphe avec gestion d'erreur
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