import os
import re
import pymysql
from typing import List, TypedDict, Dict, Any, Optional
from mistralai import Mistral
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

tracing = os.getenv("LANGSMITH_TRACING")
api_key = os.getenv("LANGSMITH_API_KEY")
Mistral_api_key = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Chemin vers le dossier contenant vos documents
DOCUMENTS_PATH = "../data"

# Chat model mistral
llm = init_chat_model("mistral-large-latest", model_provider="mistralai")

# Classe DatabaseManager pour la gestion de la connexion à la base de données
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
            print(" Connexion réussie avec pymysql.")
            return True
        except pymysql.Error as erreur:
            print(f" Échec de la connexion : {erreur}")
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
            print(f" Erreur de recherche : {erreur}")
            return []

# Fonction pour extraire le numéro de dossier de la question
def extract_dossier_number(question: str) -> List[str]:
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

# Fonction pour rechercher les informations d'un dossier dans la base de données via DatabaseManager
def search_dossier_in_db(dossier_numbers: List[str]) -> List[Dict[str, Any]]:
    try:
        db_manager = DatabaseManager()
        
        # Tester la connexion avant de procéder
        if not db_manager.tester_connexion():
            print("Impossible de se connecter à la base de données")
            return []
        
        results = []
        for num in dossier_numbers:
            dossier_results = db_manager.rechercher_dossier(numero_dossier=num)
            results.extend(dossier_results)
        
        return results
    except Exception as e:
        print(f"Erreur lors de la recherche dans la base de données: {e}")
        return []

# Fonction pour convertir les résultats de la base de données en documents Langchain
def db_results_to_documents(results: List[Dict[str, Any]]) -> List[Document]:
    documents = []
    for result in results:
        # Création d'un contenu formaté à partir des données du dossier
        content = f"""
Informations sur le dossier {result.get('Numero', 'N/A')}:
- Nom de l'usager: {result.get('nom_usager', 'N/A')}
- Date de création: {result.get('date_creation', 'N/A')}
- Dernière modification: {result.get('derniere_modification', 'N/A')}
- Agent affecté: {result.get('agent_affecter', 'N/A')}
- Instructeur: {result.get('instructeur', 'N/A')}
- Statut actuel: {result.get('statut', 'N/A')}
- Statut visible par l'usager: {result.get('statut_visible_usager', 'N/A')}
- Montant: {result.get('montant', 'N/A')} €
"""
        # Création d'un document Langchain avec les métadonnées appropriées
        doc = Document(
            page_content=content,
            metadata={
                "source": "base_de_donnees",
                "type": "dossier",
                "numero": result.get('Numero', 'N/A'),
                "section": "Informations dossier",
                "page": "1",
                "update_date": result.get('derniere_modification', 'N/A')
            }
        )
        documents.append(doc)
    
    return documents

# Charger les documents depuis le dossier "data"
try:
    loader = DirectoryLoader(
        DOCUMENTS_PATH,
        glob="*.txt",  # Charger uniquement les fichiers .txt
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}  # Spécifier l'encodage approprié
    )
    docs = loader.load()

    if docs:
        print(f"{len(docs)} documents chargés avec succès.")
        for i, doc in enumerate(docs[:5]):  # Afficher les 5 premiers documents
            # Extraire le nom du fichier depuis la métadonnée "source"
            source = doc.metadata.get("source", "inconnu")
            file_title = os.path.basename(source)
            print(f"Document {i+1}:")
            print(f"Nom du fichier : {file_title}")
            print("-" * 50)
    else:
        print("Aucun document n'a été chargé.")
        # Initialiser docs comme une liste vide pour éviter les erreurs
        docs = []
except Exception as e:
    print(f"Erreur lors du chargement des documents: {e}")
    docs = []

# Découper les documents en chunks pour un meilleur traitement par le modèle
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Initialiser les embeddings Mistral
embeddings = MistralAIEmbeddings()

# Créer le vectorstore en mémoire à partir des chunks et des embeddings
vector_store = InMemoryVectorStore.from_documents(all_splits, embeddings)

# Définir le prompt pour la recherche de réponse
prompt = hub.pull("rlm/rag-prompt")

# Définition de l'état de l'application
class State(TypedDict):
    question: str
    context: List[Document]  # Liste d'objets Document - spécifier le type
    db_results: List[Dict[str, Any]]  # Résultats de la base de données
    answer: str

# Fonction de récupération des données de la base de données
def search_database(state: State) -> Dict[str, Any]:
    # Extraire les numéros de dossier de la question
    dossier_numbers = extract_dossier_number(state["question"])
    
    # Si des numéros de dossier sont trouvés, rechercher dans la base de données
    db_results = []
    if dossier_numbers:
        db_results = search_dossier_in_db(dossier_numbers)
    
    return {"db_results": db_results}

# Fonction de récupération (retrieval) des documents pertinents
def retrieve(state: State) -> Dict[str, Any]:
    try:
        # Récupérer les documents pertinents depuis le vectorstore
        retrieved_docs = vector_store.similarity_search(state["question"], k=3)
        
        # Convertir les résultats de la base de données en documents
        db_docs = db_results_to_documents(state["db_results"])
        
        # Combiner les documents de la base de données et les documents du vectorstore
        combined_docs = db_docs + retrieved_docs
        
        return {"context": combined_docs}
    except Exception as e:
        print(f"Erreur dans la fonction retrieve: {e}")
        return {"context": []}  # Retourner une liste vide en cas d'erreur

# Fonction de génération de la réponse
def generate(state: State) -> Dict[str, Any]:
    try:
        # Vérifier si le contexte est vide
        if not state["context"]:
            return {"answer": "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question."}
        
        # Agrégation du contenu des documents récupérés avec détails sur les sources
        docs_details = []
        for doc in state["context"]:
            source = doc.metadata.get("source", "Source inconnue")
            
            # Traitement spécial pour les documents de la base de données
            if source == "base_de_donnees":
                file_name = f"Base de données - Dossier {doc.metadata.get('numero', 'inconnu')}"
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
                "update_date": update_date
            })

        # Agrégation du contenu des documents
        docs_content = "\n\n".join(doc["content"] for doc in docs_details)

        # Formatage des sources avec section, page et date
        formatted_sources = "\n".join([
            f"[Document: {doc['file_name']}, Section: {doc['section']}, Page: {doc['page']}, Mise à jour: {doc['update_date']}]"
            for doc in docs_details
        ])

        # Instructions système mises à jour pour inclure les informations de dossier
        system_instructions = (
            "Tu es un instructeur expert du dispositif KAP Numérique. Tu réponds à des questions en te basant uniquement sur les documents officiels fournis "
            "et les informations de dossier extraites de la base de données.\n\n"
            "Consignes de réponse :\n"
            "1. Commence ta réponse en répétant la question posée, par exemple : 'En réponse à votre question : \"[question]\", voici les informations demandées :'\n"
            "2. Fournis une réponse concise et structurée.\n"
            "3. Utilise des listes à puces pour organiser les informations.\n"
            "4. Si la question concerne un dossier spécifique, vérifie d'abord les informations de la base de données. "
            "   Indique clairement le statut actuel du dossier, la date de dernière modification, et toute information pertinente.\n"
            "5. Pour les questions sur le paiement, si le statut est 'Mandatement', indique que le paiement est en cours de traitement "
            "   et devrait être effectué dans les jours suivants.\n"
            "6. Cites systématiquement les sources avec le format suivant : [Document: Nom du document, Section: Nom de la section, Page: Numéro de page, Mise à jour: Date].\n\n"
            "Maintenant, réponds à la question en respectant ce format."
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
            return {"answer": f"Erreur lors de la génération de la réponse : {e}"}
    except Exception as e:
        print(f"Erreur dans la fonction generate: {e}")
        return {"answer": f"Une erreur s'est produite lors du traitement de votre demande: {e}"}

# Construction du graphe d'application
def build_graph():
    try:
        # Définir les nœuds du graphe
        workflow = StateGraph(State)
        
        # Ajouter les nœuds individuellement pour mieux contrôler
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
        print(f"Erreur lors de la construction du graphe: {e}")
        raise



# Construire le graphe
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