import streamlit as st
import os 
import time 
import pandas as pd 
from typing import Dict,Any,List,Optional 
import logging 
import json 

#configuration du logging pour streamlit 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streamlit_app")

#  variable d'état
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "general"  
    

#import du module RAG 
try:
    import rag_module
    rag_import_success = True
    logger.info("Modue RAG importé avec succès.")
except ImportError as e:
    rag_import_success = False 
    error_message = f"Erreur d'importation du module RAG: {e}"
    logger.error(error_message)
    st.error(error_message)
    
#configuration de la page streamlit 
st.set_page_config(
    page_title="Assistant KAP Numérique",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "KAP Numérique est un assistant virtuel qui vous aide à trouver des informations sur les Bénéficiaire et le dispositif général du KAP Numérique.",
    }
)


#initialisation de la session 
def init_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False 
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'rag_components' not in st.session_state:
        st.session_state.rag_components = None
        
    if 'active_dossier' not in st.session_state:
        st.session_state.active_dossier = None
        
    if 'sources_expanded' not in st.session_state:
        st.session_state.sources_expanded = False
        
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = True
        
    if 'show_db_results' not in st.session_state:
        st.session_state.show_db_results = True
        
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
        
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    
#initialiser l'etat de session 
init_session_state()


# Styles CSS personnalisés pour améliorer l'interface
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e6f3ff;
    }
    .source-box {
        border-left: 3px solid #4682B4;
        padding-left: 10px;
        background-color: #f9f9f9;
    }
    .header-container {
        display: flex;
        align-items: center;
    }
    .system-status-badge {
        margin-left: 10px;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .system-online {
        background-color: #d4edda;
        color: #155724;
    }
    .system-offline {
        background-color: #f8d7da;
        color: #721c24;
    }
    .notification-count {
        background-color: #007bff;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        margin-left: 5px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f8f9fa;
        padding: 10px;
        text-align: center;
        font-size: 0.8rem;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)


def initialize_rag_system() -> bool:
    """ 
     Initialise le système RAG et configure les composants nécessaires.
     Affiche une barre de progression pendant l'initialisation.
        
     Returns:
        bool: True si l'initialisation réussit, False sinon
    """
    #Réinitialiser les eventuels message d'erreurs précédents
    st.session_state.error_message = None 
    
    progres_bar = st.progress(0)
    status_text = st.empty()
    
    
    try:
        #Etape : 1 Vérification des prérequis 
        status_text.text("Vérification des prérequis...")
        progres_bar.progress(10)
        time.sleep(0.5) #simule le temps de traitement
        
        #Etape : 2 Connexion à la base de données 
        status_text.text("Connexion à la base de données...")
        progres_bar.progress(30)
        time.sleep(0.5)
        
        
        #Etape 3 : chargement des modèles et documents 
        status_text.text("Chargement des modèles et documents...")
        progres_bar.progress(60)
        time.sleep(0.5)
        
        #Etape 4 : Initialisation du système RAG 
        status_text.text("Initialisation du système RAG...")
        progres_bar.progress(80)
        
        #Appel à la fonction d'initialisation du module RAG
        st.session_state.rag_components = rag_module.init_rag_system()
        
        #Etape 5 : Vérification des composants RAG 
        status_text.text("Finalisation de l'initialisation...")
        progres_bar.progress(100)
        time.sleep(0.5) 
        
        #nettoyer les éléments temporaires
        status_text.empty()
        progres_bar.empty()
        
        #marquer comme initialisé 
        st.session_state.initialized = True
        logger.info("Système RAG initialisé avec succès.")
        
        return True 
    
    except Exception as e:
        status_text.empty()
        progres_bar.empty()
        
        
        error_msg = f"Erreur lors de l'initalisation du system RAG :  {e}"
        st.session_state.error_message = error_msg
        logger.error(error_msg,exc_info=True) 
        
        return False
    
# Fonction pour sauvegarder l'historique de conversation 
def save_chat_history():
    """ 
        Sauvegarde l'historique du chat dans un fichier local
    """
    
    try:
        #On verifie d'abord si le dossie de sauvegarde existe
        os.makedirs("./data/chat_history",exist_ok=True)
        
        #Générer un ID unique basé sur la date et l'heure
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"./data/chat_history/chat{timestamp}.json"
        
        # Ensuite on sauvegarde l'historique 
        with open(filename,"w",encoding="utf-8") as f:
            json.dump(st.session_state.chat_history,f,ensure_ascii=False,indent=2)
            
        logger.info(f"Historique de conversation sauvegardé dans {filename}")
        return True 
    except Exception as e:
        
        logger.error(f"Erreur lors de la sauvegardede l'historique : {e}", exc_info=True)
        return False
    
#Fonction pour charger un historique de conversation précédent
def load_chat_history(filename: str) -> bool:
    """ 
    Charge un historique de conversation depuis un fichier
    """
    try:
        with open(filename,"r",encoding="utf-8") as f:
            st.session_state.chat_history = json.load(f)
        return True 
    except Exception as e :
        logger.error(f"Erreur lors du chargement de l'historique : {e}", exc_info=True)
        return False 
    
#Fonction pour traiter la question de l'instructeur 
def process_user_query(query: str)->str:
    """ 
        Traite la question de l'utilisateur .
        
        Args:
            query (str): La question posée par l'utilisateur
            
        Returns:
            str: La réponse générée par le système
    """
    if not st.session_state.initialized:
        return "Le system n'est pas encore initialisé. Veuillez initaliser le systeme RAG depuis la bare latérale."
    
    try:
        #Marquer comme en cours de traitement 
        
        st.session_state.is_processing = True 
        
        #Obtenir l'objet graph du RAG 
        graph = st.session_state.rag_components.get("graph")
        
        if not graph:
            return "Le graph RAG n'est pas initialisé correctement. Vérifiez la configuration du système."
        
        #intégrer le dossier actif dans le contexte si disponible 
        context = []
        
        # Si un dossier est actif, l'ajouter  au contexte
        if st.session_state.active_dossier:
            dossier_id = st.session_state.active_dossier.get("Numero","N/A")
            
            if "dossier" not in query.lower() and dossier_id not in query: 
                enhanced_query = f"Concernant spécifiquement le dossier {dossier_id}: {query}"
            else:
                enhanced_query = query
            
             # Ajouter les informations du dossier actif au contexte
            context.append({
                "type": "active_dossier",
                "dossier_id": dossier_id,
                "dossier_data": st.session_state.active_dossier,
                "is_primary_context": True # Flag pour indiquer que ce contexte est prioritaire
            })
            
            logger.info(f"Question traitée avec le dossier actif #{dossier_id}")
        else:
            enhanced_query = query
    
        #Etat inital pour le graph
        initial_state = {
            "question":enhanced_query,
            "context": context,
            "db_results": [st.session_state.active_dossier] if st.session_state.active_dossier else [],
            "answer": "",
            "history": st.session_state.chat_history[-5:] if len(st.session_state.chat_history) > 0 else [],
            "force_dossier_id": dossier_id if st.session_state.active_dossier else None  # Forcer l'utilisation de ce dossier
        }
        
        #executer le graphe avec timeout pour eviter les blocages
        start_time = time.time()
        result = graph.invoke(initial_state)
        processing_time = time.time() - start_time #temps de traitement 
        
        logger.info(f"Requete traitée en {processing_time:.2f} secondes")
        
        #sauvegarder les résulats pour les afficher 
        st.session_state.last_result = result
        
        if st.session_state.active_dossier:
            dossier_prefix = f"📋 *Information concernant le dossier {dossier_id}:*\n\n"
            return dossier_prefix + result["answer"]
        
        #Enrichir la réponse avec les métadonnées
        elif result.get("db_results") and st.session_state.show_db_results:
            dossier_info = f"\n\n*Informations trouvées dans {len(result['db_results'])} dossier(s)*"
            return result["answer"] + dossier_info
        
        else:
            return result["answer"]
        
    except Exception as e:
        error_msg = f"Une erreur s'est produite lors du traitement de votre demande:  {str(e)}"
        logger.error(f"Erreur de traitement de la requête: {e}",exc_info=True)
        return error_msg
    
    finally : 
        #marquer comme traitement terminé 
        st.session_state.is_processing = False 
        
# Fonction pour rechercher un dossier spécifique 

def search_dossier(dossier_id: str)-> Optional[Dict]:
    """
    Recherche un dossier spécifique dans la base de données
    
    Args:
        dossier_id (str): L'identifiant du dossier à rechercher
        
    Returns:
        Optional[Dict]: Les données du dossier si trouvé, None sinon
    """
    
    if not st.session_state.initialized:
        st.error("Le système n'est pas encore initialisé.")
        return None 
    
    try:
        # Récuperer la fonction de recherche de dossier du module RAG 
        search_function = st.session_state.rag_components.get("rechercher_dossier")
        
        
        if not search_function:
            st.error("La fonction de recherche  de dossier n'est pas disponible.")
            return None 
        
        # Effectuer la recherche 
        results = search_function(dossier_id)
        
        
        
        #Verifie si un dossier à ete retrouver
        if results and len(results) > 0:
            #prendre le premier resultat
            result = results[0]
            
            #stocker le resultat comme dossier actif
            st.session_state.active_dossier = result
            
            return result
        else:
            st.warning(f"Aucun dossier trouvé avec l'identifiant {dossier_id}.")
            return None 
    
    except Exception as e:
        logger.error(f"Erreur lors de la recherche du dossier : {e}",exc_info=True)
        st.error(f"Une erreur s'est produite lors de la recherche de dossier : {str(e)}")
        return None 
    
#Fonction pour formater et afficher une source 
def display_source(source, index):
    """ 
    Affiche une source de manière formatée
    """
    st.markdown(f"**Source {index+1}:** *{source.metadata.get('source','Document interne')}*")
    
    #Affichage des métadonnées importantes 
    if source.metadata:
        metadata_str = ",".join([f"{k}: {v}" for k,v in source.metadata.items() if k not in ['source','content','text'] and v])
        
        if metadata_str:
            st.caption(metadata_str)
            
    #Affiche le contenu avec mise en evidence si court, sinon avec expanseur
    content = source.page_content
    if len(content) > 500:
        with st.expander("Voir le contenu complet"):
            st.markdown(f'<div class="source-box">{content}</div>',unsafe_allow_html=True)
        st.text(content[:500] + "...")
    else:
        st.markdown(f'<div class="source-box">{content}</div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
#===== INTERFACE PRINCIPALE ====
# Ajouter des onglets en haut de l'interface principale
mode_tabs = st.tabs(["💬 Questions Générales", "📁 Consultation de Dossier"])

#Titre principal avec indicateur d'état du système 
col1,col2 = st.columns([3,1])
with col1:
    st.title("🤖 Assistant KAP Numérique")
with col2:
    if st.session_state.initialized:
        st.markdown('<div class="system-status-badge system-online">Système en ligne</div>',unsafe_allow_html=True)
    else:
         st.markdown('<div class="system-status-badge system-offline">Système hors ligne</div>',unsafe_allow_html=True)
st.markdown("---")

# Barre latérale avec les informations et contrôles

with st.sidebar:
    st.header("Paramètres du système")
    
    #Bouton d'initialisation du système
    if not st.session_state.initialized:
        if st.button(" Initialiser le système RAG ", use_container_width=True):
            initialize_rag_system()
    else:
        st.success("✅ Système RAG initialisé et opérationnel")
    
        # Afficher les informations sur l'etat du  système
        components = st.session_state.rag_components
        
        if components:
            with st.expander(" Etat du système",expanded=True):
                # Etat de la connexion à la base de données
                
                if components.get("db_connected",False): 
                    st.success("✅ Connecté à la base de données")
                else:
                    st.warning("⚠️ Base de données non connectée")

            #Nombre de documents chargés
            docs= components.get("docs",[])
            st.info(f"📄 {len(docs)} documents chargés")
            
            #compteur de performance 
            if "performance" in components:
                perf = components["performance"]
                st.metric(label="Temps de réponse moyen", value=f"{perf.get('avg_response_time',0):.2f} secondes")
                
            #Affichage des catégories de documents 
            if docs:
                doc_categories = {}
                for doc in docs:
                    category = doc.metadata.get("category","non_classifié")
                    if category in doc_categories:
                        doc_categories[category] += 1
                    else:
                        doc_categories[category] = 1

                #  Afficher un graphiqe pour la répartition
                if doc_categories:
                    st.markdown("### Répartition des documents")
                    chart_data = pd.DataFrame({
                        'Categorie': list(doc_categories.keys()),
                        'Nombre': list(doc_categories.values())
                    })
                    st.bar_chart(chart_data.set_index('Categorie'))
                    
              
                    
    #Option d'afichage regroupées
    with st.expander("⚙️ Options d'affichage", expanded=True):
        st.session_state.show_sources = st.checkbox("Afficher les sources citées", value=st.session_state.show_sources)
        st.session_state.show_db_results = st.checkbox("Afficher les détails des dossiers", value=st.session_state.show_db_results)
        st.session_state.sources_expanded = st.checkbox("Développer les sources par défaut", value=st.session_state.sources_expanded)
        
    #gestion de l'historique
    with st.expander("🕒 Gestion de l'historique", expanded=False):
        #Ajout d'un bouton pour effacer l'historique du chat 
        
        if st.button("🗑️ Effacer l'historique", use_container_width=True):
            if st.session_state.chat_history:
                #proposer de sauvegarder avant d'effacer
                if st.checkbox("Sauvegarder avant d'effacer ?", value=True):
                    save_chat_history()
            
            st.session_state.chat_history = []
            st.rerun()  
        
        # Option pour sauvegarder l'historique manuellement
        if st.button("💾 Sauvegarder l'historique", use_container_width=True):
            if save_chat_history():
                st.success("Historique sauvegardé avec succès.")
            else:
                st.error("Erreur lors de la sauvegarde de l'historique.")
    
    #section d'aide et documentation 
    with st.expander("❓ Aide", expanded=False):
        st.markdown("""
            ### Comment utiliser l'assistant
        
            1. **Initialiser le système** depuis la barre latérale
            2. **Poser des questions** dans la zone de saisie en bas
            3. **Rechercher un dossier** spécifique via la section dédiée
            
            ### Exemples de questions
            - "Quels sont les critères d'éligibilité pour le dispositif KAP?"
            - "Comment faire une demande de financement?"
            - "Quelle est la procédure pour le suivi d'un dossier?  
                    """)
    
    #Affichage des erreurs si présentes 
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        if st.button("Effacer ce message"):
            st.session_state.error_message = None 
            st.experimental_rerun()
  
# Premier onglet : Questions générales
with mode_tabs[0]:
    if st.session_state.app_mode != "general":
        st.session_state.app_mode = "general"
        st.session_state.active_dossier = None  # Désactiver tout dossier actif
        st.rerun()
    
    st.subheader("Questions sur le dispositif KAP Numérique")
    
     
#Zone d'affichage des messages du chat avec scrolling
st.subheader("💬 Conversation")
chat_container = st.container(height=400)

with chat_container:
    if not st.session_state.chat_history:
        st.info("👋 Bonjour ! Je suis l'assistant du dispositif KAP Numérique. Comment puis-je vous aider aujourd'hui ?")
    else:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.chat_message("user",avatar="👤").write(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    content = st.write(message["content"])
                    
                    # On verifie si content est une chaîne de caractères
                    if not isinstance(content, str):
                        content = str(content) if content is not None else ""
                    
                    # Vérifier si ce message concerne un dossier spécifique
                    # Par celle-ci (avec une vérification du type)
                    # Vérifier si ce message concerne un dossier spécifique
                    is_dossier_message = "dossier" in content.lower() and any(str(i) in content for i in range(10))
                    
                    
                    if is_dossier_message:
                        # Ajouter un style pour les messages concernant un dossier
                        st.markdown(f"""
                        <div style="border-left: 4px solid #28a745; padding-left: 10px;">
                            {content}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.write(content)
                    
                     # Afficher le badge de dossier actif
                    if st.session_state.active_dossier:
                        dossier_id = st.session_state.active_dossier.get('Numero','N/A')
                        st.caption(f"🟢 Dossier actif: #{dossier_id}")
                    
                    #si c'est le dernier message de l'assistant et qu'il y'a un dossier actif 
                    if i==len(st.session_state.chat_history) -1 and st.session_state.active_dossier:
                        st.caption(f"Dossier actif: #{st.session_state.active_dossier.get('Numero','N/A')}")
                        
#Affichage des destails du dossier s'il y'en a et si l'option est activée
if(st.session_state.get('show_db_results',True) and 
    st.session_state.get('last_result') and
    st.session_state.get('last_result').get('db_results')
    ):
    
    with st.expander("📋 Détails du dossier", expanded=True):
        db_results = st.session_state.last_result["db_results"]
        if db_results:
            #Creation d'un DataFrame propore avec les colonne importante en premier 
            df = pd.DataFrame(db_results)
            
            #Reorganiser les colonnes si nécéssaire
            important_cols = ['Numero','date_creation','derniere_modification','nom_usager','agent_affecter','instructeur','statut','montant'    
            ]
            
            cols = [col for col in important_cols if col in df.columns] + [col for col in df.columns if col not in important_cols]
            
            
            #Afficher avec formatage 
            st.dataframe(
                df[cols],
                use_container_width=True,
                column_config={
                    "Numero": st.column_config.TextColumn("Numéro de dossier", width="150px"),
                    "date_creation": st.column_config.DateColumn("Date de création", format="DD/MM/YYYY"),
                    "nom_usager": st.column_config.TextColumn("Nom de l'usager"),
                    "agent_affecter": st.column_config.TextColumn("Agent affecté"),
                    "instructeur": st.column_config.TextColumn("Instructeur"),
                    "statut": st.column_config.TextColumn("Statut",width="small"),
                    "montant": st.column_config.NumberColumn("Montant", format="$%.2f")
                }
            )
            
            #actions sur le dossiers 
                # Actions sur les dossiers
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Définir comme dossier actif", key="set_active"):
                    if len(db_results) > 0:
                        st.session_state.active_dossier = db_results[0]
                        st.success(f"Dossier #{db_results[0].get('Numero', 'N/A')} défini comme dossier actif")
                        st.rerun()
            
            with col2:
                if st.button("Poser une question sur ce dossier", key="ask_details"): 
                    # Définir le dossier comme actif
                    
                    if len(db_results) > 0:
                        dossier_id = db_results[0].get('Numero', 'inconnu')
                        st.session_state.active_dossier = db_results[0]
                        # Ajouter une question prédéfinie à l'historique
                        question = f"Résume-moi le dossier #{dossier_id}"
                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.rerun()
        else:
            st.info("Aucune information de dossier disponible pour cette requête.")
    
# Zone de saisie pour la question de l'utilisateur avec instructions
st.markdown("### Posez votre question !!")

# Suggestions de questions rapides
quick_questions = [
    "Comment fonctionne le dispositif KAP Numérique?",
    "Quels sont les critères d'éligibilité?",
    "Quels documents fournir pour un dossier?"
]

# Afficher les boutons de questions rapides si pas encore de conversation
if len(st.session_state.chat_history) < 2:
    cols = st.columns(len(quick_questions))
    for i, col in enumerate(cols):
        if col.button(quick_questions[i]):
            st.session_state.chat_history.append({"role": "user", "content": quick_questions[i]})
            st.rerun()
            
#Zone de saisie principale
help_text = "Système initialisé, posez votre question..." if st.session_state.initialized else "Veuillez initialiser le système depuis la barre latérale"

user_input = st.chat_input(help_text, disabled=not st.session_state.initialized or st.session_state.is_processing)

#traitement de l'entrée 
if user_input:
    #Ajouter la question à l'historique
    st.session_state.chat_history.append({"role":"user","content":user_input})
    
    #afficher immédiatement la quesiton 
    st.rerun()
    
#si le dernier message est de l'utilisateur, traiter la réponse
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    #Afficher un sinner pendant le traitement 
    
    with st.spinner("🔍 Recherche d'informations en cours..."):
        query = st.session_state.chat_history[-1]["content"]
        response = process_user_query(query)
        
    #Ajouter la réponse à l'historique
    st.session_state.chat_history.append({"role": "assistant","content":response})
    
    #Actualiser l'affichage
    st.rerun()
    
#section recherche rapide de dossier en bas de page
st.markdown("---")
st.subheader("🔍 Recherche rapide de dossier")

col1,col2,col3 = st.columns([3,1,1])

with col1:
    dossier_number = st.text_input("Numéro de dossier (ex: 82-2069)",key="dossier_search",placeholder="Entrez le numéro du dossier...")
    
with col2:
    search_button = st.button("🔎 Rechercher", use_container_width=True, disabled=not st.session_state.initialized or not dossier_number)
    
with col3:
     if st.session_state.active_dossier:
         if st.button("❌ Effacer dossier actif", use_container_width=True):
             st.session_state.active_dossier = None 
             st.rerun()
    
#traitement de la recherche de dossier
if search_button and dossier_number:
    result = search_dossier(dossier_number)
    
    if result:
        st.success(f"Dossier  {dossier_number} trouvé !")
        
        # Ajout d'un badge visuel en haut de la section dossier
        st.markdown("""
            <div style="
                background-color: #d4edda;
                color: #155724;
                padding: 10px;
                border-radius: 5px;
                margin-bottom:20px;
                display:flex;
                align-items:center; 
            ">
            <span style="font-weight:bold; margin-right:10px;">🟢 DOSSIER ACTIF</span>
            <span>Toutes les questions posées concerneront ce dossier jusqu'à ce que vous en sélectionniez un autre ou le désactiviez.</span>
        
        """,unsafe_allow_html=True)
        
        
        
        #Afficher un aperçu des informations du dossier dans un format compact
        col1,col2,col3,col4,col5,col6,col7 = st.columns(7)
        
        with col1:
            st.metric("Numero" , result.get("Numero","N/A"))
        with col2:
            st.metric("Nom usager", result.get("nom_usager","N/A"))
        with col3:
            st.metric("Statut", result.get("statut","N/A"))
        with col4:
            st.metric("Montant", result.get("montant","N/A"))
        with col5:
            st.metric("Date création", result.get("date_creation","N/A"))
        with col6:
            st.metric("Agent affecté", result.get("agent_affecter","N/A"))
        with col7:
            st.metric("Instructeur", result.get("instructeur","N/A"))
            
        #Afficher toutes les informations dans un tableaux extensible 
        
        with st.expander("Voir toutes les details du dossier", expanded=True):
            
            #convertir en dataframe pour un affichage plus propre
            result_df = pd.DataFrame([result])
            st.dataframe(result_df)
            
            #option pour poser des questions sur ce dossier
            if st.button("Poser des questions sur ce dossier",key="ask_details"):
                #ajouter une question predefinir à l'historique
                question = f"Donne-moi un résumé du dossier {dossier_number}"
                st.session_state.chat_history.append({"role":"user","content":question})
                st.rerun()
    else:
        st.warning(f"Aucun dossier trouvé avec l'identifiant {dossier_number}")
        
#Section d'affichage des sources si activée 
if(st.session_state.get('show_sources',True) and 
    st.session_state.get('last_result') and
    st.session_state.get('last_result').get('context')
    ):
    
    st.markdown("---")
    
    with st.expander("📚 Sources consultées", expanded=st.session_state.sources_expanded):
        context = st.session_state.last_result["context"]
        if context:
            st.markdown(f"### {len(context)} sources utilisées pour générer la réponse")
            
            #option d'affichage des sources
            view_mode = st.radio("Mode d'affichage", ("Liste détailée","Vue tableau"), horizontal=True)
            
            if view_mode == "Liste détaillée":
                for i, source in enumerate(context):
                    display_source(source, i)
            else:
                 # Créer un DataFrame avec les sources pour une vue compacte
                source_data = []
                for i, source in enumerate(context):
                    source_data.append({
                        "Source":source.metadata.get("source","Document interne"),
                        "Type": source.metadata.get("type","non spécifié"),
                        "Contenu": source.page_content[:100] + "..." if len(source.page_content) > 100 else source.page_content,
                    })
                if source_data:
                    st.dataframe(pd.DataFrame(source_data), use_container_width=True)
        else:
            st.info("Aucune source disponible pour cette requête.")
        
# Footer avec des informations sur l'applicaiton
st.markdown("""
<div class="footer">
    © 2025 KAP Numérique - Assistant basé sur un système RAG | v1.0.0 | Créer par Chahalane Bériche
    <a href="mailto:support@kap-numerique.fr">Assistance</a>
</div>
""", unsafe_allow_html=True)
        