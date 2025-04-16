import streamlit as st
import os
import time
import rag_module
import pandas as pd
import logging
import json
from datetime import date, timedelta # Importer date 
from typing import Dict, Any, List, Optional



# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streamlit_app")

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Assistant KAP Numérique",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@kap-numerique.fr',
        'Report a bug': "mailto:support@kap-numerique.fr",
        'About': """
        **Assistant KAP Numérique Beta v1.0**

        Cet assistant utilise un système RAG (Retrieval Augmented Generation)
        pour répondre aux questions des instructeurs du dispositif KAP Numérique.
        Il peut fournir des informations générales ou spécifiques à un dossier.
        """
    }
)

# --- Variables d'état (Session State) ---
def init_session_state():
    """Initialise les variables d'état nécessaires pour l'application."""
    defaults = {
        'app_mode': "general",          # Mode actuel: "general" ou "dossier"
        'initialized': False,           # Statut d'initialisation du système RAG
        'chat_history': [],             # Historique de la conversation
        'is_processing': False,         # Indicateur de traitement en cours
        'rag_components': None,         # Composants du système RAG (graph, fonctions, etc.)
        'sources_expanded': False,      # État de l'expandeur des sources
        'show_sources': True,           # Afficher/masquer les sources
        'show_db_results': True,        # Afficher/masquer les détails des dossiers trouvés
        'last_result': None,            # Dernier résultat obtenu du système RAG
        'error_message': None,          # Message d'erreur à afficher
        'system_status_msg': "Système non initialisé.", # Message d'état du système
        'dossier_search_results': None # Résultats de la dernière recherche de dossier
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialiser l'état de session au démarrage
init_session_state()

# --- Importation et Initialisation du Module RAG ---
try:
    rag_import_success = True
    logger.info("Module RAG importé avec succès.")

    # --- Initialisation Automatique ---
    # On Vérifie si le système est déjà initialisé dans cette session
    if not st.session_state.initialized:
        initialization_placeholder = st.empty() # Créer un espace pour le message/spinner
        with initialization_placeholder.container():
            with st.spinner("⏳ Initialisation de l'assistant en cours, veuillez patienter..."):
                try:
                    start_time = time.time()
                    # Appel de la fonction d'initialisation
                    st.session_state.rag_components = rag_module.init_rag_system()
                    st.session_state.initialized = True
                    st.session_state.system_status_msg = "✅ Système prêt et connecté."
                    st.session_state.error_message = None
                    duration = time.time() - start_time
                    #logger.info(f"Système RAG initialisé avec succès en {duration:.2f} secondes.")
                    # Optionnel: Afficher un message de succès temporaire
                    st.success(f"Assistant initialisé en {duration:.2f}s !")
                    time.sleep(2) # Garder le message de succès visible un instant
                except Exception as e:
                    logger.error(f"Erreur critique lors de l'initialisation du RAG: {e}", exc_info=True)
                    st.session_state.initialized = False
                    st.session_state.error_message = f"❌ Erreur d'initialisation: {e}. L'assistant risque de ne pas fonctionner."
                    st.session_state.system_status_msg = "❌ Erreur d'initialisation."
                    # Afficher l'erreur persistante si l'initialisation échoue
                    st.error(st.session_state.error_message)

        # Nettoyer le placeholder une fois l'initialisation terminée (succès ou échec géré)
        initialization_placeholder.empty()
except ImportError as e:
    st.session_state.initialized = False # Marquer comme non initialisé si l'import échoue
    st.session_state.system_status_msg = "❌ Module RAG introuvable."
    
# --- Styles CSS Personnalisés ---
st.markdown("""
<style>
    /* --- Base & Layout --- */
    .stApp {
        /* background-color: #ffffff; */ /* White background */
    }
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px; /* Space between tabs */
	}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
		background-color: transparent;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
  		background-color: #0d6efd; /* Blue for active tab */
        color: white;
	}
    /* --- Sidebar --- */
    .stSidebar {
        /* background-color: #f8f9fa; /* Light gray background */
        /* border-right: 1px solid #dee2e6; */
    }
    .stSidebar .stButton button {
        background-color: #0d6efd; /* Blue buttons */
        color: white;
        border-radius: 0.3rem;
        transition: background-color 0.3s ease;
    }
    .stSidebar .stButton button:hover {
        background-color: #0b5ed7; /* Darker blue on hover */
    }
    .stSidebar .stExpander {
        border: 1px solid #dee2e6;
        border-radius: 0.3rem;
    }

    /* --- Chat --- */
    .stChatMessage {
        border-radius: 0.5rem;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid transparent; /* Base border */
    }
    .stChatMessage[data-testid="chat-message-container-user"] {
        background-color: #e7f3ff; /* Light blue for user messages */
        border-left: 5px solid #0d6efd; /* Blue accent */
    }
    .stChatMessage[data-testid="chat-message-container-assistant"] {
        background-color: #f8f9fa; /* Light gray for assistant messages */
        border-left: 5px solid #ffc107; /* Yellow accent */
    }

    /* --- Sources & Details --- */
    .source-box {
        border-left: 4px solid #ffc107; /* Yellow accent for sources */
        padding: 0.5rem 1rem;
        margin-top: 0.5rem;
        background-color: #fffbeb; /* Very light yellow background */
        border-radius: 0.3rem;
        font-size: 0.9rem;
    }
    .dossier-details-box {
        border: 1px solid #cfe2ff; /* Light blue border */
        background-color: #f2f7ff; /* Very light blue background */
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
     .dossier-active-banner {
        background-color: #d1ecf1; /* Light cyan background */
        color: #0c5460; /* Dark cyan text */
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 1px solid #bee5eb;
        display: flex;
        align-items: center;
        font-size: 0.95rem;
    }
    .dossier-active-banner span {
        margin-left: 10px;
    }

    /* --- Status Indicators --- */
    .system-status-badge {
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block; /* To allow margin */
        margin-top: 10px; /* Align better with title */
    }
    .system-online {
        background-color: #d4edda; /* Green */
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .system-offline {
        background-color: #f8d7da; /* Red */
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
     .system-initializing {
        background-color: #fff3cd; /* Yellow */
        color: #856404;
        border: 1px solid #ffeeba;
    }

    /* --- Footer --- */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0d6efd; /* Blue footer */
        color: white; /* White text */
        padding: 8px 0;
        text-align: center;
        font-size: 0.8rem;
        border-top: 1px solid #0b5ed7;
        z-index: 99; /* Ensure it's above other elements */
    }
    .footer a {
        color: #ffc107; /* Yellow links */
        text-decoration: none;
    }
     .footer a:hover {
        text-decoration: underline;
    }

    /* --- Misc --- */
     h1, h2, h3 {
        color: #0d6efd; /* Blue headers */
     }

</style>
""", unsafe_allow_html=True)

# --- Fonctions Utilitaires ---
def initialize_rag_system() -> bool:
    """
    Initialise le système RAG en appelant la fonction du module importé.
    Affiche une barre de progression et des messages d'état.

    Returns:
        bool: True si l'initialisation réussit, False sinon.
    """
    if not rag_import_success:
        st.session_state.error_message = "Le module RAG n'a pas pu être importé. Impossible d'initialiser."
        st.session_state.system_status_msg = "Échec Import RAG."
        logger.error("Tentative d'initialisation alors que l'import RAG a échoué.")
        return False

    st.session_state.error_message = None
    st.session_state.system_status_msg = "Initialisation en cours..."
    st.session_state.initialized = False 

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.info("🚀 Démarrage de l'initialisation du système RAG...")

    try:
        status_text.info("1/3 Préparation de l'initialisation...")
        progress_bar.progress(25)
        time.sleep(0.3)

        status_text.info("2/3 Appel de la fonction d'initialisation RAG...")
        progress_bar.progress(60)
        # --- Appel de la Vfonction d'initialisation du module RAG---
        if hasattr(rag_module, 'init_rag_system') and callable(rag_module.init_rag_system):
             st.session_state.rag_components = rag_module.init_rag_system()
             
             if not isinstance(st.session_state.rag_components, dict) or 'graph' not in st.session_state.rag_components:
                 logger.warning("init_rag_system did not return a dictionary with a 'graph' key.")
        else:
             raise RuntimeError("Le module RAG importé n'a pas de fonction 'init_rag_system' exécutable.")
        # ----------------------------------------------------
        progress_bar.progress(90)
        time.sleep(0.5) 

        status_text.info("3/3 Finalisation et vérification...")
        progress_bar.progress(100)
        time.sleep(0.3)

        # Nettoyer les éléments temporaires
        status_text.success("✅ Système RAG initialisé avec succès !")
        progress_bar.empty()
        time.sleep(1.5) 
        status_text.empty() 

        # Marquer comme initialisé
        st.session_state.initialized = True
        st.session_state.system_status_msg = "Système opérationnel."
        logger.info("Système RAG initialisé avec succès via rag_module.init_rag_system.")
        return True

    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        error_msg = f"❌ Erreur lors de l'initialisation du système RAG : {e}"
        st.session_state.error_message = error_msg
        st.session_state.system_status_msg = "Échec de l'initialisation."
        logger.error(error_msg, exc_info=True)
        st.session_state.initialized = False
        st.session_state.rag_components = None 
        return False
    finally:
        try:
            progress_bar.empty()
        except Exception:
            pass
        try:
            status_text.empty()
        except Exception:
            pass

def get_dossier_data_for_timeline(numero_dossier: str) -> Optional[Dict]:
    """
    Récupère les données d'un dossier spécifique pour la timeline.
    Utilise les résultats déjà chargés si possible, sinon refait une recherche.
    """
    # 1. Essayer de trouver dans les résultats déjà affichés
    if 'dossier_search_results' in st.session_state and st.session_state.dossier_search_results:
        for dossier in st.session_state.dossier_search_results:
            if dossier.get('Numero') == numero_dossier:
                return dossier

    # 2. Si non trouvé (ou si on veut être sûr d'avoir les dernières infos), refaire une recherche exacte
    logger.info(f"Données non trouvées dans le cache local pour {numero_dossier}, recherche en base...")
    search_function = st.session_state.rag_components.get("rechercher_dossier")
    if callable(search_function):
        try:
            # Recherche exacte par Numero
            results = search_function(search_term=numero_dossier)
            if results:
                return results[0] # Retourne le premier (et normalement unique) résultat
            else:
                logger.warning(f"Impossible de retrouver le dossier {numero_dossier} pour la timeline.")
                return None
        except Exception as e:
            logger.error(f"Erreur en rechargeant le dossier {numero_dossier}: {e}")
            return None
    return None


def display_dossier_timeline(numero_dossier: str):
    """Affiche une timeline simplifiée pour un dossier."""
    st.markdown(f"#### Historique Simplifié du Dossier {numero_dossier}")

    dossier_data = get_dossier_data_for_timeline(numero_dossier)

    if not dossier_data:
        st.warning("Impossible de charger les données du dossier pour afficher l'historique.")
        return

    # Création des événements de la timeline (version simplifiée)
    events = []

    # 1. Création
    date_creation = dossier_data.get('date_creation')
    if date_creation:
         # Formater la date pour l'affichage
         try:
            date_str = date_creation.strftime('%d %B %Y') if isinstance(date_creation, date) else str(date_creation)
            events.append({
                "date": date_str,
                "event": "📅 Création du dossier",
                "details": f"Dossier initié pour {dossier_data.get('nom_usager', 'N/A')}."
            })
         except AttributeError: # Au cas où ce n'est ni une date ni une string formatable
             events.append({"date": str(date_creation), "event": "📅 Création du dossier", "details": "Date de création enregistrée."})


    # 2. Dernière Modification (si différente de la création et non nulle)
    date_modif = dossier_data.get('derniere_modification')
    if date_modif and date_modif != date_creation : # Vérifier si la date est différente
         try:
            # Essayer de formater date ET heure si possible
            if hasattr(date_modif, 'strftime'):
                 date_str = date_modif.strftime('%d %B %Y à %H:%M')
            else:
                 date_str = str(date_modif)

            events.append({
                "date": date_str,
                "event": "✍️ Dernière Modification",
                "details": f"Le dossier a été mis à jour. Statut actuel : {dossier_data.get('statut', 'N/A')}."
                         f" (Instructeur: {dossier_data.get('instructeur', 'N/A')})"
            })
         except Exception as e: # Gestion d'erreur générique pour le formatage
              logger.error(f"Erreur formatage date modif {date_modif}: {e}")
              events.append({"date": str(date_modif), "event": "✍️ Dernière Modification", "details": f"Statut: {dossier_data.get('statut', 'N/A')}"})

    # --- Option 1: Affichage simple avec st.markdown ---
    if events:
        st.markdown("**Événements Clés :**")
        # Trier par date (même si ici c'est simple, bonne pratique pour une vraie timeline)
        # On suppose que les dates sont comparables. Si ce sont des strings, le tri peut être incorrect.
        # Pour un tri robuste, il faudrait convertir en objets date/datetime.
        # events.sort(key=lambda x: datetime.strptime(x['date'], '...format...') if isinstance(x['date'], str) else x['date']) # Exemple complexe

        for event in events:
            st.markdown(f"- **{event['date']} :** {event['event']}")
            st.caption(f"  > {event['details']}")
    else:
        st.info("Aucun événement historique majeur à afficher pour ce dossier (basé sur les données disponibles).")



def display_dossier_details_enhanced(dossier: Dict, index: int):
    """Affiche les détails d'un dossier de manière plus structurée et visuelle."""
    numero_dossier = dossier.get('Numero', f'Inconnu_{index}')
    container_key = f"dossier_container_{numero_dossier}_{index}" # Clé unique pour le conteneur

    with st.container(border=True): # Utiliser une bordure pour séparer visuellement
        st.subheader(f"📁 Dossier : {dossier.get('Numero', 'N/A')}")

        col1, col2, col3 = st.columns(3)
        with col1:
            # Utiliser st.metric pour les infos clés
            st.metric(label="Statut", value=dossier.get('statut', 'N/A'))
        with col2:
            # Formatter le montant en euros
            montant = dossier.get('montant', 0)
            try:
                 montant_formate = f"{float(montant):,.2f} €".replace(",", " ").replace(".", ",") if montant else "N/A"
            except (ValueError, TypeError):
                 montant_formate = "N/A"
            st.metric(label="Montant Demandé", value=montant_formate)
        with col3:
             # Utiliser st.metric pour la date importante
             date_crea = dossier.get('date_creation', 'N/A')
             # Essayer de formater la date si elle existe
             try:
                 if isinstance(date_crea, date):
                      date_crea_formatee = date_crea.strftime('%d/%m/%Y')
                 elif isinstance(date_crea, str): # Si c'est une chaîne, on essaie de la garder
                      date_crea_formatee = date_crea
                 else:
                      date_crea_formatee = "N/A"
             except Exception:
                  date_crea_formatee = "N/A"

             st.metric(label="Date Création", value=date_crea_formatee)


        st.markdown("**Informations Usager**")
        st.text(f"Nom: {dossier.get('nom_usager', 'N/A')}")
        # Ajoutez d'autres infos usager si disponibles

        st.markdown("**Assignations et Suivi**")
        assign_col1, assign_col2, assign_col3 = st.columns(3)
        with assign_col1:
            st.text(f"Agent Affecté: {dossier.get('agent_affecter', 'N/A')}")
        with assign_col2:
            st.text(f"Instructeur: {dossier.get('instructeur', 'N/A')}")
        with assign_col3:
             st.text(f"Valideur: {dossier.get('valideur', 'N/A')}")


        # Afficher les dates importantes restantes
        date_modif = dossier.get('derniere_modification', 'N/A')
        try:
             if isinstance(date_modif, date):
                  date_modif_formatee = date_modif.strftime('%d/%m/%Y %H:%M') # Ajouter l'heure si disponible
             elif isinstance(date_modif, str):
                  date_modif_formatee = date_modif
             else:
                 date_modif_formatee = "N/A"
        except Exception:
             date_modif_formatee = "N/A"
        st.caption(f"Dernière modification : {date_modif_formatee}")


        # --- Expander pour les détails complets (Tableau comme avant) ---
        with st.expander("Voir tous les champs bruts"):
            try:
                # Filtrer pour éviter les objets non sérialisables si besoin
                displayable_dossier = {k: v for k, v in dossier.items() if isinstance(v, (str, int, float, bool, date))} # Adapter les types si nécessaire
                df_dossier = pd.DataFrame.from_dict(displayable_dossier, orient='index', columns=['Valeur'])
                df_dossier.index.name = "Champ"
                st.dataframe(df_dossier, use_container_width=True)
            except Exception as e:
                logger.error(f"Erreur lors de la création du DataFrame pour les détails du dossier {numero_dossier}: {e}")
                st.warning("Impossible d'afficher les détails bruts sous forme de tableau.")
                # st.json(dossier) # Fallback en JSON

        # --- Bouton pour la Timeline (Étape 5) ---
        if st.button(f"🕒 Voir l'historique du dossier {numero_dossier}", key=f"timeline_btn_{numero_dossier}_{index}"):
            display_dossier_timeline(numero_dossier) # Fonction à créer à l'étape 5

        st.markdown("---") # Séparateur léger entre les dossiers
def display_dossier_details_enhanced(dossier: Dict, index: int):
    """Affiche les détails d'un dossier de manière plus structurée et visuelle."""
    numero_dossier = dossier.get('Numero', f'Inconnu_{index}')
    container_key = f"dossier_container_{numero_dossier}_{index}" # Clé unique pour le conteneur

    with st.container(border=True): # Utiliser une bordure pour séparer visuellement
        st.subheader(f"📁 Dossier : {dossier.get('Numero', 'N/A')}")

        col1, col2, col3 = st.columns(3)
        with col1:
            # Utiliser st.metric pour les infos clés
            st.metric(label="Statut", value=dossier.get('statut', 'N/A'))
        with col2:
            # Formatter le montant en euros
            montant = dossier.get('montant', 0)
            try:
                 montant_formate = f"{float(montant):,.2f} €".replace(",", " ").replace(".", ",") if montant else "N/A"
            except (ValueError, TypeError):
                 montant_formate = "N/A"
            st.metric(label="Montant Demandé", value=montant_formate)
        with col3:
             # Utiliser st.metric pour la date importante
             date_crea = dossier.get('date_creation', 'N/A')
             # Essayer de formater la date si elle existe
             try:
                 if isinstance(date_crea, date):
                      date_crea_formatee = date_crea.strftime('%d/%m/%Y')
                 elif isinstance(date_crea, str): # Si c'est une chaîne, on essaie de la garder
                      date_crea_formatee = date_crea
                 else:
                      date_crea_formatee = "N/A"
             except Exception:
                  date_crea_formatee = "N/A"

             st.metric(label="Date Création", value=date_crea_formatee)


        st.markdown("**Informations Usager**")
        st.text(f"Nom: {dossier.get('nom_usager', 'N/A')}")
        # Ajoutez d'autres infos usager si disponibles

        st.markdown("**Assignations et Suivi**")
        assign_col1, assign_col2, assign_col3 = st.columns(3)
        with assign_col1:
            st.text(f"Agent Affecté: {dossier.get('agent_affecter', 'N/A')}")
        with assign_col2:
            st.text(f"Instructeur: {dossier.get('instructeur', 'N/A')}")
        with assign_col3:
             st.text(f"Valideur: {dossier.get('valideur', 'N/A')}")


        # Afficher les dates importantes restantes
        date_modif = dossier.get('derniere_modification', 'N/A')
        try:
             if isinstance(date_modif, date):
                  date_modif_formatee = date_modif.strftime('%d/%m/%Y %H:%M') # Ajouter l'heure si disponible
             elif isinstance(date_modif, str):
                  date_modif_formatee = date_modif
             else:
                 date_modif_formatee = "N/A"
        except Exception:
             date_modif_formatee = "N/A"
        st.caption(f"Dernière modification : {date_modif_formatee}")


        # --- Expander pour les détails complets (Tableau comme avant) ---
        with st.expander("Voir tous les champs bruts"):
            try:
                # Filtrer pour éviter les objets non sérialisables si besoin
                displayable_dossier = {k: v for k, v in dossier.items() if isinstance(v, (str, int, float, bool, date))} # Adapter les types si nécessaire
                df_dossier = pd.DataFrame.from_dict(displayable_dossier, orient='index', columns=['Valeur'])
                df_dossier.index.name = "Champ"
                st.dataframe(df_dossier, use_container_width=True)
            except Exception as e:
                logger.error(f"Erreur lors de la création du DataFrame pour les détails du dossier {numero_dossier}: {e}")
                st.warning("Impossible d'afficher les détails bruts sous forme de tableau.")
                # st.json(dossier) # Fallback en JSON

        # --- Bouton pour la Timeline (Étape 5) ---
        if st.button(f"🕒 Voir l'historique du dossier {numero_dossier}", key=f"timeline_btn_{numero_dossier}_{index}"):
            display_dossier_timeline(numero_dossier) # Fonction à créer à l'étape 5

        st.markdown("---") # Séparateur léger entre les dossiers

def save_chat_history():
    """Sauvegarde l'historique de conversation dans un fichier JSON."""
    try:
        history_dir = "./data/chat_history"
        os.makedirs(history_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(history_dir, f"chat_{timestamp}.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)

        logger.info(f"Historique de conversation sauvegardé dans {filename}")
        return True, filename
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de l'historique : {e}", exc_info=True)
        return False, None

def load_chat_history(filename: str) -> bool:
    """Charge un historique de conversation depuis un fichier JSON."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            st.session_state.chat_history = json.load(f)
        logger.info(f"Historique chargé depuis {filename}")

        st.session_state.last_result = None
        return True
    except FileNotFoundError:
        logger.error(f"Fichier historique non trouvé: {filename}")
        st.session_state.error_message = f"Le fichier {filename} n'a pas été trouvé."
        return False
    except json.JSONDecodeError:
        logger.error(f"Erreur de décodage JSON dans le fichier: {filename}")
        st.session_state.error_message = f"Le fichier {filename} est corrompu ou mal formaté."
        return False
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'historique : {e}", exc_info=True)
        st.session_state.error_message = f"Une erreur inattendue s'est produite lors du chargement: {e}"
        return False

#Fonction pour traiter la question de l'instructeur 
def process_user_query(query: str) -> str:
    """ 
    Traite la question de l'utilisateur en utilisant le système RAG.
    
    Args:
        query (str): La question posée par l'utilisateur
        
    Returns:
        str: La réponse générée par le système
    """
    if not st.session_state.initialized:
        return "Le système n'est pas encore initialisé. Veuillez initialiser le système RAG depuis la barre latérale."
    
    try:
        # Marquer comme en cours de traitement 
        st.session_state.is_processing = True 
        
        # Obtenir l'objet graph du RAG 
        graph = st.session_state.rag_components.get("graph")
        if not graph:
            return "Le graph RAG n'est pas initialisé correctement. Vérifiez la configuration du système."
          
        enhanced_query = query  # On utilise directement la requête de l'utilisateur
    
        # État initial pour le graph
        initial_state = {
            "question": enhanced_query,
            "context": [],
            "db_results": [],
            "answer": "",
            "history": st.session_state.chat_history[-5:] if len(st.session_state.chat_history) > 0 else []
        }
    
        # Exécuter le graphe avec timeout pour éviter les blocages
        start_time = time.time()
        result = graph.invoke(initial_state)
        processing_time = time.time() - start_time  # temps de traitement 
    
        logger.info(f"Requête traitée en {processing_time:.2f} secondes")
        
        # Sauvegarder le résultat pour affichage
        st.session_state.last_result = result
        # Ici, on retourne simplement la réponse du système 
        return result["answer"]
    
    except Exception as e:
        error_msg = f"Une erreur s'est produite lors du traitement de votre demande : {str(e)}"
        logger.error(f"Erreur de traitement de la requête: {e}", exc_info=True)
        return error_msg
    
    finally:
        # Marquer le traitement comme terminé 
        st.session_state.is_processing = False 



def search_dossier(dossier_id: str) -> Optional[List[Dict]]:
    """
    Recherche un ou plusieurs dossiers spécifiques via la fonction du module RAG.

    Args:
        dossier_id (str): L'identifiant (ou terme de recherche) du dossier.

    Returns:
        Optional[List[Dict]]: Une liste de dictionnaires contenant les données des dossiers trouvés, ou None en cas d'erreur.
    """
    if not st.session_state.initialized or not rag_import_success:
        st.error("Le système RAG n'est pas initialisé. Veuillez l'activer depuis la barre latérale.")
        return None
    if not st.session_state.rag_components or 'rechercher_dossier' not in st.session_state.rag_components:
         logger.error("Fonction 'rechercher_dossier' manquante dans st.session_state.rag_components.")
         st.error("La fonction de recherche de dossier est manquante dans les composants RAG.")
         return None

    try:
        # --- Appel de la fonction de recherche ---
        search_function = st.session_state.rag_components.get("rechercher_dossier")
        if not callable(search_function):
             logger.error("Le composant 'rechercher_dossier' n'est pas une fonction exécutable.")
             st.error("Le composant 'rechercher_dossier' n'est pas une fonction exécutable.")
             return None
        # ---------------------------------------------

        logger.info(f"Recherche réelle du dossier avec ID/terme: {dossier_id}")
        results = search_function(dossier_id) 
        logger.info(f"Résultats de la recherche réelle pour '{dossier_id}': {len(results)} trouvé(s).")

        if not isinstance(results, list):
             logger.error(f"La fonction rechercher_dossier a retourné un type inattendu: {type(results)}")
             st.error("La recherche de dossier a retourné un format de données incorrect.")
             results = [] 

        st.session_state.dossier_search_results = results

        if not results:
            st.warning(f"Aucun dossier trouvé correspondant à '{dossier_id}'.")
            return []
        else:
             # Return the list of found dossiers
             return results

    except Exception as e:
        logger.error(f"Erreur lors de la recherche réelle du dossier '{dossier_id}': {e}", exc_info=True)
        st.error(f"Une erreur s'est produite lors de la recherche du dossier : {str(e)}")
        st.session_state.dossier_search_results = None
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


def display_dossier_summary(dossier: Dict):
    """Affiche un résumé compact d'un dossier."""
    st.markdown(f"""
    <div class="dossier-details-box">
        <h4>Dossier #{dossier.get('Numero', 'N/A')}</h4>
        <p>
            <strong>Usager:</strong> {dossier.get('nom_usager', 'N/A')} |
            <strong>Statut:</strong> {dossier.get('statut', 'N/A')} |
            <strong>Montant:</strong> {dossier.get('montant', 'N/A')} € |
            <strong>Création:</strong> {dossier.get('date_creation', 'N/A')}
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_dossier_details(dossier: Dict):
    """Affiche les détails complets d'un dossier dans un DataFrame."""
    st.markdown("---")
    st.subheader(f"Détails du Dossier #{dossier.get('Numero', 'N/A')}")
    
    # Convertir le dictionnaire du dossier en DataFrame pour une meilleure affichage (transposé)
    try:
        # Filtrer les valeurs scalaires pour créer le DataFrame
        displayable_dossier = {k: v for k, v in dossier.items() if isinstance(v, (str, int, float, bool))}
        df_dossier = pd.DataFrame.from_dict(displayable_dossier, orient='index', columns=['Valeur'])
        df_dossier.index.name = "Champ"
        st.dataframe(df_dossier, use_container_width=True)
    except Exception as e:
        logger.error(f"ERREUR lors de la création du DataFrame Pour les details du dossiers: {e}")
        st.warning("Impossible d'afficher les détails du dossier sous forme de tableau.")
        st.json(dossier)  # Affichage en JSON si la conversion échoue

    # Bouton pour poser une question sur le dossier
    # NOTE : La fonctionnalité de définir ce dossier comme "actif" a été supprimée.
    dossier_num = dossier.get('Numero', '')
    if st.button(f"❓ Poser une question sur ce dossier ({dossier_num})", key=f"ask_about_{dossier_num}"):
        # Vous pouvez choisir d'effectuer une action ici sans définir le dossier comme actif.
        # Par exemple, vous pouvez pré-remplir la zone de texte de la question ou effectuer une autre action.
        question = f"Peux-tu me donner plus de détails sur le dossier {dossier_num} ?"
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.rerun()  # Redémarrer pour traiter la question



# ===== INTERFACE PRINCIPALE =====

# Vérifier si l'initialisation a échoué et bloquer si nécessaire
if not st.session_state.initialized and rag_import_success:
     st.error("L'initialisation de l'assistant a échoué. Veuillez vérifier les logs ou contacter le support.")
     st.stop() # Arrêter l'exécution si l'initialisation est critique et a échoué
elif not rag_import_success:
     st.error("Le module RAG n'a pas pu être chargé. L'application ne peut pas démarrer.")
     st.stop()


# --- Barre Latérale  ---
with st.sidebar:
    st.image("logo_region_reunion.png", width=150) #  Logo
    st.header("⚙️ Contrôle Système")
    
with st.sidebar:
    st.title("Assistant KAP Numérique")
    # Afficher le statut du système
    status_color = "red"
    if st.session_state.initialized:
        status_color = "green"
    elif "Initialisation" in st.session_state.system_status_msg:
         status_color = "orange"
    st.markdown(f"**Statut:** <span style='color:{status_color};'>{st.session_state.system_status_msg}</span>", unsafe_allow_html=True)

    
    # Section Options d'Affichage
    with st.expander("👁️ Options d'Affichage", expanded=True):
        st.session_state.show_sources = st.toggle("Afficher les sources citées", value=st.session_state.show_sources, key="toggle_sources")
        st.session_state.show_db_results = st.toggle("Afficher les détails des dossiers trouvés", value=st.session_state.show_db_results, key="toggle_db_details")
        st.session_state.sources_expanded = st.toggle("Développer les sources par défaut", value=st.session_state.sources_expanded, key="toggle_expand_sources")

    # Section Gestion de l'Historique
    with st.expander("🕒 Gestion de l'Historique", expanded=False):
        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            if st.button("💾 Sauvegarder", use_container_width=True, key="save_hist"):
                success, filename = save_chat_history()
                if success:
                    st.success(f"Historique sauvegardé: `{filename}`")
                else:
                    st.error("Échec de la sauvegarde.")
        with col_hist2:
             if st.button("🗑️ Effacer", use_container_width=True, key="clear_hist"):
                if st.session_state.chat_history:
                    st.session_state.chat_history = []
                    st.session_state.last_result = None
                    st.session_state.active_dossier = None
                    st.success("Historique effacé.")
                    st.rerun()
                else:
                     st.info("L'historique est déjà vide.")

    # Section Aide
    with st.expander("❓ Aide", expanded=False):
        st.markdown("""
        **Comment utiliser l'assistant :**
        1.  Cliquez sur **Initialiser le Système RAG**.
        2.  Utilisez l'onglet **Questions Générales** pour des infos sur le dispositif.
        3.  Utilisez l'onglet **Consultation de Dossier** pour rechercher un dossier spécifique par son numéro et poser des questions dessus.
        4.  Le dossier recherché devient le **dossier actif** pour les questions suivantes dans cet onglet.

        **Exemples de questions :**
        * *Général:* "Quels sont les critères d'éligibilité ?"
        * *Dossier (après recherche):* "Quel est le statut actuel de ce dossier ?"
        * *Dossier (après recherche):* "Y a-t-il des documents manquants ?"
        """)

    # Affichage permanent des erreurs
    if st.session_state.error_message:
        st.error(f"🚨 Erreur: {st.session_state.error_message}")
        if st.button("❌ Effacer le message d'erreur", key="clear_error_btn"):
            st.session_state.error_message = None
            st.rerun()


# --- Zone Principale ---

st.title("🤖 Assistant KAP Numérique")
st.caption("Votre assistant intelligent pour le dispositif KAP Numérique.")

# Définir les onglets pour les modes
tab_general, tab_dossier = st.tabs(["💬 Questions Générales", "📁 Consultation de Dossier"])

# --- Onglet Questions Générales ---
with tab_general:
    st.header("Posez une question générale")
    st.markdown("Utilisez cette section pour des questions sur le fonctionnement général du dispositif, les procédures, les critères, etc.")

    # Indicateur si un dossier est actif

    # Conteneur pour l'historique du chat
    chat_container_general = st.container(height=500) 
    with chat_container_general:
        if not st.session_state.chat_history:
       
            welcome_msg = "👋 Bonjour ! Posez-moi une question sur le dispositif KAP Numérique."
            if not rag_import_success:
                 welcome_msg = "⚠️ Le module RAG n'a pas pu être chargé. Fonctionnalités limitées."
            elif not st.session_state.initialized:
                 welcome_msg = "👋 Bonjour ! Veuillez initialiser le système RAG depuis la barre latérale."
            st.info(welcome_msg)
        else:
       
            for message in st.session_state.chat_history:
                avatar = "👤" if message["role"] == "user" else "🤖"
                with st.chat_message(message["role"], avatar=avatar):
                    st.write(message["content"]) 

    # Affichage des sources et détails si applicable au dernier message 
    if st.session_state.last_result and isinstance(st.session_state.last_result, dict):

         if st.session_state.app_mode == "general":
             # Afficher les sources si activé
            sources = st.session_state.last_result.get('context', []) 
            
            #Affiche la liste des sources
            if st.session_state.show_sources and isinstance(sources, list) and sources:
                    with st.expander("📚 Sources consultées", expanded=st.session_state.sources_expanded):
                        context = st.session_state.last_result["context"]
                        if context:
                            st.markdown(f"### {len(context)} sources utilisées pour générer la réponse")
                            
                            #option d'affichage des sources
                            view_mode = st.radio("Mode d'affichage", ("Liste détailée"), horizontal=True)
                            
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

            # Afficher les détails des dossiers trouvés si activé 
            db_results = st.session_state.last_result.get('db_results', []) 
            if st.session_state.show_db_results and isinstance(db_results, list) and db_results:
                 with st.expander("📋 Dossiers Mentionnés", expanded=True):
                     st.caption("La dernière réponse a potentiellement utilisé des informations des dossiers suivants :")
                     for dossier in db_results:
                         if isinstance(dossier, dict):
                              display_dossier_summary(dossier)
                              dossier_num = dossier.get('Numero', '')
                  
                              if st.button(f"🔍 Consulter ce dossier (#{dossier_num})", key=f"switch_to_{dossier_num}"):
                                  st.session_state.active_dossier = dossier
                                  st.session_state.app_mode = "dossier"
                                  st.info(f"Dossier #{dossier_num} défini comme actif. Allez à l'onglet 'Consultation de Dossier' pour interagir.")
                                  time.sleep(2)
                                  st.rerun()
                         else:
                              st.warning(f"Format de résultat de dossier inattendu: {type(dossier)}")


        
# --- Onglet Consultation de Dossier ---
# --- Onglet Consultation de Dossier ---
with tab_dossier:
    st.header("Rechercher et consulter un dossier spécifique")
    st.markdown("Utilisez les filtres et/ou la barre de recherche pour trouver des dossiers.")

    # --- Section des Filtres ---
    with st.expander("🔍 Afficher les Filtres", expanded=False): # Commence fermé
        # Récupérer les composants RAG pour accéder aux fonctions DB
        rag_components = st.session_state.get('rag_components', {})
        get_distinct_values_func = rag_components.get('get_distinct_values')

        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            # Filtre Statut
            statuts_options = ["Tous"] # Option par défaut
            if get_distinct_values_func:
                try:
                    distinct_statuts = get_distinct_values_func('statut')
                    statuts_options.extend(distinct_statuts)
                except Exception as e:
                    logger.warning(f"Impossible de charger les statuts distincts: {e}")
                    st.warning("Impossible de charger la liste des statuts.")
            # Utiliser st.session_state pour garder la valeur sélectionnée
            st.session_state.filter_statut = st.selectbox(
                "Filtrer par Statut",
                options=statuts_options,
                index=0, # "Tous" par défaut
                key="filter_statut_selectbox" # Clé unique
            )

            # Filtre Date de Création (Début)
            st.session_state.filter_date_debut = st.date_input(
                "Dossiers créés après le",
                value=None, # Pas de date par défaut
                key="filter_date_debut_input"
            )

        with col_filter2:
            # Filtre Instructeur
            instructeurs_options = ["Tous"]
            if get_distinct_values_func:
                try:
                    distinct_instructeurs = get_distinct_values_func('instructeur')
                    instructeurs_options.extend(distinct_instructeurs)
                except Exception as e:
                     logger.warning(f"Impossible de charger les instructeurs distincts: {e}")
                     st.warning("Impossible de charger la liste des instructeurs.")
            st.session_state.filter_instructeur = st.selectbox(
                "Filtrer par Instructeur",
                options=instructeurs_options,
                index=0, # "Tous" par défaut
                key="filter_instructeur_selectbox"
            )

            # Filtre Date de Création (Fin)
            st.session_state.filter_date_fin = st.date_input(
                "Dossiers créés avant le",
                value=None, # date.today() pourrait être une option
                key="filter_date_fin_input"
            )

    # --- Zone de Recherche (Texte) ---
    search_col, btn_col = st.columns([4, 1])
    with search_col:
        dossier_search_input = st.text_input(
            "Rechercher par N° de dossier ou Nom usager...", # Placeholder mis à jour
            key="dossier_search_input_fuzzy", # Clé potentiellement nouvelle ou mise à jour
            placeholder="Entrez un numéro exact (XX-YYYY) ou un terme...",
            label_visibility="collapsed"
        )
    with btn_col:
        search_button = st.button(
            "🔎 Rechercher",
            key="search_dossier_button_combined", # Clé potentiellement nouvelle ou mise à jour
            use_container_width=True,
            # Désactivé si RAG non prêt OU si l'input ET les filtres sont vides (optionnel)
            disabled=not st.session_state.initialized
        )

    # --- Logique de Recherche et Affichage ---
    if search_button: # Se déclenche quand le bouton est cliqué
        # Récupérer les valeurs des filtres depuis session_state
        term = dossier_search_input # Le terme de la barre de recherche
        statut_filter = st.session_state.filter_statut if st.session_state.filter_statut != "Tous" else None
        instructeur_filter = st.session_state.filter_instructeur if st.session_state.filter_instructeur != "Tous" else None
        date_debut_filter = st.session_state.filter_date_debut
        date_fin_filter = st.session_state.filter_date_fin

        # Vérifier si au moins un critère est fourni (optionnel mais évite recherche vide)
        if not term and not statut_filter and not instructeur_filter and not date_debut_filter and not date_fin_filter:
             st.warning("Veuillez entrer un terme de recherche ou sélectionner au moins un filtre.")
             st.session_state.dossier_search_results = [] # Vider les résultats précédents
        else:
             # Appel de la fonction de recherche Backend MISE À JOUR
             search_function = st.session_state.rag_components.get("rechercher_dossier")
             if callable(search_function):
                  with st.spinner(f"Recherche des dossiers..."):
                     try:
                        # Passer les arguments nommés mis à jour
                        found_dossiers = search_function(
                            search_term=term if term else None,
                            statut=statut_filter,
                            instructeur=instructeur_filter,
                            date_debut_creation=date_debut_filter,
                            date_fin_creation=date_fin_filter
                        )
                        st.session_state.dossier_search_results = found_dossiers # Stocker les résultats
                        if found_dossiers:
                            st.success(f"{len(found_dossiers)} dossier(s) trouvé(s).")
                        else:
                            st.info("Aucun dossier ne correspond à vos critères de recherche.")
                     except Exception as e:
                         logger.error(f"Erreur lors de l'appel à rechercher_dossier depuis Streamlit: {e}", exc_info=True)
                         st.error(f"Une erreur est survenue pendant la recherche: {e}")
                         st.session_state.dossier_search_results = None
             else:
                  st.error("La fonction de recherche de dossier n'est pas disponible.")
                  st.session_state.dossier_search_results = None

    # --- Affichage Amélioré des Résultats (toujours visible si des résultats existent en session_state) ---
    if 'dossier_search_results' in st.session_state and st.session_state.dossier_search_results is not None:
         results_to_display = st.session_state.dossier_search_results
         if not results_to_display:
             # Afficher le message "Aucun dossier..." seulement si une recherche a été effectuée (bouton cliqué)
             if search_button: # Ou une autre variable pour savoir si une recherche a été tentée
                  st.info("Aucun dossier ne correspond à vos critères de recherche.")
             # else: # Si aucune recherche n'a encore été faite, ne rien afficher ou un message par défaut
             #    st.info("Utilisez les filtres ou la barre de recherche ci-dessus.")
         else:
             st.markdown(f"--- \n### {len(results_to_display)} Dossier(s) Trouvé(s)")
             for index, dossier in enumerate(results_to_display):
                 if isinstance(dossier, dict):
                     # Utiliser la fonction d'affichage améliorée (Étape 4)
                     display_dossier_details_enhanced(dossier, index) # Nouvelle fonction à créer
                 else:
                     st.warning(f"Format de résultat de dossier inattendu à l'index {index}: {type(dossier)}")
                     st.json(dossier) # Afficher le JSON brut en cas de problème

# --- Footer ---
st.markdown("""
<div class="footer">
    © 2025 KAP Numérique - Assistant RAG v1.0
    <a href="mailto:support@kap-numerique.fr" target="_blank">Assistance</a>
</div>
""", unsafe_allow_html=True)
