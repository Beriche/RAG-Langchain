import streamlit as st
import pandas as pd
from datetime import date
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)
# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def get_dossier_data_for_timeline(numero_dossier: str) -> Optional[Dict]:
    """
    Récupère les données d'un dossier spécifique pour la timeline.
    Utilise les résultats déjà chargés si possible, sinon refait une recherche.
    """
    if 'dossier_search_results' in st.session_state and st.session_state.dossier_search_results:
        for dossier in st.session_state.dossier_search_results:
            if dossier.get('Numero') == numero_dossier:
                return dossier

    logger.info(f"Données non trouvées en cache local pour {numero_dossier} (timeline), recherche en base...")
    if st.session_state.rag_components and callable(st.session_state.rag_components.get("search_function")):
        search_function = st.session_state.rag_components.get("search_function")
        try:
            results = search_function(numero_dossier=numero_dossier) 
            if results:
                return results[0]
            else:
                logger.warning(f"Impossible de retrouver le dossier {numero_dossier} pour la timeline.")
                return None
        except Exception as e:
            logger.error(f"Erreur en rechargeant le dossier {numero_dossier} pour timeline: {e}")
            return None
    else:
        logger.error("Fonction de recherche de dossier non disponible dans rag_components pour la timeline.")
        return None


def display_dossier_timeline(numero_dossier: str):
    """Affiche une timeline simplifiée pour un dossier."""
    st.markdown(f"#### Historique Simplifié du Dossier {numero_dossier}")
    dossier_data = get_dossier_data_for_timeline(numero_dossier)

    if not dossier_data:
        st.warning("Impossible de charger les données du dossier pour afficher l'historique.")
        return

    events = []
    date_creation = dossier_data.get('date_creation')
    if date_creation:
         try:
            date_str = date_creation.strftime('%d %B %Y') if isinstance(date_creation, date) else str(date_creation)
            events.append({
                "date": date_str, "event": "📅 Création du dossier",
                "details": f"Dossier initié pour {dossier_data.get('nom_usager', 'N/A')}."
            })
         except AttributeError:
             events.append({"date": str(date_creation), "event": "📅 Création du dossier", "details": "Date création enregistrée."})

    date_modif = dossier_data.get('derniere_modification')
    if date_modif and date_modif != date_creation :
         try:
            date_str = date_modif.strftime('%d %B %Y à %H:%M') if hasattr(date_modif, 'strftime') else str(date_modif)
            events.append({
                "date": date_str, "event": "✍️ Dernière Modification",
                "details": f"Statut: {dossier_data.get('statut', 'N/A')}. Instructeur: {dossier_data.get('instructeur', 'N/A')}."
            })
         except Exception as e:
              logger.error(f"Erreur formatage date modif {date_modif}: {e}")
              events.append({"date": str(date_modif), "event": "✍️ Dernière Modification", "details": f"Statut: {dossier_data.get('statut', 'N/A')}"})

    if events:
        st.markdown("**Événements Clés :**")
        for event in events:
            st.markdown(f"- **{event['date']} :** {event['event']}")
            st.caption(f"  > {event['details']}")
    else:
        st.info("Aucun événement historique majeur à afficher (données disponibles).")


def display_dossier_details_enhanced(dossier: Dict, index: int):
    """Affiche les détails d'un dossier de manière plus structurée et visuelle."""
    numero_dossier = dossier.get('Numero', f'Inconnu_{index}')
    with st.container(border=True):
        st.subheader(f"📁 Dossier : {dossier.get('Numero', 'N/A')}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Statut", value=dossier.get('statut', 'N/A'))
        with col2:
            montant = dossier.get('montant', 0)
            montant_formate = f"{float(montant):,.2f} €".replace(",", " ").replace(".", ",") if montant else "N/A"
            st.metric(label="Montant Demandé", value=montant_formate)
        with col3:
             date_crea = dossier.get('date_creation', 'N/A')
             date_crea_formatee = date_crea.strftime('%d/%m/%Y') if isinstance(date_crea, date) else str(date_crea)
             st.metric(label="Date Création", value=date_crea_formatee)

        st.markdown("**Informations Usager**")
        st.text(f"Nom: {dossier.get('nom_usager', 'N/A')}")

        st.markdown("**Assignations et Suivi**")
        assign_col1, assign_col2, assign_col3 = st.columns(3)
        with assign_col1: st.text(f"Agent Affecté: {dossier.get('agent_affecter', 'N/A')}")
        with assign_col2: st.text(f"Instructeur: {dossier.get('instructeur', 'N/A')}")
        with assign_col3: st.text(f"Valideur: {dossier.get('valideur', 'N/A')}")

        date_modif = dossier.get('derniere_modification', 'N/A')
        date_modif_formatee = date_modif.strftime('%d/%m/%Y %H:%M') if hasattr(date_modif, 'strftime') else str(date_modif)
        st.caption(f"Dernière modification : {date_modif_formatee}")

        if st.button(f"🕒 Voir l'historique du dossier {numero_dossier}", key=f"timeline_btn_{numero_dossier}_{index}"):
            display_dossier_timeline(numero_dossier)
        st.markdown("---")


def display_source(source_doc: Any, index: int, expanded_by_default: bool = False):
    """ Affiche une source de manière formatée. """
    try:
        source_name = source_doc.metadata.get('source', 'Document interne')
        st.markdown(f"**Source {index + 1}:** *{source_name}*")

        if source_doc.metadata:
            metadata_str = ", ".join([
                f"{k}: {v}" for k, v in source_doc.metadata.items()
                if k not in ['source', 'content', 'text'] and v
            ])
            if metadata_str:
                st.caption(metadata_str)

        content = source_doc.page_content
        if len(content) > 500:
           
            with st.expander("Voir le contenu complet", expanded=expanded_by_default):
                st.markdown(f'<div class="source-box">{content}</div>', unsafe_allow_html=True)
            st.text(content[:500] + "...") # Aperçu toujours visible en dehors de l'expander
        else:
            st.markdown(f'<div class="source-box">{content}</div>', unsafe_allow_html=True)
        st.markdown("---")
    except AttributeError:
        logger.error(f"L'objet source {index+1} n'a pas les attributs attendus (metadata, page_content). Type: {type(source_doc)}")
        st.warning(f"Impossible d'afficher correctement la source {index+1}.")

def display_dossier_summary(dossier: Dict):
    """Affiche un résumé compact d'un dossier."""
    date_creation_val = dossier.get('date_creation', 'N/A')
    if isinstance(date_creation_val, date):
        date_creation_str = date_creation_val.strftime('%d/%m/%Y')
    else:
        date_creation_str = str(date_creation_val)
    
    montant_val = dossier.get('montant', 'N/A')
    montant_str = f"{float(montant_val):,.2f}".replace(",", " ").replace(".", ",") if isinstance(montant_val, (int, float)) else str(montant_val)


    st.markdown(f"""
    <div class="dossier-details-box">
        <h4>Dossier #{dossier.get('Numero', 'N/A')}</h4>
        <p>
            <strong>Usager:</strong> {dossier.get('nom_usager', 'N/A')} |
            <strong>Statut:</strong> {dossier.get('statut', 'N/A')} |
            <strong>Montant:</strong> {montant_str} € |
            <strong>Création:</strong> {date_creation_str}
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_dossier_details_df(dossier: Dict):
    """Affiche les détails complets d'un dossier dans un DataFrame."""
    st.markdown("---")
    st.subheader(f"Détails du Dossier #{dossier.get('Numero', 'N/A')}")
    try:
        displayable_dossier = {k: v for k, v in dossier.items() if isinstance(v, (str, int, float, bool, date))}
        for key, value in displayable_dossier.items():
            if isinstance(value, date):
                displayable_dossier[key] = value.strftime('%d/%m/%Y')

        df_dossier = pd.DataFrame.from_dict(displayable_dossier, orient='index', columns=['Valeur'])
        df_dossier.index.name = "Champ"
        st.dataframe(df_dossier, use_container_width=True)
    except Exception as e:
        logger.error(f"Erreur création DataFrame détails dossier: {e}")
        st.warning("Impossible d'afficher les détails du dossier en tableau.")
        st.json(dossier)

    dossier_num = dossier.get('Numero', '')
    if st.button(f"❓ Poser une question sur ce dossier ({dossier_num})", key=f"ask_about_{dossier_num}_df"):
        question = f"Peux-tu me donner plus de détails sur le dossier {dossier_num} ?"
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.rerun()