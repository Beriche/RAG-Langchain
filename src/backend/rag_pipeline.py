# src/backend/rag_pipeline.py

import os
import re
import logging
import json
from datetime import date
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore # Ajout pour type hinting
from langgraph.graph import StateGraph, END, START
from functools import partial # IMPORTANT: Importer partial

# PAS D'IMPORT DE .rag_orchestrator ICI

logger = logging.getLogger(__name__)

class State(Dict):
    """Structure pour représenter l'état du système RAG."""
    question: str
    context: List[Document]
    db_results: List[Dict[str, Any]]
    answer: str

def extract_dossier_number(question: str) -> List[str]:
    # ... (inchangé) ...
    pattern = r'\b(\d{2}[-\s]\d{2,4})\b'
    matches = re.findall(pattern, question, re.IGNORECASE)
    results = []
    seen = set()
    for match in matches:
        normalized_match = re.sub(r'\s', '-', match).strip()
        if normalized_match not in seen:
            results.append(normalized_match)
            seen.add(normalized_match)
    if results:
         logger.info(f"Numéros de dossier potentiels extraits: {results}")
    else:
         logger.debug(f"Aucun numéro de dossier trouvé dans: '{question[:100]}...'")
    return results


def db_resultats_to_documents(resultats: List[Dict[str, Any]]) -> List[Document]:
    # ... (inchangé) ...
    documents = []
    if not resultats:
        return documents
    logger.info(f"Conversion de {len(resultats)} résultat(s) BDD en Documents Langchain.")
    for i, resultat in enumerate(resultats):
        content_parts = [f"**Informations Dossier BDD {resultat.get('Numero', 'N/A')} (Résultat {i+1})**"]
        for key, value in resultat.items():
             value_str = value.strftime('%d/%m/%Y') if isinstance(value, date) else (str(value) if value is not None else 'N/A')
             formatted_key = key.replace('_', ' ').capitalize()
             content_parts.append(f"- {formatted_key}: {value_str}")
        content = "\n".join(content_parts)
        metadata = {
            "source": "base_de_donnees", "category": "dossier_bdd", "type": "db_data",
            "numero_dossier": resultat.get('Numero', 'N/A'), "db_result_index": i + 1,
            "update_date": resultat.get('derniere_modification', None)
        }
        metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
        doc = Document(page_content=content.strip(), metadata=metadata)
        documents.append(doc)
    return documents

# --- Noeuds du Graphe (modifiés pour accepter les dépendances) ---
def search_database_node(state: State, db_man_obj: Optional[Any], db_conn_status_val: bool) -> Dict[str, Any]:
    """Noeud: Recherche dans la BDD."""
    logger.info("Noeud: search_database")
    question = state["question"]
    dossier_numbers = extract_dossier_number(question)
    db_results = []

    if not db_conn_status_val or not db_man_obj:
        logger.warning("Recherche BDD annulée: connexion/manager non disponible via dépendances.")
        return {"db_results": []}

    if dossier_numbers:
        num_to_search = dossier_numbers[0]
        logger.info(f"Tentative de recherche BDD pour le numéro: {num_to_search}")
        try:
            # Utilise db_man_obj (qui sera l'instance de DatabaseManager)
            db_results = db_man_obj.rechercher_dossier(numero_dossier=num_to_search)
            logger.info(f"{len(db_results)} résultats BDD trouvés pour {num_to_search}.")
        except Exception as e:
            logger.error(f"Erreur pendant l'appel à rechercher_dossier: {e}", exc_info=True)
            db_results = []
    else:
         logger.info("Aucun numéro de dossier détecté dans la question pour la recherche BDD.")
    return {"db_results": db_results}

def retrieve_node(
    state: State,
    rules_vs: Optional[VectorStore],
    official_vs: Optional[VectorStore],
    echanges_vs: Optional[VectorStore]
) -> Dict[str, Any]:
    """Noeud: Récupère documents depuis BDD (via state) et vector stores (via dépendances)."""
    logger.info("Noeud: retrieve")
    question = state["question"]
    db_docs = db_resultats_to_documents(state.get("db_results", []))
    
    relevant_rules, relevant_official_docs, relevant_echanges = [], [], []

    if rules_vs:
        try:
            relevant_rules = rules_vs.similarity_search(question, k=3)
            logger.info(f"Retrieve - {len(relevant_rules)} règles.")
        except Exception as e: logger.error(f"Erreur VS règles: {e}", exc_info=True)
    else: logger.warning("rules_vector_store (dépendance) non disponible.")

    if official_vs:
        try:
            relevant_official_docs = official_vs.similarity_search(question, k=5)
            logger.info(f"Retrieve - {len(relevant_official_docs)} docs officiels.")
        except Exception as e: logger.error(f"Erreur VS officiels: {e}", exc_info=True)
    else: logger.warning("official_docs_vector_store (dépendance) non disponible.")

    if echanges_vs:
        try:
            relevant_echanges = echanges_vs.similarity_search(question, k=3)
            logger.info(f"Retrieve - {len(relevant_echanges)} échanges.")
        except Exception as e: logger.error(f"Erreur VS échanges: {e}", exc_info=True)
    else: logger.warning("echanges_vector_store (dépendance) non disponible.")

    combined_docs = db_docs + relevant_rules + relevant_official_docs + relevant_echanges
    logger.info(f"Retrieve - Combinés avant dédup: {len(combined_docs)}")
    
    seen_content = set()
    unique_docs = []
    for doc in combined_docs:
        dedup_key = (doc.page_content, doc.metadata.get("source_file", doc.metadata.get("source")))
        if dedup_key not in seen_content:
            unique_docs.append(doc)
            seen_content.add(dedup_key)
    if len(unique_docs) < len(combined_docs):
         logger.info(f"Retrieve - Après dédup: {len(unique_docs)}")
    return {"context": unique_docs}

def generate_node(state: State, llm_model: Optional[Any]) -> Dict[str, Any]:
    """Noeud: Génère la réponse avec le LLM (via dépendance)."""
    logger.info("Noeud: generate")
    question = state["question"]
    context_docs = state.get("context", [])

    if llm_model is None:
        logger.error("Génération impossible: LLM (dépendance) non initialisé.")
        return {"answer": "Erreur: Service de génération non disponible."}

    # ... (le reste de la logique de generate est inchangé, sauf qu'elle utilise llm_model) ...
    if not context_docs:
        logger.warning("Génération: Aucun contexte fourni.")
        db_results_in_state = state.get("db_results", [])
        if db_results_in_state:
             db_docs_only = db_resultats_to_documents(db_results_in_state)
             db_content_only = "\n\n".join([doc.page_content for doc in db_docs_only])
             return {"answer": f"Infos BDD:\n\n{db_content_only}\n\nPas d'infos complémentaires."}
        return {"answer": "Désolé, aucune information pertinente trouvée."}

    rules_content, db_content, official_content, echanges_content, other_content = [], [], [], [], []
    docs_details_list = []
    source_counter = 1
    for doc in context_docs:
        category = doc.metadata.get("category", "inconnu")
        doc_type = doc.metadata.get("type", "inconnu")
        source_file = doc.metadata.get("source_file", os.path.basename(doc.metadata.get("source", "Source inconnue")))
        source_id = f"SOURCE {source_counter}"
        content_with_id = f"[{source_id}]\n{doc.page_content.strip()}"
        docs_details_list.append({"id": source_id, "file": source_file, "category": category, "type": doc_type})
        if category == "regles" or doc_type == "rule_document": rules_content.append(content_with_id)
        elif category == "dossier_bdd" or doc_type == "db_data": db_content.append(content_with_id)
        elif category == "docs_officiels" or doc_type == "knowledge_document": official_content.append(content_with_id)
        elif category == "echanges" or doc_type == "echange_document": echanges_content.append(content_with_id)
        else: other_content.append(f"[{source_id} - Cat: {category}]\n{doc.page_content.strip()}")
        source_counter += 1
    
    context_sections = []
    if db_content: context_sections.append("**INFOS DOSSIER BDD (Priorité 1):**\n" + "\n\n---\n\n".join(db_content))
    if rules_content: context_sections.append("**RÈGLES (Priorité 2):**\n" + "\n\n---\n\n".join(rules_content))
    if official_content: context_sections.append("**DOCS OFFICIELS (Priorité 3):**\n" + "\n\n---\n\n".join(official_content))
    if echanges_content: context_sections.append("**ÉCHANGES (Style/Ton - Priorité 4):**\n" + "\n\n---\n\n".join(echanges_content))
    if other_content: context_sections.append("**AUTRES (Priorité 5):**\n" + "\n\n---\n\n".join(other_content))
    context_string_for_prompt = "\n\n".join(context_sections)
    formatted_sources_list = "\n".join([f"- [{d['id']}] {d['file']} (Cat: {d['category']}, Type: {d['type']})" for d in docs_details_list])

    system_instructions = (
        "Tu es un assistant expert KAP Numérique pour agents instructeurs. "
        "Réponds précisément basé EXCLUSIVEMENT sur le contexte fourni, en respectant la hiérarchie des sources (BDD > Règles > Officiels > Echanges pour style > Autres). "
        "Cite tes sources avec `[SOURCE X]`. Si info manquante, dis-le. Format Markdown. Ne t'identifie pas comme IA."
    )
    user_prompt = (
        f"**Question Agent:**\n{question}\n\n"
        f"**Contexte Fourni:**\n---\n{context_string_for_prompt}\n---\n\n"
        f"**Liste Sources:**\n{formatted_sources_list}\n\n"
        f"**Réponse (Markdown, justifiée avec [SOURCE X]):**"
    )
    logger.info(f"Génération - Prompt LLM (longueur approx: {len(user_prompt)} chars).")
    try:
        messages = [{"role": "system", "content": system_instructions}, {"role": "user", "content": user_prompt}]
        # Utilise llm_model passé en argument
        response = llm_model.invoke(messages, temperature=0.1, max_tokens=2000)
        final_answer = response.content
        logger.info("Génération - Réponse LLM reçue.")
        return {"answer": final_answer}
    except Exception as e:
        logger.error(f"Génération - Erreur LLM: {e}", exc_info=True)
        return {"answer": f"Erreur technique génération. Détails: {e}"}


def build_graph_with_deps(
    db_man_dep: Optional[Any], # Peut être DatabaseManager ou None
    db_conn_status_dep: bool,
    llm_dep: Optional[Any],     # Peut être ChatOpenAI ou None
    rules_vs_dep: Optional[VectorStore],
    official_vs_dep: Optional[VectorStore],
    echanges_vs_dep: Optional[VectorStore]
) -> StateGraph: # StateGraph retourné, pas workflow.compile() directement ici
    """Construit et compile le graphe LangGraph en liant les dépendances."""
    logger.info("Construction du graphe RAG avec dépendances...")
    workflow = StateGraph(State)

    # Lier les dépendances aux fonctions des noeuds en utilisant functools.partial
    bound_search_database = partial(search_database_node, db_man_obj=db_man_dep, db_conn_status_val=db_conn_status_dep)
    bound_retrieve = partial(retrieve_node, rules_vs=rules_vs_dep, official_vs=official_vs_dep, echanges_vs=echanges_vs_dep)
    bound_generate = partial(generate_node, llm_model=llm_dep)

    workflow.add_node("search_database", bound_search_database)
    workflow.add_node("retrieve", bound_retrieve)
    workflow.add_node("generate", bound_generate)

    workflow.add_edge(START, "search_database")
    workflow.add_edge("search_database", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    try:
        graph = workflow.compile()
        logger.info("Graphe RAG compilé avec succès (avec dépendances).")
        return graph
    except Exception as e:
        logger.error(f"Erreur critique lors de la compilation du graphe (avec dépendances): {e}", exc_info=True)
        raise