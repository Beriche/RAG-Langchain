import os
import re
import logging
from datetime import date
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langgraph.graph import StateGraph, END, START
from functools import partial
from langchain.retrievers import EnsembleRetriever # Ajout pour EnsembleRetriever
from rank_bm25 import BM25Okapi # Pour type hinting
import concurrent.futures # Ajout pour la parallélisation

# Import des utilitaires BM25 et du retriever personnalisé
from .bm25_utils import BM25Retriever 

# Type hint pour les composants BM25 (BM25Okapi, List[Document])
BM25Components = Optional[Tuple[BM25Okapi, List[Document]]]


logger = logging.getLogger(__name__)

class State(Dict):
    """Structure pour représenter l'état du système RAG."""
    question: str
    context: List[Document]
    db_results: List[Dict[str, Any]]
    answer: str
    history: List[Dict[str, str]]

#Extraction du numéro de dossier depuis le prompt user s'il en existe
def extract_dossier_number(question: str) -> List[str]:
    """
     Extrait les numéros de dossier (formats XX-XX, XX-XXX, XX-XXXX avec
     séparateur '-' ou ' ') de la question. 
    """
   
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
    """
       Convertit les résultats de la base de données (liste de dictionnaires) en Documents Langchain.
    """

    documents = []
    if not resultats:
        return documents
    logger.info(f"Conversion de {len(resultats)} résultat(s) BDD en Documents Langchain.")
    for i, resultat in enumerate(resultats):
        # Formatage du contenu pour lisibilité
        content_parts = [f"**Informations Dossier BDD {resultat.get('Numero', 'N/A')} (Résultat {i+1})**"]
        for key, value in resultat.items():
             # Formate les dates si présentes
             value_str = value.strftime('%d/%m/%Y') if isinstance(value, date) else (str(value) if value is not None else 'N/A')
            # Formater la ligne (ex: Nom Usager: Dupont)
             formatted_key = key.replace('_', ' ').capitalize()
             content_parts.append(f"- {formatted_key}: {value_str}")
        content = "\n".join(content_parts)
        
        # Création des métadonnées spécifiques à la BDD
        metadata = {
            "source": "base_de_donnees", "category": "dossier_bdd", "type": "db_data",
            "numero_dossier": resultat.get('Numero', 'N/A'), "db_result_index": i + 1,
            "update_date": resultat.get('derniere_modification', None)
        }
        metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
        doc = Document(page_content=content.strip(), metadata=metadata)
        documents.append(doc)
    return documents


def search_database(state: State, db_manager: Optional[Any], db_conn_status_val: bool) -> Dict[str, Any]:
    """Noeud: Recherche dans la BDD sur les numéros extraits."""
    logger.info("Noeud: search_database")
    
    question = state["question"]
    dossier_numbers = extract_dossier_number(question)
    
    db_results = []

    if not db_conn_status_val or not db_manager:
        logger.warning("Recherche BDD annulée: connexion ou manager non initialisé")
        return {"db_results": []}

      # Stratégie actuelle: recherche du premier numéro trouvé.
    if dossier_numbers:
        num_to_search = dossier_numbers[0]
        logger.info(f"Tentative de recherche BDD pour le numéro: {num_to_search}")
        try:
            db_results = db_manager.rechercher_dossier(numero_dossier=num_to_search)
            logger.info(f"{len(db_results)} résultats BDD trouvés pour {num_to_search}.")
        except Exception as e:
            logger.error(f"Erreur pendant l'appel à rechercher_dossier: {e}", exc_info=True)
            db_results = []
    else:
         logger.info("Aucun numéro de dossier détecté dans la question pour la recherche BDD.")
    return {"db_results": db_results}

def retrieve(
    state: State,
    # FAISS VectorStores
    rules_vs: Optional[VectorStore],
    official_vs: Optional[VectorStore],
    echanges_vs: Optional[VectorStore],
    user_vs: Optional[VectorStore],
    # BM25 Components (BM25Okapi index, List[Document] corpus)
    rules_bm25_comps: BM25Components,
    official_bm25_comps: BM25Components,
    echanges_bm25_comps: BM25Components,
    user_bm25_comps: BM25Components,
    # Configuration pour la recherche (k pour FAISS, k pour BM25, poids pour EnsembleRetriever)
    k_faiss: int = 3, 
    k_bm25: int = 3,
    ensemble_weights: Optional[List[float]] = None, # e.g., [0.5, 0.5]
    max_workers: int = 4 # Nombre de threads pour la parallélisation
) -> Dict[str, Any]:
    """
    Noeud: Récupère documents depuis BDD (via state) et de manière hybride (FAISS + BM25)
    depuis les différentes sources de documents, avec récupération parallélisée par source.
    """
    logger.info("Noeud: retrieve (Hybride FAISS + BM25) - Parallélisé")
    question = state["question"]
    db_docs = db_resultats_to_documents(state.get("db_results", []))

    if ensemble_weights is None:
        ensemble_weights = [0.5, 0.5] # Poids par défaut si non fournis

    # Fonction d'aide pour la récupération hybride par source (reste identique)
    def get_hybrid_docs_for_source(
        source_name: str,
        faiss_vs: Optional[VectorStore],
        bm25_components: BM25Components
    ) -> Tuple[str, List[Document]]: # Retourne aussi le nom pour identifier les résultats
        retrieved_for_source: List[Document] = []
        faiss_retriever, bm25_retriever_custom = None, None

        # ... (logique interne de get_hybrid_docs_for_source reste la même) ...
        if faiss_vs:
            faiss_retriever = faiss_vs.as_retriever(search_kwargs={'k': k_faiss})
        
        if bm25_components:
            bm25_idx, bm25_docs_corpus = bm25_components
            if bm25_idx and bm25_docs_corpus:
                bm25_retriever_custom = BM25Retriever(bm25_index=bm25_idx, docs=bm25_docs_corpus, k=k_bm25)
        
        if faiss_retriever and bm25_retriever_custom:
            logger.debug(f"Retrieve - {source_name}: Utilisation de EnsembleRetriever (FAISS + BM25).")
            ensemble = EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever_custom],
                weights=ensemble_weights
            )
            try:
                retrieved_for_source = ensemble.invoke(question)
            except Exception as e:
                logger.error(f"Retrieve - Erreur EnsembleRetriever pour {source_name}: {e}", exc_info=True)
        elif faiss_retriever:
            logger.debug(f"Retrieve - {source_name}: Utilisation de FAISS Retriever uniquement.")
            try:
                retrieved_for_source = faiss_retriever.invoke(question)
            except Exception as e:
                logger.error(f"Retrieve - Erreur FAISS Retriever pour {source_name}: {e}", exc_info=True)
        elif bm25_retriever_custom:
            logger.debug(f"Retrieve - {source_name}: Utilisation de BM25Retriever uniquement.")
            try:
                retrieved_for_source = bm25_retriever_custom.invoke(question)
            except Exception as e:
                logger.error(f"Retrieve - Erreur BM25Retriever pour {source_name}: {e}", exc_info=True)
        else:
            logger.warning(f"Retrieve - {source_name}: Aucun retriever disponible.")
            
        logger.info(f"Retrieve - {source_name}: {len(retrieved_for_source)} documents trouvés.")
        return source_name, retrieved_for_source

    # Définir les tâches de récupération pour chaque source
    tasks = [
        ("UserDocs", user_vs, user_bm25_comps),
        ("Rules", rules_vs, rules_bm25_comps),
        ("OfficialDocs", official_vs, official_bm25_comps),
        ("Echanges", echanges_vs, echanges_bm25_comps),
    ]

    results_dict = {}
    logger.info(f"Lancement de la récupération parallèle pour {len(tasks)} sources avec max_workers={max_workers}...")
    # Exécution parallèle
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre les tâches
        future_to_source = {executor.submit(get_hybrid_docs_for_source, name, vs, bm25): name for name, vs, bm25 in tasks}
        
        # Récupérer les résultats au fur et à mesure qu'ils sont complétés
        for future in concurrent.futures.as_completed(future_to_source):
            source_name_res = future_to_source[future]
            try:
                name, docs = future.result()
                results_dict[name] = docs
                logger.info(f"Récupération parallèle terminée pour: {name}")
            except Exception as exc:
                logger.error(f"Erreur lors de la récupération parallèle pour {source_name_res}: {exc}", exc_info=True)
                results_dict[source_name_res] = [] # Assurer une liste vide en cas d'erreur

    logger.info("Récupération parallèle terminée pour toutes les sources.")

    # Récupérer les résultats dans l'ordre souhaité
    relevant_user_docs = results_dict.get("UserDocs", [])
    relevant_rules = results_dict.get("Rules", [])
    relevant_official_docs = results_dict.get("OfficialDocs", [])
    relevant_echanges = results_dict.get("Echanges", [])

    # Combiner les résultats dans l'ordre de priorité: BDD > Règles > Officiels > User > Échanges
    combined_docs = (
        db_docs + 
        relevant_rules + 
        relevant_official_docs + 
        relevant_user_docs + 
        relevant_echanges
    )
    logger.info(f"Retrieve - Documents combinés avant déduplication: {len(combined_docs)} "
                f"(BDD: {len(db_docs)}, Règles: {len(relevant_rules)}, "
                f"Officiels: {len(relevant_official_docs)}, User: {len(relevant_user_docs)}, Echanges: {len(relevant_echanges)})")

    # Déduplication simple basée sur le contenu exact 
    seen_content = set()
    unique_docs = []
    for doc in combined_docs:
        # Clé de déduplication: contenu + source principale,pour différencier légèrement
        dedup_key = (doc.page_content, doc.metadata.get("source_file", doc.metadata.get("source")))
        if dedup_key not in seen_content:
            unique_docs.append(doc)
            seen_content.add(dedup_key)
    if len(unique_docs) < len(combined_docs):
        
         logger.info(f"Retrieve - Documents après déduplication: {len(unique_docs)}")
    return {"context": unique_docs}

def generate(state: State, llm_model: Optional[Any]) -> Dict[str, Any]:
    """Noeud: Génère la réponse avec le LLM, le contexte et un prompt structuré."""
    logger.info("Noeud: generate")
    question = state["question"]
    context_docs = state.get("context", [])
    chat_history = state.get("history", [])
    
    if llm_model is None:
        logger.error("Génération impossible: LLM  non initialisé.")
        return {"answer": "Erreur: Service de génération non disponible."}

    if not context_docs:
        logger.warning("Génération: Aucun document de contexte fourni après l'étape retrieve.")
        
        # Vérifier si la BDD avait des résultats même si le reste est vide
        db_results_in_state = state.get("db_results", [])
        if db_results_in_state:
             db_docs_only = db_resultats_to_documents(db_results_in_state)
             db_content_only = "\n\n".join([doc.page_content for doc in db_docs_only])
             return {"answer": f"Infos BDD:\n\n{db_content_only}\n\nPas d'infos complémentaires."}
        return {"answer": "Désolé, aucune information pertinente trouvée."}

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
        docs_details_list.append({"id": source_id, "file": source_file, "category": category, "type": doc_type})
        
        
        # Classifier pour le prompt basé sur la catégorie/type
        if category == "regles" or doc_type == "rule_document": rules_content.append(content_with_id)
        elif category == "dossier_bdd" or doc_type == "db_data": db_content.append(content_with_id)
        elif category == "docs_officiels" or doc_type == "knowledge_document": official_content.append(content_with_id)
        elif category == "echanges" or doc_type == "echange_document": echanges_content.append(content_with_id)
        else: other_content.append(f"[{source_id} - Cat: {category}]\n{doc.page_content.strip()}")
        
        source_counter += 1
        
    # Construire les sections du contexte pour le prompt
    context_sections = []
    if db_content: context_sections.append("**INFOS DOSSIER BDD (Priorité 1):**\n" + "\n\n---\n\n".join(db_content))
    if rules_content: context_sections.append("**RÈGLES (Priorité 2):**\n" + "\n\n---\n\n".join(rules_content))
    if official_content: context_sections.append("**DOCS OFFICIELS (Priorité 3):**\n" + "\n\n---\n\n".join(official_content))
    if echanges_content: context_sections.append("**ÉCHANGES (Style/Ton - Priorité 4):**\n" + "\n\n---\n\n".join(echanges_content))
    if other_content: context_sections.append("**AUTRES (Priorité 5):**\n" + "\n\n---\n\n".join(other_content))
    
    context_string_for_prompt = "\n\n".join(context_sections)
    
    # Formater la liste des sources pour la référence
    formatted_sources_list = "\n".join([f"- [{d['id']}] {d['file']} (Cat: {d['category']}, Type: {d['type']})" for d in docs_details_list])
    
    # Formatage de l'historique pour le prompt
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
   

    # Instructions Système 
    system_instructions = (
        "Tu es un assistant expert spécialisé dans le dispositif du KAP Numérique, tu es conçu pour aider les agents instructeurs.\n"
        "Ta mission est de fournir des réponses précises, structurées et professionnelles basées **exclusivement** sur les informations fournies dans le contexte. Pas d'hallucination\n\n"
        
        "**GESTION DE L'HISTORIQUE ET QUESTIONS DE SUIVI/CLARIFICATION:**\n"
        " - Utilise l'historique des conversations fourni ('Historique de la conversation') pour comprendre le contexte des questions de suivi. Par exemple, si la question précédente concernait un dossier spécifique, une question comme 'Quel est son statut ?' se réfère à ce dossier.\n"
        " - Si une question de l'utilisateur est vague ou manque d'informations cruciales pour fournir une réponse précise (ex: 'Suis-je éligible ?' sans détails sur l'entreprise), tu DOIS poser une question de clarification ciblée pour obtenir les informations manquantes (ex: 'Pourriez-vous préciser votre secteur d'activité ou fournir votre code APE ?', 'De quel dispositif parlez-vous ?'). Ne réponds pas vaguement si tu peux demander une précision.\n"
        
        "**RÈGLES IMPÉRATIVES POUR LA GÉNÉRATION DE RÉPONSE:**\n"
        "**HIÉRARCHIE STRICTE DES SOURCES:** Analyse le contexte en respectant l'ordre de priorité suivant:\n"
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
        "    - Formatage Markdown: Utilise **gras**, *italique*, listes à puces/numérotées, et des titres (`## Titre` ou `### Sous-titre`) pour structurer clairement ta réponse. Préserve la structure hiérarchique si elle existe dans le contexte (par exemple, les différentes catégories de projets).\n"
        "    - Tableaux: Si le contexte contient des données sous forme de tableau pertinentes pour la question (par exemple, conditions, éligibilité/inéligibilité, codes APE, montants), tu DOIS présenter ces données sous forme de tableau Markdown dans ta réponse. Inclus TOUTES les colonnes et lignes pertinentes du tableau source. Assure-toi explicitement que les colonnes comme 'Code APE' sont présentes si elles le sont dans la source.\n"
        "    - Exhaustivité: Synthétise TOUTES les informations pertinentes du contexte pour répondre complètement à la question. Ne résume pas de manière excessive si cela omet des détails importants ou des catégories entières présentes dans le contexte.\n"
        "    - Citations: Justifie les points clés en citant la source exacte `[SOURCE X]`. Si multiple: `[SOURCE 1, SOURCE 3]`.\n"
        "5.  **INTERDICTIONS:**\n"
        "    - Ne mentionne jamais que tu es une IA ou un chatbot.\n"
        "    - N'invite pas à contacter un autre agent.\n"
        "    - N'invite pas à contacter un instructeur par mail, car c'est déjà l'instructeur qui te pose les question.\n"
        "    - N'utilise pas les échanges comme source factuelle.\n\n"
        "    - Lorsqu'une question posé est du style mail, avec objet, formule de politesse génére une réponse aussi du même style\n\n"
        "    - **NE PAS inclure les coordonnées du service Kap Numérik ou de la Région Réunion (mail, téléphone, horaires) dans la réponse.** L'agent instructeur (ton utilisateur) possède déjà ces informations et n'a pas besoin que tu les répétes. Cela s'applique même si ces informations sont présentes dans le contexte. L'objectif est de fournir une réponse que l'agent peut directement utiliser sans avoir à supprimer ces informations redondantes.\n"
        
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
        f"**Historique de la conversation (pour contexte uniquement):**\n{formatted_history}\n\n"
        f"**Question de l'Agent Instructeur:**\n{question}\n\n"
        f"**Contexte Fourni (analyser en respectant la hiérarchie indiquée dans les instructions système):**\n"
        f"---\n{context_string_for_prompt}\n---\n\n"
        f"**Liste des Sources du Contexte:**\n{formatted_sources_list}\n\n"
        f"**Réponse pour l'Agent (basée EXCLUSIVEMENT sur le contexte, justifiée avec [SOURCE X], format Markdown):**"
    )

    logger.info(f"Génération - Prompt final préparé (longueur approx: {len(user_prompt)} chars). Appel du LLM...")
    
    # Appel du modèle LLM
    try:
        messages = [{"role": "system", "content": system_instructions}, 
                    {"role": "user", "content": user_prompt}]
        
        # Utilise llm_model passé en argument
        #response = llm_model.invoke(messages, temperature=0.1, max_tokens=2000)
        response = llm_model.invoke(messages)
        final_answer = response.content
        logger.info("Génération - Réponse LLM reçue.")
        return {"answer": final_answer}
    except Exception as e:
        logger.error(f"Génération - Erreur LLM: {e}", exc_info=True)
        return {"answer": f"Erreur technique génération. Détails: {e}"}


def build_graph_with_deps(
    db_man_dep: Optional[Any],
    db_conn_status_dep: bool,
    llm_dep: Optional[Any],
    # FAISS VectorStores
    rules_vs_dep: Optional[VectorStore],
    official_vs_dep: Optional[VectorStore],
    echanges_vs_dep: Optional[VectorStore],
    user_vs_dep: Optional[VectorStore],
    # BM25 Components
    rules_bm25_comps_dep: BM25Components,
    official_bm25_comps_dep: BM25Components,
    echanges_bm25_comps_dep: BM25Components,
    user_bm25_comps_dep: BM25Components,
    # Configuration pour la recherche (optionnel, avec des valeurs par défaut dans retrieve)
    k_faiss_dep: int = 3,
    k_bm25_dep: int = 3,
    ensemble_weights_dep: Optional[List[float]] = None,
    max_workers_dep: int = 4 # Ajout du paramètre pour le nombre de workers
) -> StateGraph:
    """Construit et compile le graphe LangGraph en liant les dépendances, y compris BM25."""
    logger.info("Construction du graphe RAG avec dépendances (FAISS + BM25)...")
    workflow = StateGraph(State)

    bound_search_database = partial(search_database, db_manager=db_man_dep, db_conn_status_val=db_conn_status_dep)
    
    # Lier toutes les nouvelles dépendances à la fonction retrieve, y compris max_workers
    bound_retrieve = partial(
        retrieve,
        rules_vs=rules_vs_dep, official_vs=official_vs_dep, 
        echanges_vs=echanges_vs_dep, user_vs=user_vs_dep,
        rules_bm25_comps=rules_bm25_comps_dep, official_bm25_comps=official_bm25_comps_dep,
        echanges_bm25_comps=echanges_bm25_comps_dep, user_bm25_comps=user_bm25_comps_dep,
        k_faiss=k_faiss_dep, k_bm25=k_bm25_dep, ensemble_weights=ensemble_weights_dep,
        max_workers=max_workers_dep # Passer le nombre de workers
    )
    bound_generate = partial(generate, llm_model=llm_dep)

    # Ajout des noeuds
    workflow.add_node("search_database", bound_search_database)
    workflow.add_node("retrieve", bound_retrieve)
    workflow.add_node("generate", bound_generate)
    
    # Définition des transitions (flux de travail)
    workflow.add_edge(START, "search_database")# Commence par la recherche BDD
    workflow.add_edge("search_database", "retrieve")# Puis récupération vectorielle

    workflow.add_edge("retrieve", "generate")# Puis la récuperation
    workflow.add_edge("generate", END)## Termine par la génération de la réponse
    
    # Compilation du graphe
    try:
        graph = workflow.compile()
        logger.info("Graphe RAG compilé avec succès (avec dépendances).")
        return graph
    except Exception as e:
        logger.error(f"Erreur critique lors de la compilation du graphe (avec dépendances): {e}", exc_info=True)
        raise
