import os
import json
import glob
import logging
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

def format_table(table_data: Dict[str, Any]) -> str:
    """Formate un dictionnaire représentant une table en Markdown."""
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    title = table_data.get("title")
    if not headers or not rows:
        return ""
    table_str = ""
    if title:
        table_str += f"**{title}**\n"
    header_line = " | ".join(str(h).strip() for h in headers)
    separator = " | ".join(["---"] * len(headers))
    row_lines = []
    for row in rows:
        cells = []
        for h in headers:
            cell_content = row.get(h, "")
            # Si le contenu est une liste, joindre avec des sauts de ligne DANS la cellule
            if isinstance(cell_content, list):
                 cell_str = ", ".join(str(item).strip() for item in cell_content if str(item).strip())
            else:
                 cell_str = str(cell_content).strip()
            cells.append(cell_str)
        row_lines.append(" | ".join(cells))
    table_str += "\n".join([header_line, separator] + row_lines)
    return table_str.strip()

def format_content(block_type: str, content: Any) -> str:
    """
    Formate le contenu d'un bloc selon son type en texte lisible,
    gérant les structures simples et complexes/imbriquées.
    """
    text_parts = [] # Collecte les morceaux de texte formaté
    
    if not content:
        return ""
    
    # --- Types simples ---
    if block_type in ["text", "paragraph", "footer_info", "link"]:
        # Pour footer_info et link, le contenu est un dict, on cherche 'text'
        if isinstance(content, dict):
            text = content.get("text", "")
            if block_type == "link":
                 text = f"Lien : {text}"  # Préciser que c'est un lien

        elif isinstance(content, str):
             text = content
        else:
             text = str(content)
        cleaned_text = text.strip()
        if cleaned_text:
            text_parts.append(cleaned_text)
            
    elif block_type == "subheading":
        if isinstance(content, dict) and "text" in content:
             text_parts.append(f"**{content['text'].strip()}**")
        elif isinstance(content, str):
             text_parts.append(f"**{content.strip()}**")
    # --- Type Listes ---         
    elif block_type == "list":
        items_to_format = []
        list_content = content.get("items", []) if isinstance(content, dict) else (content if isinstance(content, list) else [])
        
        for item in list_content:
            item_text = json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item).strip()
            if item_text:
                 items_to_format.append(f"- {item_text}")
        if items_to_format:
             text_parts.append("\n".join(items_to_format))
             
    elif block_type == "list_structured":
        title = content.get("title", "")
        items = content.get("items", [])
        if title:
            text_parts.append(f"**{title}**")
        structured_list_parts = []
        for item in items:
            item_str_parts = []
            # Formatage spécifique pour la structure "demandeur/documents"
            if "demandeur" in item and "documents" in item:
                 item_str_parts.append(f"  * Demandeur : {item['demandeur']}")
                 docs = item['documents']
                 if isinstance(docs, list):
                     doc_lines = [f"    - {doc}" for doc in docs if str(doc).strip()]
                     if doc_lines:
                          item_str_parts.append("  * Documents :\n" + "\n".join(doc_lines))
                 else:
                     item_str_parts.append(f"  * Documents : {docs}")
            else:
                 for key, value in item.items():
                     item_str_parts.append(f"  * {key.capitalize()} : {value}")
            structured_list_parts.append("\n".join(item_str_parts))
        if structured_list_parts:
             text_parts.append("\n\n".join(structured_list_parts))
             
    # --- Questions/Réponses ---         
    elif block_type == "qa":
        if isinstance(content, dict):
            question = content.get("question", "")
            answer_raw = content.get("answer") or content.get("answer_list")
            answer_heading = content.get("answer_heading")
            answer_intro = content.get("answer_intro")
            
            # Format Question
            if question:
                 text_parts.append(f"**Question :** {str(question).strip()}")
            answer_formatted_parts = []
            if answer_heading:
                answer_formatted_parts.append(f"*{answer_heading}*")
            if answer_intro:
                answer_formatted_parts.append(answer_intro.strip())
                
            if isinstance(answer_raw, list):
                # Cas 1: Liste de strings (answer_list)
                 if all(isinstance(item, str) for item in answer_raw):
                      list_items = [f"- {item.strip()}" for item in answer_raw if item.strip()]
                      if list_items:
                          answer_formatted_parts.append("\n".join(list_items))
                # Cas 2: Liste de dictionnaires (structure complexe dans 'answer')         
                 elif all(isinstance(item, dict) for item in answer_raw):
                      for sub_block in answer_raw:
                          sub_type = sub_block.get("type")
                          sub_content = sub_block.get("content", sub_block.get("text", sub_block.get("items", sub_block.get("table", sub_block))))
                          formatted_sub_content = format_content(sub_type, sub_content)
                          if formatted_sub_content:
                              answer_formatted_parts.append(formatted_sub_content)
            # Cas 3: Réponse simple (string)                  
            elif isinstance(answer_raw, str):
                 cleaned_answer = answer_raw.strip()
                 if cleaned_answer:
                      answer_formatted_parts.append(cleaned_answer)
            
            # Assembler la partie Réponse          
            if answer_formatted_parts:
                 text_parts.append("**Réponse :**\n" + "\n".join(answer_formatted_parts))
     # --- Type Tables ---             
    elif block_type in ["qa_table", "table"]:
        # Si c'est qa_table, il y a une question
        if isinstance(content, dict):
            question = content.get("question", "")
            if block_type == "qa_table" and question:
                 text_parts.append(f"**Question :** {str(question).strip()}")
                 
            table_data = content.get("table", content)
            formatted_table = format_table(table_data)
            if formatted_table:
                 text_parts.append(f"**Tableau :**\n{formatted_table}")
                 
    # --- type Steps étape à suivre ---             
    elif block_type == "qa_steps":
         if isinstance(content, dict):
             question = content.get("question", "")
             steps = content.get("steps", [])
             if question:
                  text_parts.append(f"**Question :** {str(question).strip()}")
             steps_formatted_parts = []
             for step in steps:
                 step_num = step.get("step", "?")
                 title = step.get("title", "")
                 description = step.get("description", "")
                 step_part = f"**Étape {step_num} : {title}**"
                 if isinstance(description, list):
                      desc_items = [f"- {d.strip()}" for d in description if d.strip()]
                      if desc_items:
                           step_part += "\n" + "\n".join(desc_items)
                 elif isinstance(description, str) and description.strip():
                      step_part += f"\n{description.strip()}"
                 steps_formatted_parts.append(step_part)
             if steps_formatted_parts:
                  text_parts.append("**Étapes :**\n" + "\n\n".join(steps_formatted_parts))
    # --- Contact Info (souvent imbriqué) ---              
    elif block_type == "contact_info":
         if isinstance(content, dict):
             org = content.get("organisation", "")
             methods = content.get("methods", [])
             avail = content.get("availability", "")
             contact_parts = []
             
             if org:
                  contact_parts.append(f"- Organisation : {org}")
             if methods:
                  method_lines = [f"  - {m.get('type', '?')} : {m.get('value', 'N/A')}" for m in methods]
                  contact_parts.append("- Méthodes de contact :\n" + "\n".join(method_lines))
             if avail:
                  contact_parts.append(f"- Disponibilité : {avail}")
             if contact_parts:
                 text_parts.append("**Informations de Contact :**\n" + "\n".join(contact_parts))
    # --- Gestion des types inconnus ou non explicitement gérés ---             
    else:
        # Essaye une conversion simple en string, si pertinent
        str_content = str(content).strip()
        if len(str_content) > 20:
            logger.warning(f"Type de bloc '{block_type}' non géré explicitement, tentative de conversion str.")
            text_parts.append(f"[{block_type.upper()}] : {str_content}")
    # Retourner le texte assemblé pour ce bloc
    return "\n".join(text_parts).strip()

def load_official_docs_from_json(official_docs_path: str) -> List[Document]:
    """Charge les documents JSON en extrayant tous les types de blocs pour le contexte RAG."""
    docs: List[Document] = []
    if not os.path.isdir(official_docs_path):
        logger.warning(f"Répertoire non trouvé : {official_docs_path}")
        return docs
    
    json_files = glob.glob(os.path.join(official_docs_path, "*.json"))
    logger.info(f"{len(json_files)} fichiers JSON trouvés dans {official_docs_path}.")
    
    for json_file in json_files:
        logger.info(f"--- Traitement de {os.path.basename(json_file)} ---")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            doc_meta = data.get("document_metadata", {})
            global_meta = data.get("global_block_metadata", {})
            blocks_data = []
            block_count = 0  # Compteur pour le log final du fichier
            
            for page in data.get("pages", []):
                page_num = page.get("page_number", "N/P")
                for section in page.get("sections", []):
                    title = section.get("section_title", "Sans Titre")
                    sec_id = section.get("section_id", "N/A")
                    for block_index, block in enumerate(section.get("content_blocks", [])):
                        block_type = block.get("type", "unknown")
                        content = block.get("content")
                        block_id = block.get("metadata", {}).get("block_id", f"sec_{sec_id}_idx_{block_index}") # Créer un ID si absent
                        
                        logger.debug(f"    Traitement Bloc: Type='{block_type}', ID='{block_id}'")
                        
                         # --- Appel de la fonction format_content ---
                        text = format_content(block_type, content)
                        
                        if text:
                            header = f"Page {page_num}"
                            if title != "Sans Titre":
                                header += f" - Section : {title}"
                            # Ajouter l'ID du bloc et le type peut aider au débogage ou à un ciblage fin

                            blocks_data.append(f"### {header} (Type : {block_type} / ID: {block_id})\n\n{text}")
                            block_count += 1
                            
            if not blocks_data:
                logger.warning(f"Aucun contenu textuel pertinent extrait de {os.path.basename(json_file)}")
                continue
            full_content = "\n\n---\n\n".join(blocks_data)
            metadata = {
                "source": json_file, "source_file": os.path.basename(json_file),
                "document_title": doc_meta.get("document_title"), "program_name": doc_meta.get("program_name"),
                "category": "docs_officiels", "type": "knowledge_document",
                "date_update": global_meta.get("date_update") or doc_meta.get("date_update"),
                "tags": doc_meta.get("tags", []), "priority": doc_meta.get("priority", 100),
            }
            metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
            docs.append(Document(page_content=full_content, metadata=metadata))
            logger.info(f"Document créé pour {os.path.basename(json_file)} avec {block_count} bloc(s) textuel(s) extrait(s).")
        except json.JSONDecodeError as e:
            logger.error(f"Erreur JSON dans {json_file} : {e}")
        except Exception as e:
            logger.error(f"Erreur inattendue lors du traitement de {json_file} : {e}", exc_info=True)
    logger.info(f"Chargement terminé. {len(docs)} documents créés au total.")
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
                    "source": json_file, "source_file": os.path.basename(json_file),
                    "category": "regles", "type": "rule_document",
                    "rule_number": rule.get("rule_number", "N/A"), "title": rule.get("title", "Sans titre"),
                    "tags": rule.get("metadata", {}).get("tags", []),
                    "priority": global_metadata.get("priority", 50),
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

def load_all_documents(official_docs_path: str, echanges_path: str, regles_path: str) -> Tuple[List[Document], List[Document], List[Document]]:
    """
    Charge tous les documents (officiels, échanges, règles) et les retourne en listes séparées.
    Les chemins sont passés en argument pour plus de flexibilité.
    """
    logger.info("Début du chargement de tous les types de documents...")
    # Charger les règles
    rules_docs = load_rules_from_json(regles_path)
    # Charger les documents officiels
    official_docs = load_official_docs_from_json(official_docs_path)
    # Charger les documents d'échanges (TXT)
    echanges_docs: List[Document] = []
    
    if os.path.isdir(echanges_path):
        try:
            echanges_loader = DirectoryLoader(
                echanges_path, glob="**/*.txt", loader_cls=TextLoader, 
                loader_kwargs={"encoding": "utf-8"}, recursive=True,
                show_progress=True, use_multithreading=True,
            )
            
            loaded_echanges = echanges_loader.load()
            
            logger.info(f"Chargement échanges: {len(loaded_echanges)} fichiers TXT trouvés et chargés.")
            
            # Ajouter/Vérifier les métadonnées pour chaque document d'échange
            for doc in loaded_echanges:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata["category"] = "echanges"
                doc.metadata["type"] = "echange_document"
                source_path = doc.metadata.get("source", "inconnu.txt")
                doc.metadata["source_file"] = os.path.basename(source_path)
                doc.metadata["source"] = source_path
                doc.metadata = {k: v for k, v in doc.metadata.items() if v is not None and v != ""}
            echanges_docs.extend(loaded_echanges)
        except Exception as e:
            logger.error(f"Erreur lors du chargement des documents d'échanges depuis {echanges_path}: {e}", exc_info=True)
    else:
        logger.warning(f"Répertoire des échanges non trouvé: {echanges_path}")
    logger.info(f"Chargement complet: {len(official_docs)} officiels, {len(echanges_docs)} échanges, {len(rules_docs)} règles.")
    return official_docs, echanges_docs, rules_docs

def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Découpe une liste de documents."""
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    splits = splitter.split_documents(documents)
    logger.info(f"{len(documents)} documents découpés en {len(splits)} chunks (size: {chunk_size}, overlap: {chunk_overlap}).")
    return splits



def load_user_uploaded_documents(user_uploads_path: str) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(user_uploads_path):
        logger.warning(f"Répertoire des uploads utilisateur non trouvé: {user_uploads_path}")
        return docs

    # Chargement des TXT
    txt_loader = DirectoryLoader(
        user_uploads_path, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, recursive=True,
        show_progress=False, use_multithreading=False, # Simplifier pour moins de fichiers
    )
    loaded_txt = txt_loader.load()
    for doc in loaded_txt:
        doc.metadata["category"] = "user_uploaded"
        doc.metadata["type"] = "user_document"
        doc.metadata["source_file"] = os.path.basename(doc.metadata.get("source", "inconnu.txt"))
    docs.extend(loaded_txt)
    logger.info(f"{len(loaded_txt)} fichiers TXT chargés depuis les uploads utilisateur.")

    # Chargement des PDF
    pdf_files = glob.glob(os.path.join(user_uploads_path, "*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pdf_pages = loader.load_and_split() # PyPDFLoader charge et split déjà
            for page_doc in pdf_pages: # Chaque page est un Document
                page_doc.metadata["category"] = "user_uploaded"
                page_doc.metadata["type"] = "user_document"
                page_doc.metadata["source_file"] = os.path.basename(pdf_file)
                page_doc.metadata["source"] = pdf_file # Garder le chemin complet
            docs.extend(pdf_pages)
            logger.info(f"Fichier PDF chargé et splitté: {pdf_file} ({len(pdf_pages)} pages)")
        except Exception as e:
            logger.error(f"Erreur chargement PDF {pdf_file}: {e}")
    
    logger.info(f"Total {len(docs)} documents chargés depuis les uploads utilisateur.")
    return docs