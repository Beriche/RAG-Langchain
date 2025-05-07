import os
import mysql.connector
import logging
import re
from datetime import date
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__) # Utilise le nom du module actuel

class DatabaseManager:
    """Gestionnaire de connexion et d'interrogation de la base de données."""

    def __init__(self):
        """Initialise la configuration de la base de données."""
        self.config = {
            'user': os.getenv('SQL_USER'),
            'password': os.getenv('SQL_PASSWORD', ''),
            'host': os.getenv('SQL_HOST', 'localhost'),
            'database': os.getenv('SQL_DB'),
            'port': int(os.getenv('SQL_PORT', '3306'))
        }
        if not all([self.config['user'], self.config['host'], self.config['database']]):
            logger.error("Variables d'environnement manquantes pour la connexion DB: SQL_USER, SQL_HOST ou SQL_DB ne sont pas définies.")

    def _is_config_valid(self) -> bool:
        """Vérifie si la configuration essentielle est présente."""
        valid = all([self.config['user'], self.config['host'], self.config['database']])
        if not valid:
             logger.warning("Configuration DB incomplète (SQL_USER, SQL_HOST, SQL_DB).")
        return valid

    def tester_connexion(self) -> bool:
        """Teste la connexion à la base de données."""
        if not self._is_config_valid():
            return False
        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(**self.config, connect_timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            logger.info("Connexion réussie à la base de données.")
            return True
        except mysql.connector.Error as erreur:
            logger.error(f"Échec de la connexion à la base de données: {erreur}")
            return False
        except Exception as e:
            logger.error(f"Erreur inattendue lors du test de connexion DB: {e}", exc_info=True)
            return False
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

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
        Recherche des dossiers dans la base de données avec gestion des erreurs et fermeture de connexion.
        """
        if not self._is_config_valid():
            return []

        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(**self.config, connect_timeout=10) # Timeout un peu plus long pour query
            cursor = conn.cursor(dictionary=True) # Résultats sous forme de dict

            base_query = "SELECT * FROM dossiers"
            conditions = []
            parametres = []

            # Priorité au numéro de dossier s'il est fourni directement
            if numero_dossier:
                conditions.append("Numero = %s")
                parametres.append(numero_dossier.strip())
                logger.info(f"Recherche BDD par numéro direct: {numero_dossier.strip()}")
            # Sinon, analyser search_term
            elif search_term:
                cleaned_term = search_term.strip()
                # Format exact XX-YYYY ou XX YYYY
                is_exact_numero = re.fullmatch(r'\d{2}[-\s]?\d{4}', cleaned_term)
                if is_exact_numero:
                    # Normaliser au format XX-YYYY pour la recherche
                    normalized_numero = re.sub(r'\s', '-', cleaned_term)
                    conditions.append("Numero = %s")
                    parametres.append(normalized_numero)
                    logger.info(f"Recherche BDD par numéro exact détecté dans search_term: {normalized_numero}")
                else:
                    conditions.append("(Numero LIKE %s OR nom_usager LIKE %s)")
                    fuzzy_term = f"%{cleaned_term}%"
                    parametres.extend([fuzzy_term, fuzzy_term])
                    logger.info(f"Recherche BDD floue (numéro/nom) pour: {cleaned_term}")

            # Autres filtres
            if statut and statut.lower() != "tous":
                conditions.append("statut = %s")
                parametres.append(statut)
            if instructeur and instructeur.lower() != "tous":
                conditions.append("instructeur = %s")
                parametres.append(instructeur)
            if date_debut_creation:
                conditions.append("date_creation >= %s")
                parametres.append(date_debut_creation)
            if date_fin_creation:
                 conditions.append("date_creation <= %s")
                 parametres.append(date_fin_creation)
                 # Vérification simple de cohérence
                 if date_debut_creation and date_debut_creation > date_fin_creation:
                      logger.warning("Date de début postérieure à la date de fin dans la recherche BDD.")

            # Critères kwargs (utiliser avec prudence si les clés viennent de l'extérieur)
            for cle, valeur in kwargs.items():
                if valeur is not None:
                    # Exemple simple, pourrait nécessiter une validation des clés
                    conditions.append(f"`{cle}` = %s") # Backticks pour noms de colonnes
                    parametres.append(valeur)

            # Construction de la requête finale
            requete = base_query
            if conditions:
                requete += " WHERE " + " AND ".join(conditions)
            requete += " ORDER BY derniere_modification DESC" # Trier par défaut

            # Appliquer la limite seulement si ce n'est pas une recherche par numéro exact
            apply_limit = True
            if numero_dossier or (search_term and re.fullmatch(r'\d{2}[-\s]?\d{4}', search_term.strip())):
                apply_limit = False

            if apply_limit and limit is not None and limit > 0:
                requete += " LIMIT %s"
                parametres.append(limit)

            logger.info(f"Exécution requête BDD: {requete} | Params: {parametres}")
            cursor.execute(requete, tuple(parametres)) # Exécuter avec un tuple de paramètres
            resultats = cursor.fetchall()
            logger.info(f"{len(resultats)} dossiers trouvés dans la BDD.")
            return resultats

        except mysql.connector.Error as erreur:
            logger.error(f"Erreur lors de la recherche BDD: {erreur}")
            return []
        except Exception as e:
            logger.error(f"Erreur inattendue dans rechercher_dossier: {e}", exc_info=True)
            return []
        finally:
            # Assurer la fermeture du curseur et de la connexion
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()
                
