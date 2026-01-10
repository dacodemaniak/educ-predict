# 🎓 EduPredict : Plateforme MLOps d'Aide à la Réussite Scolaire

Solution *industrielle* de "Machine Learning as a Service" (MLaaS) conçue pour prédire le risque d'échec scolaire.

Il s'agit d'une architecture complète, agnostique et modulaire permettant de gérer le cycle de vie complet d'un modèle d'IA : de l'entraînement dynamique à l'inférence monitorée.

## 🚀 Fonctionnalités Clés

- Pipeline Modulaire : Architecture basée sur les Design Pattern **Chain of Responsibily** et **Strategy**, ainsi que sur **Dependency Inversion Principle** permettant d'interchanger les méthodes de nettoyage et les algorithmes (Random Forest, Régression Logistique) sans modifier le cœur du code.
- Laboratoire d'Expérimentation : Système de versioning des configurations (YAML). Permet de tester de nouvelles hypothèses, avec une validation complète (PyDantic), de les archiver pour une reproductibilité totale.
- Dualité de Prédiction : Choix dynamique entre deux modèles "Champions" : **Accuracy** optimisé pour la fiabilité globale des statistiques. **AUC** optimisé pour le dépistage précoce et la sensibilité aux profils à risque.
- Audit & Traçabilité : Journalisation complète de chaque requête (Inputs, Outputs, UserID, Date) dans un format JSON structuré.
- Architecture Agnostique : Déploiement via Docker Compose et workflows CI/CD compatibles GitHub Actions et GitLab CI.

## 🏗️ Architecture Technique

La solution est découpée en deux services principaux orchestrés par Docker :

- **Backend (FastAPI)** :Gestion du cycle de vie ML (Entraînement/Inférence). Validation des schémas de données via Pydantic.Points d'entrée de santé (/health) et de configuration (/configuration).
- **Frontend (Streamlit)** :Interface "Professeur" pour les diagnostics individuels. Interface "Expert" pour le pilotage du pipeline et l'édition des configurations.

## 🛠️ Installation et Lancement

Prérequis: 
- Docker & Docker Compose
- (Optionnel) Un serveur MLflow pour le tracking

### Démarrage rapideBash

**Cloner le dépôt**
```shell
git clone https://github.com/dacodemaniak/educ-predict.git
cd edupredict
```

**Entraîner les données**
Vous pouvez lancer un premier entraînement directement à partir du Notebook : final_notebook
Deux modèles seront générés dans le dossier "backend/models"

**Lancer la plateforme**

```shell
docker compose up -d  --build
```

L'interface est alors accessible sur :

- **UI Streamlit** : http://localhost:8501
- **API Documentation** : http://localhost:8000/docs

**Utilisation locale**

```shell
uvicorn backend.student_api:app --reload --host 127.0.0.1 --port 8000
streamlit run ./frontend/streamlit_app.py # IHM
```

## 📊 Analyse des Métriques & Performance

Le système permet une analyse fine via deux métriques pivots, essentielles pour l'interprétation pédagogique :
1. Précision Globale (**Accuracy**)Utilisée pour minimiser le nombre total d'erreurs de classification. C'est l'indicateur de performance "standard",
2. Capacité de Séparation (**AUC - ROC**)Essentielle pour le dépistage. Une AUC élevée garantit que le modèle sait classer un élève "en danger" au-dessus d'un élève "en réussite", quel que soit le seuil de décision choisi.

## ⚙️ API Reference (Endpoints)

| Méthode | Route | Description |
| ---- | ---- | ---- |
| GET | /health | État de santé de l'API et présence des modèles |
| POST | /predict/{strategy} | Inférence avec choix du modèle (*accuracy* vs *auc*) |
| POST| /train | Lance l'entraînement monitoré (Background Task). |
| GET | /configuration | Récupère la configuration YAML de référence. |
| POST| /configuration/experiment| Valide et sauvegarde une nouvelle configuration expérimentale |

## Testing

- **api** : 
```shell
python -m pytest ./backend/tests/unit/test_api.py
```
