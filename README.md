# 🌍 My_ML_APP - AutoML App by Josias Nteme

Bienvenue sur **My_ML_APP**, une application Web intuitive conçue avec **Streamlit** pour automatiser l’ensemble du processus de **Machine Learning**, de l'exploration des données jusqu’à la génération du meilleur modèle, prêt à être téléchargé.

## 🚀 Fonctionnalités

- 📥 **Importation de données** CSV
- 📊 **Profiling automatique** des données avec YData Profiling
- 🤖 **Sélection automatique de modèles** (régression ou classification) avec PyCaret
- 📈 **Visualisation des performances**
  - Courbes ROC, Matrices de confusion, Importance des variables, etc.
- 💾 **Téléchargement du meilleur modèle** entraîné

## 📦 Technologies utilisées

- [Streamlit](https://streamlit.io/) – Interface utilisateur
- [PyCaret](https://pycaret.org/) – Entraînement automatique de modèles ML
- [pandas](https://pandas.pydata.org/) – Manipulation de données
- [ydata-profiling](https://github.com/ydataai/ydata-profiling) – Analyse exploratoire des données
- [streamlit-pandas-profiling](https://github.com/pandas-profiling/pandas-profiling) – Intégration des rapports EDA

## 🖼️ Interface utilisateur

L’application comprend :
- Une **barre latérale** avec instructions et lien vers l’auteur
- Une **zone centrale** où l'utilisateur peut :
  - Importer un fichier
  - Générer un rapport EDA
  - Sélectionner une tâche ML (régression ou classification)
  - Lancer l’entraînement automatique
  - Télécharger le meilleur modèle

## 🛠️ Installation

1. Clone ce dépôt :
   ```bash
   git clone https://github.com/josias76/my_ml_app.git
   cd my_ml_app
   ```

2. Installe les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Lance l'application :
   ```bash
   streamlit run my_ml_app
   ```

## 📁 Structure du projet

```
my_ml_app/
├── app.py                 # Code principal de l'application
├── style.css              # Feuille de style personnalisée
├── logo1.jpeg             # Logo SDA Consulting
├── my_app.png             # Image par défaut affichée
├── best_class_model.pkl   # (Généré après entraînement)
├── best_reg_model.pkl     # (Généré après entraînement)
```

![Uploading my_app.png…]()



## 👤 Auteur

Développé par [Josias Nteme](https://www.linkedin.com/in/josias-nteme-95757721a/)  
📍 SDA Consulting

## 📜 Licence

Ce projet est à but éducatif. Tous droits réservés © 2025.
