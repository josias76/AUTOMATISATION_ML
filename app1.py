import streamlit as st
from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg, save_model as save_model_reg, plot_model as plot_model_reg
from pycaret.classification import setup as setup_class, compare_models as compare_models_class, save_model as save_model_class, plot_model as plot_model_class


st.set_page_config(page_title="My_ML_APP", page_icon="🌍", layout="wide")
url = "https://www.linkedin.com/in/josias-nteme-95757721a/"

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

def main():
    st.title("🤖 AutoML App")
    st.sidebar.write("[Author: Josias Nteme](%s)" % url)
    st.sidebar.markdown(
        "**Bienvenue sur votre assistant intelligent pour l'automatisation du Machine Learning !**\n"
        "Cette application vous permet de :\n"
        "1. Importer vos données\n"
        "2. Explorer automatiquement les variables\n"
        "3. Entraîner plusieurs modèles\n"
        "4. Évaluer les performances\n"
        "5. Télécharger le meilleur modèle\n"
        "Simplifiez vos tâches de data science avec une interface intuitive et rapide."
    )

    file = st.file_uploader("Upload your dataset in CSV format", type=["csv"])

    if file is not None:
        data = load_data(file)
        st.dataframe(data.head())

        if st.button("Profiling du Dataset"):
            profile = ProfileReport(data, explorative=True)
            st_profile_report(profile)

        target = st.selectbox("Sélectionnez la variable cible", data.columns)
        task = st.selectbox("Sélectionnez le type de tâche ML", ["Regression", "Classification"])
        






if __name__ == '__main__':
    main()

# Bande de bas de page
st.markdown("""
    <hr style="border-top: 1px solid #4CAF50; margin-top: 50px;"/>
    <div style="text-align: center; color: #888; font-size: 0.9em;">
        &copy; 2025 <strong>Josias Nteme</strong> - Tous droits réservés.
    </div>
""", unsafe_allow_html=True)
