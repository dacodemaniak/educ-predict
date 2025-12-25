import streamlit as st
import requests
import json

st.set_page_config(page_title="EduPredict Dashboard", layout="wide")

API_URL = "http://backend:8000"

st.sidebar.title("🛡️ EduPredict Admin")
menu = st.sidebar.radio("Navigation", ["Prediction", "Labo"])

# --- ONGLET PRÉDICTION ---
if menu == "Prediction":
    st.title("🎓 Risk Analysis")
    
    with st.expander("Model options", expanded=True):
        strategy = st.selectbox("AI Strategy", ["accuracy", "auc"], 
                                help="Accuracy: Overall precision. AUC: Better detection of risks.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            school = st.selectbox("School", ["GP", "MS"])
            age = st.number_input("Age", 15, 22, 17)
        with c2:
            studytime = st.slider("Study Time", 1, 4, 2)
            absences = st.number_input("Absences", 0, 100, 0)
        with c3:
            g1 = st.slider("Note G1", 0, 20, 10)
            g2 = st.slider("Note G2", 0, 20, 10)
        
        submitted = st.form_submit_button("Launch Prediction")

    if submitted:
        payload = {
            "school": school, "sex": "F", "age": age, "studytime": studytime,
            "failures": 0, "absences": absences, "G1": g1, "G2": g2
        }
        res = requests.post(f"{API_URL}/predict?strategy={strategy}", json=payload)
        
        if res.status_code == 200:
            data = res.json()
            if data["is_failure"]:
                st.error(f"⚠️ Fail risk (Probability: {data['probability']:.2%})")
            else:
                st.success(f"✅ Sucess profile (Fail probability: {data['probability']:.2%})")
        else:
            st.error("Backend error during prediction.")

# --- SETTINGS ---
elif menu == "Labo":
    st.title("🧪 Training options labo")
    
    # History of configurations
    history = requests.get(f"{API_URL}/configuration/history").json()["history"]
    selected_file = st.selectbox("Load a version :", ["Base (pipeline_config.yaml)"] + history)
    
    if selected_file == "Base (pipeline_config.yaml)":
        config_data = requests.get(f"{API_URL}/configuration").json()
    else:
        config_data = requests.get(f"{API_URL}/configuration/{selected_file}").json()

    new_config_str = st.text_area("Editor YAML (JSON format)", value=json.dumps(config_data, indent=4), height=400)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Create an Experiment"):
            save_res = requests.post(f"{API_URL}/configuration/experiment", json=json.loads(new_config_str))
            if save_res.status_code == 200:
                st.success(f"Settings saved : {save_res.json()['experiment_id']}")
    
    with col2:
        if st.button("🚀 Train with this settings"):
            train_res = requests.post(f"{API_URL}/train", params={"config_id": selected_file})
            st.warning("Training launched in background...")