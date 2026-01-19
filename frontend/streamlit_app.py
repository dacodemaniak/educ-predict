import os
import streamlit as st
import requests
import json

st.set_page_config(page_title="EduPredict Dashboard", layout="wide")

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")

API_URL = f"http://{API_HOST}:{API_PORT}"
st.sidebar.info(f"Connecté à : {API_URL}")

st.sidebar.title("🛡️ EduPredict Admin")
menu = st.sidebar.radio("Navigation", ["Prediction", "Run", "Labo"])

# --- ONGLET PRÉDICTION ---
if menu == "Prediction":
    st.title("🎓 Risk Analysis")
    
    with st.expander("Model options", expanded=True):


        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                school = st.selectbox("School", ["GP", "MS"])
                sex = st.selectbox("Gender", ["F", "M"])
                age = st.number_input("Age", 15, 22, 17)
                address = st.selectbox("Area", ["U", "R"], help="U: Urbain, R: Rural")
            with c2:
                mjob = st.selectbox("Mother occupation", ["teacher", "health", "services", "at_home", "other"])
                fjob = st.selectbox("Father occupation", ["teacher", "health", "services", "at_home", "other"])
                studytime = st.slider("Week study time", 1, 4, 2)
                failures = st.number_input("Passed fails", 0, 4, 0)
            with c3:
                absences = st.number_input("Absences", 0, 93, 0)
                g1 = st.slider("Note G1", 0, 20, 10)
                g2 = st.slider("Note G2", 0, 20, 10)
                #strategy = st.radio("Stratégie AI", ["accuracy", "auc"])
                strategy = st.selectbox("AI Strategy", ["accuracy", "auc"], 
                            help="Accuracy: Overall precision. AUC: Better detection of risks.")
            
            submitted = st.form_submit_button("Launch Prediction")

        if submitted:
            payload = {
                "school": school, "sex": sex, "age": age, "address": address,
                "famsize": "GT3", "Pstatus": "T", "Medu": 4, "Fedu": 4,
                "Mjob": mjob, "Fjob": fjob, "reason": "course", "guardian": "mother",
                "traveltime": 1, "studytime": studytime, "failures": failures,
                "schoolsup": "no", "famsup": "no", "paid": "no", "activities": "no",
                "nursery": "yes", "higher": "yes", "internet": "yes",
                "famrel": 4, "freetime": 3, "goout": 3, "absences": absences,
                "G1": g1, "G2": g2
            }

            with st.spinner("Sending request..."):
                response = requests.post(f"{API_URL}/predict/{strategy}", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    prob = data["probability"]
                    advice = data["strategic_advice"]
                    if data["is_failure"]:
                        st.error(f"⚠️ Risk detected ({prob:.2%})")
                        # Explainability
                        if data["strategic_advice"]:
                            cols = st.columns(len(data["strategic_advice"]))
                            st.subheader("🛠️ Improvement ways")
                            for i, advice in enumerate(data["strategic_advice"]):
                                with cols[i]:
                                    st.metric(label="Factor", value=advice["feature"])
                                    st.caption(advice["message"])
                        else:
                            print("Unable to read strategic_advice attribute")
                    else:
                        st.success(f"✅ Success predicted (Risk : {prob:.2%})")
                else:
                    st.error(f"Error {response.status_code} : {response.text}")
elif menu == "Run":
    st.title("🚀 Train and track")

    strat_tab = st.selectbox("Choose the strategy:", ["accuracy", "auc"])

    # Get data from API
    metrics_res = requests.get(f"{API_URL}/monitoring/metrics/{strat_tab}")

    if metrics_res.status_code == 200:
        data = metrics_res.json()
        run_id = data["run_id"]
        scenario = data["scenario"]

        # Metrics as cards
        cols = st.columns(len(data["metrics"]))
        for i, (name, val) in enumerate(data["metrics"].items()):
            cols[i].metric(label=name.upper(), value=f"{val:.4f}")

        # Graphics
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("**Performance curve**")
            # On détermine le nom du fichier selon l'algo (stocké en tag dans MLflow)
            img_name = f"roc_curve_{scenario}.png" if strat_tab == "auc" else "rf_precision_recall.png"
            st.image(f"{API_URL}/monitoring/artifact/{run_id}/{img_name}")
            
        with c2:
            st.write("**Explicability (Features)**")
            if strat_tab == "accuracy":
                feat_img = "rf_feature_importance.png"
            else:
                feat_img = f"confusion_matrix_{scenario}.png"

            st.image(f"{API_URL}/monitoring/artifact/{run_id}/{feat_img}")
    else:
        st.warning("En attente du premier entraînement conforme...")     
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
        if st.button("💾 Create a manual Experiment"):
            save_res = requests.post(f"{API_URL}/configuration/experiment", json=json.loads(new_config_str))
            if save_res.status_code == 200:
                st.success(f"Settings saved : {save_res.json()['experiment_id']}")
                if st.button("🚀 Train with this settings"):
                    train_res = requests.post(f"{API_URL}/train", params={"config_id": selected_file})
                    st.warning("Training launched in background...")
    
    with col2:
        # Option Smart Retraining
        use_optuna = st.checkbox("Enable automatic optimization (Optuna)", value=False, help="Looking for best hyper parameters before model finalization")
        if st.button("Smart training"):
            with st.spinner("Pipeline initialization..."):
                # Call the API with use_tuning
                params = {"use_tuning": use_optuna}
                res = requests.post(f"{API_URL}/train", params=params)
                
                if res.status_code == 200:
                    st.success(f"Pipeline start in background (Mode: {'Optimized' if use_optuna else 'Standard'})")
                    st.link_button("Follow in MLFlow", f"{API_URL}/mlflow")
                else:
                    st.error("Failed to load the pipeline")        