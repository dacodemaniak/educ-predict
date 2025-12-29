# Utilisation d'une image légère
FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances
# Note : streamlit et requests sont obligatoires ici
COPY requirements.frontend.txt .
RUN pip install --no-cache-dir -r requirements.frontend.txt

# Copie du code source
COPY frontend/. ./

# Exposition du port Streamlit
EXPOSE 8501

# Configuration pour éviter les problèmes de réseau dans Docker
# --server.address=0.0.0.0 est crucial pour que le port soit accessible
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]