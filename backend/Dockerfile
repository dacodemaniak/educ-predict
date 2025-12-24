FROM python:3.10-slim

WORKDIR /app

# Install required dependencies for scikit-learn/pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (check requirements.txt path, root of the project)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for experiments and models
RUN mkdir -p outputs/experiments outputs/logs models

COPY . .

EXPOSE 8000

# Launch with Uvicorn
CMD ["uvicorn", "student_api:app", "--host", "0.0.0.0", "--port", "8000"]