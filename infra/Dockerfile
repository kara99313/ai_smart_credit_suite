# ---- Base Python (paramétrable) ----
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# ---- Environnement d'exécution propre & outils système ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# ---- Dossier de travail ----
WORKDIR /app

# ---- Dépendances Python ----
# 1) copie requirements seul (pour bénéficier du cache de couche)
COPY requirements.txt /app/requirements.txt

# 2) mise à jour des outils d'install
RUN pip install --upgrade pip setuptools wheel

# 3) installer PyTorch CPU depuis l'index officiel (plus rapide & évite résolution lente)
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# 4) installer le reste des libs
RUN pip install -r /app/requirements.txt

# ---- Code de l’application ----
COPY . /app

# Les commandes de démarrage sont définies dans docker-compose.yml (web/api)
