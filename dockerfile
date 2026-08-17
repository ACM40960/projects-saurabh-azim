# ============================================================
# MediGuide - Dockerfile
# AI-Powered Analysis for Intelligent Healthcare Assistance
# ============================================================
#
# Container workflow:
#
# Python 3.11
#      ↓
# Install dependencies
#      ↓
# Copy MediGuide application
#      ↓
# Start Flask application
#
# Authors:
# Saurabh Kumbhar - 25204974
# Azim Hassan   - 25203062
# ============================================================


# ------------------------------------------------------------
# 1. Base Image
# ------------------------------------------------------------

# Use a lightweight Python 3.11 image to match the
# MediGuide development environment.
FROM python:3.11-slim


# ------------------------------------------------------------
# 2. Python Runtime Configuration
# ------------------------------------------------------------

# Prevent Python from generating .pyc cache files.
ENV PYTHONDONTWRITEBYTECODE=1

# Send Python output directly to Docker logs.
ENV PYTHONUNBUFFERED=1


# ------------------------------------------------------------
# 3. Application Directory
# ------------------------------------------------------------

# All MediGuide files will live inside /app.
WORKDIR /app


# ------------------------------------------------------------
# 4. Install Python Dependencies
# ------------------------------------------------------------

# Copy requirements separately first.
#
# Docker can cache this layer, so dependencies do not
# need to be installed again whenever only source code changes.
COPY requirements.txt ./


# Upgrade pip and install MediGuide dependencies.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt


# ------------------------------------------------------------
# 5. Copy MediGuide Application
# ------------------------------------------------------------

# Copy the remaining project files into the container.
COPY . .


# ------------------------------------------------------------
# 6. Application Port
# ------------------------------------------------------------

# MediGuide Flask server listens on port 8080.
EXPOSE 8080


# ------------------------------------------------------------
# 7. Start MediGuide
# ------------------------------------------------------------

CMD ["python", "app.py"]