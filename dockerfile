# ============================================================
# MediGuide - Dockerfile
# AI-Powered Analysis for Intelligent Healthcare Assistance
# ============================================================
#
# This Dockerfile:
# 1. Uses a lightweight Python image
# 2. Creates the application working directory
# 3. Installs Python dependencies
# 4. Copies the MediGuide source code
# 5. Exposes the Flask application port
# 6. Starts the MediGuide backend
#
# Authors:
# Saurabh Kumbhar - 25204974
# Azim Hassan   - 25203062
# ============================================================


# ------------------------------------------------------------
# 1. Base Python Image
# ------------------------------------------------------------

# Use Python 3.11 to match the MediGuide development environment.
FROM python:3.11-slim


# ------------------------------------------------------------
# 2. Python Runtime Configuration
# ------------------------------------------------------------

# Prevent Python from creating .pyc cache files.
ENV PYTHONDONTWRITEBYTECODE=1

# Send Python output directly to the terminal/container logs.
ENV PYTHONUNBUFFERED=1


# ------------------------------------------------------------
# 3. Application Working Directory
# ------------------------------------------------------------

WORKDIR /app


# ------------------------------------------------------------
# 4. Install Python Dependencies
# ------------------------------------------------------------

# Copy requirements separately first.
#
# Docker can cache this layer, so dependencies are not
# reinstalled every time only application code changes.
COPY requirements.txt .


# Upgrade pip and install MediGuide dependencies.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ------------------------------------------------------------
# 5. Copy MediGuide Application
# ------------------------------------------------------------

# Copy the project source code into the container.
COPY . .


# ------------------------------------------------------------
# 6. Application Port
# ------------------------------------------------------------

# MediGuide Flask application runs on port 8080.
EXPOSE 8080


# ------------------------------------------------------------
# 7. Start MediGuide
# ------------------------------------------------------------

CMD ["python3", "app.py"]