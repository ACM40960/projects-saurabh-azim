#!/bin/bash

# Create the main source code directory
mkdir -p src

# Create a directory for research/experimentation notebooks
mkdir -p research

# --- Creating files ---

# Marks 'src' as a Python package
touch src/__init__.py

# File for helper/utility functions
touch src/helper.py

# File for storing prompt templates or prompt-related logic
touch src/prompt.py

# File to store environment variables (API keys, secrets, config)
touch .env

# Setup script for packaging/installing the project as a module
touch setup.py

# Main application entry point
touch app.py

# Jupyter notebook for experiments and trial runs
touch research/trials.ipynb

# File listing all Python package dependencies
touch requirements.txt

# Confirmation message
echo "Directory and files created successfully!."