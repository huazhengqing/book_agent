#!/bin/bash


# apt update
# apt install -y python3 python3-pip python3-full
echo
echo


if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "Error: Python is not installed. Please install Python 3.6+ and try again."
    exit 1
fi
$PYTHON -c "import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)" || {
    echo "Error: Python 3.6+ is required. You have $($PYTHON --version)"
    exit 1
}
echo "Using $($PYTHON --version)"
echo


# if [ -d "venv" ]; then
#     echo "Removing previous virtual environment..."
#     rm -rf venv
# fi
echo "Creating new virtual environment..."
# $PYTHON -m venv venv
if [ ! -d "venv" ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
fi
echo

echo "Activating virtual environment..."
source venv/bin/activate
echo

echo "Upgrading pip..."
# pip install --upgrade pip
echo


# pip install --upgrade huggingface_hub
# pip install gemini-cli
# pip install google-generativeai


echo "Installing all project dependencies..."
pip install -r requirements.txt
echo


# python3 -m sentence_transformers download BAAI/bge-small-zh
# python3 -m sentence_transformers download all-MiniLM-L6-v2
# python3 -m nltk.downloader punkt
# python3 download.py


HF_ENDPOINT=https://hf-mirror.com hf download BAAI/bge-small-zh --local-dir ./models/bge-small-zh 
HF_ENDPOINT=https://hf-mirror.com hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/all-MiniLM-L6-v2 



echo "Installing main package in development mode..."
pip install -v -e .
echo


# api_key_env_file="recursive/api_key.env"
# if [ ! -f "$api_key_env_file" ]; then
#     echo "Creating empty API key file..."
#     touch "$api_key_env_file"
# fi

echo
echo "Environment setup complete! You can now run the application with:"
echo
echo "Or activate the environment manually with:"
echo "source venv/bin/activate"
echo

# Deactivate virtual environment
deactivate