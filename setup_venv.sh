#!/bin/bash

# Name of the virtual environment
VENV_NAME="rag_env"

# Create virtual environment
python3 -m venv $VENV_NAME

# Activate virtual environment
source $VENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install transformers torch sentence-transformers psycopg2-binary numpy boto3 PyPDF2 nltk python-dotenv

# Verify installation
PACKAGES=("transformers" "torch" "sentence-transformers" "psycopg2-binary" "numpy" "boto3" "PyPDF2" "nltk" "python-dotenv")

echo "Verifying installation..."
for PACKAGE in "${PACKAGES[@]}"; do
    if ! pip show "$PACKAGE" > /dev/null 2>&1; then
        echo "Failed to install package: $PACKAGE"
        exit 1
    fi
done

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

echo "Required packages have been installed and verified."

echo "To deactivate the virtual environment, run 'deactivate'."