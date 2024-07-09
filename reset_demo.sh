#!/bin/bash

# Load environment variables from .env file if present
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

# Reset script for RAG demo

# Clean the database
echo "Cleaning the database..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
DROP TABLE IF EXISTS document_chunks;
DROP EXTENSION IF EXISTS vector;
EOF

echo "Database reset complete."

# Remove Hugging Face cache
echo "Removing Hugging Face cache..."
rm -rf ~/.cache/huggingface

# Remove SentenceTransformer cache
echo "Removing SentenceTransformer cache..."
rm -rf ~/.cache/torch/sentence_transformers

# Remove any temporary files or caches if needed
# (Add commands here if you have any temporary files to remove)

echo "Reset complete. Your environment is now in a clean starting state."
