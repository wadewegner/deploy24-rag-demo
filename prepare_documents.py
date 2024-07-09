import os
from dotenv import load_dotenv
import boto3
import io
import re
from botocore.exceptions import ClientError
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
import nltk
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# DigitalOcean Spaces configuration from environment variables
SPACE_NAME = os.getenv("SPACE_NAME")
FOLDER_NAME = os.getenv("FOLDER_NAME")

# PostgreSQL connection details from environment variables
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Chunking configuration
CHUNK_SIZE = 1500  # Increase from 1000
CHUNK_OVERLAP = 300  # Increase from 200

# Initialize the embedding model
model = SentenceTransformer('all-mpnet-base-v2')

# Set up the client for DigitalOcean Spaces
session = boto3.session.Session()
client = session.client('s3',
                        region_name='us-east-1',
                        endpoint_url='https://nyc3.digitaloceanspaces.com',
                        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

def list_documents():
    """List all documents in the specified folder of the Space."""
    try:
        response = client.list_objects_v2(Bucket=SPACE_NAME, Prefix=FOLDER_NAME)
        return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'] != FOLDER_NAME]
    except ClientError as e:
        print(f"Error listing documents: {e}")
        return []

def download_document(key):
    """Download a document from the Space."""
    try:
        response = client.get_object(Bucket=SPACE_NAME, Key=key)
        return response['Body'].read()
    except ClientError as e:
        print(f"Error downloading document {key}: {e}")
        return None

def extract_text(content, filename):
    """Extract text from PDF or decode text file."""
    if filename.lower().endswith('.pdf'):
        try:
            pdf = PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error processing PDF {filename}: {e}")
            return ""
    else:
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return content.decode('iso-8859-1')
            except UnicodeDecodeError:
                print(f"Error: {filename} couldn't be decoded with UTF-8 or ISO-8859-1.")
                return ""

def clean_text(text):
    """Clean the extracted text."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split the text into overlapping chunks."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped_chunks.append(chunk)
        else:
            overlap_text = chunks[i-1][-chunk_overlap:]
            overlapped_chunks.append(overlap_text + " " + chunk)

    return overlapped_chunks

def process_documents():
    """Process documents from Spaces and return chunks."""
    documents = list_documents()
    print(f"Found {len(documents)} documents")

    all_chunks = []

    for doc in documents:
        print(f"Processing {doc}")
        content = download_document(doc)
        if content:
            text = extract_text(content, doc)
            cleaned_text = clean_text(text)
            chunks = chunk_text(cleaned_text)
            all_chunks.extend([(doc, i, chunk) for i, chunk in enumerate(chunks)])
            print(f"Extracted {len(chunks)} chunks")
        print("------------------------")

    print(f"Total chunks extracted: {len(all_chunks)}")
    return all_chunks

def create_table():
    """Create the table for storing document chunks and their embeddings."""
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()
    
    cur.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
    
    CREATE TABLE IF NOT EXISTS document_chunks (
        id SERIAL PRIMARY KEY,
        document_name TEXT,
        chunk_number INTEGER,
        chunk_text TEXT,
        embedding vector(768)
    );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def insert_chunks(chunks):
    """Insert chunks and their embeddings into the database."""
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()
    
    # Generate embeddings for all chunks
    texts = [chunk[2] for chunk in chunks]
    embeddings = model.encode(texts)
    
    # Prepare data for insertion
    data = [(chunk[0], chunk[1], chunk[2], embedding.tolist()) for chunk, embedding in zip(chunks, embeddings)]
    
    # Insert data
    execute_values(cur, """
    INSERT INTO document_chunks (document_name, chunk_number, chunk_text, embedding)
    VALUES %s
    """, data)
    
    conn.commit()
    cur.close()
    conn.close()

def main():
    # Process documents and get chunks
    chunks = process_documents()
    
    # Create the table if it doesn't exist
    create_table()
    
    # Insert chunks and their embeddings into the database
    insert_chunks(chunks)
    
    print(f"Inserted {len(chunks)} chunks with embeddings into the database.")

if __name__ == "__main__":
    main()
