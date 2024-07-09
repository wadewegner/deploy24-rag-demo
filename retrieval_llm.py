import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

import os
from dotenv import load_dotenv
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load environment variables from .env file
load_dotenv()

# PostgreSQL connection details from environment variables
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Initialize the embedding model
model = SentenceTransformer('all-mpnet-base-v2')  # More powerful but slower

# Initialize the LLM
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def string_to_array(s):
    return [float(x) for x in s.strip('[]').split(',')]

def get_similar_chunks(query, n=5):
    query_embedding = model.encode(query)

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()

    cur.execute("""
    SELECT document_name, chunk_number, chunk_text, embedding::text
    FROM document_chunks
    """)

    results = cur.fetchall()
    similarities = []
    for doc_name, chunk_num, chunk_text, embedding_str in results:
        embedding_array = string_to_array(embedding_str)
        similarity = np.dot(query_embedding, embedding_array)
        similarities.append((doc_name, chunk_num, chunk_text, similarity))

    similarities.sort(key=lambda x: x[3], reverse=True)
    top_results = similarities[:n]

    cur.close()
    conn.close()

    return top_results

def generate_response(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    outputs = llm.generate(
        inputs.input_ids,
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        no_repeat_ngram_size=3,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    query = input("Enter your question: ")
    similar_chunks = get_similar_chunks(query)

    print(f"\nTop {len(similar_chunks)} chunks similar to '{query}':")
    context = ""
    for doc_name, chunk_num, chunk_text, similarity in similar_chunks:
        print(f"\nDocument: {doc_name}")
        print(f"Chunk number: {chunk_num}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Text: {chunk_text[:200]}...")

        if len(tokenizer.encode(context + chunk_text)) <= 900:
            context += chunk_text + " "
        else:
            break

    response = generate_response(query, context)
    print("\nGenerated Response:")
    print(response)

if __name__ == "__main__":
    main()
