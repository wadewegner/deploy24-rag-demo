# Deploy24-RAG-Demo

This is the code used to run the RAG demo using GPU Droplets in Paddy Srinivasan's Deploy 2024 keynote.

This project demonstrates a Retrieval-Augmented Generation (RAG) system using Python. It retrieves relevant document chunks based on query embeddings and generates detailed responses using a large language model (LLM).

## Prerequisites

1. **DigitalOcean (DO) Account**
2. **Python 3.8+**

## Setup Instructions

### 1. Configuring DigitalOcean Resources

#### 1.1 Create DigitalOcean Spaces

1. Log in to your DigitalOcean account and navigate to the control panel.
2. Click on "Spaces" and create a new Space (e.g., `wade-rag`).
3. Note your `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from the API section in your DO account.

#### 1.2 Add your documents to the DO Space

In your space, create a folder and add the files as PDFs or documents that you want to include in your RAG. Note the name of the folder to use below.

#### 1.3 Create a Managed PostgreSQL Database

1. Go to the "Databases" section and create a new PostgreSQL database cluster.
2. Configure the database and note the connection details (host, port, user, password, database name).
3. Connect to your PostgreSQL database and install the `pgvector` extension.

```
CREATE EXTENSION IF NOT EXISTS vector;
```

### 1.4 Create a GPU (or CPU) Droplet

1. Go to the "Droplets" section and create a new Droplet.
2. Note the IP address for use later.
3. Setup an SSH key so you can remote into the environment.

### 2. Setting Up the Python Environment

First, you'll want to SSH into the Droplet. Note the IP address of your droplet, and from a terminal type `ssh root@YOUR-IP-ADDRESS`.

#### 2.1 Clone the Repository

```bash
git clone https://github.com/wadewegner/deploy24-rag-demo.git
cd deploy24-rag-demo
```

#### 2.2 Create the .env File

Create a `.env` file in the project directory with the following content:

```
DB_NAME=
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=25060

AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

SPACE_NAME=
FOLDER_NAME=
```

Be sure to add all your info!

### 3. Running the Setup Script

Run the setup script to create and set up the virtual environment:

```
chmod +x setup_venv.sh
sudo ./setup_venv.sh # Installs all the required dependencies
source rag_env/bin/activate  # Activate the virtual environment
```

### 4. Prepare the Documents

Run the `prepare_documents.py` script to process and store document embeddings in the database:

```
python prepare_documents.py
```

### 5. Running the Retrieval and Generation Script

Run the `retrieval_llm.py` script to query the system and generate responses:

```
python retrieval_llm.py
```

## Clean Up

To reset the environment, run the reset script:

```
chmod +x reset_demo.sh
./reset_demo.sh
```
