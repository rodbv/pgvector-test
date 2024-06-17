# Vector Databases: pgvector and LangChain

This project explores the use of vector databases, specifically focusing on `pgvector` and `LangChain`. 

## Overview

Vector databases allow you to store vectors and perform efficient nearest neighbor searches. This project uses `pgvector` for PostgreSQL, which is a vector extension for PostgreSQL. It also explores `LangChain`, a language model that uses `pgvector`.

For more detailed information, please refer to this [blog post](https://bugbytes.io/posts/vector-databases-pgvector-and-langchain/). 

The main difference between this repo and the blog post is that we're running Ollama locally instead of using OpenAI to create the embeddings, which is not in their free offering.

## Installation

This project requires Python 3.6 or later. Here are the steps to set up the project:

1. Clone the repository:
    ```bash
    git clone git@github.com:rodbv/pgvector-test.git
    cd pgvector-test
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv .env
    source .env/bin/activate## Pulling Different Models with Ollama

Ollama allows you to pull different models for your project. Here's how you can do it:

1. First, make sure you have `ollama` installed. If not, you can install it using pip:
    ```bash
    pip install ollama
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Running Postgres with pgVector support from Docker.

3. Running Postgres with pgVector support from Docker.

    You can run a PostgreSQL instance with `pgvector` support using Docker. Here's how you can do it:

    First, pull the `pgvector` image from Docker Hub:
    ```bash
    docker pull pgvector/pgvector:pg16
    ```

    Then, run a container from the pulled image. Replace `<your_password>` with your desired PostgreSQL password:
    ```bash
    docker run --name pgvector -e POSTGRES_PASSWORD=<your_password> -p 5432:5432 -d pgvector/pgvector:pg16
    ```

    This command will start a PostgreSQL server with `pgvector` support on port 5432. 

    To connect to the PostgreSQL server, you can use any PostgreSQL client with the following connection details:
    - Host: `localhost`
    - Port: `5432`
    - User: `postgres`
    - Password: `<your_password>`

    Remember to replace `<your_password>` with the actual password you used when starting the container.


## Usage

After installation, you can run the main script (replace `main.py` with your actual main script name):

```bash
python main.py
```

## Pulling Different Models with Ollama

Ollama allows you to pull different models for your project. Here's how you can do it:

1. To pull a model, use the `ollama pull` command followed by the model name. For example, to pull the `bert-base-uncased` model, you would run:
    ```bash
    ollama pull bert-base-uncased
    ```

2. The model will be downloaded and stored in a directory named `.ollama` in your home directory. You can use the model by passing the model name to the `run` function in `main.py`

