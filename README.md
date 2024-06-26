# Vector Databases: pgvector and LangChain

This project explores the use of vector databases, specifically focusing on `pgvector` and `LangChain`. 

## Overview

Vector databases allow you to store vectors and perform efficient nearest neighbor searches. This project uses `pgvector` for PostgreSQL, which is a vector extension for PostgreSQL. It also explores `LangChain`, a language model that uses `pgvector`.

For more detailed information, please refer to this [blog post](https://bugbytes.io/posts/vector-databases-pgvector-and-langchain/). 

The main difference between this repo and the blog post is that we're running Ollama locally instead of using OpenAI to create the embeddings, which is not in their free offering.

## Installation

This project requires Python 3.6 or later, and a local instance of Ollama. Here are the steps to set up the project:

1. Clone the repository:
    ```bash
    git clone git@github.com:rodbv/pgvector-test.git
    cd pgvector-test
    ```

1. Create a virtual environment and activate it:
    ```bash
    python3 -m venv .env
    source .env/bin/activate## Pulling Different Models with Ollama
    ```

1. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

1. Installign Ollama locally

To install Ollama locally, please refer to their documentation (it's easy stuff): [https://github.com/ollama/ollama](https://github.com/ollama/ollama).

Once installed, you can start it by running the command 

```
ollama serve
```

..and then this command to get the default model we're using:

```
ollama pull nomic-embed-text
```


2. Running Postgres with pgVector support from Docker.

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

After installation, you can run the main script passing a query:

```bash
python main.py --query "Find parts of the speech related to Brazil"
```

## Pulling Different Models with Ollama

Ollama allows you to pull different models for your project. Here's how you can do it:

1. To pull a model, use the `ollama pull` command followed by the model name. For example, to pull the `orca-mini` model, you would run:
    ```bash
    ollama pull orca-mini
    ```

2. The model will be downloaded and stored in a directory named `.ollama` in your home directory. You can use the model by passing the parameter `--model`.

