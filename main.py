from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
import argparse

CONNECTION_STRING = "postgresql+psycopg2://postgres:pass@localhost:5432/vector_db"
COLLECTION_NAME = "state_of_union_vectors_{model}"


def run(prompt: str, model: str):
    loader = TextLoader("state_of_the_union.txt", encoding="utf-8")
    documents = loader.load()

    embeddings = OllamaEmbeddings(model=model)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    db = PGVector.from_documents(
        embedding=embeddings,
        documents=texts,
        collection_name=COLLECTION_NAME.format(model=model),
        connection_string=CONNECTION_STRING,
        use_jsonb=True,
    )

    results = db.similarity_search_with_score(prompt, k=3)

    for doc in results:
        yield doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the script with a specific model and query."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nomic-embed-text",
        help='The model to use. Default is "nomic-embed-text".',
    )
    parser.add_argument("--query", type=str, required=True, help="The query to run.")

    args = parser.parse_args()

    started_at = datetime.now()
    for result in run(args.query, model=args.model):
        print(result)
    elapsed = datetime.now() - started_at
    print(f"Elapsed time: {elapsed}")
