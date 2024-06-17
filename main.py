from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

CONNECTION_STRING = "postgresql+psycopg2://postgres:pass@localhost:5432/vector_db"
COLLECTION_NAME = "state_of_union_vectors_full_text_orca_mini"


def run(prompt: str, model: str = None):
    loader = TextLoader("state_of_the_union.txt", encoding="utf-8")
    documents = loader.load()

    embeddings = OllamaEmbeddings(model=model if model else "nomic-embed-text")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    db = PGVector.from_documents(
        embedding=embeddings,
        documents=texts,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        use_jsonb=True,
    )

    results = db.similarity_search_with_score(prompt, k=3)

    for doc in results:
        yield doc


if __name__ == "__main__":
    started_at = datetime.now()
    for result in run(
        "Which sentences refer to Brazil directly or indirectly?", model="orca-mini"
    ):
        print(result)
    elapsed = datetime.now() - started_at
    print(f"Elapsed time: {elapsed}")
