import json
import chromadb # Lightweight local vector DB — no infrastructure needed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
TARGET_DIMS = 1536  # [Default value]Consider shrinking to 512 for performance, if you have millions of docs


def load_documents(filepath="data/sample_docs.json"):
    with open(filepath, "r") as f:
        return json.load(f)

def get_embedding(text, model=EMBEDDING_MODEL, dims=TARGET_DIMS):
    response = client.embeddings.create(
        input=text, 
        model=model, 
        dimensions=dims
    )
    return response.data[0].embedding

def ingest(filepath="data/sample_docs.json"):
    docs = load_documents(filepath)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete existing collection if it exists and then recreates
    try:
        chroma_client.delete_collection("knowledge_base")
    except Exception:
        pass
    
    collection = chroma_client.create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )

    for doc in docs:
        embedding = get_embedding(doc["content"])
        
        collection.add(
            ids=[doc["id"]],
            documents=[doc["content"]],
            metadatas=[{"title": doc["title"]}],
            embeddings=[embedding]
        )
        print(f"Ingested: {doc['title']} (Size: {len(embedding)})")

    print(f"\nIngested {len(docs)} documents at {TARGET_DIMS} dimensions.")
    return collection

if __name__ == "__main__":
    ingest()