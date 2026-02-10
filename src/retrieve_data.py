import chromadb
from openai import OpenAI
from dotenv import load_dotenv
 
load_dotenv()
client = OpenAI()

# --- ------------- ---
EMBEDDING_MODEL = "text-embedding-3-small"
TARGET_DIMS = 1536  # Matches your ingest_data settings
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
# ---------------------------------

def get_embedding(text, model=EMBEDDING_MODEL, dims=TARGET_DIMS):
    """Generates a vector using the config variables."""
    response = client.embeddings.create(
        input=text, 
        model=model, 
        dimensions=dims
    )
    return response.data[0].embedding

def retrieve(query, n_results=2):
    # Connect using config path and name
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    # Convert query using the same dimensions as the stored data
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    retrieved = []
    # Loop through the matches found by Chroma
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "content": results["documents"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None
        })

    return retrieved

if __name__ == "__main__":
    query_text = "What are the rate limits?"
    print(f"\nSearching for: '{query_text}' (Dims: {TARGET_DIMS})...")
    
    matches = retrieve(query_text)
    
    for r in matches:
        print(f"[{r['title']}] (distance: {r['distance']:.4f})")
        print(f"  {r['content'][:100]}...\n")