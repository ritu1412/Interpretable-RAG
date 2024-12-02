import chromadb
from constants import CHROMADB_DIR, OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
import openai

openai.api_key = OPENAI_API_KEY

def create_chroma_client():
    client = chromadb.PersistentClient(path=CHROMADB_DIR)
    return client

def get_embedding(text, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS):
    response = openai.embeddings.create(
        input=[text],
        model=model,
        dimensions=dimensions  # If dimensions is None, default dimensions are used
    )
    embedding = response.data[0].embedding
    return embedding

def index_chunks(chunks):
    client = create_chroma_client()
    collection = client.get_or_create_collection('document_chunks')
    embeddings = [get_embedding(chunk) for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection

def retrieve_similar_chunks(query, top_k=5):
    client = create_chroma_client()
    collection = client.get_collection('document_chunks')
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0], results['ids'][0]  # Assuming single query