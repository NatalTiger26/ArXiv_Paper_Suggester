import torch
import chromadb
from sentence_transformers import SentenceTransformer


# 1. Check device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")


# 2. Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device,
)

# 3. Dummy documents
documents = [
    "Neural quantum states can represent many-body wavefunctions.",
    "Variational Monte Carlo is used to study quantum spin systems.",
    "Large language models can assist scientific discovery.",
    "General relativity describes gravity as the curvature of spacetime.",
    "Reinforcement learning trains agents through interaction with environments.",
]

ids = ["paper1", "paper2", "paper3", "paper4", "paper5"]

# 4. Generate embeddings
embeddings = model.encode(
    documents,
    convert_to_numpy=True,
)

print(f"Embedding shape: {embeddings.shape}")


# 5. Create an in-memory ChromaDB collection
client = chromadb.Client()

collection = client.create_collection(
    name="test_papers"
)

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
)


# 6. Query
query = "quantum many-body wavefunctions and spin systems"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
)[0]

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3,
)


# 7. Display results
print("\nQuery:")
print(query)

print("\nTop results:")
for i, (doc, distance) in enumerate(
    zip(results["documents"][0], results["distances"][0]),
    start=1,
):
    print(f"{i}. {doc}")
    print(f"   distance: {distance:.4f}")


print("\n✅ Embedding + ChromaDB test passed!")
