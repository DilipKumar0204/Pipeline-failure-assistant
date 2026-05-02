"""
PROGRAM 2 — Query Vector DB + Get Recommendations from Azure OpenAI
---------------------------------------------------------------------
✅ Loads vector DB from disk (built by Program 1 / updated by Program 1b)
✅ Embeds user query using the SAME HuggingFace model (no Azure for embeddings)
✅ Retrieves top-K most relevant documents via semantic search
✅ Sends [system prompt + retrieved context + user query] to Azure OpenAI GPT
✅ Returns intelligent recommendations

Install:
    pip install openai sentence-transformers chromadb

Run:
    python program2_recommend.py
"""

import os
import json
import chromadb
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# CONFIG — Azure OpenAI (only used here in Program 2)
# ─────────────────────────────────────────────

# Get the script's directory for building absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO: Set these as environment variables or replace with your own values
AZURE_ENDPOINT     = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://your-resource-name.openai.azure.com/")
AZURE_API_KEY      = os.environ.get("AZURE_OPENAI_API_KEY", "your-api-key-here")
AZURE_API_VERSION  = "2024-12-01-preview"
CHAT_MODEL         = "gpt-4o"              # your Azure chat deployment name

VECTOR_STORE_PATH  = os.path.join(SCRIPT_DIR, "vector_store")
TOP_K              = 5                     # number of relevant docs to retrieve

# ─────────────────────────────────────────────
# AZURE OPENAI CLIENT (chat only)
# ─────────────────────────────────────────────

azure_client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)

# ─────────────────────────────────────────────
# STEP 1 — LOAD VECTOR DB + CONFIG FROM DISK
# ─────────────────────────────────────────────

def load_vectordb() -> tuple[chromadb.Collection, SentenceTransformer]:
    """Load ChromaDB collection and the same HuggingFace model used in Program 1."""
    config_path = os.path.join(VECTOR_STORE_PATH, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "Vector store config not found. Run program1_build_vectordb.py first."
        )

    with open(config_path) as f:
        config = json.load(f)

    embedding_model = config["embedding_model"]
    collection_name = config["collection"]

    print(f"Loading vector DB from '{VECTOR_STORE_PATH}/'")
    print(f" Embedding model : {embedding_model}")

    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collection = chroma_client.get_collection(name=collection_name)

    model = SentenceTransformer(embedding_model)

    print(f"[OK] Vector DB loaded — {collection.count()} documents available.\n")
    return collection, model

# ─────────────────────────────────────────────
# STEP 2 — RETRIEVE RELEVANT DOCS (HuggingFace embed + ChromaDB search)
# ─────────────────────────────────────────────

def retrieve_context(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    query: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """Embed the query locally and find the most similar docs in the vector DB."""
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "title":      meta.get("title", "Untitled"),
            "source":     meta.get("source", "api"),
            "content":    doc,
            "similarity": round(1 - dist, 4),
        })

    return retrieved

# ─────────────────────────────────────────────
# STEP 3 — GET RECOMMENDATIONS FROM AZURE OPENAI
# ─────────────────────────────────────────────

def get_recommendations(user_context: str, retrieved_docs: list[dict]) -> str:
    """
    Build a RAG prompt:
        System prompt + Retrieved DB context + User query
    → Azure OpenAI returns tailored recommendations.
    """

    # Format retrieved docs into a readable context block
    context_block = ""
    for i, doc in enumerate(retrieved_docs, start=1):
        context_block += (
            f"[Document {i}]  similarity={doc['similarity']}  source={doc['source']}\n"
            f"Title   : {doc['title']}\n"
            f"Content : {doc['content']}\n\n"
        )

    system_prompt = """You are an expert DevOps and CI/CD troubleshooting assistant powered by a vector knowledge base.
You receive:
  1. A set of relevant documents retrieved from a vector database (containing pipeline documentation, error patterns, and solutions)
  2. Pipeline failure logs and error details

Your job is to:
- Identify the root cause of pipeline failures
- Provide specific solutions based on similar past issues in the knowledge base
- Give step-by-step remediation instructions
Always cite which document(s) or past solutions informed your diagnosis."""

    user_prompt = f"""## Retrieved Documents from Vector DB (top {len(retrieved_docs)} matches):
{context_block}
## User's Current Context / Query:
{user_context}

## Instructions:
Based on the retrieved documents and the user's context, provide:
1. **Summary** — key findings from the retrieved documents relevant to the query
2. **Recommendations** — numbered, specific, and actionable
3. **Caveats** — any limitations or things to verify
"""

    print("Sending context to Azure OpenAI GPT for recommendations...\n")

    response = azure_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=1000,
    )

    return response.choices[0].message.content

# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

def recommend(user_context: str) -> None:
    print("=" * 60)
    print(f"User Context:\n   {user_context}")
    print("=" * 60 + "\n")

    # 1. Load DB + embedding model
    collection, model = load_vectordb()

    # 2. Retrieve top-K similar docs
    print(f" Retrieving top {TOP_K} relevant documents from vector DB...")
    retrieved_docs = retrieve_context(collection, model, user_context, top_k=TOP_K)

    print("Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"   #{i} [score={doc['similarity']}] [{doc['source']}] {doc['title']}")
    print()

    # 3. Get recommendations from Azure OpenAI
    recommendation = get_recommendations(user_context, retrieved_docs)

    print("=" * 60)
    print("RECOMMENDATIONS FROM AZURE OPENAI:")
    print("=" * 60)
    print(recommendation)
    print("=" * 60)

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Read error logs from file using absolute path
    error_file_path = os.path.join(SCRIPT_DIR, "new_data", "error_logs.txt")
    
    if os.path.exists(error_file_path):
        print(f"Reading error logs from: {error_file_path}\n")
        with open(error_file_path, "r", encoding="utf-8") as f:
            current_context = f.read().strip()
    else:
        print(f"[WARNING] Error file not found: {error_file_path}")
        print("Please create the file and paste your pipeline error logs in it.\n")
        current_context = input("Or enter your error logs here:\n> ").strip()

    if not current_context:
        print("[ERROR] No context provided. Exiting.")
        exit(1)

    recommend(current_context)