"""
PROGRAM 1 — Build & Save Vector DB  (One-Time Job)
---------------------------------------------------
✅ Reads local .yaml / .yml / .txt log files from ./data/
✅ Uses HuggingFace sentence-transformers for embeddings (no Azure here)
✅ Saves vector DB to disk → ./vector_store/

Install:
    pip install pyyaml sentence-transformers chromadb

Run:
    python program1_build_vectordb.py
"""

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HF_EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # free local HuggingFace model (~80 MB)
VECTOR_STORE_PATH  = os.path.join(SCRIPT_DIR, "vector_store")       # folder where DB is saved on disk
COLLECTION_NAME    = "posts_collection"
DATA_FOLDER        = os.path.join(SCRIPT_DIR, "data")               # ← put your .yaml and .txt files here

# ─────────────────────────────────────────────
# STEP 1 — LOAD LOCAL FILES (.yaml + .txt)
# ─────────────────────────────────────────────

def load_yaml_file(filepath: str) -> list[dict]:
    """
    Parse a YAML file into documents.
    Supports two formats:
      - A list of dicts (each dict becomes one document)
      - A single dict (the whole file becomes one document)
    """
    import yaml
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f)

    filename = os.path.basename(filepath)
    docs = []

    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # Use 'title' key if present, else first key's value
                title = item.get("title") or item.get("name") or f"{filename} — item {i+1}"
                body  = "\n".join(f"{k}: {v}" for k, v in item.items())
            else:
                title = f"{filename} — item {i+1}"
                body  = str(item)

            docs.append({
                "id":     f"{filename}__item_{i+1}",
                "title":  title,
                "body":   body,
                "source": filepath,
                "type":   "yaml",
            })

    elif isinstance(data, dict):
        title = data.get("title") or data.get("name") or filename
        body  = "\n".join(f"{k}: {v}" for k, v in data.items())
        docs.append({
            "id":     f"{filename}__full",
            "title":  title,
            "body":   body,
            "source": filepath,
            "type":   "yaml",
        })

    return docs


def load_txt_file(filepath: str) -> list[dict]:
    """
    Parse a TXT log file into documents.
    Each non-empty line becomes a separate document so logs are
    individually searchable. You can change CHUNK_LINES to group
    multiple lines into a single chunk instead.
    """
    CHUNK_LINES = 10   # group every N lines into one document (set 1 for line-by-line)
    filename = os.path.basename(filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip() for l in f if l.strip()]  # skip blank lines

    docs = []
    for i in range(0, len(lines), CHUNK_LINES):
        chunk  = lines[i : i + CHUNK_LINES]
        chunk_text = "\n".join(chunk)
        chunk_num  = (i // CHUNK_LINES) + 1

        docs.append({
            "id":     f"{filename}__chunk_{chunk_num}",
            "title":  f"{filename} — lines {i+1}–{i+len(chunk)}",
            "body":   chunk_text,
            "source": filepath,
            "type":   "txt_log",
        })

    return docs


def load_all_files(data_folder: str) -> list[dict]:
    """
    Walk through the data folder and load all .yaml / .yml / .txt files.
    Returns a flat list of document dicts ready for embedding.
    """
    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Data folder '{data_folder}' not found.\n"
            f"Create it and place your .yaml / .txt files inside."
        )

    all_docs = []
    supported = (".yaml", ".yml", ".txt")

    for root, _, files in os.walk(data_folder):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported:
                continue

            fpath = os.path.join(root, fname)
            print(f"Reading: {fpath}")

            try:
                if ext in (".yaml", ".yml"):
                    docs = load_yaml_file(fpath)
                elif ext == ".txt":
                    docs = load_txt_file(fpath)

                all_docs.extend(docs)
                print(f"      -> {len(docs)} document(s) extracted")

            except Exception as e:
                print(f"Skipped (error: {e})")

    print(f"\n   Total documents loaded: {len(all_docs)}\n")
    return all_docs

# ─────────────────────────────────────────────
# STEP 2 — GENERATE EMBEDDINGS (HuggingFace, local)
# ─────────────────────────────────────────────

def get_embeddings(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    """Embed texts locally — no API key, no Azure, fully offline."""
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings.tolist()

# ─────────────────────────────────────────────
# STEP 3 — PERSIST TO DISK
# ─────────────────────────────────────────────

def build_and_save_vectordb(docs: list[dict]) -> None:
    ids       = [doc["id"]                          for doc in docs]
    documents = [f"{doc['title']}. {doc['body']}"   for doc in docs]
    metadatas = [{
        "title":  doc["title"],
        "source": doc.get("source", "unknown"),
        "type":   doc.get("type", "unknown"),
    } for doc in docs]

    # Load HuggingFace model (downloads once, then cached)
    print("Loading HuggingFace embedding model...")
    model = SentenceTransformer(HF_EMBEDDING_MODEL)
    print(f" Model '{HF_EMBEDDING_MODEL}' ready.\n")

    print(f"Generating embeddings for {len(documents)} docs...\n")
    embeddings = get_embeddings(documents, model)

    # Save to disk using ChromaDB PersistentClient
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

    # Fresh build: remove old collection if it exists
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(" Cleared existing collection for fresh build.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Save config so Program 1b and Program 2 know which model was used
    with open(os.path.join(VECTOR_STORE_PATH, "config.json"), "w") as f:
        json.dump({
            "embedding_model": HF_EMBEDDING_MODEL,
            "collection": COLLECTION_NAME,
            "total_docs": collection.count(),
            "data_folder": DATA_FOLDER,
        }, f, indent=2)

    print(f"\n Vector DB saved to '{VECTOR_STORE_PATH}/'")
    print(f" {collection.count()} documents indexed")
    print(f"\n Add more data      : python program1b_add_context.py")
    print(f" Get recommendations: python program2_recommend.py")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"Scanning data folder: {DATA_FOLDER}\n")
    docs = load_all_files(DATA_FOLDER)

    if not docs:
        print("No documents found. Add .yaml or .txt files to the data folder.")
        return

    build_and_save_vectordb(docs)

if __name__ == "__main__":
    main()