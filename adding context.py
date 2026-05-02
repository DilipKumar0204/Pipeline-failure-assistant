"""
PROGRAM 1b — Add Extra Context to Existing Vector DB
------------------------------------------------------
✅ Loads the existing vector DB from disk (built by Program 1)
✅ Reads NEW .yaml / .txt files from ./new_data/ folder
✅ Embeds using the SAME HuggingFace model as Program 1
✅ Appends to existing collection — does NOT rebuild from scratch
✅ Skips duplicate IDs automatically

Install:
    pip install pyyaml sentence-transformers chromadb

Run:
    python program1b_add_context.py
"""

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# CONFIG (auto-loaded from config.json saved by Program 1)
# ─────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(SCRIPT_DIR, "vector_store")
NEW_DATA_FOLDER   = os.path.join(SCRIPT_DIR, "new_data")    # ← drop your NEW .yaml / .txt files here

def load_config() -> dict:
    config_path = os.path.join(VECTOR_STORE_PATH, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "config.json not found. Please run program1_build_vectordb.py first."
        )
    with open(config_path) as f:
        return json.load(f)

# ─────────────────────────────────────────────
# STEP 1 — LOAD EXISTING VECTOR DB
# ─────────────────────────────────────────────

def load_vectordb(collection_name: str) -> tuple[chromadb.Collection, set]:
    """Load the persisted collection and return existing IDs to avoid duplicates."""
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTOR_STORE_PATH}'.\n"
            "Run program1_build_vectordb.py first."
        )

    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collection = chroma_client.get_collection(name=collection_name)

    # Fetch existing IDs to prevent duplicate inserts
    existing = collection.get(include=[])
    existing_ids = set(existing["ids"])

    print(f" Loaded vector DB — {collection.count()} documents already indexed.")
    print(f"   Existing IDs tracked: {len(existing_ids)}\n")
    return collection, existing_ids

# ─────────────────────────────────────────────
# STEP 2 — LOAD NEW FILES FROM ./new_data/
# ─────────────────────────────────────────────

def load_yaml_file(filepath: str) -> list[dict]:
    import yaml
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f)

    filename = os.path.basename(filepath)
    docs = []

    if isinstance(data, list):
        for i, item in enumerate(data):
            title = (item.get("title") or item.get("name") or f"{filename} — item {i+1}") if isinstance(item, dict) else f"{filename} — item {i+1}"
            body  = "\n".join(f"{k}: {v}" for k, v in item.items()) if isinstance(item, dict) else str(item)
            docs.append({"id": f"{filename}__item_{i+1}", "title": title, "body": body, "source": filepath, "type": "yaml"})
    elif isinstance(data, dict):
        title = data.get("title") or data.get("name") or filename
        body  = "\n".join(f"{k}: {v}" for k, v in data.items())
        docs.append({"id": f"{filename}__full", "title": title, "body": body, "source": filepath, "type": "yaml"})

    return docs


def load_txt_file(filepath: str) -> list[dict]:
    CHUNK_LINES = 10
    filename = os.path.basename(filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip() for l in f if l.strip()]

    docs = []
    for i in range(0, len(lines), CHUNK_LINES):
        chunk = lines[i : i + CHUNK_LINES]
        chunk_num = (i // CHUNK_LINES) + 1
        docs.append({
            "id":     f"{filename}__chunk_{chunk_num}",
            "title":  f"{filename} — lines {i+1}–{i+len(chunk)}",
            "body":   "\n".join(chunk),
            "source": filepath,
            "type":   "txt_log",
        })
    return docs


def get_new_documents(new_data_folder: str) -> list[dict]:
    """Scan new_data/ folder for .yaml and .txt files to add."""
    if not os.path.exists(new_data_folder):
        raise FileNotFoundError(
            f"Folder '{new_data_folder}' not found.\n"
            f"Create it and place your new .yaml / .txt files inside."
        )

    all_docs = []
    supported = (".yaml", ".yml", ".txt")

    for root, _, files in os.walk(new_data_folder):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported:
                continue
            fpath = os.path.join(root, fname)
            print(f" Reading: {fpath}")
            try:
                docs = load_yaml_file(fpath) if ext in (".yaml", ".yml") else load_txt_file(fpath)
                all_docs.extend(docs)
                print(f"      -> {len(docs)} document(s) extracted")
            except Exception as e:
                print(f" Skipped (error: {e})")

    print(f"\n   Total new documents found: {len(all_docs)}\n")
    return all_docs

# ─────────────────────────────────────────────
# STEP 3 — FILTER OUT DUPLICATES
# ─────────────────────────────────────────────

def filter_new_docs(new_docs: list[dict], existing_ids: set) -> list[dict]:
    """Skip documents whose ID already exists in the vector DB."""
    fresh = [doc for doc in new_docs if doc["id"] not in existing_ids]
    skipped = len(new_docs) - len(fresh)
    if skipped:
        print(f" Skipped {skipped} duplicate ID(s).")
    return fresh

# ─────────────────────────────────────────────
# STEP 4 — EMBED + APPEND TO VECTOR DB
# ─────────────────────────────────────────────

def add_to_vectordb(
    collection: chromadb.Collection,
    new_docs: list[dict],
    model: SentenceTransformer,
    config: dict,
) -> None:
    """Embed new documents and append them to the existing collection."""

    ids       = [doc["id"]    for doc in new_docs]
    documents = [f"{doc['title']}. {doc['body']}" for doc in new_docs]
    metadatas = [{
        "title":  doc["title"],
        "source": doc.get("source", "unknown"),
    } for doc in new_docs]

    print(f" Generating embeddings for {len(documents)} new docs...\n")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=32).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Update config with new total count
    config["total_docs"] = collection.count()
    with open(os.path.join(VECTOR_STORE_PATH, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n Added {len(new_docs)} new documents to the vector DB!")
    print(f"   -> Total documents now: {collection.count()}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # Load config from Program 1
    config = load_config()
    embedding_model = config["embedding_model"]
    collection_name = config["collection"]

    print(f" Vector Store  : {VECTOR_STORE_PATH}/")
    print(f" Embedding Model: {embedding_model}")
    print(f" Collection    : {collection_name}\n")

    # Step 1: Load existing DB
    collection, existing_ids = load_vectordb(collection_name)

    # Step 2: Get new documents from new_data folder
    print(f" Scanning new data folder: {NEW_DATA_FOLDER}\n")
    new_docs = get_new_documents(NEW_DATA_FOLDER)

    # Step 3: Filter duplicates
    new_docs = filter_new_docs(new_docs, existing_ids)
    if not new_docs:
        print(" No new documents to add — all IDs already exist in the DB.")
        return

    # Step 4: Load the SAME HuggingFace model as Program 1
    print(f" Loading HuggingFace model '{embedding_model}'...")
    model = SentenceTransformer(embedding_model)

    # Step 5: Embed and append
    add_to_vectordb(collection, new_docs, model, config)

    print(f"\n Get recommendations: python program2_recommend.py")


if __name__ == "__main__":
    main()