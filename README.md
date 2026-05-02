# Pipeline Failure Assistant

When a pipeline fails, the usual process is — check the logs, dig through old tickets, try to remember if this happened before, guess a fix, try again. That loop can take hours.

I built this to short-circuit that loop.

It plugs into the end of your pipeline. When something breaks, it pulls the failure context, searches through your historical logs and config files, and tells you what the issue is — whether it's a recurring one with a known fix, or a brand new bug it's seeing for the first time.

---

## The problem it solves

Say your pipeline fails at 2am. The error is vague. You don't remember if you've seen it before. Your fix from 3 months ago is buried somewhere in a log file or a YAML config that nobody reads.

This tool surfaces that. It doesn't just say "here's the error" — it says:

> *"This same DB connection timeout occurred on March 3rd. The fix was increasing the pool size from 10 to 25 in services.yaml. Here's the exact block that was changed."*

And if it's a brand new issue you've never seen before, it still helps — it searches your configs and logs for related context, reasons about the failure, and suggests a fix based on what it finds. You're not on your own either way.

That's the difference between a 3-hour debug session and a 10-minute fix.

---

## How it works

**Step 1 — Index your history (run once)**

Point it at your logs, YAML configs, past incident notes — anything text-based. It reads them, generates embeddings locally using HuggingFace, and saves a searchable vector database to disk.

**Step 2 — Keep it updated (run when new files come in)**

As new logs and configs accumulate, you append them to the existing database. No rebuild needed.

**Step 3 — Hook it into your pipeline**

At the point where your pipeline fails, call Program 2 with the failure context. It searches the historical database for similar past failures, then sends the retrieved context + current error to Azure OpenAI to generate a recommendation — including what was done last time if it finds a match.

```
Pipeline fails
      ↓
current error / context
      ↓
search historical logs + configs  ←  ChromaDB (local, on disk)
      ↓
top matching past incidents
      ↓
Azure OpenAI  →  "here's the issue, here's the fix, this happened before on X date"
```

**Index your historical data:**
```bash
cp your_logs/*.txt data/
cp your_configs/*.yaml data/
python program1_build_vectordb.py
```

**Add new logs over time:**
```bash
cp new_logs/* new_data/
python program1b_add_context.py
```

**Query when a failure happens:**
```python
# at the end of your pipeline's failure handler
current_context = "AuthService timeout — token validation failed after 5000ms, pod restarted"
python program2_recommend.py
```

---

## What it needs from you

- Your historical log files (`.txt`)
- Your config files (`.yaml`) — especially ones that have been changed as part of past fixes
- Azure OpenAI access for the recommendation step
- The rest runs fully locally

---

## A few things worth knowing

- The quality of recommendations depends on what you feed it — the more historical context (old logs, fixed configs, incident notes), the better it gets
- TXT logs are chunked every 10 lines by default — adjust `CHUNK_LINES` if your log format is dense or sparse
- If you change the HuggingFace embedding model, delete `vector_store/` and reindex — embeddings from different models can't be mixed
- Azure OpenAI is only involved at the final recommendation step — all the indexing and search is local and free

---

Contributions welcome. Obvious next step would be auto-triggering Program 2 from a CI/CD webhook rather than calling it manually.
