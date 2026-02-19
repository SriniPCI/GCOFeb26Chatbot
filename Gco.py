# =========================
# ONE-CELL COLAB SCRIPT (FIXED)
# Section chunking + SQLite FTS5 (BM25) retrieval + Azure OpenAI
# Fixes: "no such column: day" by generating safe FTS MATCH terms (no hyphens/punct)
# =========================

import sys, os, re, sqlite3, subprocess
from pathlib import Path

# ---------- Install dependency ----------
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "openai"])

# ---------- Secrets helpers (Colab Secrets preferred, env fallback) ----------
def _colab_get(name: str):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None

def _get_any(*names, default=None):
    for n in names:
        v = _colab_get(n)
        if v is not None and str(v).strip():
            return str(v)
        v = os.getenv(n)
        if v is not None and str(v).strip():
            return str(v)
    return default

AZURE_OPENAI_ENDPOINT = _get_any("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY  = _get_any("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY")
AZURE_API_VERSION     = _get_any("AZURE_API_VERSION", "AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = _get_any("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT_NAME", default="gpt-4o-mini")

missing = [k for k,v in {
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_API_VERSION": AZURE_API_VERSION
}.items() if not v]

if missing:
    import getpass
    print("⚠️ Missing settings:", ", ".join(missing))
    print("Add them via Colab 🔑 Secrets (recommended) OR paste below (masked for key).")
    if not AZURE_OPENAI_ENDPOINT:
        AZURE_OPENAI_ENDPOINT = input("Paste AZURE_OPENAI_ENDPOINT: ").strip()
    if not AZURE_OPENAI_API_KEY:
        AZURE_OPENAI_API_KEY = getpass.getpass("Paste AZURE_OPENAI_API_KEY (masked): ").strip()
    if not AZURE_API_VERSION:
        AZURE_API_VERSION = input("Paste AZURE_API_VERSION: ").strip()

print("✅ Azure config loaded (values hidden). Deployment =", AZURE_OPENAI_DEPLOYMENT)

# ---------- Ensure repo exists ----------
REPO_DIR = Path("/content/GCOFeb26Chatbot")
KB_PATH  = REPO_DIR / "knowledge_base.txt"
INV_DB_PATH = REPO_DIR / "inventory.db"
REPO_URL = "https://github.com/HTRahman/GCOFeb26Chatbot.git"

def ensure_repo():
    if KB_PATH.exists():
        return
    if REPO_DIR.exists():
        import shutil
        shutil.rmtree(REPO_DIR, ignore_errors=True)
    subprocess.check_call(["git", "clone", "-q", REPO_URL, str(REPO_DIR)])

ensure_repo()

kb_text = KB_PATH.read_text(encoding="utf-8").strip()
print(f"✅ KB loaded from {KB_PATH} (chars={len(kb_text)})")

# =========================
# 1) SECTION-BASED CHUNKING
# =========================
def parse_kb_sections(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections = []
    if not lines:
        return sections
    sections.append(("Company", lines[0]))  # company header line
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            sections.append((k.strip(), v.strip()))
        else:
            sections.append(("General", ln))
    sections.append(("All", text))          # whole KB as a backstop
    return sections

sections = parse_kb_sections(kb_text)

# =========================
# 2) SQLITE FTS5 INDEX + SAFE MATCH QUERY
# =========================
# FTS5 virtual table + MATCH queries + bm25 ranking [1](https://github.com/)[3](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github)
FTS_ENABLED = True
kb_conn = sqlite3.connect(":memory:")
cur = kb_conn.cursor()

try:
    cur.execute("CREATE VIRTUAL TABLE kb_fts USING fts5(title, content);")
    cur.executemany("INSERT INTO kb_fts(title, content) VALUES (?, ?);", sections)
    kb_conn.commit()
except sqlite3.OperationalError as e:
    FTS_ENABLED = False
    print("⚠️ FTS5 not available. Falling back to simple scoring. Error:", e)

# Synonyms to improve recall: address -> location, etc.
SYNONYMS = {
    "address": ["location"],
    "where": ["location"],
    "opening": ["office", "hours"],
    "open": ["office", "hours"],
    "closing": ["office", "hours"],
    "close": ["office", "hours"],
    "phone": ["contact"],
    "email": ["contact", "support"],
    "returns": ["return"],
    "refund": ["return", "returns"],
    "shipping": ["delivery"],
    "next day": ["next", "day", "delivery"],
    "company name": ["company", "techgear"],
}

def normalize_query(q: str) -> str:
    ql = q.lower().strip()
    for phrase, adds in SYNONYMS.items():
        if phrase in ql:
            ql += " " + " ".join(adds)
    # IMPORTANT: turn hyphens into spaces to avoid FTS parser issues (next-day -> next day)
    ql = ql.replace("-", " ")
    # Keep only safe chars for tokenization
    ql = re.sub(r"[^a-z0-9\s]", " ", ql)
    ql = re.sub(r"\s+", " ", ql).strip()
    return ql

def safe_fts_tokens(q: str):
    # Extract only alphanumeric tokens so MATCH never sees punctuation that can confuse parser
    # This avoids errors like "no such column: day".
    return re.findall(r"[a-z0-9]+", q)

def fts_match_string(q: str) -> str:
    toks = safe_fts_tokens(q)
    if not toks:
        return ""
    # OR across tokens for robust recall on a tiny KB
    return " OR ".join(toks)

def kb_retrieve(query: str, topk: int = 5):
    qn = normalize_query(query)
    if not qn:
        return []

    if FTS_ENABLED:
        match = fts_match_string(qn)
        if not match:
            return []
        try:
            rows = kb_conn.execute(
                "SELECT title, content, bm25(kb_fts) AS rank "
                "FROM kb_fts WHERE kb_fts MATCH ? ORDER BY rank LIMIT ?;",
                (match, topk)
            ).fetchall()
            return [{"title": t, "content": c, "rank": r} for (t, c, r) in rows]
        except sqlite3.OperationalError as e:
            # Safety fallback: if MATCH string ever errors, degrade gracefully
            # (Still avoids hard-crashing your demo.)
            print("⚠️ FTS MATCH error, falling back to simple scoring. Error:", e)
            # fall through to simple scoring below

    # Simple overlap scoring fallback
    tokens = set(safe_fts_tokens(qn))
    scored = []
    for t, c in sections:
        blob = f"{t} {c}".lower()
        score = sum(1 for tok in tokens if tok in blob)
        if score > 0:
            scored.append((score, t, c))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [{"title": t, "content": c, "rank": -s} for (s, t, c) in scored[:topk]]

# =========================
# 3) OPTIONAL: inventory.db lookup (SQLite)
# =========================
def looks_like_inventory(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["price", "cost", "stock", "inventory", "available", "how much", "how many"])

def inventory_lookup(query: str, limit: int = 5):
    if not INV_DB_PATH.exists():
        return None

    conn = sqlite3.connect(str(INV_DB_PATH))
    cur = conn.cursor()

    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    qn = normalize_query(query)
    toks = safe_fts_tokens(qn)
    token = toks[-1] if toks else qn

    preferred = ["name", "item", "product", "sku", "title", "description", "size", "colour", "color"]
    for table in tables:
        cols = cur.execute(f"PRAGMA table_info({table});").fetchall()
        colnames = [c[1] for c in cols]
        lower_cols = [c.lower() for c in colnames]
        search_cols = [colnames[i] for i,c in enumerate(lower_cols) if c in preferred or any(p in c for p in preferred)]
        if not search_cols:
            continue

        where = " OR ".join([f"LOWER({c}) LIKE ?" for c in search_cols])
        params = [f"%{token}%"] * len(search_cols)
        sql = f"SELECT * FROM {table} WHERE {where} LIMIT {limit};"
        try:
            rows = cur.execute(sql, params).fetchall()
            if rows:
                conn.close()
                return {"table": table, "columns": colnames, "rows": rows}
        except Exception:
            continue

    conn.close()
    return None

# =========================
# 4) AZURE OPENAI (GROUNDED ANSWERING)
# =========================
from openai import AzureOpenAI

FALLBACK = "I'm sorry, I cannot answer your query at the moment."

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION
)

def answer(query: str) -> str:
    q = query.strip()
    if not q:
        return ""

    # Tier 1: inventory
    if looks_like_inventory(q):
        inv = inventory_lookup(q)
        if inv:
            return f"Inventory match in '{inv['table']}':\n" + "\n".join(map(str, inv["rows"]))

    # Tier 2: KB retrieval via FTS5
    hits = kb_retrieve(q, topk=5)
    if not hits:
        return FALLBACK

    evidence = "\n".join([f"{h['title']}: {h['content']}" for h in hits])

    system_prompt = (
        "You are a TechGear UK support chatbot. "
        "Answer ONLY using the KNOWLEDGE BASE evidence provided. "
        f"If the answer is not explicitly stated in the evidence, reply exactly: {FALLBACK} "
        "Use UK English."
    )
    user_prompt = f"KNOWLEDGE BASE EVIDENCE:\n{evidence}\n\nQUESTION:\n{q}"

    resp = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0
    )

    out = (resp.choices[0].message.content or "").strip()
    return out if out else FALLBACK

# =========================
# 5) TESTS + LOOP
# =========================
tests = [
    "What is the company name?",
    "What is your address?",
    "Do you offer next day delivery?",
    "What are your office hours on Saturday?",
    "What is your returns policy?",
    "How can I contact support?"
]

print("\n🧪 Smoke test:")
for t in tests:
    print("\nQ:", t)
    print("A:", answer(t))

print("\n💬 TechGear UK Chatbot (type 'exit' to quit)")
while True:
    try:
        q = input("> ").strip()
    except EOFError:
        break
    if not q:
        continue
    if q.lower() == "exit":
        break
    print(answer(q))