import os
import io
import json
import re
import sqlite3
import hashlib
import uuid
from pathlib import Path
from datetime import datetime, date, timedelta
from urllib.parse import quote

import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageOps
import pytesseract
import shutil

tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
elif os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"
DB_PATH = DATA_DIR / "sutra.db"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = os.getenv("SUTRA_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("SUTRA_OLLAMA_MODEL", "llama3.2:3b")
TOP_K = int(os.getenv("SUTRA_TOP_K", "5"))


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                sha256 TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page INTEGER,
                text TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT DEFAULT 'user'
            );

            CREATE TABLE IF NOT EXISTS audits (
                id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


init_db()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_text(text: str, chunk_size=900, overlap=120):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + 200:
                end = boundary
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_pdf(file_bytes: bytes):
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install -r requirements.txt")
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for idx, page in enumerate(doc, start=1):
            pages.append((idx, page.get_text("text")))
    return pages


def extract_docx(file_bytes: bytes):
    if DocxDocument is None:
        raise RuntimeError("python-docx is not installed. Run: pip install -r requirements.txt")
    doc = DocxDocument(io.BytesIO(file_bytes))
    text = "\n".join(p.text for p in doc.paragraphs)
    return [(None, text)]

def extract_image(data: bytes):
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = ImageOps.autocontrast(image)

    text = pytesseract.image_to_string(image)

    return [(None, text)]


def extract_file(name: str, data: bytes):
    ext = Path(name).suffix.lower()

    if ext == ".pdf":
        return extract_pdf(data)

    if ext in {".txt", ".md", ".csv"}:
        return [(None, data.decode("utf-8", errors="replace"))]

    if ext == ".docx":
        return extract_docx(data)

    if ext in {".jpg", ".jpeg", ".png"}:
        return extract_image(data)

    raise ValueError("Supported formats: PDF, DOCX, TXT, MD, CSV, JPG, JPEG, PNG")


def add_document(name: str, file_bytes: bytes):
    digest = sha256_bytes(file_bytes)
    with get_db() as conn:
        existing = conn.execute("SELECT id, name FROM documents WHERE sha256 = ?", (digest,)).fetchone()
        if existing:
            return False, f"Already indexed: {existing['name']}"

    pages = extract_file(name, file_bytes)
    doc_id = str(uuid.uuid4())
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    target = DOCS_DIR / f"{doc_id[:8]}_{safe_name}"
    target.write_bytes(file_bytes)

    rows = []
    for page_no, text_value in pages:
        chunks = split_text(text_value)
        for idx, chunk in enumerate(chunks):
            rows.append((str(uuid.uuid4()), doc_id, idx, page_no, chunk))

    with get_db() as conn:
        conn.execute(
            "INSERT INTO documents(id, name, path, sha256, created_at) VALUES(?,?,?,?,?)",
            (doc_id, name, str(target), digest, now_iso()),
        )
        conn.executemany(
            "INSERT INTO chunks(id, document_id, chunk_index, page, text) VALUES(?,?,?,?,?)", rows
        )
        conn.execute(
            "INSERT INTO audits(id, action, payload, created_at) VALUES(?,?,?,?)",
            (str(uuid.uuid4()), "document_ingested", json.dumps({"name": name, "chunks": len(rows)}), now_iso()),
        )
    return True, f"Indexed {name} ({len(rows)} chunks)"


def delete_document(doc_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT path, name FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not row:
            return
        try:
            Path(row["path"]).unlink(missing_ok=True)
        except OSError:
            pass
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.execute(
            "INSERT INTO audits(id, action, payload, created_at) VALUES(?,?,?,?)",
            (str(uuid.uuid4()), "document_deleted", json.dumps({"name": row["name"]}), now_iso()),
        )


def retrieve(query: str, top_k=TOP_K):
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.text, c.page, d.name AS document_name
            FROM chunks c JOIN documents d ON c.document_id = d.id
            """
        ).fetchall()
    if not rows:
        return []

    texts = [r["text"] for r in rows]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts + [query])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    indices = sims.argsort()[::-1][:top_k]
    results = []
    for i in indices:
        if sims[i] <= 0:
            continue
        r = rows[int(i)]
        results.append({
            "score": float(sims[i]),
            "text": r["text"],
            "page": r["page"],
            "document_name": r["document_name"],
        })
    return results


def get_memories():
    with get_db() as conn:
        return conn.execute("SELECT * FROM memories ORDER BY created_at DESC").fetchall()


def add_memory(content: str, source="user"):
    content = content.strip()
    if not content:
        return
    with get_db() as conn:
        conn.execute(
            "INSERT INTO memories(id, content, created_at, source) VALUES(?,?,?,?)",
            (str(uuid.uuid4()), content, now_iso(), source),
        )
        conn.execute(
            "INSERT INTO audits(id, action, payload, created_at) VALUES(?,?,?,?)",
            (str(uuid.uuid4()), "memory_created", json.dumps({"content": content}), now_iso()),
        )


def delete_memory(memory_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT content FROM memories WHERE id = ?", (memory_id,)).fetchone()
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.execute(
            "INSERT INTO audits(id, action, payload, created_at) VALUES(?,?,?,?)",
            (str(uuid.uuid4()), "memory_deleted", json.dumps({"content": row["content"] if row else ""}), now_iso()),
        )


def export_memories():
    return json.dumps([dict(r) for r in get_memories()], indent=2)


def ollama_status():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        if r.ok:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            return True, models
    except requests.RequestException:
        pass
    return False, []


def call_ollama(prompt: str, system: str):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


def fallback_answer(query, context, memories):
    if not context:
        return "I don't have enough local evidence yet. Upload the relevant document or add the fact to Memory Vault."
    snippets = []
    for item in context[:3]:
        snippets.append(item["text"][:350])
    answer = "Based on your local indexed data, the most relevant evidence is:\n\n" + "\n\n".join(f"• {s}" for s in snippets)
    if memories:
        answer += "\n\nMemory currently available to SUTRA: " + "; ".join(m["content"] for m in memories[:3])
    return answer


def ask_sutra(query: str):
    context = retrieve(query)
    memories = get_memories()
    source_lines = []
    for i, item in enumerate(context, start=1):
        page_label = f"p.{item['page']}" if item["page"] else "document"
        source_lines.append(f"SOURCE {i} — {item['document_name']} ({page_label})\n{item['text']}")
    memory_lines = "\n".join(f"- {m['content']}" for m in memories[:20]) or "(no saved memories)"
    system = (
        "You are SUTRA, a privacy-first local academic assistant. "
        "Answer only from the supplied local evidence and user memories. "
        "Do not invent deadlines, names, dates, or requirements. "
        "If the evidence is insufficient, say so. Cite source numbers like [S1], [S2]. "
        "When using memory, say [M]. Keep answers practical and concise."
    )
    prompt = f"""User question:\n{query}\n\nLOCAL EVIDENCE:\n{chr(10).join(source_lines) or '(none)'}\n\nUSER MEMORY:\n{memory_lines}"""
    ok, _ = ollama_status()
    if ok:
        try:
            answer = call_ollama(prompt, system)
            mode = "local LLM"
        except Exception as exc:
            answer = fallback_answer(query, context, memories)
            mode = f"demo fallback ({exc.__class__.__name__})"
    else:
        answer = fallback_answer(query, context, memories)
        mode = "demo fallback — start Ollama for generative answers"
    with get_db() as conn:
        conn.execute(
            "INSERT INTO audits(id, action, payload, created_at) VALUES(?,?,?,?)",
            (str(uuid.uuid4()), "chat_query", json.dumps({"query": query, "mode": mode}), now_iso()),
        )
    return answer, context, mode


def build_ics(tasks):
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//SUTRA//EN", "CALSCALE:GREGORIAN"
    ]
    for task in tasks:
        start_dt = task["start"]
        end_dt = start_dt + timedelta(minutes=task.get("duration", 60))
        fmt = lambda dt: dt.strftime("%Y%m%dT%H%M%S")
        uid = str(uuid.uuid4())
        title = task["title"].replace("\n", " ")
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{fmt(datetime.now())}",
            f"DTSTART:{fmt(start_dt)}",
            f"DTEND:{fmt(end_dt)}",
            f"SUMMARY:{title}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def extract_bullets(text):
    parts = []
    for line in text.splitlines():
        line = re.sub(r"^[-*•]\s*", "", line.strip())
        if line and len(line) > 3:
            parts.append(line)
    return parts[:8]


def render_header():
    st.set_page_config(page_title="SUTRA — Sovereign AI", page_icon="🧠", layout="wide")
    st.markdown(
        """
        <style>
        .sutra-card { padding: 1rem; border: 1px solid rgba(120,120,120,.25); border-radius: 14px; margin-bottom: 1rem; }
        .pill { display:inline-block; padding:.2rem .55rem; border-radius:99px; background:#eef2ff; margin-right:.35rem; font-size:.82rem; }
        .small { color:#6b7280; font-size:.86rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🧠 SUTRA")
    st.caption("Sovereign University Trust & Reasoning Agent")
    st.markdown('<span class="pill">Local-first</span><span class="pill">Evidence-backed</span><span class="pill">Human-approved actions</span><span class="pill">Memory control</span>', unsafe_allow_html=True)


def sidebar():
    st.sidebar.header("SUTRA controls")
    ok, models = ollama_status()
    if ok:
        st.sidebar.success("Ollama: connected")
        st.sidebar.caption(f"Model: {OLLAMA_MODEL}")
        with st.sidebar.expander("Detected local models"):
            st.write(models or "No models reported")
    else:
        st.sidebar.warning("Ollama: offline")
        st.sidebar.caption("The app still works in retrieval/demo mode.")
        st.sidebar.code("ollama serve\nollama pull llama3.2:3b", language="bash")
    with st.sidebar.expander("Privacy"):
        st.write("Documents, memories, and audit logs are stored in the local data/ folder. No cloud API is required by SUTRA itself.")


def dashboard():
    with get_db() as conn:
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        actions = conn.execute("SELECT COUNT(*) FROM audits WHERE action LIKE 'action_%' OR action LIKE 'calendar_%'").fetchone()[0]
    cols = st.columns(4)
    for col, value, label in zip(cols, [docs, chunks, memories, actions], ["Documents", "Knowledge chunks", "Memories", "Actions"]):
        col.metric(label, value)


def knowledge_tab():
    st.subheader("Knowledge base")
    st.write("Drop in the documents you want SUTRA to know. Everything is indexed locally.")
    files = st.file_uploader("Upload course/college files", type=["pdf", "docx", "txt", "md", "csv", "jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Index selected files", type="primary", disabled=not files):
        for f in files or []:
            try:
                ok, msg = add_document(f.name, f.getvalue())
                (st.success if ok else st.info)(msg)
            except Exception as exc:
                st.error(f"Could not index {f.name}: {exc}")
    with get_db() as conn:
        docs = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    if docs:
        st.markdown("### Indexed documents")
        for d in docs:
            c1, c2, c3 = st.columns([5, 2, 1])
            c1.write(d["name"])
            c2.caption(d["created_at"])
            if c3.button("Delete", key=f"del_{d['id']}"):
                delete_document(d["id"])
                st.rerun()
    else:
        st.info("No documents yet. Upload your syllabus, timetable, notices, or lab manuals.")


def chat_tab():
    st.subheader("Ask SUTRA")
    query = st.text_area("Ask a question", placeholder="What deadlines do I have this week?", height=100)
    if st.button("Ask", type="primary", disabled=not query.strip()):
        with st.spinner("Retrieving local evidence and reasoning…"):
            answer, context, mode = ask_sutra(query.strip())
        st.markdown(answer)
        st.caption(f"Mode: {mode}")
        if context:
            with st.expander("Evidence used"):
                for i, item in enumerate(context, start=1):
                    page_label = f"page {item['page']}" if item["page"] else "document"
                    st.markdown(f"**[S{i}] {item['document_name']} — {page_label}**")
                    st.write(item["text"])


def memory_tab():
    st.subheader("Memory Vault")
    st.write("SUTRA only uses saved memories that you can inspect and delete here.")
    with st.form("memory_form"):
        content = st.text_input("Add a memory", placeholder="I prefer studying after 7 PM")
        submitted = st.form_submit_button("Save memory")
    if submitted and content.strip():
        add_memory(content)
        st.success("Memory saved.")
        st.rerun()

    memories = get_memories()
    if not memories:
        st.info("Memory Vault is empty.")
    for m in memories:
        c1, c2 = st.columns([8, 1])
        c1.write(m["content"])
        c1.caption(f"Created {m['created_at']} · source: {m['source']}")
        if c2.button("Delete", key=f"mem_{m['id']}"):
            delete_memory(m["id"])
            st.rerun()
    st.download_button("Export memory as JSON", export_memories(), file_name="sutra-memory.json", mime="application/json")


def actions_tab():
    st.subheader("Action desk")
    st.write("SUTRA proposes actions. You approve them before anything leaves the app.")
    prompt = st.text_area("Describe your week", placeholder="Physics lab Thursday, C assignment Friday, Maths internal Monday", height=100)
    if st.button("Generate a proposed plan", type="primary", disabled=not prompt.strip()):
        context = retrieve(prompt, top_k=6)
        memories = get_memories()
        evidence = "\n".join(item["text"][:500] for item in context)
        memory_text = "\n".join(m["content"] for m in memories[:10])
        system = (
            "You are a scheduling assistant. Create a concise study plan. "
            "Do not fabricate fixed deadlines. Mark uncertain items as NEEDS CONFIRMATION."
        )
        llm_prompt = f"User notes: {prompt}\nLocal evidence:\n{evidence}\nMemory:\n{memory_text}"
        ok, _ = ollama_status()
        if ok:
            try:
                plan = call_ollama(llm_prompt, system)
            except Exception:
                plan = None
        else:
            plan = None
        if not plan:
            plan = "Suggested blocks:\n- 60 min: review the highest-urgency course\n- 90 min: complete the nearest assignment\n- 30 min: prepare for the next lab\n- 45 min: active revision\nNEEDS CONFIRMATION: exact deadlines and preferred hours."
        st.session_state["proposed_plan"] = plan
        st.session_state["plan_evidence"] = context

    if st.session_state.get("proposed_plan"):
        st.markdown("### Proposed plan")
        st.write(st.session_state["proposed_plan"])
        st.info("Nothing is scheduled yet. Approve only after checking the dates/times.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Approve demo calendar", type="primary"):
                tomorrow = datetime.now().replace(hour=19, minute=0, second=0, microsecond=0) + timedelta(days=1)
                tasks = [
                    {"title": "SUTRA plan — study block 1", "start": tomorrow, "duration": 60},
                    {"title": "SUTRA plan — study block 2", "start": tomorrow + timedelta(hours=2), "duration": 90},
                ]
                ics = build_ics(tasks)
                st.session_state["ics"] = ics
                with get_db() as conn:
                    conn.execute("INSERT INTO audits(id, action, payload, created_at) VALUES(?,?,?,?)", (str(uuid.uuid4()), "action_calendar_approved", json.dumps({"tasks": len(tasks)}), now_iso()))
                st.success("Action approved. Review the generated calendar file before importing it.")
        with c2:
            if st.button("Reject / clear"):
                st.session_state.pop("proposed_plan", None)
                st.session_state.pop("ics", None)
                st.rerun()

    if st.session_state.get("ics"):
        st.download_button("Download calendar (.ics)", st.session_state["ics"], file_name="sutra-plan.ics", mime="text/calendar")


def audit_tab():
    st.subheader("Audit log")
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM audits ORDER BY created_at DESC LIMIT 100").fetchall()
    for row in rows:
        st.write(f"**{row['created_at']}** — `{row['action']}` — {row['payload']}")


def main():
    render_header()
    sidebar()
    dashboard()
    tabs = st.tabs(["💬 Ask", "📚 Knowledge", "🧠 Memory Vault", "⚡ Actions", "🧾 Audit"])
    with tabs[0]:
        chat_tab()
    with tabs[1]:
        knowledge_tab()
    with tabs[2]:
        memory_tab()
    with tabs[3]:
        actions_tab()
    with tabs[4]:
        audit_tab()


if __name__ == "__main__":
    main()
