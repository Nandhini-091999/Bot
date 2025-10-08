import os 
import re
import uuid
import tempfile
import smtplib
import json
import socket
from typing import Dict, Any, Tuple
from email.message import EmailMessage
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, flash, session
)
import pandas as pd
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# -----------------------------
# 1) Load config
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DB_URI = os.getenv("DB_URI")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# Mail env vars
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
UPSTREAM_EMAIL = os.getenv("UPSTREAM_EMAIL")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in .env")
if not DB_URI:
    raise ValueError("DB_URI missing in .env")

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

OUT_DIR = os.path.join(tempfile.gettempdir(), "ai_it_support_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 2) DB connection helper
# -----------------------------
def run_sql_to_df(sql: str) -> pd.DataFrame:
    engine = create_engine(DB_URI)
    df = pd.read_sql(sql, engine)
    return df

# -----------------------------
# 3) LLM (Groq)
# -----------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0
)

# -----------------------------
# 4) Safety helpers
# -----------------------------
WRITE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|GRANT|REVOKE)\b",
    re.IGNORECASE
)

def is_safe_sql(sql: str) -> bool:
    if WRITE_PATTERN.search(sql):
        return False
    return sql.strip().upper().startswith("SELECT")

# -----------------------------
# 5) Table detection (Enhanced)
# -----------------------------
ORDER_KEYWORDS = {
    "order", "orders", "order_id", "order header", "status", "consignment",
    "order_type", "creation_date", "ship_by_date", "deliver_by_date"
}

SKU_KEYWORDS = {
    "sku", "sku_id", "stroke", "description", "tdept", "color", "item", "product"
}

LOCATION_KEYWORDS = {
    "location", "warehouse", "zone", "site", "pick_sequence", "modes", "site_code"
}

INVENTORY_KEYWORDS = {
    "inventory", "stock", "qty_on_hand", "qty_allocated", "availability", "balance"
}

def detect_table(question: str) -> str:
    """Detects which primary table(s) the query likely involves."""
    q = question.lower()

    if any(k in q for k in ORDER_KEYWORDS):
        if any(k in q for k in SKU_KEYWORDS):
            return "order_header JOIN order_line"
        return "order_header"

    if any(k in q for k in SKU_KEYWORDS):
        if any(k in q for k in INVENTORY_KEYWORDS):
            return "sku JOIN inventory"
        return "sku"

    if any(k in q for k in LOCATION_KEYWORDS):
        if any(k in q for k in INVENTORY_KEYWORDS):
            return "location JOIN inventory"
        return "location"

    if any(k in q for k in INVENTORY_KEYWORDS):
        return "inventory"

    return "order_header"

# -----------------------------
# 6) Prompt generator
# -----------------------------
def build_sql_prompt():
    system_text = """
You are an expert SQL generator for MySQL. Return ONLY a single SELECT query.
Analyze the user's question and use appropriate joins if required.

Available tables:
- order_header(order_id, status, consignment, order_type, customer_id, postcode,
               creation_date, ship_by_date, deliver_by_date, last_updated_by, last_updated_date)
- order_line(order_id, line_id, sku_id, qty_ordered, qty_tasked, allocate)
- sku(sku_id, stroke, description, tdept, color)
- location(location_id, location_type, pick_sequence, site_code, zone, modes)
- inventory(inventory_id, sku_id, location_id, qty_on_hand, qty_allocated)

Relationships:
- order_header.order_id = order_line.order_id
- order_line.sku_id = sku.sku_id
- inventory.sku_id = sku.sku_id
- inventory.location_id = location.location_id

Rules:
- Always generate safe SELECT statements only.
- If user asks about stock, inventory, SKU details, or locations → join relevant tables.
- Always include LIMIT 1000 unless the user specifies otherwise.
- Use clear JOIN syntax and relevant WHERE filters inferred from question.
- Never generate DDL or DML (INSERT/UPDATE/DELETE).

Return ONLY the SQL query without explanation.
"""
    return system_text

# -----------------------------
# 7) SQL generation
# -----------------------------
def generate_sql_for_question(question: str) -> str:
    sys_prompt = build_sql_prompt()
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("user", f"User question: {question}\n\nReturn ONLY the SQL query.")
    ])
    resp = (prompt | llm).invoke({"question": question})
    sql = resp.content.strip()

    if sql.startswith("```"):
        sql = sql.strip("`")
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()
    return sql

# -----------------------------
# 8) File save helpers
# -----------------------------
def save_df_to_files(df: pd.DataFrame) -> dict:
    unique = uuid.uuid4().hex
    csv_path = os.path.join(OUT_DIR, f"results_{unique}.csv")
    xlsx_path = os.path.join(OUT_DIR, f"results_{unique}.xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return {"csv": csv_path, "excel": xlsx_path}

# -----------------------------
# 9) Session chat helpers
# -----------------------------
def reset_chat():
    session["chat"] = [
        {"role": "assistant", "content": "👋 Hi — I'm your AI Support Bot. Ask me anything about orders, SKUs, locations, or inventory."}
    ]
    session["state"] = "collect"
    session.pop("pending_sql", None)
    session.pop("latest_files", None)

def append_message(role: str, content: str):
    msgs = session.get("chat", [])
    msgs.append({"role": role, "content": content})
    session["chat"] = msgs

# -----------------------------
# 10–12 unchanged
# -----------------------------
ISSUE_STORE: Dict[str, Dict[str, Any]] = {}
SERVER_CHAT_STORE: Dict[str, list] = {}

def send_mail_to_upstream(subject: str, body: str, to: str = None) -> bool:
    if not all([SMTP_USER, SMTP_PASS, UPSTREAM_EMAIL]):
        print("SMTP config missing. Skipping email.")
        return False

    host = SMTP_HOST or "smtp.gmail.com"
    debug = str(os.getenv("SMTP_DEBUG", "0")).strip() in {"1", "true", "yes"}

    recipients = to.split(",") if to else [e.strip() for e in UPSTREAM_EMAIL.split(",") if e.strip()]
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    # If user explicitly set port or SSL, respect it; else try SSL:465 then STARTTLS:587
    env_use_ssl = os.getenv("SMTP_USE_SSL")
    env_port = os.getenv("SMTP_PORT")

    attempts = []
    if env_port or env_use_ssl is not None:
        use_ssl = str(env_use_ssl or "").lower() in {"1", "true", "yes"}
        port = int(env_port) if env_port else (465 if use_ssl else 587)
        attempts.append((use_ssl, host, port))
    else:
        attempts.extend([
            (True, host, 465),   # SSL first (matches working code)
            (False, host, 587),  # STARTTLS fallback
        ])

    errors = []
    for use_ssl, h, p in attempts:
        try:
            if use_ssl:
                with smtplib.SMTP_SSL(h, p, timeout=25) as server:
                    if debug:
                        server.set_debuglevel(1)
                    server.login(SMTP_USER, SMTP_PASS)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(h, p, timeout=25) as server:
                    if debug:
                        server.set_debuglevel(1)
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(SMTP_USER, SMTP_PASS)
                    server.send_message(msg)
            print("Mail sent to upstream:", recipients)
            return True
        except (socket.timeout, smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected, OSError) as e:
            errors.append(f"{h}:{p} ssl={use_ssl} -> {repr(e)}")
            continue

    print("SMTP failure: all attempts failed:\n" + "\n".join(errors))
    return False
        
def create_and_send_issue(user_question: str, sql: str, table: str, details: str) -> Tuple[str, str, bool]:
    issue_id = uuid.uuid4().hex[:8]
    convo_token = uuid.uuid4().hex[:12]
    ISSUE_STORE[issue_id] = {
        "question": user_question,
        "sql": sql,
        "table": table,
        "details": details,
        "status": "open"
    }
    subject = f"[AI Chatbot] Missing Data Escalation — {table} ({issue_id})"
    body = f"Conversation token: {convo_token}\nUser Question: {user_question}\nSQL: {sql}\nDetails: {details}"
    ok = send_mail_to_upstream(subject, body)
    return issue_id, convo_token, ok

# -----------------------------
# 13) Flask route (MODIFIED PART)
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if "chat" not in session:
        reset_chat()

    chat_messages = session.get("chat", [])
    state = session.get("state", "collect")

    if request.method == "POST":
        action = request.form.get("action", "message")
        user_text = request.form.get("message", "").strip()

        # NEW: handle chat close request
        if user_text.lower() in {"close", "end", "stop", "bye", "exit", "close chat"}:
            append_message("assistant", "✅ Chat session closed. You can start a new request anytime.")
            reset_chat()
            return redirect(url_for("index"))

        if action == "reset":
            reset_chat()
            return redirect(url_for("index"))

        if not user_text:
            flash("Please enter a message.", "warning")
            return redirect(url_for("index"))

        append_message("user", user_text)

        try:
            if state == "collect":
                sql = generate_sql_for_question(user_text)
                if not is_safe_sql(sql):
                    append_message("assistant", "❌ I couldn’t generate a safe SQL for that. Please rephrase.")
                    return redirect(url_for("index"))

                session["pending_sql"] = sql
                append_message("assistant", "I found a query for that. Do you want me to run it? (Yes/No)")
                session["state"] = "confirm"
                return redirect(url_for("index"))

            elif state == "confirm":
                if user_text.lower() in {"yes", "y"}:
                    sql = session.get("pending_sql")
                    df = run_sql_to_df(sql)
                    if df.empty:
                        issue_id, token, ok = create_and_send_issue(
                            user_text, sql, detect_table(user_text), "No rows returned"
                        )
                        if ok:
                            append_message("assistant", f"No data found. Issue **{issue_id}** escalated upstream.")
                        else:
                            append_message("assistant", f"No data found, but failed to send email. Issue ID: {issue_id}")
                        session["state"] = "collect"
                        return redirect(url_for("index"))

                    files = save_df_to_files(df)
                    session["latest_files"] = files
                    preview = df.head(20).to_html(classes="table table-striped table-sm", index=False)
                    append_message("assistant", f"Here are the results:<br>{preview}")
                    append_message("assistant", f'<a href="{url_for("download_file", filetype="csv")}">Download CSV</a> | <a href="{url_for("download_file", filetype="excel")}">Download Excel</a>')
                    append_message("assistant", "Would you like to make another request or type 'close chat' to end?")
                    session["state"] = "collect"  # 🔹 Allow continuous multi-request flow
                    return redirect(url_for("index"))
                else:
                    append_message("assistant", "Okay, not running that query. Ask me another question.")
                    session["state"] = "collect"
                    return redirect(url_for("index"))

        except Exception as e:
            append_message("assistant", f"⚠️ Error: {e}")
            session["state"] = "collect"
            return redirect(url_for("index"))

    return render_template("index.html", chat_messages=chat_messages, state=state)

# -----------------------------
# 14) Download route
# -----------------------------
@app.route("/download/<filetype>")
def download_file(filetype):
    files = session.get("latest_files")
    if not files or filetype not in files:
        flash("File not available.", "warning")
        return redirect(url_for("index"))
    return send_file(files[filetype], as_attachment=True)

# -----------------------------
# 15) Main entry
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
