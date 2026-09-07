import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medcore_patients.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id  TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            age         INTEGER,
            sex         TEXT,
            ward        TEXT,
            notes       TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _next_patient_id(conn):
    row = conn.execute("SELECT patient_id FROM patients ORDER BY patient_id DESC LIMIT 1").fetchone()
    if not row:
        return "MED-0001"
    try:
        n = int(row["patient_id"].split("-")[1])
    except (IndexError, ValueError):
        n = 0
    return f"MED-{n + 1:04d}"


def create_patient(name: str, age: int, sex: str, ward: str, notes: str = "") -> str:
    """Insert a new patient, returning the auto-generated patient_id."""
    conn = _connect()
    patient_id = _next_patient_id(conn)
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO patients (patient_id, name, age, sex, ward, notes, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (patient_id, name, age, sex, ward, notes, now, now),
    )
    conn.commit()
    conn.close()
    return patient_id


def get_all_patients() -> list:
    conn = _connect()
    rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_patient(patient_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_patient(patient_id: str, **fields) -> bool:
    """Update any subset of {name, age, sex, ward, notes} for a patient."""
    allowed = {"name", "age", "sex", "ward", "notes"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    conn = _connect()
    updates["updated_at"] = datetime.now().isoformat(timespec="seconds")
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [patient_id]
    cur = conn.execute(f"UPDATE patients SET {set_clause} WHERE patient_id = ?", values)
    conn.commit()
    changed = cur.rowcount > 0
    conn.close()
    return changed


def delete_patient(patient_id: str) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def count_patients() -> int:
    conn = _connect()
    n = conn.execute("SELECT COUNT(*) AS c FROM patients").fetchone()["c"]
    conn.close()
    return n