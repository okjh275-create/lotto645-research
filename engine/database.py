from pathlib import Path
import sqlite3

DB_PATH = Path("data") / "lotto.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS draw_history(
        round INTEGER PRIMARY KEY,
        n1 INTEGER NOT NULL,
        n2 INTEGER NOT NULL,
        n3 INTEGER NOT NULL,
        n4 INTEGER NOT NULL,
        n5 INTEGER NOT NULL,
        n6 INTEGER NOT NULL,
        bonus INTEGER NOT NULL
    );
    """)

    conn.commit()
    conn.close()

    print(f"Database created : {DB_PATH}")