"""
database.py

SQLite Database Manager
Lotto645 Research Platform v4.0
"""

from pathlib import Path
import sqlite3


DB_PATH = Path("data") / "lotto.db"


def get_connection():
    """SQLite 연결 반환"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database():
    """데이터베이스 및 테이블 생성"""

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS draw_history (
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

    print(f"Database initialized : {DB_PATH}")


if __name__ == "__main__":
    initialize_database()