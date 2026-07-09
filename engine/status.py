from pathlib import Path
import sqlite3

DB_PATH = Path("data") / "lotto.db"


def database_status():

    if not DB_PATH.exists():
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM draw_history")
    total = cur.fetchone()[0]

    cur.execute("SELECT MIN(round), MAX(round) FROM draw_history")
    first_round, latest_round = cur.fetchone()

    conn.close()

    print("=" * 50)
    print("Lotto645 Research Platform")
    print("=" * 50)
    print(f"Database      : {DB_PATH}")
    print(f"Stored Draws  : {total}")
    print(f"First Round   : {first_round}")
    print(f"Latest Round  : {latest_round}")
    print("=" * 50)