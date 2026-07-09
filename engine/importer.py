from pathlib import Path
import sqlite3

DB_PATH = Path("data") / "lotto.db"
TXT_PATH = Path("data") / "history.txt"


def import_history():

    if not TXT_PATH.exists():
        raise FileNotFoundError(f"{TXT_PATH} 가 존재하지 않습니다.")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    inserted = 0
    updated = 0

    with TXT_PATH.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 8:
                print(f"Skip : {line}")
                continue

            nums = list(map(int, parts))

            cur.execute("""
            INSERT OR REPLACE INTO draw_history
            (round,n1,n2,n3,n4,n5,n6,bonus)
            VALUES(?,?,?,?,?,?,?,?)
            """, nums)

            inserted += 1

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM draw_history")
    total = cur.fetchone()[0]

    cur.execute("SELECT MAX(round) FROM draw_history")
    latest = cur.fetchone()[0]

    conn.close()

    print("=" * 50)
    print("Import Finished")
    print("=" * 50)
    print(f"Imported : {inserted}")
    print(f"Total DB : {total}")
    print(f"Latest Round : {latest}")