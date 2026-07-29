from pathlib import Path
import sqlite3
from collections import Counter


class StatisticsEngine:

    def __init__(self):
        self.db_path = Path("data") / "lotto.db"

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def load_draws(self):
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT round, n1, n2, n3, n4, n5, n6
            FROM draw_history
            ORDER BY round DESC
        """)

        rows = cur.fetchall()
        conn.close()

        return rows

    def frequency(self, last_n=None, until_round=None):
        rows = self.load_draws()

        if last_n is not None:
            rows = rows[:last_n]

        counter = Counter()

        for row in rows:
            counter.update(row[1:])

        return counter

    def gap(self, until_round=None):
        rows = self.load_draws()

        latest_round = rows[0][0]
        gaps = {}

        for number in range(1, 46):
            gap = latest_round

            for row in rows:
                if number in row[1:]:
                    gap = latest_round - row[0]
                    break

            gaps[number] = gap

        return gaps

    def get_draw(self, round_no):
        """
        특정 회차의 당첨번호 반환
        """

        conn = self._connect()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT n1, n2, n3, n4, n5, n6
            FROM draw_history
            WHERE round=?
            """,
            (round_no,),
        )

        row = cur.fetchone()
        conn.close()

        if row is None:
            return None

        return list(row)

    def latest_round(self):
        """
        DB의 최신 회차 반환
        """

        rows = self.load_draws()

        if not rows:
            return None

        return rows[0][0]