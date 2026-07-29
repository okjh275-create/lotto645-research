from pathlib import Path
import sqlite3
from itertools import combinations
from collections import Counter


class PairEngine:

    def __init__(self):
        self.db_path = Path("data") / "lotto.db"

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def load_draws(self, until_round=None):

        conn = self._connect()
        cur = conn.cursor()

        if until_round is None:
            cur.execute("""
                SELECT round, n1, n2, n3, n4, n5, n6
                FROM draw_history
                ORDER BY round DESC
            """)
        else:
            cur.execute("""
                SELECT round, n1, n2, n3, n4, n5, n6
                FROM draw_history
                WHERE round <= ?
                ORDER BY round DESC
            """, (until_round,))

        rows = cur.fetchall()
        conn.close()

        return rows

    def pair_frequency(self, last_n=None, until_round=None):

        rows = self.load_draws(
            until_round=until_round,
        )

        if last_n is not None:
            rows = rows[:last_n]

        counter = Counter()

        for row in rows:
            for pair in combinations(sorted(row[1:]), 2):
                counter[pair] += 1

        return counter

    def number_scores(self, last_n=50, until_round=None):
        """
        번호별 Pair Score 계산
        """

        pair_freq = self.pair_frequency(
            last_n=last_n,
            until_round=until_round,
        )

        scores = Counter()

        for (a, b), count in pair_freq.items():
            scores[a] += count
            scores[b] += count

        return scores