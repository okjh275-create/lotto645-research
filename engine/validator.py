from pathlib import Path
import sqlite3


class Validator:
    def __init__(self):
        self.db_path = Path("data") / "lotto.db"

    def run(self):
        if not self.db_path.exists():
            print("Database not found.")
            return

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            SELECT round,n1,n2,n3,n4,n5,n6,bonus
            FROM draw_history
            ORDER BY round
        """)

        rows = cur.fetchall()

        range_error = 0
        order_error = 0
        duplicate_number = 0
        bonus_error = 0

        rounds = []

        for row in rows:

            rnd = row[0]
            nums = list(row[1:7])
            bonus = row[7]

            rounds.append(rnd)

            # 번호 범위 검사
            if any(n < 1 or n > 45 for n in nums + [bonus]):
                range_error += 1
                print(f"[Range Error] Round {rnd}: {nums} Bonus={bonus}")

            # 오름차순 검사
            if nums != sorted(nums):
                order_error += 1
                print(f"[Order Error] Round {rnd}")
                print(f"  Stored : {nums}")
                print(f"  Sorted : {sorted(nums)}")

            # 번호 중복 검사
            if len(nums) != len(set(nums)):
                duplicate_number += 1
                print(f"[Duplicate Number] Round {rnd}: {nums}")

            # 보너스 중복 검사
            if bonus in nums:
                bonus_error += 1
                print(f"[Bonus Error] Round {rnd}: Bonus={bonus}")

        expected = list(range(min(rounds), max(rounds) + 1))
        missing = sorted(set(expected) - set(rounds))

        conn.close()

        print("=" * 50)
        print("Validation Report")
        print("=" * 50)
        print(f"Stored Draws     : {len(rows)}")
        print(f"First Round      : {min(rounds)}")
        print(f"Latest Round     : {max(rounds)}")
        print(f"Missing Rounds   : {len(missing)}")
        print(f"Range Errors     : {range_error}")
        print(f"Order Errors     : {order_error}")
        print(f"Duplicate Number : {duplicate_number}")
        print(f"Bonus Errors     : {bonus_error}")
        print("=" * 50)

        if (
            len(missing) == 0
            and range_error == 0
            and order_error == 0
            and duplicate_number == 0
            and bonus_error == 0
        ):
            print("DATABASE VALID")
        else:
            print("DATABASE INVALID")