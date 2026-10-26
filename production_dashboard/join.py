import csv
from pathlib import Path

def read(name):
    with Path(name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

production, quality = read("production.csv"), read("quality.csv")

index = {r["lot_id"]: r for r in production}

assert len(index) == len(production), "생산 LOT 중복"

unmatched = [q["lot_id"] for q in quality if q["lot_id"] not in index]

joined = [{**q, "equipment_id": index[q["lot_id"]]["equipment_id"]}
          for q in quality if q["lot_id"] in index]

print("생산 LOT:", len(production), "검사행:", len(quality))
print("연결행:", len(joined), "미매칭:", unmatched)
