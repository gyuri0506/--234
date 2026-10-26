import pandas as pd

frame = pd.read_csv("production.csv", encoding="utf-8-sig", dtype={"lot_id": "string"})
print(frame.dtypes)
print(frame.head(3).to_string(index=False))