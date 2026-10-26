
import csv
from pathlib import Path

base = Path(__file__).resolve().parent

source = base / "로그" / "처리일지.csv"
output = base / "결과" / "처리일지.csv"

# 원본 파일 확인
if not source.exists():
    raise FileNotFoundError(f"원본 파일 없음: {source}")

# 원본 데이터 읽기
with source.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    if not columns:
        raise ValueError("CSV 헤더가 없습니다.")

    rows = list(reader)

# 기존 열 유지, 새 열 하나 추가
if "duration_ms" in columns:
    raise ValueError("duration_ms 열이 이미 존재합니다.")

new_columns = columns + ["duration_ms"]

# 결과 폴더 생성
output.parent.mkdir(parents=True, exist_ok=True)

# 새로운 파일 저장
with output.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=new_columns)
    writer.writeheader()

    for row in rows:
        row["duration_ms"] = ""
        writer.writerow(row)

print("새 CSV 파일 생성 완료!")
print("원본 파일은 수정하지 않았습니다.")
print("저장 위치:", output)
