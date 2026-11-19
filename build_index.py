
from pathlib import Path
from findex.index import build_index_parallel

corpus_dir = Path("data")
output_file = Path("index.pkl")

if not corpus_dir.is_dir():
    raise SystemExit("Помилка: папку data не знайдено")

index = build_index_parallel(
    corpus_dir=str(corpus_dir),
    workers=1,
    executor_type="serial",
)

if not index.metadata:
    raise SystemExit("Помилка: документи для індексації не знайдено")

index.save_pickle(str(output_file))

print("Індекс успішно створено!")
print(f"Документів: {len(index.metadata)}")
print(f"Унікальних термінів: {len(index.index)}")
print(f"Файл індексу: {output_file.resolve()}")
