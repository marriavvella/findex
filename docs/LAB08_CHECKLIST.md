# Лабораторна №8 — перевірка після копіювання

Усі команди запускати з кореня репозиторію у VS Code.

## Встановлення і тести

```powershell
uv sync --all-groups
uv run pytest -q
uv run pytest benchmarks/test_benchmark_lab08.py --benchmark-only
uv run ruff check .
uv run pyright
```

## Профілювання (до і після оптимізації)

Для коректного порівняння старого та нового скорера потрібно виконати ті самі команди на збереженій гілці лабораторної №7 та на оновленій гілці. Результати не підставляти без вимірювань.

```powershell
uv run python -m cProfile -s cumulative -m findex.cli index corpus.jsonl --out index.pkl
uv run python -m cProfile -s cumulative -m findex.cli search "python AND search AND engine" --index index.pkl
uv run py-spy record -o docs/search_flame.svg -- uv run findex search "python AND search AND engine" --index index.pkl
```

Якщо py-spy на Windows повідомляє про відсутність дозволу, відкрити PowerShell від адміністратора або запустити профілювання через `uv run py-spy record -o docs/search_flame.svg -- python -m findex.cli search ...`.

## Вимірювання

Зробити таблицю для частого терміна, рідкісного терміна і тритермінового запиту: старий скорер (мс), NumPy (мс), прискорення. Записати 3 найзатратніші функції за cProfile/py-spy і класифікацію часу за Scalene. Запустити тест на корпусі ×10 для альтернативи семантичному пошуку (memmap або стиснення).

## Перед здачею

- Перевірити GitHub Actions.
- Доповнити README живою адресою, GIF, трьома числами, зведеними таблицями та обмеженнями.
- Записати 2–3-хвилинне відео й додати посилання до README.
- Оновити розгорнутий сервіс, перевірити його доступність.
- Після перевірок створити теги `lab-08` і `v1.0.0`.
