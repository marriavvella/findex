

import pytest
from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


@pytest.fixture
def corpus(tmp_path):
    """Створює тестовий корпус."""

    (tmp_path / "doc1.txt").write_text(
        "apple banana apple garden",
        encoding="utf-8",
    )

    (tmp_path / "doc2.txt").write_text(
        "banana orange garden",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def saved_index(corpus, tmp_path):
    """Створює готовий тестовий індекс."""

    from findex.index import InvertedIndex

    index = InvertedIndex()
    index.build(str(corpus))

    path = tmp_path / "index.pkl"
    index.save_pickle(str(path))

    return path


def test_cli_build_success(corpus, tmp_path):
    """Перевіряє успішну побудову індексу."""

    output = tmp_path / "result.pkl"

    result = runner.invoke(
        app,
        ["build", str(corpus), "--out", str(output)],
    )

    assert result.exit_code == 0
    assert "Index built successfully!" in result.output
    assert output.exists()


def test_cli_build_json(corpus, tmp_path):
    """Перевіряє збереження індексу у JSON."""

    output = tmp_path / "result.json"

    result = runner.invoke(
        app,
        ["build", str(corpus), "--out", str(output)],
    )

    assert result.exit_code == 0
    assert output.exists()


def test_cli_build_invalid_directory(tmp_path):
    """Перевіряє обробку неіснуючої папки."""

    missing = tmp_path / "missing"

    result = runner.invoke(
        app,
        ["build", str(missing)],
    )

    assert result.exit_code == 1
    assert "Corpus directory not found" in result.output


def test_cli_build_invalid_extension(corpus, tmp_path):
    """Перевіряє заборону неправильного формату."""

    output = tmp_path / "result.txt"

    result = runner.invoke(
        app,
        ["build", str(corpus), "--out", str(output)],
    )

    assert result.exit_code == 1
    assert "Output must be .pkl or .json" in result.output


def test_cli_search_bm25(saved_index):
    """Перевіряє пошук із BM25."""

    result = runner.invoke(
        app,
        [
            "search",
            "apple",
            "--index",
            str(saved_index),
            "--scorer",
            "bm25",
        ],
    )

    assert result.exit_code == 0
    assert "Found 1 results" in result.output
    assert "Doc 1" in result.output


def test_cli_search_tfidf(saved_index):
    """Перевіряє пошук із TF-IDF."""

    result = runner.invoke(
        app,
        [
            "search",
            "banana",
            "--index",
            str(saved_index),
            "--scorer",
            "tfidf",
        ],
    )

    assert result.exit_code == 0
    assert "Found 2 results" in result.output


def test_cli_search_missing_index(tmp_path):
    """Перевіряє пошук без файлу індексу."""

    missing = tmp_path / "missing.pkl"

    result = runner.invoke(
        app,
        [
            "search",
            "apple",
            "--index",
            str(missing),
        ],
    )

    assert result.exit_code == 1
    assert "Index file not found" in result.output


def test_cli_search_invalid_scorer(saved_index):
    """Перевіряє неправильний алгоритм ранжування."""

    result = runner.invoke(
        app,
        [
            "search",
            "apple",
            "--index",
            str(saved_index),
            "--scorer",
            "unknown",
        ],
    )

    assert result.exit_code == 1
    assert "Scorer must be bm25 or tfidf" in result.output


def test_cli_search_invalid_top_k(saved_index):
    """Перевіряє обмеження top_k."""

    result = runner.invoke(
        app,
        [
            "search",
            "apple",
            "--index",
            str(saved_index),
            "--top-k",
            "0",
        ],
    )

    assert result.exit_code != 0


def test_cli_stats_success(saved_index):
    """Перевіряє виведення статистики індексу."""

    result = runner.invoke(
        app,
        [
            "stats",
            "--index",
            str(saved_index),
        ],
    )

    assert result.exit_code == 0
    assert "Findex Statistics" in result.output
    assert "Documents: 2" in result.output
    assert "Unique terms:" in result.output
    assert "Total tokens:" in result.output


def test_cli_stats_missing_index(tmp_path):
    """Перевіряє статистику неіснуючого індексу."""

    result = runner.invoke(
        app,
        [
            "stats",
            "--index",
            str(tmp_path / "missing.pkl"),
        ],
    )

    assert result.exit_code == 1
    assert "Index file not found" in result.output


def test_cli_help():
    """Перевіряє відображення довідки CLI."""

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "build" in result.output
    assert "search" in result.output
    assert "stats" in result.output
    assert "crawl" in result.output
    assert "serve" in result.output
