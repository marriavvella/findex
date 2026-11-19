
import sys

from collections import Counter

from findex import stats


def test_stats_main_output(tmp_path, monkeypatch, capsys):
    """Перевіряє виведення статистики корпусу."""

    (tmp_path / "doc1.txt").write_text(
        "apple banana apple",
        encoding="utf-8",
    )

    (tmp_path / "doc2.txt").write_text(
        "banana orange",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["stats", str(tmp_path)],
    )

    stats.main()

    output = capsys.readouterr().out

    assert "Оброблено документів: 2" in output
    assert "Загальна кількість токенів: 5" in output
    assert "apple" in output
    assert "banana" in output


def test_stats_with_limit(tmp_path, monkeypatch, capsys):
    """Перевіряє обмеження кількості документів."""

    for i in range(3):
        (tmp_path / f"doc{i}.txt").write_text(
            "apple banana",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["stats", str(tmp_path), "--limit", "1"],
    )

    stats.main()

    output = capsys.readouterr().out

    assert "Оброблено документів: 1" in output
    assert "Загальна кількість токенів: 2" in output


def test_stats_empty_corpus(tmp_path, monkeypatch, capsys):
    """Перевіряє статистику порожнього корпусу."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["stats", str(tmp_path)],
    )

    stats.main()

    output = capsys.readouterr().out

    assert "Оброблено документів: 0" in output
    assert "Загальна кількість токенів: 0" in output


def test_stats_without_arguments(monkeypatch):
    """Перевіряє повідомлення про відсутність аргументів."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["stats"],
    )

    try:
        stats.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected SystemExit")


def test_stats_top_terms(tmp_path, monkeypatch, capsys):
    """Перевіряє відображення найчастіших термінів."""

    (tmp_path / "doc.txt").write_text(
        "apple apple apple banana banana orange",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["stats", str(tmp_path)],
    )

    stats.main()

    output = capsys.readouterr().out

    assert "apple: 3" in output
    assert "banana: 2" in output
    assert "orange: 1" in output
