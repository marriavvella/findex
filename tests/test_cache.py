
from findex.index import InvertedIndex
from findex.utils import cached_search_ids


def test_search_cache(tmp_path, capsys):
    (tmp_path / "doc1.txt").write_text(
        "alice in wonderland",
        encoding="utf-8"
    )

    index = InvertedIndex()
    index.build(str(tmp_path))

    first_result = cached_search_ids(index, "alice")
    first_output = capsys.readouterr().out

    second_result = cached_search_ids(index, "alice")
    second_output = capsys.readouterr().out

    assert first_result == second_result
    assert len(first_result) == 1

    assert "[CACHE MISS]" in first_output
    assert "[CACHE HIT]" in second_output


def test_cache_after_rebuild(tmp_path, capsys):
    document = tmp_path / "doc1.txt"
    document.write_text("alice", encoding="utf-8")

    index = InvertedIndex()
    index.build(str(tmp_path))

    assert len(cached_search_ids(index, "alice")) == 1
    capsys.readouterr()

    document.write_text("wonderland", encoding="utf-8")
    index.build(str(tmp_path))

    result = cached_search_ids(index, "alice")
    output = capsys.readouterr().out

    assert result == []
    assert "[CACHE MISS]" in output
