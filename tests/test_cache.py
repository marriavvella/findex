
import logging

from findex.index import InvertedIndex
from findex.utils import cached_search_ids


def test_search_cache(tmp_path, caplog):
    (tmp_path / "doc1.txt").write_text(
        "alice in wonderland",
        encoding="utf-8",
    )

    index = InvertedIndex()
    index.build(str(tmp_path))

    with caplog.at_level(logging.DEBUG, logger="findex.utils"):
        first_result = cached_search_ids(index, "alice")
        first_output = caplog.text

        caplog.clear()

        second_result = cached_search_ids(index, "alice")
        second_output = caplog.text

    assert first_result == second_result
    assert len(first_result) == 1

    assert "[CACHE MISS]" in first_output
    assert "[CACHE HIT]" in second_output


def test_cache_after_rebuild(tmp_path, caplog):
    document = tmp_path / "doc1.txt"
    document.write_text("alice", encoding="utf-8")

    index = InvertedIndex()
    index.build(str(tmp_path))

    with caplog.at_level(logging.DEBUG, logger="findex.utils"):
        assert len(cached_search_ids(index, "alice")) == 1

        caplog.clear()

        document.write_text("wonderland", encoding="utf-8")
        index.build(str(tmp_path))

        result = cached_search_ids(index, "alice")
        output = caplog.text

    assert result == []
    assert "[CACHE MISS]" in output
