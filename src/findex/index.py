
import json
import pickle
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from pathlib import Path
from multiprocessing import get_context
import concurrent.futures
from functools import cached_property

from findex.corpus import iter_documents
from findex.tokenize import tokenize_text
from findex.models import DocMeta


class InvertedIndex:
    """Інвертований індекс із використанням NumPy-масивів."""

    def __init__(self, keep_positions: bool = False):
        self.keep_positions = keep_positions
        self.index: Dict[str, Dict[str, np.ndarray]] = {}
        self.metadata: Dict[int, DocMeta] = {}
        self.doc_lengths: np.ndarray = np.array([], dtype=np.int32)

    @property
    def num_docs(self) -> int:
        """Кількість документів в індексі."""
        return len(self.metadata)

    @cached_property
    def avg_doc_length(self) -> float:
        """Середня кількість токенів у документі."""
        if not self.metadata:
            return 0.0
        return sum(meta.length for meta in self.metadata.values()) / self.num_docs

    def doc_length(self, doc_id: int) -> int:
        """Довжина документа за його ID; KeyError для невідомого ID."""
        return self.metadata[doc_id].length

    def df(self, term: str) -> int:
        """Кількість документів, що містять термін."""
        posting = self.index.get(term)
        return 0 if posting is None else len(posting["doc_ids"])

    def __enter__(self) -> "InvertedIndex":
        """Дозволяє використовувати with open_index(path) as ix."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Файли закриті в load_pickle/load_json; винятки не приховуємо."""
        return False

    # Спеціальні методи Python — лабораторна №3

    def __len__(self) -> int:
        """Кількість унікальних термінів в індексі."""
        return len(self.index)

    def __contains__(self, term: str) -> bool:
        """Перевірка наявності терміна в індексі."""
        return term in self.index

    def __getitem__(self, term: str) -> Dict[str, np.ndarray]:
        """Повертає постинги терміна або викликає KeyError."""
        return self.index[term]

    def __iter__(self):
        """Ітерація за термінами індексу."""
        return iter(self.index)

    def __repr__(self) -> str:
        """Текстове представлення індексу."""
        return (
            f"InvertedIndex("
            f"documents={len(self.metadata)}, "
            f"terms={len(self.index)})"
        )

    def build(self, corpus_dir: str):
        """Послідовна побудова інвертованого індексу."""

        temp_index = defaultdict(
            lambda: {"doc_ids": [], "tfs": []}
        )

        # Очищення попередніх даних перед новою індексацією
        self.index = {}
        self.metadata = {}
        self.doc_lengths = np.array([], dtype=np.int32)
        self.__dict__.pop("avg_doc_length", None)

        # Генератор повертає об'єкти Document
        for doc_id, document in enumerate(
            iter_documents(Path(corpus_dir)), start=1
        ):
            filename = str(document.path)
            tokens = list(tokenize_text(document.text))
            doc_len = len(tokens)

            self.metadata[doc_id] = DocMeta(
                doc_id=doc_id,
                filename=filename,
                length=doc_len
            )

            counts = Counter(tokens)

            for token, count in counts.items():
                temp_index[token]["doc_ids"].append(doc_id)
                temp_index[token]["tfs"].append(count)

        # Перетворення списків у NumPy-масиви
        for term, data in temp_index.items():
            self.index[term] = {
                "doc_ids": np.array(
                    data["doc_ids"], dtype=np.int32
                ),
                "tfs": np.array(
                    data["tfs"], dtype=np.int32
                )
            }

        # Формування масиву довжин документів
        if self.metadata:
            max_id = max(self.metadata.keys())

            lengths_arr = np.zeros(
                max_id + 1, dtype=np.int32
            )

            for doc_id, meta in self.metadata.items():
                lengths_arr[doc_id] = meta.length

            self.doc_lengths = lengths_arr

    def save_pickle(self, filepath: str):
        """Збереження індексу у форматі Pickle."""

        with open(filepath, "wb") as f:
            pickle.dump(
                (
                    self.keep_positions,
                    self.metadata,
                    self.index,
                    self.doc_lengths
                ),
                f
            )

    @classmethod
    def load_pickle(cls, filepath: str) -> "InvertedIndex":
        """Завантаження індексу з Pickle."""

        with open(filepath, "rb") as f:
            (
                keep_positions,
                metadata,
                index_data,
                doc_lengths
            ) = pickle.load(f)

        obj = cls(keep_positions=keep_positions)
        obj.metadata = metadata
        obj.index = index_data
        obj.doc_lengths = doc_lengths

        return obj

    def save_json(self, filepath: str):
        """Збереження індексу у форматі JSON."""

        json_index = {
            term: {
                "doc_ids": data["doc_ids"].tolist(),
                "tfs": data["tfs"].tolist()
            }
            for term, data in self.index.items()
        }

        data = {
            "keep_positions": self.keep_positions,
            "metadata": {
                str(doc_id): {
                    "doc_id": meta.doc_id,
                    "filename": meta.filename,
                    "length": meta.length
                }
                for doc_id, meta in self.metadata.items()
            },
            "index": json_index
        }

        with open(
            filepath, "w", encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load_json(cls, filepath: str) -> "InvertedIndex":
        """Завантаження індексу з JSON."""

        with open(
            filepath, "r", encoding="utf-8"
        ) as f:
            data = json.load(f)

        obj = cls(
            keep_positions=data["keep_positions"]
        )

        obj.metadata = {
            int(k): DocMeta(
                v["doc_id"],
                v["filename"],
                v["length"]
            )
            for k, v in data["metadata"].items()
        }

        obj.index = {
            term: {
                "doc_ids": np.array(
                    item["doc_ids"], dtype=np.int32
                ),
                "tfs": np.array(
                    item["tfs"], dtype=np.int32
                )
            }
            for term, item in data["index"].items()
        }

        if obj.metadata:
            max_id = max(obj.metadata.keys())

            lengths_arr = np.zeros(
                max_id + 1, dtype=np.int32
            )

            for doc_id, meta in obj.metadata.items():
                lengths_arr[doc_id] = meta.length

            obj.doc_lengths = lengths_arr

        return obj


def open_index(filepath: str) -> InvertedIndex:
    """Відкриття індексу залежно від формату."""

    path = Path(filepath)

    if path.suffix == ".json":
        return InvertedIndex.load_json(filepath)

    return InvertedIndex.load_pickle(filepath)


def build_partial(
    file_paths: List[str],
    start_doc_id: int,
    keep_positions: bool = False
) -> Tuple[
    Dict[str, Dict[str, List[int]]],
    Dict[int, DocMeta]
]:
    """Побудова часткового індексу."""

    local_index = defaultdict(
        lambda: {"doc_ids": [], "tfs": []}
    )

    local_metadata: Dict[int, DocMeta] = {}
    doc_id = start_doc_id

    for fp in file_paths:
        path = Path(fp)

        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        tokens = list(tokenize_text(text))
        doc_len = len(tokens)

        local_metadata[doc_id] = DocMeta(
            doc_id=doc_id,
            filename=str(path),
            length=doc_len
        )

        counts = Counter(tokens)

        for token, count in counts.items():
            local_index[token]["doc_ids"].append(doc_id)
            local_index[token]["tfs"].append(count)

        doc_id += 1

    serializable_index = {
        term: {
            "doc_ids": data["doc_ids"],
            "tfs": data["tfs"]
        }
        for term, data in local_index.items()
    }

    return serializable_index, local_metadata


def merge(
    partials
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[int, DocMeta]
]:
    """Об'єднання часткових індексів."""

    merged_temp = defaultdict(
        lambda: {"doc_ids": [], "tfs": []}
    )

    merged_metadata: Dict[int, DocMeta] = {}

    for index_part, meta_part in partials:
        merged_metadata.update(meta_part)

        for term, data in index_part.items():
            merged_temp[term]["doc_ids"].extend(
                data["doc_ids"]
            )
            merged_temp[term]["tfs"].extend(
                data["tfs"]
            )

    final_index = {}

    for term, data in merged_temp.items():
        doc_ids = np.array(
            data["doc_ids"], dtype=np.int32
        )

        tfs = np.array(
            data["tfs"], dtype=np.int32
        )

        # Сортування постингів за doc_id
        sort_idx = np.argsort(doc_ids)

        final_index[term] = {
            "doc_ids": doc_ids[sort_idx],
            "tfs": tfs[sort_idx]
        }

    return final_index, merged_metadata


def build_index_parallel(
    corpus_dir: str,
    workers: int = 1,
    executor_type: str = "serial",
    keep_positions: bool = False
) -> InvertedIndex:
    """Послідовна або паралельна побудова індексу."""

    if workers < 1:
        raise ValueError("workers must be >= 1")

    path = Path(corpus_dir)

    files = sorted([
        str(p)
        for p in path.glob("**/*")
        if p.is_file() and p.suffix in (".txt", ".md")
    ])

    if not files:
        return InvertedIndex(
            keep_positions=keep_positions
        )

    chunk_size = max(1, len(files) // workers)

    chunks = []
    current_id = 1

    for i in range(0, len(files), chunk_size):
        chunk_files = files[i:i + chunk_size]

        if chunk_files:
            chunks.append(
                (chunk_files, current_id)
            )

            current_id += len(chunk_files)

    partials = []

    if executor_type == "serial":
        for chunk_files, start_id in chunks:
            partials.append(
                build_partial(
                    chunk_files,
                    start_id,
                    keep_positions
                )
            )

    elif executor_type == "threads":
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = [
                executor.submit(
                    build_partial,
                    cf,
                    sid,
                    keep_positions
                )
                for cf, sid in chunks
            ]

            partials = [
                future.result()
                for future in futures
            ]

    elif executor_type == "processes":
        ctx = get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx
        ) as executor:

            futures = [
                executor.submit(
                    build_partial,
                    cf,
                    sid,
                    keep_positions
                )
                for cf, sid in chunks
            ]

            partials = [
                future.result()
                for future in futures
            ]

    else:
        raise ValueError(
            f"Unknown executor type: {executor_type}"
        )

    index_data, metadata = merge(partials)

    idx = InvertedIndex(
        keep_positions=keep_positions
    )

    idx.index = index_data
    idx.metadata = metadata

    if metadata:
        max_id = max(metadata.keys())

        lengths_arr = np.zeros(
            max_id + 1, dtype=np.int32
        )

        for doc_id, meta in metadata.items():
            lengths_arr[doc_id] = meta.length

        idx.doc_lengths = lengths_arr

    return idx


# Сумісність з іншими модулями Findex
Index = InvertedIndex
