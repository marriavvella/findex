
import re
from pathlib import Path

from findex.index import Index
from findex.tokenize import tokenize


class ASTNode:
    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def __and__(self, other: "ASTNode") -> "And":
        return And(self, other)

    def __or__(self, other: "ASTNode") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)


class Term(ASTNode):
    def __init__(self, term: str) -> None:
        self.term = " ".join(tokenize(term))

    def evaluate(self, index: Index) -> set[int]:
        postings = index.index.get(self.term)

        if postings is None:
            return set()

        return {
            int(doc_id)
            for doc_id in postings["doc_ids"]
        }

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Term)
            and self.term == other.term
        )

    def __repr__(self) -> str:
        return f"Term('{self.term}')"


class Phrase(ASTNode):
    def __init__(self, phrase: str) -> None:
        self.tokens: list[str] = list(tokenize(phrase))

    def evaluate(self, index: Index) -> set[int]:
        if not self.tokens:
            return set()

        # Відбираємо документи, які містять усі слова
        candidates: set[int] = Term(
            self.tokens[0]
        ).evaluate(index)

        for token in self.tokens[1:]:
            candidates &= Term(token).evaluate(index)

        # Перевіряємо точний порядок слів
        result: set[int] = set()
        phrase_length = len(self.tokens)

        for doc_id in candidates:
            filename = index.metadata[doc_id].filename

            try:
                with Path(filename).open(
                    "r", encoding="utf-8"
                ) as file:
                    document_tokens = list(
                        tokenize(file.read())
                    )
            except (OSError, UnicodeError):
                continue

            for i in range(
                len(document_tokens) - phrase_length + 1
            ):
                if (
                    document_tokens[i:i + phrase_length]
                    == self.tokens
                ):
                    result.add(doc_id)
                    break

        return result

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Phrase)
            and self.tokens == other.tokens
        )

    def __repr__(self) -> str:
        return f"Phrase({' '.join(self.tokens)})"


class And(ASTNode):
    def __init__(
        self,
        left: ASTNode,
        right: ASTNode,
    ) -> None:
        self.left = left
        self.right = right

    def evaluate(self, index: Index) -> set[int]:
        return (
            self.left.evaluate(index)
            & self.right.evaluate(index)
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, And)
            and self.left == other.left
            and self.right == other.right
        )

    def __repr__(self) -> str:
        return f"And({self.left}, {self.right})"


class Or(ASTNode):
    def __init__(
        self,
        left: ASTNode,
        right: ASTNode,
    ) -> None:
        self.left = left
        self.right = right

    def evaluate(self, index: Index) -> set[int]:
        return (
            self.left.evaluate(index)
            | self.right.evaluate(index)
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Or)
            and self.left == other.left
            and self.right == other.right
        )

    def __repr__(self) -> str:
        return f"Or({self.left}, {self.right})"


class Not(ASTNode):
    def __init__(self, child: ASTNode) -> None:
        self.child = child

    def evaluate(self, index: Index) -> set[int]:
        return (
            set(index.metadata.keys())
            - self.child.evaluate(index)
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Not)
            and self.child == other.child
        )

    def __repr__(self) -> str:
        return f"Not({self.child})"


def parse_query(query: str) -> ASTNode:
    tokens: list[str] = re.findall(
        r'"[^"]*"|\'[^\']*\'|\(|\)|AND\b|OR\b|NOT\b|[^\s()]+',
        query,
        flags=re.IGNORECASE,
    )

    def parse_primary(pos: int) -> tuple[ASTNode, int]:
        if pos >= len(tokens):
            return Term(""), pos

        token = tokens[pos]

        if token.upper() == "NOT":
            child, pos = parse_primary(pos + 1)
            return Not(child), pos

        if token == "(":
            node, pos = parse_or(pos + 1)

            if pos < len(tokens) and tokens[pos] == ")":
                pos += 1

            return node, pos

        if token.startswith(('"', "'")):
            return Phrase(token[1:-1]), pos + 1

        return Term(token), pos + 1

    def parse_and(pos: int) -> tuple[ASTNode, int]:
        node, pos = parse_primary(pos)

        while pos < len(tokens):
            token = tokens[pos].upper()

            if token in ("OR", ")"):
                break

            if token == "AND":
                pos += 1

            right, pos = parse_primary(pos)
            node = And(node, right)

        return node, pos

    def parse_or(pos: int) -> tuple[ASTNode, int]:
        node, pos = parse_and(pos)

        while (
            pos < len(tokens)
            and tokens[pos].upper() == "OR"
        ):
            right, pos = parse_and(pos + 1)
            node = Or(node, right)

        return node, pos

    node, _ = parse_or(0)
    return node
