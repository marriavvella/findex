
import re
from typing import Set
from findex.index import Index
from findex.tokenize import tokenize


class ASTNode:
    def evaluate(self, index: Index) -> Set[int]:
        raise NotImplementedError

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


class Term(ASTNode):
    def __init__(self, term: str):
        self.term = " ".join(tokenize(term))

    def evaluate(self, index: Index) -> Set[int]:
        postings = index.index.get(self.term)
        if postings is None:
            return set()
        return {int(doc_id) for doc_id in postings["doc_ids"]}

    def __eq__(self, other):
        return isinstance(other, Term) and self.term == other.term

    def __repr__(self):
        return f"Term('{self.term}')"


class Phrase(ASTNode):
    def __init__(self, phrase: str):
        self.tokens = list(tokenize(phrase))

    def evaluate(self, index: Index) -> Set[int]:
        if not self.tokens:
            return set()

        # Поточний індекс зберігає doc_ids і tfs,
        # але не позиції слів.
        # Тому поки шукаємо документи, що містять усі слова.
        result = Term(self.tokens[0]).evaluate(index)

        for token in self.tokens[1:]:
            result &= Term(token).evaluate(index)

        return result

    def __eq__(self, other):
        return isinstance(other, Phrase) and self.tokens == other.tokens

    def __repr__(self):
        return f"Phrase({' '.join(self.tokens)})"


class And(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left, self.right = left, right

    def evaluate(self, index: Index) -> Set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def __eq__(self, other):
        return (
            isinstance(other, And)
            and self.left == other.left
            and self.right == other.right
        )

    def __repr__(self):
        return f"And({self.left}, {self.right})"


class Or(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left, self.right = left, right

    def evaluate(self, index: Index) -> Set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def __eq__(self, other):
        return (
            isinstance(other, Or)
            and self.left == other.left
            and self.right == other.right
        )

    def __repr__(self):
        return f"Or({self.left}, {self.right})"


class Not(ASTNode):
    def __init__(self, child: ASTNode):
        self.child = child

    def evaluate(self, index: Index) -> Set[int]:
        return set(index.metadata.keys()) - self.child.evaluate(index)

    def __eq__(self, other):
        return isinstance(other, Not) and self.child == other.child

    def __repr__(self):
        return f"Not({self.child})"


def parse_query(query: str) -> ASTNode:
    tokens = re.findall(
        r'"[^"]*"|\'[^\']*\'|\(|\)|AND\b|OR\b|NOT\b|[^\s()]+',
        query,
        flags=re.IGNORECASE,
    )

    def parse_primary(pos):
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

    def parse_and(pos):
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

    def parse_or(pos):
        node, pos = parse_and(pos)

        while pos < len(tokens) and tokens[pos].upper() == "OR":
            right, pos = parse_and(pos + 1)
            node = Or(node, right)

        return node, pos

    node, _ = parse_or(0)
    return node
