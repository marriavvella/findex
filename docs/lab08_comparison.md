# Lab 08: Python loops vs NumPy

Synthetic corpus: 500 documents.

Median of 100 timed runs after 10 warm-ups. Times are in milliseconds.

This compares a Python-loop reference implementation with the current NumPy implementation. It is not a measurement of an earlier Git revision.

| Algorithm | Query | Python (ms) | NumPy (ms) | Speedup |
|---|---|---:|---:|---:|
| BM25 | Frequent | 0.7121 | 0.1940 | 3.67x |
| BM25 | Rare | 0.2913 | 0.1753 | 1.66x |
| BM25 | Three terms | 0.5086 | 0.2485 | 2.05x |
| TF-IDF | Frequent | 0.7350 | 0.1786 | 4.12x |
| TF-IDF | Rare | 0.2874 | 0.1637 | 1.76x |
| TF-IDF | Three terms | 0.5088 | 0.2140 | 2.38x |
