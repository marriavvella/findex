from findex.tokenize import tokenize


def test_tokenize_basic():
    text = "Привіт, світ! Це тестовий рядок."
    tokens = list(tokenize(text))
    assert tokens == ["привіт", "світ", "це", "тестовий", "рядок"]


def test_tokenize_unicode_nfc():
    # Перевірка, що різні варіанти написання символу 'é' дають однаковий результат
    text1 = "café"
    text2 = "cafe\u0301"
    assert list(tokenize(text1)) == list(tokenize(text2))


def test_tokenize_empty():
    assert list(tokenize("")) == []
