import re


def clean_text(text):

    text = text.lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()
    words = [w for w in words if len(w) > 2]

    return " ".join(words)
