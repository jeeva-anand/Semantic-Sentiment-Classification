import numpy as np
from gensim.models import Word2Vec


def train_w2v(tokenized_text, size=200):

    model = Word2Vec(
        sentences=tokenized_text,
        vector_size=size,
        window=5,
        min_count=2,
        workers=2,
        sg=1,
        seed=42,
        negative=10,
        hs=0
    )

    return model


def get_sentence_vector(model, tokens, size=200):

    vec = np.zeros(size)
    count = 0

    for w in tokens:
        if w in model.wv:
            vec += model.wv[w]
            count += 1

    if count != 0:
        vec = vec / count

    return vec
