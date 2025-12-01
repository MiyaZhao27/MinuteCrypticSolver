from gensim.models import KeyedVectors

model = None


def get_model():
    global model
    if model is None:
        print("Loading fast GloVe 50d (Word2Vec format)...")
        model = KeyedVectors.load_word2vec_format(
            "glove50_word2vec.txt", binary=False)
        print("Loaded.")
    return model
