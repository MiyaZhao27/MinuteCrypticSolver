import gensim.downloader as api

# created a function that gets the GloVe model furthered explained at:
# https://huggingface.co/fse/glove-wiki-gigaword-50

def get_model():
    print("Loading GloVe 50d embeddings from Gensim API…")
    model = api.load("glove-wiki-gigaword-50")
    print("Done!")
    return model

