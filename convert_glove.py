from gensim.scripts.glove2word2vec import glove2word2vec

glove_input = "glove.6B.50d.txt"
word2vec_output = "glove50_word2vec.txt"

print("Converting GloVe -> Word2Vec format...")
glove2word2vec(glove_input, word2vec_output)
print("Done! Saved as glove50_word2vec.txt")


# download from the stanford https://nlp.stanford.edu/data/glove.6B.zip
# extract glove.6B.50d.txt
