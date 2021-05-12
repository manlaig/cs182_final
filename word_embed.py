import gensim
from gensim.models import KeyedVectors

MAX_REVIEW_SIZE = 2048

print("Loading word2vec model")
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print("Finished loading word2vec model")

# returns the word2vec representation
def embed(word):
    if word in model:
        return model[word]
    return [0.] * 300

# returns the word2vec representation for a sentence
def embed_review(text):
    if len(text) > MAX_REVIEW_SIZE:
        text = text[:MAX_REVIEW_SIZE]
    out = []
    for word in text.split(" "):
        out += [embed(word)]
    while len(out) < MAX_REVIEW_SIZE:
        out += [[0.] * 300]
    return out

# embeds the label (e.g. "3.0") into a one-hot vector
def embed_label(text):
    stars = float(text)
    out = []
    count = 0.0
    while count <= 5.0:
        out += [1 if count == stars else 0]
        count += 0.5
    return out
