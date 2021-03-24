from gensim.test.utils import common_texts
from gensim.models import FastText
import random
import numpy as np

# target_word_candidates = similar_by_vector(target_word_vector,top=3)

model = FastText(sentences=common_texts, window=5, min_count=1, workers=4)
model.save("./embedding/ft.model")
model = FastText.load("./embedding/ft.model")

word = model.wv['betroot']
print(model.wv.vectors_vocab)

print(model.wv.similar_by_vector(word))