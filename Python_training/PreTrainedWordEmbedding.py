from gensim.models import KeyedVectors

# load the Stanford GloVe model
filename = './embedding/glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['prince', 'woman'], negative=['man'], topn=1)
print(result[0][0])
vector = model.get_vector(str(result[0][0]))
print(model.similar_by_vector(vector, topn=2))