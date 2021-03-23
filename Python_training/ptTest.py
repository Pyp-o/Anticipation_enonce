import fasttext

model = fasttext.load_model('./fasttext/data/fil9.bin')
print(model.get_nearest_neighbors('asparagus'))