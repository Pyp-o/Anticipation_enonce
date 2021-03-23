import fasttext

model = fasttext.load_model('./fasttext/data/fil9.bin') #can't add trained model to git
print(model.get_nearest_neighbors('asparagus'))