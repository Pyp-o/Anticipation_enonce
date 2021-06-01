import timeit


setup = """
from gensim.models import KeyedVectors

print("importing GloVe...")
filename = '../embedding/glove.6B.100d.txt.word2vec'            #relative path from main
glove = KeyedVectors.load_word2vec_format(filename, binary=False)
print("GloVe imported")

word = "test"
vector = glove.get_vector(word)
"""
stmt1='v=glove.get_vector(word)'
stmt2='w=glove.similar_by_vector(vector, topn=1)'


timer1 = timeit.timeit(stmt=stmt1, setup=setup, number=100000)
timer2 = timeit.timeit(stmt=stmt2, setup=setup, number=100000)

print("temps encodage:",timer1)
print("temps decodage",timer2)