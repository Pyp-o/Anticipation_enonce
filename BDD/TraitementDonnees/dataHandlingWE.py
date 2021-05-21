import dataPrep
import pickle
from gensim.models import KeyedVectors

def printDATA(data):
    print(data[0])
    print(data[1])
    print(data[2])
    print("###")

def prepareData(filename):
    #importing and preparing data
    df = open(filename, encoding='utf-8')    #relative path from main
    data = dataPrep.parseDialogs(df)
    df.close()

    data = dataPrep.parseUtterances(data)
    data = dataPrep.parsePhrase(data)

    #data = dataPrep.removePunctuation(data)
    #make each phrase as an entry of array
    data = dataPrep.dataAsArray(data)
    data = dataPrep.rmSpaces(data)

    #import GloVe
    print("importing GloVe...")
    filename = '../embedding/glove.6B.100d.txt.word2vec'            #relative path from main
    glove = KeyedVectors.load_word2vec_format(filename, binary=False)
    print("GloVe imported")
    #convert words to ix
    data = dataPrep.convertPhrasetoWE(data, glove)

    with open('WEdata.txt', 'wb') as fp:
        pickle.dump(data, fp)

    return data, glove