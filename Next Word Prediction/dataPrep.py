import string
import re
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import matplotlib.pyplot as plt

""" -------------------------------------------------------------------------
# each dialog is separated by a return                                      #
# parse the whole in multiple dialog                                        #
-------------------------------------------------------------------------"""
def parseDialogs(tab):
    data = []
    for t in tab:
        data.append(t.split("\n"))
    return data

""" -------------------------------------------------------------------------
# every dialog is separated with "__eou__"                                  #
# parse each dialog into multiple utterance                                 #
-------------------------------------------------------------------------"""
def parseUtterances(tab):
    data = []
    for t in tab:
        data.append(t[0].split("__eou__"))
    data = deleteLastElement(data)
    return data

""" -------------------------------------------------------------------------
# some utterances have multiple phrases                                     #
# parse data to split every phrase                                          #
-------------------------------------------------------------------------"""
def parsePhrase(tab):
    data = []
    for d in tab:
        for e in d:
            c = e.split()[len(e.split())-1]
            phrase = re.split('\! |\? |\.', e)[0]
            if c=='.' or c=='!' or c=='?':
                data.append(phrase+c)
            else :
                data.append(phrase)

    #data = deleteLastElement(data)
    return data

""" -------------------------------------------------------------------------
# limit length of each phrase                                               #
# input : array containing only one phrase by entry                         #
-------------------------------------------------------------------------"""
def limitLength(tab, length):
    data = []
    for i in range(len(tab)):
        if len(tab[i]) == length:
            data.append(tab[i])
    return data

""" -------------------------------------------------------------------------
# limit length of each phrase                                               #
# input : array containing only one phrase by entry                         #
-------------------------------------------------------------------------"""
def limitLength2(tab, min=4, max=10):
    data = []
    zero = np.zeros(100)
    length = len(tab)
    for i in range(length):
        perc = int(i*100/length)
        l = len(tab[i])
        """ #large data print
        if perc % 20 == 0:
            print(f'epoch:{perc}/{100}')
        """
        if min<=l<=max:
            utt = tab[i]
            if l%2!=0:
                utt.append(zero)
                l = l+1
            if l!=max:
                n = max-l
                n = int(n/2)
                for i in range(n):
                    utt.insert(0, zero)
                    utt.append(zero)
            data.append(utt)
    return data

""" -------------------------------------------------------------------------
# remove punctuation                                                        #
-------------------------------------------------------------------------"""
def removePunctuation(tab):
    data = []
    for d in tab:
        for u in d:
            data.append(u.translate(str.maketrans('', '', string.punctuation)))
    return data

""" -------------------------------------------------------------------------
# previous function introduce empty entries                                 #
# delete these                                                              #
-------------------------------------------------------------------------"""
def deleteLastElement(tab):
    for t in tab:
        del t[-1]
    return tab

""" -------------------------------------------------------------------------
# transform data to have only phrases as entries of array                   #
-------------------------------------------------------------------------"""
def dataAsArray(tab):
    data = []
    for t in tab:
        data.append(t)
    return data

""" -------------------------------------------------------------------------
# constrcut vocabulary                                                      #
# order is random
-------------------------------------------------------------------------"""
def vocabConstruct(tab):
    data = []
    for t in tab:                           # for dialogue
        for d in t.split():                 # for utterance
            if d.lower() not in data:      # test with word lowered
                data.append(d.lower())     # word lowered in vocab
    return set(data)

""" -------------------------------------------------------------------------
# encode word                                                               #
# multiple encoding possibilities : index, one-hot encoding, word embedding
-------------------------------------------------------------------------"""
def encodeWord(vocab, type='oneHot'):
    if type == 'index':         #
        word_to_ix = {word: i for i, word in enumerate(vocab)}
        ix_to_word = {i: word for i, word in enumerate(vocab)}
        return word_to_ix, ix_to_word

    elif type == 'oneHot':
        data = []
        n_features = len(vocab)

        for i in range(n_features):
            ar = list(np.zeros(n_features, dtype=int))
            ar[i] = 1
            data.append(tuple(ar))
        vocab = tuple(vocab)
        data = tuple(data)

        word_to_oneHot = {vocab[i]: data[i] for i in range(n_features)}
        oneHot_to_word = {data[i]: vocab[i] for i in range(n_features)}
        word_to_ix = {word: i for i, word in enumerate(vocab)}
        ix_to_word = {i: word for i, word in enumerate(vocab)}

        return word_to_oneHot, oneHot_to_word, word_to_ix , ix_to_word, n_features

    elif type == 'wordEncoding':
        raise NameError('Not implemented yet')
        return word_to_ix, ix_to_word
    else:
        raise NameError('Invalid encoding type')
        return word_to_ix, ix_to_word

""" -------------------------------------------------------------------------
# split X and y from dataset                                                #
-------------------------------------------------------------------------"""
def splitX_y(dataset, length):
    X = []
    y = []
    for phrase in dataset:
        X.append(phrase[:length])
        y.append(phrase[length:])
    return X,y

def splitX_y2(dataset):
    X = []
    y = []
    for phrase in dataset:
        length = int(len(phrase)/2)
        X.append(phrase[:length])
        y.append(phrase[length:])
    return X,y

def sliding_XY(data):
    X = []
    Y = []
    zero = np.zeros(100)    #vector full of 0 for padding
    vec_zero = []

    for i in range(len(data[0])):
        vec_zero.append(zero)

    for phrase in data:
        if phrase[0].any() != zero.any():
            min = 0
        if phrase[len(phrase) - 1].any() != zero.any():
            max = len(phrase) - 2
        for i in range(len(phrase)):
            if phrase[i].any() != zero.any():
                min = i
                break
        for i in range(len(phrase) - 1, min, -1):
            if phrase[i].any() != zero.any():
                max = i
                break

        for i in range(min + 1, max + 1):
            x = vec_zero.copy()
            y = vec_zero.copy()
            for j in range(0, i):
                x[len(phrase) - i + j] = phrase[j]
            for j in range(len(y)):
                y[j] = phrase[i]
            X.append(x)
            Y.append(y)
            x = vec_zero.copy()
            y = vec_zero.copy()

    return X, Y

""" -------------------------------------------------------------------------
# remove spaces injected during parsing and cleaning data                   #
-------------------------------------------------------------------------"""
def rmSpaces(dataset):
    data = []
    for phrase in dataset:
        ph = ""
        for i in range(len(phrase.split())):
            if i < len(phrase.split()):
                ph += phrase.split()[i].lower() + " "
            else:
                ph += phrase.split()[i].lower()
        data.append(ph)
    return data

""" -------------------------------------------------------------------------
# convert words to ix                                                       #
# input : array [["word1", "word2" ...],["word1", "word2" ...]]             #
-------------------------------------------------------------------------"""
def convertPhrasetoIx(dataset, word_to_ix):
    data = []
    fail = 0
    for phrase in dataset:
        encodedPhrase = []
        for word in phrase.split():
            try:
                encodedPhrase.append(word_to_ix.get_vector(word.lower()))
            except:
                encodedPhrase = []
                break
        if encodedPhrase!=[]:
            data.append(encodedPhrase)
    return data

""" -------------------------------------------------------------------------
# convert words to vector                                                    #
# input : array [["v1", "v2", ...],["v1", "v2", ...]]                          #
-------------------------------------------------------------------------"""
def convertPhrasetoWE(dataset, glove):
    data = []
    for phrase in dataset:
        encodedPhrase = []
        for word in phrase.split():
            try:
                encodedPhrase.append(glove.get_vector(word.lower()))
            except:
                encodedPhrase = []
                break
        if encodedPhrase!=[]:
            data.append(encodedPhrase)
    return data

""" -------------------------------------------------------------------------
# convert an array shape ['word0', 'word1', 'word2', 'word3' ...] to ix     #
-------------------------------------------------------------------------"""
def convertWordstoIx(dataset, word_to_ix):
    data = []
    for i in range(len(dataset)):
        data.append(word_to_ix[dataset[i].lower()])
    return data

""" -------------------------------------------------------------------------
# convert an array shape ['ix0', 'ix1', 'ix2', 'ix3' ...] to word           #
-------------------------------------------------------------------------"""
def convertIxtoPhrase(dataset, ix_to_word):
    data = []
    for phrase in dataset:
        p = []
        for i in range(len(phrase)):
            if phrase[i]-int(phrase[i])==0.5:
                print("!")
                p.append(ix_to_word[math.ceil(phrase[i])])  #arrondi inférieur
                p.append(ix_to_word[math.floor(phrase[i])]) #arrondi supérieur
            else :
                p.append(ix_to_word[round(phrase[i])])
        data.append(p)
    return data

""" -------------------------------------------------------------------------
# crate a scaler and fit it                                                 #
# allow to fit on the whole dataset, not just the parsed one                #
-------------------------------------------------------------------------"""
def fitScaler(dataset, word_to_ix, min=-1, max=1):
    data = []
    for phrase in dataset:
        for word in phrase.split():
            data.append(word)
    test = convertWordstoIx(data, word_to_ix)
    test = np.reshape(test, (-1, 1))
    scaler = MinMaxScaler(feature_range=(min, max))
    scaler = scaler.fit(test)
    return scaler

""" -------------------------------------------------------------------------
# reverse predicted tensor from gpu to cpu and from torch.tensor            #
# to numpy.array                                                            #
-------------------------------------------------------------------------"""
def reverseTensor(tensors):
    converted = []
    for tensor in tensors:
        a = []
        tensor = tensor.cpu()
        tensor = tensor.detach().numpy()
        for value in tensor[0]:
            a.append(value[0])
        converted.append(a)
    return converted

def oneHotClean(dataset, ix_to_word):
    data = []
    for phrase in dataset:
        for words in phrase:
            ph = []
            for word in words:
                w = []
                index = 0
                max = -10
                for i in range(len(word)):
                    if word[i] > max:
                        max = word[i]
                        index = i
                for i in range(len(word)):
                    if i == index:
                        w.append(1)
                    else:
                        w.append(0)
                ph.append(ix_to_word[tuple(w)])
            data.append(ph)
    return data

def reverseOneHot(dataset, ix_to_word):
    data = []
    for phrase in dataset:
        ph = []
        for words in phrase:
            ph.append(ix_to_word[words])
        data.append(ph)
    return data

def reverseEmbed(dataset, embed):
    data = []
    for phrase in dataset:
        ph = []
        for word in phrase:
            ph.append(embed.similar_by_vector(word, topn=1))
        data.append(ph)
    return data

def plotLoss(loss):
    plt.figure()
    plt.plot(np.log10(loss))
    plt.title('Learning curve')
    plt.ylabel('loss: log10(MSE)')
    plt.xlabel('epoch')
    plt.show()
    plt.close('all')
    return 0