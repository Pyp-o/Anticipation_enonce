import string
import re


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
            data.append(re.split('\! |\? |\.', e))
    data = deleteLastElement(data)
    return data

""" -------------------------------------------------------------------------
# limit length of each phrase                                               #
# input : array containing only one phrase by entry                         #
-------------------------------------------------------------------------"""
def limitLength(tab, length):
    data = []
    for i in range(len(tab)):
        if len(tab[i].split()) == length:
            data.append(tab[i])
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
# multiple encoding possibilities : index
-------------------------------------------------------------------------"""
def encodeWord(vocab, type='index'):
    if type == 'index':         #
        word_to_ix = {word: i for i, word in enumerate(vocab)}
        ix_to_word = {i: word for i, word in enumerate(vocab)}
    elif type == 'binaire':
        raise NameError('Not implemented yet')
    elif type == 'hexa':
        raise NameError('Not implemented yet')
    elif type == 'oneHot':
        raise NameError('Not implemented yet')
    elif type == 'notConitnuousIndex':
        raise NameError('Not implemented yet')
    else:
        NameError('Invalid encoding type')
    return word_to_ix, ix_to_word

""" -------------------------------------------------------------------------
# split X and y from dataset
-------------------------------------------------------------------------"""
def splitX_y(dataset, length):
    X = []
    y = []
    for phrase in dataset:
        X.append(phrase.split()[:length])
        y.append(phrase.split()[length:])
    return X,y