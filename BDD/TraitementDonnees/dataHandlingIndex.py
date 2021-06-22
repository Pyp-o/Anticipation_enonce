import dataPrep
import pickle
import sys

def prepareData(min=-1, max=1):
    #importing and preparing data
    df = open(r'../../DataBase/dialog/dialogues_text.txt', encoding='utf-8')    #relative path from main
    data = dataPrep.parseDialogs(df)
    df.close()
    data = dataPrep.parseUtterances(data)
    data = dataPrep.parsePhrase(data)
    #data = dataPrep.removePunctuation(data)
    #make each phrase as an entry of array
    data = dataPrep.dataAsArray(data)
    data = dataPrep.rmSpaces(data)
    vocab = dataPrep.vocabConstruct(data)
    #vocab = sorted(vocab, key=str.lower)  # tri du vocabulaire par ordre alphabétique -> permet d'avoir des mots proches lorsque l'erreur de prédiction est faible
    word_to_ix, ix_to_word = dataPrep.encodeWord(vocab, 'index')
    #convert words to ix
    data = dataPrep.convertPhrasetoIx(data, word_to_ix)
    scaler = dataPrep.fitScaler(data, word_to_ix, min, max)

    with open('Indexdata.txt', 'wb') as fp:
        pickle.dump(data, fp)
    with open('Vocabdata.txt', 'wb') as fp:
        pickle.dump(vocab, fp)
    with open('WordToIxdata.txt', 'wb') as fp:
        pickle.dump(word_to_ix, fp)
    with open('IxToWorddata.txt', 'wb') as fp:
        pickle.dump(ix_to_word, fp)

    return data, word_to_ix, ix_to_word, scaler