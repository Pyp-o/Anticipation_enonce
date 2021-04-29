import dataPrep
import pickle
import sys

def prepareData():
    # importing and preparing data
    df = open(r'../../DataBase/dialog/dialogues_text.txt', encoding='utf-8')  # relative path from main
    data = dataPrep.parseDialogs(df)
    df.close()

    data = dataPrep.parseUtterances(data)
    data = dataPrep.parsePhrase(data)
    data = dataPrep.removePunctuation(data)
    # make each phrase as an entry of array
    data = dataPrep.dataAsArray(data)
    data = dataPrep.rmSpaces(data)
    print("data prepared")
    vocab = dataPrep.vocabConstruct(data)
    vocab = sorted(vocab, key=str.lower)  # tri du vocabulaire par ordre alphabétique -> permet d'avoir des mots proches lorsque l'erreur de prédiction est faible
    print("vocab constructed")
    word_to_oneHot, oneHot_to_word, word_to_ix, ix_to_word, n_features, oneHot_to_ix, ix_to_oneHot = dataPrep.encodeWord(vocab, "oneHot")
    data = dataPrep.convertPhrasetoIx(data, word_to_oneHot)
    print("words converted to oneHot")

    """
    with open('OneHotdata0.txt', 'wb') as fp:
        pickle.dump(data[:int(len(data)/3)], fp)
    with open('OneHotdata1.txt', 'wb') as fp:
        pickle.dump(data[int(len(data)/3):2*int(len(data)/3)], fp)
    with open('OneHotdata1.txt', 'wb') as fp:
        pickle.dump(data[2*int(len(data)/3):], fp)
    with open('Vocabdata.txt', 'wb') as fp:
        pickle.dump(vocab, fp)
    with open('WordToIxdata.txt', 'wb') as fp:
        pickle.dump(word_to_ix, fp)
    with open('IxToWorddata.txt', 'wb') as fp:
        pickle.dump(ix_to_word, fp)
    with open('WordToOneHotdata.txt', 'wb') as fp:
        pickle.dump(word_to_oneHot, fp)
    with open('OneHotToWorddata.txt', 'wb') as fp:
        pickle.dump(oneHot_to_word, fp)
    """

    return data, vocab, word_to_oneHot, oneHot_to_word, word_to_ix, ix_to_word, n_features, oneHot_to_ix, ix_to_oneHot