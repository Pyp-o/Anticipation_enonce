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

    return data, vocab, word_to_oneHot, oneHot_to_word, word_to_ix, ix_to_word, n_features, oneHot_to_ix, ix_to_oneHot