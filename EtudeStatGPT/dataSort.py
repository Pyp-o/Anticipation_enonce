import pickle
import dataPrep

def saveDictFromRaw(file, output, min=2, max=18, reduction=None):
    #importing and preparing data
    df = open(file, encoding='utf-8')    #relative path from main
    data = dataPrep.parseDialogs(df)
    df.close()
    data = dataPrep.parseUtterances(data)
    data = dataPrep.parsePhrase2(data)

    if reduction != None:
        data = data[:reduction]

    dictionary = []
    for j in range(min, max):
        tab = []
        for i in range(len(data)):
            if len(data[i].split())==j:
                tab.append(data[i])
        dictionary.append(tab)

    del data

    with open(output, 'wb') as fp:
        pickle.dump(dictionary, fp)

def dictToSorted(file, output):
    data = pickle.load(open(file, 'rb'))

    sorted_data = []
    for dict in data:
        sorted_data.append(sorted(dict))

    with open(output, 'wb') as fp:
        pickle.dump(sorted_data, fp)

def cleanSortedData(file, output):
    data = pickle.load(open(file, 'rb'))

    prev_phrase = ""
    dat = []
    for dict in data:
        dictionary = []
        for phrase in dict:
            if prev_phrase != phrase:
                dictionary.append(phrase)
                prev_phrase = phrase
        dat.append(dictionary)
    del data

    with open(output, 'wb') as fp:
        pickle.dump(dat, fp)


############    MAIN    ############
RAW_FILE = '../../DataBase/dialog/dialogues_text.txt'
EDITED_FILE = "./Data/ParsedData.txt"
SORTED_FILE = "./Data/SortedData.txt"
CLEAN_FILE = "./Data/CleanData.txt"

saveDictFromRaw(RAW_FILE, output=EDITED_FILE)
dictToSorted(EDITED_FILE, SORTED_FILE)
cleanSortedData(SORTED_FILE, CLEAN_FILE)

data = pickle.load(open(CLEAN_FILE, 'rb'))
