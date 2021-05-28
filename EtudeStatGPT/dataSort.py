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

def DictToSorted(file, output):
    data = pickle.load(open(file, 'rb'))

    sorted_data = []
    for dict in data:
        sorted_data.append(sorted(dict))

    with open(output, 'wb') as fp:
        pickle.dump(sorted_data, fp)

############    MAIN    ############
RAW_FILE = '../../DataBase/dialog/dialogues_text.txt'
EDITED_FILE = "./ParsedData.txt"
SORTED_FILE = "./SortedData.txt"

#saveDictFromRaw(RAW_FILE)
#DictToSorted(EDITED_FILE, SORTED_FILE)

data = pickle.load(open(SORTED_FILE, 'rb'))