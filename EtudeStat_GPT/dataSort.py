import pickle
import dataPrep
import sys

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


############    MAIN    ############
RAW_FILE = '../../DataBase/dialog/dialogues_text.txt'
EDITED_FILE = "./ParsedData.txt"
#saveDictFromRaw(RAW_FILE)
data = pickle.load(open(EDITED_FILE, 'rb'))

#sort by alphabetical order
for dict in data:
    for phrase in dict:
        sys.exit()
