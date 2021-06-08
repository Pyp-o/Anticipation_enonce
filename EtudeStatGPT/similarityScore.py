import pickle, sys
from seedSelector import selectDict

LENGTH = range(3, 18)    #sentence length
NUMBER = 200            #number of phrases per dict
inputLength= range(2,17)    #length of input, indepent from sentence length

def main():
    dataFile = "./Data/CleanData.txt"
    data = pickle.load(open(dataFile, 'rb'))
    for inputLen in inputLength:
        for leng in LENGTH:
            if inputLen >= leng:
                print("passed")
                continue
            FILE = "./Predictions/InputLength_" + str(inputLen) + "/prediction_" + str(leng) + "_" + str(NUMBER) + "_" + str(inputLen) + ".txt"
            OUTPUT_FILE = "./Predictions/InputLength_" + str(inputLen) + "/scores_" + str(leng) + "_" + str(NUMBER) + "_" + str(inputLen) + ".txt"
            #predictions = open(FILE).read().split("\n")

            predictions = cleanSortedData(FILE)
            scores = []
            for predictPhrase in predictions:
                index = selectDict(data, len(predictPhrase.split()))  # get index of dict for each phrase
                dict = data[index]
                scores.append(getScore(predictPhrase, dict))

            with open(OUTPUT_FILE, 'wb') as fp:
                pickle.dump(scores, fp)
    return


def getScore(predictPhrase, dict):      #get
    predictPhrase = predictPhrase.split()
    n_juste = 0
    max_juste = 0
    for phrase in dict:
        n_juste = 0
        phrase = phrase.split()
        for i in range(len(phrase)):
            if phrase[i].lower() == predictPhrase[i].lower():
                n_juste+=1
        if n_juste == len(predictPhrase):
            return (predictPhrase, len(predictPhrase), n_juste)
        elif n_juste > max_juste:
            max_juste = n_juste
            ph = phrase
    return (predictPhrase, len(predictPhrase), max_juste)


def cleanSortedData(file):      #delete all double phrases in the sorted dict
    data = open(file).read().split("\n")
    prev_phrase = ""
    dat = []
    for phrase in data:
        if prev_phrase != phrase and phrase != "":
            dat.append(phrase)
            prev_phrase = phrase
    del data

    for i in range(len(dat)):
        pr = dat[i].split(".")
        if len(pr) > 1:
            dat[i] = dat[i].split(".")[0] + " ."
            continue
        pr = dat[i].split("!")
        if len(pr) > 1:
            dat[i] = dat[i].split("!")[0] + " !"
            continue
        pr = dat[i].split("?")
        if len(pr) > 1:
            dat[i] = dat[i].split("?")[0] + " ?"

    return dat

#-----------------------------------------------------------------------------------------------------------------------
#main()
"""
for inputLen in inputLength:
    for leng in LENGTH:
        if inputLen >= leng:
            continue
        FILE = "./Predictions/InputLength_" + str(inputLen) + "/prediction_" + str(leng) + "_" + str(NUMBER) + "_" + str(inputLen) + ".txt"
        OUTPUT_FILE = "./Predictions/InputLength_" + str(inputLen) + "/scores_" + str(leng) + "_" + str(NUMBER) + "_" + str(inputLen) + ".txt"


OUTPUT_FILE = "./Predictions/InputLength_" + str(10) + "/scores_" + str(11) + "_" + str(200) + "_" + str(10) + ".txt"
score = pickle.load(open(OUTPUT_FILE, 'rb'))
print(score)
"""
#TODO idée : faire l'étude des phrases qui n'ont pas été prédites pour déceler une faible régularité