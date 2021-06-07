import pickle, sys
from seedSelector import selectDict

LENGTH = range(2, 18)    #sentence length
NUMBER = 200            #number of phrases per dict
inputLength= range(1,17)    #length of input, indepent from sentence length

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

            predictions = cleanSortedData(FILE, FILE)

            scores = []
            for predictPhrase in predictions:
                index = selectDict(data, len(predictPhrase.split()))  # get index of dict for each phrase
                dict = data[index]
                scores.append(getScore(predictPhrase, dict))
            print(scores)

    return

def getScore(predictPhrase, dict):      #get
    predictPhrase = predictPhrase.split()
    n_juste = 0
    max_juste = 0
    for phrase in dict:
        n_juste = 0
        phrase = phrase.split()
        for i in range(len(phrase)):
            if phrase[i] == predictPhrase[i]:
                n_juste+=1

        if n_juste == len(predictPhrase):
            return (predictPhrase, len(predictPhrase), n_juste)
        elif n_juste > max_juste:
            max_juste = n_juste
    return (predictPhrase, len(predictPhrase), max_juste)


def cleanSortedData(file, output):      #delete all double phrases in the sorted dict
    data = open(file).read().split("\n")

    prev_phrase = ""
    dat = []
    for phrase in data:
        if prev_phrase != phrase and phrase != "":
            dat.append(phrase)
            prev_phrase = phrase
    del data

    return dat

#-----------------------------------------------------------------------------------------------------------------------
FILE = "./Predictions/InputLength_" + str(2) + "/prediction_" + str(3) + "_" + str(200) + "_" + str(2) + ".txt"
OUTPUT_FILE = "./Predictions/InputLength_" + str(2) + "/scores_" + str(3) + "_" + str(200) + "_" + str(2) + ".txt"

scores = pickle.load(open(OUTPUT_FILE, 'rb'))