import pickle
import sys
from random import randint

def selectDict(data, length):       #select index of dictionaries linked to phrase length
    index = 0
    maxLen = len(data[len(data)-1][0].split())
    minLen = len(data[0][0].split())

    if maxLen<length:
        maxlen=str(maxLen)
        raise IndexError("length too long : "+maxLen+" max expected")
    elif minLen>length:
        minLen=str(minLen)
        raise IndexError("length too short : "+minLen+" min expected")

    for i in range(len(data)):
        if len(data[i][0].split())==length:
            index = i
    return index


def selectPhraseFromDict(data, output, number, index):        #select random phrases from certain dict
    seed = []
    max = len(data[index])
    for i in range(number):
        index_sample = randint(0,max-1)
        seed.append(data[index][index_sample])

    with open(output, 'wb') as fp:
        pickle.dump(seed, fp)

    return seed

def selectPhrases(file, min_length=2, max_length=18, number=1):           #select random phrases
    data = pickle.load(open(file, 'rb'))
    LENGTH = range(min_length, max_length)
    for leng in LENGTH:
        OUTPUT = "./SelectedPhrases/Phrases_" + str(leng) + "_" + str(number) + ".txt"
        index = selectDict(data, length=leng)  # select dictionary depending on phrase length
        seed = selectPhraseFromDict(data, output=OUTPUT, number=number, index=index)  # return 'number' phrases of the dict at the 'index'
    return seed

def generateSEED(minlength=2, maxlength=18, length=range(1,17)):             #from selected phrases, generate Seeds of certain length
    LENGTH = range(minlength, maxlength)
    NUMBER = 200
    for inputLength in length:
        for leng in LENGTH:
            input = []
            output = []
            FILE = "./SelectedPhrases/Phrases_"+str(leng)+"_"+str(NUMBER)+".txt"
            INPUT_FILE = "./Seeds/input_" + str(leng) + "_" + str(NUMBER) + "_" + str(inputLength) + ".txt"
            OUTPUT_FILE = "./Seeds/output_" + str(leng) + "_" + str(NUMBER) + "_" + str(inputLength) + ".txt"
            data = pickle.load(open(FILE, 'rb'))
            for phrase in data:
                input.append(phrase.split()[:inputLength])
                output.append(phrase.split()[inputLength:])

            with open(INPUT_FILE, 'wb') as fp:
                pickle.dump(input, fp)
            with open(OUTPUT_FILE, 'wb') as fp:
                pickle.dump(output, fp)

"""--------------------------------------------------"""
CLEAN_FILE = "./Data/CleanData.txt"
SELECTED_FILE = "./SelectedPhrases/Phrases_2_200.txt"
FILE = "./Seeds/output_5_200_2.txt"

#selectPhrases(CLEAN_FILE, length=18, number=200)
generateSEED()
