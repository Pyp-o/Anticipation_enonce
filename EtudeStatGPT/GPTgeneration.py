# creation: 21-apr-2021 pierre.chevaillier@enib.Fr

import sys
import pickle
from next_word_prediction import GPT2

# global variables
eos = ['.', '!', '?']


def predict_next(sentence, depth, corpus, maxLength, num_write, max_pred):
    if depth > maxLength:   #limitation du nombre de mots à prédire
        #corpus.write(sentence + '\n')
        return num_write
    if num_write >= max_pred:   #limitation du nombre de prédictions par seed
        return num_write
    nextWords = gpt2.predict_next(sentence, width)
    for w in nextWords:
        isTerminal = w in eos
        if isTerminal:
            sentence += w
            #print(sentence)
            try:
                corpus.write(sentence + '\n')
                num_write+=1
            except:
                print("phrase error")
            return num_write
        else:
            sep = '' if len(w) == 0 or w[0] == '\'' else ' '
            newSentence = sentence + sep + w
            depth+=1
            num_write = predict_next(newSentence, depth, corpus, maxLength, num_write, max_pred)

    return num_write

def manquant():
    FILE = "./Seeds/input_" + str(17) + "_" + str(200) + "_" + str(2) + ".txt"
    OUTPUT_FILE = open("./Predictions/InputLength_" + str(2) + "/prediction_" + str(17) + "_" + str(200) + "_" + str(2) + ".txt", 'w')
    maxLength = 17 - 2 + 3  # limit time consumption
    data = pickle.load(open(FILE, 'rb'))
    for sentence in data:
        depth = 0
        num_write=0
        s = ' '.join(sentence)  # concatenate each word to have input as 1 string
        predict_next(sentence=s, depth=depth, corpus=OUTPUT_FILE, maxLength=maxLength, num_write=num_write, max_pred=max_pred)
    OUTPUT_FILE.close()

# next word predictor
gpt2 = GPT2()
depth = 0               #recursion limit
width = 2               #number of possibilities
LENGTH = range(3, 18)    #sentence length
NUMBER = 200            #number of phrases per dict
inputLength= range(6,17)    #length of input, indepent from sentence length
max_pred = 20

for inputLen in inputLength:
    for leng in LENGTH:
        print()
        print("phrase length :", leng)
        print("input length:", inputLen)
        if inputLen >= leng:
            print("passed")
            continue
        maxLength = leng-inputLen+1     #limit time consumption
        print("max length : ", maxLength)
        FILE = "./Seeds/input_" + str(leng) + "_" + str(NUMBER) + "_" + str(inputLen) + ".txt"
        OUTPUT_FILE = open("./Predictions/InputLength_" + str(inputLen) + "/prediction_" + str(leng) + "_" + str(NUMBER) + "_" + str(inputLen) + ".txt", 'w')
        data = pickle.load(open(FILE, 'rb'))
        for sentence in data:
            depth = 0
            num_write = 0
            s = ' '.join(sentence)  #concatenate each word to have input as 1 string
            n = predict_next(sentence=s, depth=depth, corpus=OUTPUT_FILE, maxLength=maxLength, num_write=num_write, max_pred=max_pred)
            #print("nombre de prédictions ", n)
        OUTPUT_FILE.close()