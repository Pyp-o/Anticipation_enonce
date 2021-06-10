# creation: 21-apr-2021 pierre.chevaillier@enib.Fr

import sys
import pickle
from next_word_prediction import GPT2

# global variables
eos = ['.', '!', '?']


def predict_next(sentence, depth, corpus):
    if depth > maxLength:
        return 1
    nextWords = gpt2.predict_next(sentence, width)
    for w in nextWords:
        isTerminal = w in eos
        if isTerminal:
            sentence += w
            #print(sentence)
            corpus.write(sentence + '\n')
            return
        else:
            sep = '' if len(w) == 0 or w[0] == '\'' else ' '
            newSentence = sentence + sep + w
            depth += 1
            predict_next(sentence, depth, corpus)

    return

# next word predictor
gpt2 = GPT2()
depth = 0               #recursion limit
width = 2               #number of possibilities
LENGTH = range(2, 18)    #sentence length
NUMBER = 200            #number of phrases per dict
inputLength= range(1,17)    #length of input, indepent from sentence length
for inputLen in inputLength:
    for leng in LENGTH:
        print("phrase length :", leng)
        print("input length:", inputLen)
        if inputLen >= leng:
            print("passed")
            continue
        maxLength = leng-inputLen+1     #limit time consumption
        FILE = "./Seeds/input_" + str(leng) + "_" + str(NUMBER) + "_" + str(inputLen) + ".txt"
        OUTPUT_FILE = open("./Predictions/InputLength_" + str(inputLen) + "/prediction_" + str(leng) + "_" + str(NUMBER) + "_" + str(inputLen) + ".txt", 'w')
        data = pickle.load(open(FILE, 'rb'))
        for sentence in data:
            depth = 0
            s = ' '.join(sentence)  #concatenate each word to have input as 1 string
            predict_next(sentence=s, depth=depth, corpus=OUTPUT_FILE)
        OUTPUT_FILE.close()