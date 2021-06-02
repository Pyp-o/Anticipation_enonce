# creation: 21-apr-2021 pierre.chevaillier@enib.Fr
import sys
import pickle
from next_word_prediction import GPT2

# global variables
eos = ['.', '!', '?']


# next word predictor
gpt2 = GPT2()


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
            predict_next(newSentence, depth, corpus)

    return


# initialText = "The course starts next"

width = 5
depth = 0
sentence = "How may I"
maxLength = 3 + 1 # last : end of sentence
#predict_next(sentence, depth)


LENGTH = range(2, 18)
NUMBER = 200

for leng in LENGTH:
    FILE = "./Seeds/input_" + str(leng) + "_" + str(NUMBER) + "_2" + ".txt"
    OUTPUT_FILE = open("./Predictions/predictions_" + str(leng) + "_" + str(NUMBER) + "_2" + ".txt", 'w')
    data = pickle.load(open(FILE, 'rb'))
    for sentence in data:
        s = ' '.join(sentence)
        print(s)
        predict_next(sentence=s, depth=depth, corpus=OUTPUT_FILE)
    OUTPUT_FILE.close()