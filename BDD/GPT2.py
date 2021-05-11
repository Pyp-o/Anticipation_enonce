# creation: 21-apr-2021 pierre.chevaillier@enib.Fr
import sys

from next_word_prediction import GPT2

# global variables
eos = ['.', '!', '?']
maxLength = 12 + 1  # last : end of sentence
width = 5
sentence = "How may I"
corpus = open('corpus_gpt2_' + sentence + str(width) + '_depth-' + str(maxLength) + '.txt', "w")

# next word predictor
gpt2 = GPT2()


def predict_next(sentence, depth):
    if depth > maxLength:
        return 1
    nextWords = gpt2.predict_next(sentence, width)
    for w in nextWords:
        isTerminal = w in eos
        if isTerminal:
            sentence += w
            #print(sentence)
            corpus.write(sentence + '__eou__' + '\n')
            return
        else:
            sep = '' if len(w) == 0 or w[0] == '\'' else ' '
            newSentence = sentence + sep + w
            depth += 1
            predict_next(newSentence, depth)

    return


# initialText = "The course starts next"

depth = 0
predict_next(sentence, depth)
corpus.close()

