####################IMPORTS######################

import pandas as pd
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import re 
import string
import enchant
from nltk.corpus import words

####################FUNCTIONS######################

def wordCount(string):
    return (len(string.strip().split(" ")))

def DTBwordCount(dtb):
	tab = []
	for d in dtb:
		if isinstance(d, str):
			tab.append(wordCount(d))
	return tab


# compte le bombre d'utterance de meme taille
def nb_len(utt_len, max):
    tab = [0] * max
    for t in utt_len:
        tab[t - 1] += 1
    return tab


def min_max(utt_len):
    min = 100
    max = 0
    for length in utt_len:
        if length > max:
            max = length
        elif length < min:
            min = length
    return min, max


def simpleTab(nb_utt_len, max):
    indice = range(max)
    nbInf = 0
    nbSup = 0

    #suppression d'une portion de tableau
    """# l = [2, 3, 4, 5]
	del l[0:2]
	# l = [4, 5]"""

    for i in range(4):
        nb_utt_len[i] = 0

    for i in indice:
        if nbInf == 50:
            nb_utt_len = nb_utt_len[: len(nb_utt_len) - i]
            indice = indice[:len(nb_utt_len) - i]
            return nb_utt_len, indice
        if nb_utt_len[i] < 150:
            nbInf += 1
        else:
            nbInf = 0

    return nb_utt_len, indice


def barplot(nb_utt_len):
    height = nb_utt_len
    y_pos = range(1, len(nb_utt_len) + 1)

    # Create bars
    plt.bar(y_pos, height)

    # Create names on the x-axis
    plt.xticks(y_pos, rotation=90)

    # Show graphic
    plt.show()

def pointplot(utt_len):
	Y = utt_len
	X = range(1, len(utt_len)+1)

	plt.scatter(X,Y)
	plt.show()


def stats(utt_len):
    moy = stat.fmean(utt_len)
    median = stat.median(utt_len)
    var = stat.variance(utt_len)
    quart1 = np.percentile(utt_len, 25)
    quart3 = np.percentile(utt_len, 75)

    printStats(moy, median, var, quart1, quart3)

    return moy, median, var, quart1, quart3


def printStats(moy, median, var, quart1, quart3):
    print("moyenne :", moy)
    print("1er quartile :", quart1)
    print("médiane :", median)
    print("3e quartile :", quart3)
    print("variance :", var)



def lexic(phrase, character):
	lex = []
	for p in phrase:
		if isinstance(p, str):
			word = re.sub('['+string.punctuation+']', '', p).split()
			for w in word:
					if w not in lex:
						if not w in character:
							lex.append(w)

	print("nb de mots différents :", len(lex))
	return lex


def randPrint(transcript):
    print(transcript[rd.randint(0, 1000)])
    print(transcript[rd.randint(0, 1000)])
    print(transcript[rd.randint(0, 1000)])
    print(transcript[rd.randint(0, 1000)])
    print(transcript[rd.randint(0, 1000)])
    print(transcript[rd.randint(0, 1000)])

#probleme avec les mots pluriels
def onlyEnglishLexic(lexic):
    lex = []
    oov = []
    word_list = words.words()
    for w in range(len(lexic)):
        if lexic[w] in word_list:
                lex.append(lexic[w])
        elif lexic[w][len(lexic[w])-1]== 's':
            lexic[w] = lexic[w][:-1]
            if lexic[w] in word_list:
                lex.append(lexic[w])
        else:
            oov.append(lexic[w]) 
    return lex, oov

def onlyEnglishLexic2(lexic):
    lex = []
    oov = []
    word_list = enchant.Dict("en ")
    for w in range(len(lexic)):
        if word_list.check(lexic[w]):
            lex.append(lexic[w])
        elif lexic[w][len(lexic[w])-1]== 's':
            lexic[w] = lexic[w][:-1]
            if word_list.check(lexic[w]):
                lex.append(lexic[w])
        else:
            oov.append(lexic[w]) 
    return lex, oov

####################MAIN######################

#load csv
df = pd.read_csv(r'../../DataBase/south_park/sp_lines.csv')

utterance = df.text
character = df.character

"""
episode = df.episode_name
season = df.season_number
eipsode = df.episode_number
"""

del(df)

#randPrint(utterance)

#var
nbWord = []
nbUtt = []
lex = []
oov = []

#remplissage du lexique
lex = lexic(utterance, character)

#delete OOV words from lexic
lex, oov = onlyEnglishLexic(lex)
print("nb de mots différents (actualisé) :", len(lex))

lex, oov = onlyEnglishLexic2(lex)
print("nb de mots différents (actualisé 2) :", len(lex))


#count number in each utterance
nbWord = DTBwordCount(utterance)

#shortest and longest utterance in nb of words
min, max = min_max(nbWord)
print("min : ", min, "max : ", max)

#count number of utterance of same length
nbUtt = nb_len(nbWord, max)

#stat calc
moy, median, var, quart1, quart3 = stats(nbWord)

#point plot
pointplot(nbUtt)

#plot [0 to 75] length utterance
del nbUtt[75:len(nbUtt)-1]
pointplot(nbUtt)

#plot [0 to 10] length utterance
del nbUtt[10:len(nbUtt)-1]
pointplot(nbUtt)
barplot(nbUtt)
