import pandas as pd
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt

#ep = pd.read_csv (r'./episodes.csv')
#id / program / episode / date

#head = pd.read_csv (r'./headlines.csv')
#headline

def wordCount(string):
	 return(len(string.strip().split(" ")))

#compte le bombre d'utterance de meme taille
def nb_len(utt_len, max):
	tab = [0] * max
	for t in utt_len:
		tab[t-1]+=1
	return tab

def min_max(utt_len):
	min = 100
	max = 0
	for length in utt_len :
		if length > max:
			max = length
		elif length < min:
			min = length
	return min, max

def simpleTab(nb_utt_len, max):
	indice = range(max)
	nbInf = 0
	nbSup = 0

	"""# l = [2, 3, 4, 5]
	del l[0:2]
	# l = [4, 5]"""

	for i in range(4):
		nb_utt_len[i]=0

	for i in indice:
		if nbInf==50:
			nb_utt_len = nb_utt_len[: len(nb_utt_len) - i]
			indice = indice[:len(nb_utt_len)-i]
			return nb_utt_len, indice
		if nb_utt_len[i] < 150:
			nbInf +=1
		else :
			nbInf = 0

	return nb_utt_len, indice


def barplot(nb_utt_len):
	height = nb_utt_len
	y_pos = range(1, len(nb_utt_len)+1)

	# Create bars
	plt.bar(y_pos, height)

	# Create names on the x-axis
	plt.xticks(y_pos)

	# Show graphic
	plt.show()

def stats(nb_utt_len):
	moy = stat.fmean(nb_utt_len)
	median = stat.median(nb_utt_len)
	var = stat.variance(nb_utt_len)
	quart1 = np.percentile(nb_utt_len, 25)
	quart3 = np.percentile(nb_utt_len, 75)

	return moy, median, var, quart1, quart3

df = pd.read_csv (r'./utterances.csv')
#episode / episode_order / speaker / utterance

#ep = df.episode
#ep_order = df.episode_order
#sp = df.speaker
utterance = df.utterance
del(df)


#number of words by utterance count
utt_len = []
for i in range(len(utterance)):
	if isinstance(utterance[i], str):
		utt_len.append(wordCount(utterance[i]))

min, max = min_max(utt_len)

nb_utt_len = nb_len(utt_len, max)


#tab simplification -> remove all 0
indice = []
nb_utt_len, indice = simpleTab(nb_utt_len, max)

#moyenne, ecart type, variance, mediane, quartiles
print(stats(utt_len))

#plot number of utterances
barplot(nb_utt_len)