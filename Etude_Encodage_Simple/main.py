import torch
import dataPrep

############### MAIN ###############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#importing and preparing data
df = open(r'../../DataBase/dialog/dialogues_text.txt', encoding='utf-8')    #beware, dataset changed : don ' t => don't
data = dataPrep.parseDialogs(df)
df.close()

data = dataPrep.parseUtterances(data)
data = dataPrep.parsePhrase(data)
data = dataPrep.removePunctuation(data)

#vocab construct random order
vocab = dataPrep.vocabConstruct(data)
"""vocab = sorted(vocab, key=str.lower) #tri du vocabulaire par ordre alphabétique"""

#make each phrase as an entry of array
data = dataPrep.dataAsArray(data)

#limit lenght of each phrase to 8 words
data = dataPrep.limitLength(data, 8)

#word encoding
word_to_ix, ix_to_word = dataPrep.encodeWord(vocab, 'index')

#split dataset into trainset and testset
ind = int(len(data)*0.7)
train = data[:ind]
test = data[ind:]

#split sets into input and output for training and testing
X_train, y_train = dataPrep.splitX_y(train, 4)
X_test, y_test = dataPrep.splitX_y(test, 4)