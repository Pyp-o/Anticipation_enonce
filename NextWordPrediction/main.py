import torch
import dataPrep
import models
import numpy as np
import random
from gensim.models import KeyedVectors

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

n_features = 100

############### MAIN ###############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :", device)

print("importing data...")
#importing and preparing data
df = open(r'../../DataBase/dialog/dialogues_text.txt', encoding='utf-8')    #beware, dataset changed : don ' t => don't
data = dataPrep.parseDialogs(df)
df.close()

print("preparing data...")
data = dataPrep.parseUtterances(data)
data = dataPrep.parsePhrase(data)
data = dataPrep.removePunctuation(data)
data = data[:2000]

#make each phrase as an entry of array
data = dataPrep.dataAsArray(data)
data = dataPrep.rmSpaces(data)

print("importing GloVe...")
#import GloVe
filename = '../embedding/glove.6B.100d.txt.word2vec'
glove = KeyedVectors.load_word2vec_format(filename, binary=False)
print("GloVe imported !")

#limit lenght of each phrase to 8 words
data = dataPrep.limitLength(data, 6)
data = data[:100]

#split dataset into trainset and testset
ind = int(len(data)*0.9)
train = data[:ind]
test = data[ind:]

print("splitting data into sets...")
#split sets into input and output for training and testing
X_train, y_train = dataPrep.splitX_y(train, 3)
X_test, y_test = dataPrep.splitX_y(test, 3)

print("converting words...")
#convert words to ix
X_train = dataPrep.convertPhrasetoIx(X_train, glove)
y_train = dataPrep.convertPhrasetoIx(y_train, glove)

X_test = dataPrep.convertPhrasetoIx(X_test, glove)
y_test = dataPrep.convertPhrasetoIx(y_test, glove)

print(len(X_train))
print(len(y_train))

print(len(X_test))
print(len(y_test))

print("converting arrays to tensors...")
T_X_train = []
T_y_train = []
T_X_test = []
T_y_test = []
#convert arrays as tensors
for i in range(len(X_train)):
    T_X_train.append(torch.FloatTensor(X_train[i]))
    T_y_train.append(torch.FloatTensor(y_train[i]))
for i in range(len(X_train)):
    T_X_train[i] = torch.reshape(T_X_train[i], (1, -1, n_features)).to(device)
    T_y_train[i] = torch.reshape(T_y_train[i], (1, -1, n_features)).to(device)

for i in range(len(X_test)):
    T_X_test.append(torch.FloatTensor(X_test[i]))
    T_y_test.append(torch.FloatTensor(y_test[i]))
for j in range(len(X_test)):
    T_X_test[j] = torch.reshape(T_X_test[j], (1, -1, n_features)).to(device)
    T_y_test[j] = torch.reshape(T_y_test[j], (1, -1, n_features)).to(device)

print("X", len(T_X_train))
print("y", len(T_y_train))
print("X_test", len(T_X_test))
print("y_test", len(T_y_test))

#del X_train, y_train

print("model declaration")
#model declaration
model = models.LSTM(hidden_size=512, nfeatures=n_features, num_layers=2).to(device) #2 couches 512 cells pour 26000 mots
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 2000

print("training model")
#model training
for i in range(epochs):
    model.train()
    for j in range(len(T_X_train)):
        y_pred = model(T_X_train[j], device).to(device)
        single_loss = loss_function(y_pred, T_y_train[j])

        single_loss.backward()
        optimizer.step()
        model.zero_grad()
    if i%5 == 1:
        print(f'epoch: {i-1:3} loss: {single_loss.item():10.10f}')

print(f'epoch: {i+1:3} loss: {single_loss.item():10.10f}')

print("model predicting")
#model predictions
predictions = []
l = len(T_X_test)
for j in range(l):
    predictions.append(model(T_X_train[j], device).to(device))

print("reverse predicted tensors to CPU")
#moving back tensors to CPU to treat tensors as numpy array
for i in range(len(predictions)):
    predictions[i] = predictions[i].cpu().detach().numpy()

inp = dataPrep.reverseEmbed(X_train[:l], glove)
out = dataPrep.reverseEmbed(y_train[:l], glove)

data = []
for phrase in predictions:
    ph = []
    for word in phrase[0]:
        ph.append(glove.similar_by_vector(word, topn=1))
    data.append(ph)

predictions = data
del data

print("input", inp)
print("expected", out)
print("predicted", predictions)

#TODO tester de nuit avec corpus entier et ~10000 epochs
#TODO modifier loss function pour l'encodage index

