import torch

import dataHandlingIndex
import dataPrep
import models
import numpy as np
import random
import pickle
from os.path import exists
import sys



#-------------- No random --------------#
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)



#-------------- Parametres --------------#
FILENAME = "./Indexdata.txt"
FILENAME1 = "./Vocabdata.txt"
FILENAME2 = "./WordToIxdata.txt"
FILENAME3 = "./IxToWorddata.txt"

SUBSAMPLE = 12000        #si 0 on prend tout le jeu de données
DATA_SUBSAMPLE = int(SUBSAMPLE/0.8) #number of phrases in the whole set
BATCH_SIZE = 120  #number oh phrases in every subsample (must respect SUBSAMPLE*BATCH_SIZE*(UTT_LEN/2)*N_FEATURES=tensor_size)
UTT_LEN = 8             #doit etre pair pour le moment

LEARNING_RATE = 0.01
N_FEATURES = 1    #1 pour index
HIDDEN_SIZE = 256
NUM_LAYERS = 2
EPOCHS = 1000



#-------------- MAIN --------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :", device)

#-------------- load data and glove
if exists(FILENAME) and exists(FILENAME1) and exists(FILENAME2) and exists(FILENAME3):
    print("importing data...")
    with open(FILENAME, "rb") as fp:
        data = pickle.load(fp)
    with open(FILENAME1, "rb") as fp:
        vocab = pickle.load(fp)
    with open(FILENAME2, "rb") as fp:
        word_to_ix = pickle.load(fp)
    with open(FILENAME3, "rb") as fp:
        ix_to_word = pickle.load(fp)
    scaler = dataPrep.fitScaler(data, word_to_ix, -1, 1)
    print("data imported !")
else:
    print("preparing data...")
    data, word_to_ix, ix_to_word, scaler = dataHandlingIndex.prepareData()
    print("data and GloVe imported !")

#-------------- limit lenght of each phrase to 8 words
data = dataPrep.limitLength(data, UTT_LEN)
if DATA_SUBSAMPLE!=0:
    data = data[:DATA_SUBSAMPLE]

#-------------- split dataset into trainset and testset
train = data[:SUBSAMPLE]
test = data[SUBSAMPLE:]
X_train, Y_train = dataPrep.splitX_y(train, int(UTT_LEN/2))
X_test, Y_test = dataPrep.splitX_y(test, int(UTT_LEN/2))

#-------------- rescale data
print(data)
data = scaler.transform(X_train)
X_train = scaler.transform(X_train)
Y_train = scaler.transform(Y_train)
X_test = scaler.transform(X_test)
Y_test = scaler.transform(Y_test)

T_X_train = []
T_y_train = []
T_X_test = []
T_y_test = []
#-------------- convert arrays as tensors
T_X_train = torch.FloatTensor(X_train)
T_y_train = torch.FloatTensor(Y_train)
T_X_train = torch.reshape(T_X_train, (-1, BATCH_SIZE, int(UTT_LEN/2), N_FEATURES)).to(device)
T_y_train = torch.reshape(T_y_train, (-1, BATCH_SIZE, int(UTT_LEN/2), N_FEATURES)).to(device)

T_X_test = torch.FloatTensor(X_test)
T_y_test = torch.FloatTensor(Y_test)
T_X_test = torch.reshape(T_X_test, (-1, int(UTT_LEN/2), N_FEATURES)).to(device)
T_y_test = torch.reshape(T_y_test, (-1, int(UTT_LEN/2), N_FEATURES)).to(device)


print("model declaration")
#model declaration
model = models.LSTM(hidden_size=HIDDEN_SIZE, nfeatures=N_FEATURES, num_layers=NUM_LAYERS).to(device)
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
losses = []
test_losses = []

print("training model")
#-------------- model training
for i in range(EPOCHS):
    model.train()
    loss = 0
    predictions = model(T_X_test).to(device)
    single_loss = loss_function(predictions, T_y_test)
    test_losses.append(single_loss.item())
    model.zero_grad()
    for j in range(len(T_X_train)):
        y_pred = model(T_X_train[j]).to(device)
        single_loss = loss_function(y_pred, T_y_train[j])
        loss += single_loss.item()
        single_loss.backward()
        optimizer.step()
        model.zero_grad()
    losses.append(loss/len(T_X_train))  #loss cumulée pour chaque epoch
    if i%5 == 1:
        print(f'epoch:{i-1:5}/{EPOCHS:3}\tloss: {single_loss.item():10.10f}')
print(f'epoch: {i+1:5}/{EPOCHS:5}\tloss: {single_loss.item():10.10f}')

print("model predicting")
#model predictions
l = len(T_X_test)
predictions = model(T_X_test).to(device)

print("reverse predicted tensors to CPU")
#moving back tensors to CPU to treat tensors as numpy array
predictions = predictions.cpu().detach().numpy()
inp = scaler.inverse_transform(X_train[:l])
out = scaler.inverse_transform(Y_train)
predictions = dataPrep.reverseTransformedPrediction(predictions, scaler)
"""
inp = dataPrep.convertIxtoPhrase(inp, ix_to_word)
out = dataPrep.convertIxtoPhrase(out, ix_to_word)
predictions = dataPrep.convertIxtoPhrase(predictions, ix_to_word)

for i in range(len(inp)):
    print(f'\ni:{i:3} input: {inp[i]}\nexpected: {out[i]}\npredicted: {predictions[i]}')
"""

dataPrep.plotLoss(losses)
dataPrep.plotLoss(test_losses)