import sys

import torch
import dataPrep
import models
import numpy as np
import random
from gensim.models import KeyedVectors
import dataHandlingWE
import pickle
from os.path import exists



#-------------- No random --------------#
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)



#-------------- Parametres --------------#
FILENAME = "WEdata.txt"
SUBSAMPLE =  1001 #number of phrases in the whole set
DATA_SUBSAMPLE = int(SUBSAMPLE/0.9)       #si 0 on prend tout le jeu de données
BATCH_SIZE = 20
MIN_LEN = 4
MAX_LEN = 10
TEST_SIZE = 50

LEARNING_RATE = 0.001
N_FEATURES = 100    #100 for GloVe
HIDDEN_SIZE = 256
NUM_LAYERS = 2
EPOCHS = 400

TRAIN_SET = "train"  #"train"
NAME = "../../models/trained_model_"+str(NUM_LAYERS)+"_"+str(HIDDEN_SIZE)+"_"+str(SUBSAMPLE)+".pt" #TODO add encoding in name of the trained model

#-------------- MAIN --------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :", device)

#-------------- load data and glove
if exists(FILENAME):
    print("importing data...")
    with open(FILENAME, "rb") as fp:
        data = pickle.load(fp)

    dat = []
    for phrase in data:
        for word in phrase :
            dat.append(word)

    data = dat
    del dat

    print("importing GloVe...")
    # import GloVe
    FILENAME = '../embedding/glove.6B.100d.txt.word2vec'
    glove = KeyedVectors.load_word2vec_format(FILENAME, binary=False)
    print("GloVe imported !")

else:
    print("preparing data...")
    data, glove = dataHandlingWE.prepareData("../../DataBase/dialog/dialogues_text.txt")
    print("data and GloVe imported !")

#-------------- limit lenght of each phrase to 8 words
#data = dataPrep.limitLength2(data, min=MIN_LEN, max=MAX_LEN)  #limit length of phrases bewteen 4 and 10 by default
if DATA_SUBSAMPLE!=0:
    data = data[:DATA_SUBSAMPLE]
else :
    SUBSAMPLE = int(len(data)*0.7)

print("SUBSAMPLE", SUBSAMPLE)

#-------------- split dataset into trainset and testset
train = data[:SUBSAMPLE]
test = data[SUBSAMPLE:]
X_train, Y_train = dataPrep.sliding_XY2(train)    #split input and output for prediction depending on each utterance length
X_test, Y_test = dataPrep.sliding_XY2(test)

print("TOTAL TRAINING SAMPLE", len(X_train))
print("VERIFICATION TOTAL TRAINING SAMPLE", len(Y_train))

print("converting arrays to tensors...")
T_X_train = []
T_y_train = []
T_X_test = []
T_y_test = []
#-------------- convert arrays as tensors
T_X_train = torch.FloatTensor(X_train)
T_y_train = torch.FloatTensor(Y_train)
T_X_train = torch.reshape(T_X_train, (-1, BATCH_SIZE, 1, N_FEATURES)).to(device)
T_y_train = torch.reshape(T_y_train, (-1, BATCH_SIZE, 1, N_FEATURES)).to(device)

T_X_test = torch.FloatTensor(X_test)
T_y_test = torch.FloatTensor(Y_test)
T_X_test = torch.reshape(T_X_test, (-1, 1, N_FEATURES)).to(device)
T_y_test = torch.reshape(T_y_test, (-1, 1, N_FEATURES)).to(device)

print(T_X_train.shape)

print("model declaration")
#-------------- model declaration
model = models.LSTM(hidden_size=HIDDEN_SIZE, nfeatures=N_FEATURES, num_layers=NUM_LAYERS).to(device)
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
losses = []

h = torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE).to(device)
c = torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE).to(device)

print("training model")
#-------------- model training
for i in range(EPOCHS):
    model.train()
    loss = 0
    for j in range(len(T_X_train)):
        h = h.to(device)
        c = c.to(device)
        y_pred, (h, c) = model(T_X_train[j], (h, c))
        single_loss = loss_function(y_pred, T_y_train[j])
        h = h.detach()
        c = c.detach()
        loss += single_loss.item()
        single_loss.backward()
        optimizer.step()
        model.zero_grad()
    losses.append(loss)  #loss cumulée pour chaque epoch
    if i%5 == 1:
        print(f'epoch:{i-1:5}/{EPOCHS:3}\tloss: {single_loss.item():10.10f}')
print(f'epoch: {i+1:5}/{EPOCHS:5}\tloss: {single_loss.item():10.10f}')

print("model predicting")
#-------------- predictions
if TRAIN_SET == "train" :
    T_X_train = T_X_train[:TEST_SIZE]
    T_X_train = torch.reshape(T_X_train, (-1, 1, N_FEATURES)).to(device)
    predictions, (_,_) = model(T_X_train)

else :
    T_X_test = T_X_test[:TEST_SIZE]
    predictions, (_,_) = model(T_X_test)

inp = dataPrep.reverseEmbed2(X_test[:TEST_SIZE], glove)
out = dataPrep.reverseEmbed2(Y_test[:TEST_SIZE], glove)

print("reverse predicted tensors to CPU")
#-------------- moving back tensors to CPU to treat tensors as numpy array
predictions = predictions.cpu().detach().numpy()
predictions = dataPrep.reverseEmbed(predictions, glove)

print(inp)
print(out)
print(predictions)

"""
for i in range(len(predictions)):
    print(f'\ni:{i:3} {inp[i]}\n{out[i]}\n{predictions[i]}')
"""
#-------------- plot loss
dataPrep.plotLoss(losses)

torch.save(model, NAME)