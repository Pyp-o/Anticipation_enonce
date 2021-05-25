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

#2*29*383

#-------------- Parametres --------------#
FILENAME = "WEdata2.txt"
SUBSAMPLE = 2000 #number of phrases in the whole set
DATA_SUBSAMPLE = int(SUBSAMPLE/0.7)       #si 0 on prend tout le jeu de données
BATCH_SIZE = 200
MIN_LEN = 4
MAX_LEN = 10
TEST_SIZE = 20

LEARNING_RATE = 0.001
N_FEATURES = 100    #100 for GloVe
HIDDEN_SIZE = 256
NUM_LAYERS = 2
EPOCHS = 500

TRAIN_SET = "train"  #"train"

#-------------- MAIN --------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :", device)

#-------------- load data and glove
if exists(FILENAME):
    print("importing data...")
    with open(FILENAME, "rb") as fp:
        data = pickle.load(fp)
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
if DATA_SUBSAMPLE!=0:
    data = data[:DATA_SUBSAMPLE]
else :
    SUBSAMPLE = int(len(data)*0.7)

print("SUBSAMPLE", SUBSAMPLE)

dat = []
for phrase in data:
    for word in phrase:
        dat.append(word)

data = dat
del dat

#-------------- split dataset into trainset and testset
train = data[:SUBSAMPLE]
test = data[SUBSAMPLE:]

print("TOTAL TRAINING SAMPLE", len(train))

print("converting arrays to tensors...")
T_X_train = []
T_X_test = []
#-------------- convert arrays as tensors
T_X_train = torch.FloatTensor(train)
T_X_train = torch.reshape(T_X_train, (-1, BATCH_SIZE, 1, N_FEATURES)).to(device)

T_X_test = torch.FloatTensor(test)
T_X_test = torch.reshape(T_X_test, (-1, 1, N_FEATURES)).to(device)

print("model declaration")
#-------------- model declaration
model = models.LSTM(hidden_size=HIDDEN_SIZE, nfeatures=N_FEATURES, num_layers=NUM_LAYERS).to(device)
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
losses = []

print("training model")
#-------------- model training
for i in range(EPOCHS):
    model.train()
    loss = 0
    for j in range(len(T_X_train)-1):
        y_pred = model(T_X_train[j]).to(device)
        single_loss = loss_function(y_pred, T_X_train[j+1])
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
    T_X_train = torch.reshape(T_X_train, (-1, int(MAX_LEN/2), N_FEATURES)).to(device)
    predictions = model(T_X_train).to(device)

    inp = dataPrep.reverseEmbed(train[:TEST_SIZE], glove)

else :
    T_X_test = T_X_test[:TEST_SIZE]
    predictions = model(T_X_test).to(device)

    inp = dataPrep.reverseEmbed(test[:TEST_SIZE], glove)

print("reverse predicted tensors to CPU")
#-------------- moving back tensors to CPU to treat tensors as numpy array
predictions = predictions.cpu().detach().numpy()
predictions = dataPrep.reverseEmbed(predictions, glove)

for i in range(len(inp)-1):
    print(f'\ni:{i:3} {inp[i]}\n{inp[i+1]}\n{predictions[i]}')

#-------------- plot loss
dataPrep.plotLoss(losses)