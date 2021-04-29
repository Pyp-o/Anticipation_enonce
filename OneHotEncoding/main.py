import torch
import dataPrep
import models
import numpy as np
import random
from os.path import exists
import pickle
import dataHandlingOneHot



#-------------- No random --------------#
SEED = 10
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)



#-------------- Parametres --------------#
FILENAME = "./OneHotdata0.txt"
FILENAME0 = "./OneHotdata1.txt"
FILENAME1 = "./OneHotdata1.txt"
FILENAME2 = "./Vocabdata.txt"
FILENAME3 = "./WordToIxdata.txt"
FILENAME4 = "./IxToWorddata.txt"
FILENAME5 = "./WordToOneHotdata.txt"
FILENAME6 = "./OneHotToWorddata.txt"

DATA_SUBSAMPLE = 200        #si 0 on prend tout le jeu de données
SUBSAMPLE = int(DATA_SUBSAMPLE*0.8) #number of phrases in the whole set
BATCH_SIZE = 1  #number oh phrases in every subsample (must respect SUBSAMPLE*BATCH_SIZE*(UTT_LEN/2)*N_FEATURES=tensor_size)
UTT_LEN = 8             #doit etre pair pour le moment

LEARNING_RATE = 0.0001
N_FEATURES = 1    #1 pour index
HIDDEN_SIZE = 512
NUM_LAYERS = 2
EPOCHS = 5

#-------------- MAIN --------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :", device)

#-------------- load data and glove
if exists(FILENAME) and exists(FILENAME0) and exists(FILENAME1) and exists(FILENAME2) and exists(FILENAME3) and exists(FILENAME4) and exists(FILENAME5) and exists(FILENAME6):
    print("importing data...")
    with open(FILENAME, "rb") as fp:
        data0 = pickle.load(fp)
    with open(FILENAME0, "rb") as fp:
        data1 = pickle.load(fp)
    with open(FILENAME1, "rb") as fp:
        data2 = pickle.load(fp)
    with open(FILENAME2, "rb") as fp:
        vocab = pickle.load(fp)
    with open(FILENAME3, "rb") as fp:
        word_to_ix = pickle.load(fp)
    with open(FILENAME4, "rb") as fp:
        ix_to_word = pickle.load(fp)
    with open(FILENAME5, "rb") as fp:
        word_to_oneHot = pickle.load(fp)
    with open(FILENAME6, "rb") as fp:
        oneHot_to_word = pickle.load(fp)

    n_features = len(word_to_oneHot["hello"])

    print("data imported !")
else:
    print("preparing data...")
    data, vocab, word_to_oneHot, oneHot_to_word, word_to_ix, ix_to_word, N_FEATURES, oneHot_to_ix, ix_to_oneHot = dataHandlingOneHot.prepareData()
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
#output back to index for CrossEntropy
Y_train = dataPrep.convertOneHotToIx(Y_train, oneHot_to_ix)
Y_test = dataPrep.convertOneHotToIx(Y_test, oneHot_to_ix)


print("convert arrays to tensors")
T_X_train = []
T_y_train = []
T_X_test = []
T_y_test = []
#-------------- convert arrays as tensors
T_X_train = torch.FloatTensor(X_train)
T_y_train = torch.FloatTensor(Y_train)
T_X_train = torch.reshape(T_X_train, (-1, BATCH_SIZE, int(UTT_LEN/2), N_FEATURES)).to(device)
T_y_train = torch.reshape(T_y_train, (-1, BATCH_SIZE, int(UTT_LEN/2), 1)).to(device)

T_X_test = torch.FloatTensor(X_test)
T_y_test = torch.FloatTensor(Y_test)
T_X_test = torch.reshape(T_X_test, (-1, int(UTT_LEN/2), N_FEATURES)).to(device)
T_y_test = torch.reshape(T_y_test, (-1, int(UTT_LEN/2), 1)).to(device)

print("T_X_train", T_X_train.shape)
print("T_y_train", T_y_train.shape)

#del X_train, y_train

print("model declaration")
#model declaration
model = models.LSTM(hidden_size=HIDDEN_SIZE, nfeatures=N_FEATURES, num_layers=NUM_LAYERS).to(device)
loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
losses = []

print("training model")
#model training
for i in range(EPOCHS):
    model.train()
    loss = 0
    for j in range(len(T_X_train)):
        y_pred = model(T_X_train[j]).to(device)
        single_loss = loss_function(y_pred, T_y_train[j])
        loss += single_loss.item()
        single_loss.backward()
        optimizer.step()
        model.zero_grad()
    losses.append(loss)  #loss cumulée pour chaque epoch
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

inp = dataPrep.reverseOneHot(X_train[:l], oneHot_to_word)
out = dataPrep.reverseOneHot(Y_train[:l], oneHot_to_word)
predictions = dataPrep.oneHotClean(predictions, oneHot_to_word)

for i in range(len(inp)):
    print(f'\ni:{i:3} input: {inp[i]}\nexpected: {out[i]}\npredicted: {predictions[i]}')

dataPrep.plotLoss(losses)