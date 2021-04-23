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
FILENAME = "./WEdata.txt"
DATA_SUBSAMPLE = 100        #si 0 on prend tout le jeu de données
SUBSAMPLE = int(DATA_SUBSAMPLE*0.8) #number of phrases in the whole set
BATCH_SIZE = 10  #number oh phrases in every subsample (must respect SUBSAMPLE*BATCH_SIZE*(UTT_LEN/2)*N_FEATURES=tensor_size)
UTT_LEN = 8             #doit etre pair pour le moment

LEARNING_RATE = 0.0001
N_FEATURES = 100    #100 pour GloVe
HIDDEN_SIZE = 512
NUM_LAYERS = 2
EPOCHS = 500



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
    data, glove = dataHandlingWE.prepareData()
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

print("converting arrays to tensors...")
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
#-------------- predictions
l = len(T_X_test)
predictions = model(T_X_test).to(device)

print("reverse predicted tensors to CPU")
#-------------- moving back tensors to CPU to treat tensors as numpy array
predictions = predictions.cpu().detach().numpy()
inp = dataPrep.reverseEmbed(X_test[:l], glove)
out = dataPrep.reverseEmbed(Y_test[:l], glove)
predictions = dataPrep.reverseEmbed(predictions, glove)

for i in range(len(inp)):
    print(f'\ni:{i:3} {inp[i]}\n{out[i]}\n{predictions[i]}')

#-------------- plot loss
dataPrep.plotLoss(losses)

#il est possible d'avoir des longueurs de phrase différentes, cela ne pose pas de problème au LSTM exemple : entrainement avec des phrases de taille 4 et de taille 3