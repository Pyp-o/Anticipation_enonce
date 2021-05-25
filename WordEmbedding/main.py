import torch
import dataPrep
import models
import numpy as np
import random
from gensim.models import KeyedVectors
import dataHandlingWE
import pickle
from os.path import exists

#TODO next word prediction without sliding window
#TODO tester l'impact du nombre de cellules sur les résultats
#TODO sécuriser et tenter d'obtenir de petits résultats

#TODO présentation dans 2 semaines semaine du 24 mai !!! attention au changement du point de vue et bien expliciter toutes les idées
#TODO envoyer mail à jérémy riviere pour soutenance de stage le 29/30 juin (début de créneau)
#TODO finir rapport au plus tard mardi 22juin
#TODO formalisation (courbes, exemples) des réussites, exemples, comment mesurer l'erreur
#TODO pourquoi ca fonctionne, pourquoi pas, intuition sur les tailles de cellules
#TODO prochaine réunion: mardi 18 17h, vendredi 21 14h30
#TODO présentation commedia semaine 24, répétion le 25 11h
#TODO draft rapport + réunion pour le 31 14h

#-------------- No random --------------#
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)



#-------------- Parametres --------------#
FILENAME = "./WEdata2.txt"
DATA_SUBSAMPLE = 2000       #si 0 on prend tout le jeu de données
SUBSAMPLE = int(DATA_SUBSAMPLE*0.9) #number of phrases in the whole set
BATCH_SIZE = 200  #number oh phrases in every subsample (must respect SUBSAMPLE*BATCH_SIZE*(UTT_LEN/2)*N_FEATURES=tensor_size)
MIN_LEN = 4
MAX_LEN = 10
TEST_SIZE = 20

LEARNING_RATE = 0.001
N_FEATURES = 100    #100 pour GloVe
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 2500

TRAIN_SET = "test"  #"test"/"train"

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
    data, glove = dataHandlingWE.prepareData('../../DataBase/dialog/dialogues_text.txt')
    print("data and GloVe imported !")

#-------------- limit lenght of each phrase to 8 words
data = dataPrep.limitLength2(data, min=MIN_LEN, max=MAX_LEN)  #limit length of phrases bewteen 4 and 10 by default
if DATA_SUBSAMPLE!=0:
    data = data[:DATA_SUBSAMPLE]
else :
    SUBSAMPLE = int(len(data)*0.7)

print("SUBSAMPLE", SUBSAMPLE)

#-------------- split dataset into trainset and testset
train = data[:SUBSAMPLE]
test = data[SUBSAMPLE:]
X_train, Y_train = dataPrep.splitX_y2(train)    #split input and output for prediction depending on each utterance length
X_test, Y_test = dataPrep.splitX_y2(test)

print("converting arrays to tensors...")
T_X_train = []
T_y_train = []
T_X_test = []
T_y_test = []
#-------------- convert arrays as tensors
T_X_train = torch.FloatTensor(X_train)
T_y_train = torch.FloatTensor(Y_train)
T_X_train = torch.reshape(T_X_train, (-1, BATCH_SIZE, int(MAX_LEN/2), N_FEATURES)).to(device)
T_y_train = torch.reshape(T_y_train, (-1, BATCH_SIZE, int(MAX_LEN/2), N_FEATURES)).to(device)

T_X_test = torch.FloatTensor(X_test)
T_y_test = torch.FloatTensor(Y_test)
T_X_test = torch.reshape(T_X_test, (-1, int(MAX_LEN/2), N_FEATURES)).to(device)
T_y_test = torch.reshape(T_y_test, (-1, int(MAX_LEN/2), N_FEATURES)).to(device)

print("model declaration")
#-------------- model declaration
model = models.LSTM(hidden_size=HIDDEN_SIZE, nfeatures=N_FEATURES, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
losses = []

h = torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)
c = torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)

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
        loss += single_loss.item()
        h = h.detach()
        c = c.detach()
        single_loss.backward()
        optimizer.step()
        model.zero_grad()
    losses.append(loss)  #loss cumulée pour chaque epoch
    if i%5 == 1:
        print(f'epoch:{i-1:5}/{EPOCHS:3}\tloss: {single_loss.item():10.10f}')
print(f'epoch: {i+1:5}/{EPOCHS:5}\tloss: {single_loss.item():10.10f}')

print("model predicting")
#-------------- predictions
if TRAIN_SET == "train":
    T_X_train = T_X_train[:TEST_SIZE]
    T_X_train = torch.reshape(T_X_train, (-1, int(MAX_LEN/2), N_FEATURES)).to(device)
    predictions, (_,_) = model(T_X_train)

    inp = dataPrep.reverseEmbed(X_train[:TEST_SIZE], glove)
    out = dataPrep.reverseEmbed(Y_train[:TEST_SIZE], glove)

else :
    T_X_test = T_X_test[:TEST_SIZE].to(device)
    predictions, (_,_) = model(T_X_test)

    inp = dataPrep.reverseEmbed(X_test[:TEST_SIZE], glove)
    out = dataPrep.reverseEmbed(Y_test[:TEST_SIZE], glove)

print("reverse predicted tensors to CPU")
#-------------- moving back tensors to CPU to treat tensors as numpy array
predictions = predictions.cpu().detach().numpy()
predictions = dataPrep.reverseEmbed(predictions, glove)

for i in range(len(inp)):
    print(f'\ni:{i:3} {inp[i]}\n{out[i]}\n{predictions[i]}')

#-------------- plot loss
dataPrep.plotLoss(losses)