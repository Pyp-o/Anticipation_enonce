import torch
import dataPrep
import models
import numpy as np
import random
import dataHandlingOneHot
from matplotlib import pyplot as plt
import os


#-------------- No random --------------#
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)



#-------------- Parametres --------------#
SUBSAMPLE = 12000        #si 0 on prend tout le jeu de données
DATA_SUBSAMPLE = int(SUBSAMPLE/0.9) #number of phrases in the whole set
BATCH_SIZE = 120  #number oh phrases in every subsample (must respect SUBSAMPLE*BATCH_SIZE*(UTT_LEN/2)*N_FEATURES=tensor_size)
UTT_LEN = 8             #doit etre pair pour le moment

LEARNING_RATE = 0.001
N_FEATURES = 1    #1 pour index
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 1000

TEST_SET = "test"
TEST_SIZE = 20

def plotTest(Losses, test_losses, output_file):
    # Loss
    plt.figure()
    plt.plot(np.log10(Losses))
    plt.plot(np.log10(test_losses), color='C2')
    plt.title('Learning curve')
    plt.ylabel('loss: log10(MSE)')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc = 'upper right')
    #plt.show()
    out = os.path.join('.', 'loss_' + output_file)
    plt.savefig(out, dpi=96, papertype='a7')
    plt.close('all')
    return


NAME = "../../models/OneHot_trained_model_layer_"+str(NUM_LAYERS)+"_Ncells_"+str(HIDDEN_SIZE)+"_size_"+str(SUBSAMPLE)+"_epochs_"+str(EPOCHS)+".pt"
PATH = "../../models/OneHot_trained_model_layer_2_Ncells_256_size_5000_epochs_1000.pt"
#-------------- MAIN --------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :", device)


print("preparing data...")
data, vocab, word_to_oneHot, oneHot_to_word, word_to_ix, ix_to_word, N_FEATURES, oneHot_to_ix, ix_to_oneHot = dataHandlingOneHot.prepareData()
print("data and GloVe imported !")

#-------------- limit lenght of each phrase to 8 words
data = dataPrep.limitLength(data, UTT_LEN)
data = data[:DATA_SUBSAMPLE]
#-------------- split dataset into trainset and testset
train = data[:SUBSAMPLE]
test = data[SUBSAMPLE:]
del data
X_train, Y_train = dataPrep.splitX_y(train, int(UTT_LEN/2))
X_test, Y_test = dataPrep.splitX_y(test, int(UTT_LEN/2))
#output back to index for CrossEntropy
Y_train = dataPrep.convertOneHotToIx(Y_train, oneHot_to_ix)
Y_test = dataPrep.convertOneHotToIx(Y_test, oneHot_to_ix)



#CTC lengths
T = int(UTT_LEN/2)
N = BATCH_SIZE
C = N_FEATURES
S = int(UTT_LEN/2)

print("convert arrays to tensors")
T_X_train = []
T_y_train = []
T_X_test = []
T_y_test = []
#-------------- convert arrays as tensors
T_X_train = torch.FloatTensor(X_train)
T_y_train = torch.LongTensor(Y_train)
#T_y_train = torch.FloatTensor(Y_train)
T_X_train = torch.reshape(T_X_train, (-1, T, N, C))
#T_y_train = torch.reshape(T_y_train, (-1, N, S))
T_y_train = torch.reshape(T_y_train, (-1, T, N, 1))

T_X_test = torch.FloatTensor(X_test)
del X_test
T_y_test = torch.LongTensor(Y_test)
#T_y_test = torch.FloatTensor(Y_test)
del Y_test
T_X_test = torch.reshape(T_X_test, (-1, T, 1, C))
#T_y_test = torch.reshape(T_y_test, (-1, 1, S))
T_y_test = torch.reshape(T_y_test, (-1, T, 1, 1))

print("T_X_train", T_X_train.shape)
print("T_y_train", T_y_train.shape)

#del X_train, y_train

print("model declaration")
#model declaration
model = models.LSTM(hidden_size=HIDDEN_SIZE, nfeatures=N_FEATURES, num_layers=NUM_LAYERS, output_size=N_FEATURES, dropout=DROPOUT).to(device)
#loss_function = torch.nn.CTCLoss(reduction='mean')
loss_function = torch.nn.CrossEntropyLoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
losses = []
test_losses = []

print("training model")
#model training
for i in range(EPOCHS):
    model.train()
    loss = 0
    test_loss = 0


    for j in range(len(T_X_train)):
        if j<len(T_X_test):
            X = T_X_test[j].to(device)
            Y = T_y_test[j]
            Y = torch.reshape(Y, (-1,)).to(device)
            y_pred, (_,_) = model(X)
            y_pred = torch.reshape(y_pred, (-1, C))
            #expt = torch.reshape(T_y_test[j], (1, S)).to(device)
            #input_lengtht = torch.full(size=(1,), fill_value=T, dtype=torch.long)
            #target_lengtht = torch.randint(low=1, high=T, size=(1,), dtype=torch.long)
            #single_loss = loss_function(yt_pred, expt, input_lengtht, target_lengtht)
            single_loss = loss_function(y_pred, Y)
            test_loss+=single_loss.item()
            model.zero_grad()

        X = T_X_train[j].to(device)
        Y = T_y_train[j]
        Y = torch.reshape(Y, (-1,)).to(device)
        y_pred, (_, _) = model(X)
        y_pred = torch.reshape(y_pred, (-1, C))
        #exp = torch.reshape(T_y_train[j], (N, S)).to(device)
        #input_length = torch.full(size = (BATCH_SIZE,), fill_value=T, dtype=torch.long)
        #target_length = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
        #single_loss = loss_function(y_pred, exp, input_length, target_length)
        single_loss = loss_function(y_pred, Y)
        loss += single_loss.item()
        single_loss.backward()
        optimizer.step()
        model.zero_grad()
        #exp.detach()
    losses.append(loss/len(T_X_train))  #loss cumulée pour chaque epoch
    test_losses.append(loss / len(T_X_test))  # loss cumulée pour chaque epoch
    if i%5 == 1:
        print(f'epoch:{i-1:5}/{EPOCHS:3}\tloss: {single_loss.item():10.10f}')
print(f'epoch: {i+1:5}/{EPOCHS:5}\tloss: {single_loss.item():10.10f}')

print("model predicting")
#-------------- predictions
if TEST_SET == "train":
    T_X_train = T_X_train[:TEST_SIZE]
    T_X_train = torch.reshape(T_X_train, (-1, int(UTT_LEN/2), N_FEATURES)).to(device)
    predictions, (_,_) = model(T_X_train)

else :
    del T_X_train
    T_X_test = T_X_test[:TEST_SIZE].to(device)
    T_X_test = torch.reshape(T_X_test, (-1, int(UTT_LEN / 2), N_FEATURES)).to(device)
    predictions, (_,_) = model(T_X_test)

print("reverse predicted tensors to CPU")
#-------------- moving back tensors to CPU to treat tensors as numpy array
inp = dataPrep.reverseOneHot(X_train[:TEST_SIZE], oneHot_to_word)
out = dataPrep.reverseOneHot(Y_train[:TEST_SIZE], ix_to_word)
predictions = dataPrep.oneHotClean(predictions, oneHot_to_word)

for i in range(len(inp)):
    print(f'\ni:{i:3} input: {inp[i]}\nexpected: {out[i]}\npredicted: {predictions[i]}')
"""
dataPrep.plotLoss(losses)
dataPrep.plotLoss(test_losses)
"""
plotTest(losses, test_losses, "ctc.png")
#torch.save(model.state_dict(), NAME)