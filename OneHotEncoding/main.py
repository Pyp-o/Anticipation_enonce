import torch
import dataPrep
import models
import numpy as np
import random
import dataHandlingOneHot



#-------------- No random --------------#
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)



#-------------- Parametres --------------#
SUBSAMPLE = 1000        #si 0 on prend tout le jeu de données
DATA_SUBSAMPLE = int(SUBSAMPLE/0.9) #number of phrases in the whole set
BATCH_SIZE = 250  #number oh phrases in every subsample (must respect SUBSAMPLE*BATCH_SIZE*(UTT_LEN/2)*N_FEATURES=tensor_size)
UTT_LEN = 8             #doit etre pair pour le moment

LEARNING_RATE = 0.001
N_FEATURES = 1    #1 pour index
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 5000

TEST_SET = "test"
TEST_SIZE = 20

NAME = "../../models/OneHot_trained_model_layer_"+str(NUM_LAYERS)+"_Ncells_"+str(HIDDEN_SIZE)+"_size_"+str(SUBSAMPLE)+"_epochs_"+str(EPOCHS)+".pt"

#-------------- MAIN --------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :", device)


print("preparing data...")
data, vocab, word_to_oneHot, oneHot_to_word, word_to_ix, ix_to_word, N_FEATURES, oneHot_to_ix, ix_to_oneHot = dataHandlingOneHot.prepareData()
print("data and GloVe imported !")

#-------------- limit lenght of each phrase to 8 words
data = dataPrep.limitLength(data, UTT_LEN)

#-------------- split dataset into trainset and testset
train = data[:SUBSAMPLE]
test = data[SUBSAMPLE:]
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
T_X_train = torch.reshape(T_X_train, (-1, T, N, C))
T_y_train = torch.reshape(T_y_train, (-1, N, S))

T_X_test = torch.FloatTensor(X_test)
T_y_test = torch.LongTensor(Y_test)
T_X_test = torch.reshape(T_X_test, (-1, int(UTT_LEN/2), N_FEATURES))
T_y_test = torch.reshape(T_y_test, (-1, int(UTT_LEN/2), 1))

print("T_X_train", T_X_train.shape)
print("T_y_train", T_y_train.shape)

#del X_train, y_train

print("model declaration")
#model declaration
model = models.LSTM(hidden_size=HIDDEN_SIZE, nfeatures=N_FEATURES, num_layers=NUM_LAYERS, output_size=N_FEATURES, dropout=DROPOUT).to(device)
loss_function = torch.nn.CTCLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
losses = []

print("training model")
#model training
for i in range(EPOCHS):
    model.train()
    loss = 0
    for j in range(len(T_X_train)):
        X = T_X_train[j].to(device)
        y_pred, (_,_) = model(X)
        exp = torch.reshape(T_y_train[j], (N, S)).to(device)
        input_length = torch.full(size = (BATCH_SIZE,), fill_value=T, dtype=torch.long)
        target_length = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
        single_loss = loss_function(y_pred, exp, input_length, target_length)
        loss += single_loss.item()
        single_loss.backward()
        optimizer.step()
        model.zero_grad()
        exp.detach()
    losses.append(loss)  #loss cumulée pour chaque epoch
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
    T_X_test = T_X_test[:TEST_SIZE].to(device)
    predictions, (_,_) = model(T_X_test)

print("reverse predicted tensors to CPU")
#-------------- moving back tensors to CPU to treat tensors as numpy array
inp = dataPrep.reverseOneHot(X_train[:TEST_SIZE], oneHot_to_word)
out = dataPrep.reverseOneHot(Y_train[:TEST_SIZE], ix_to_word)
predictions = dataPrep.oneHotClean(predictions, oneHot_to_word)



for i in range(len(inp)):
    print(f'\ni:{i:3} input: {inp[i]}\nexpected: {out[i]}\npredicted: {predictions[i]}')

dataPrep.plotLoss(losses)

torch.save(model.state_dict(), NAME)