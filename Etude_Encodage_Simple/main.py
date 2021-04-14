import torch
import dataPrep
import models
import numpy as np
import random

SEED = 10
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

encoding = 'oneHot'
n_features = 1

############### MAIN ###############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("import data")
#importing and preparing data
df = open(r'../../DataBase/dialog/dialogues_text.txt', encoding='utf-8')    #beware, dataset changed : don ' t => don't
data = dataPrep.parseDialogs(df)
df.close()

print("prepare data")
data = dataPrep.parseUtterances(data)
data = dataPrep.parsePhrase(data)
data = dataPrep.removePunctuation(data)
data = data[:2000]
#vocab construct random order
vocab = dataPrep.vocabConstruct(data)

#make each phrase as an entry of array
data = dataPrep.dataAsArray(data)
data = dataPrep.rmSpaces(data)

#word encoding
word_to_oneHot, oneHot_to_word, word_to_ix, ix_to_word, n_features = dataPrep.encodeWord(vocab, encoding)
print(n_features)

#scaler creation and fitting
if encoding == 'index':
    scaler = dataPrep.fitScaler(data, word_to_oneHot)

#limit lenght of each phrase to 8 words
data = dataPrep.limitLength(data, 6)
data = data[:50]

print("vocabulary construction")
#split dataset into trainset and testset
ind = int(len(data)*0.9)
train = data[:ind]
test = data[ind:]

print("split data into sets")
#split sets into input and output for training and testing
X_train, y_train = dataPrep.splitX_y(train, 3)
X_test, y_test = dataPrep.splitX_y(test, 3)

print("convert words into numbers")
#convert words to ix
X_train = dataPrep.convertPhrasetoIx(X_train, word_to_oneHot)
y_train = dataPrep.convertPhrasetoIx(y_train, word_to_oneHot)

X_test = dataPrep.convertPhrasetoIx(X_test, word_to_oneHot)
y_test = dataPrep.convertPhrasetoIx(y_test, word_to_oneHot)

print("rescale data")
#rescale data
if encoding == 'index':
    X_train = scaler.transform(X_train)
    y_train = scaler.transform(y_train)

    X_test = scaler.transform(X_test)
    y_test = scaler.transform(y_test)

print("convert arrays to tensors")
T_X_train = []
T_y_train = []
T_X_test = []
T_y_test = []
#convert arrays as tensors
for i in range(len(X_train)):
    T_X_train.append(torch.tensor(X_train[i], dtype=torch.float))
    T_y_train.append(torch.tensor(y_train[i]))
for i in range(len(X_train)):
    T_X_train[i] = torch.reshape(T_X_train[i], (1, -1, n_features)).to(device)
    T_y_train[i] = torch.reshape(T_y_train[i], (1, -1, n_features)).to(device)

for i in range(len(X_test)):
    T_X_test.append(torch.tensor(X_test[i], dtype=torch.float))
    T_y_test.append(torch.tensor(y_test[i]))
for j in range(len(X_test)):
    T_X_test[j] = torch.reshape(T_X_test[j], (1, -1, n_features)).to(device)
    T_y_test[j] = torch.reshape(T_y_test[j], (1, -1, n_features)).to(device)

print(T_X_train[0].shape)

#del X_train, y_train

print("model declaration")
#model declaration
model = models.LSTM(hidden_size=5, nfeatures=n_features, num_layers=2).to(device) #2 couches 512 cells pour 26000 mots
loss_function = torch.nn.CTCLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 500
batch_size = 4

loss = []

print("training model")
#model training
for i in range(epochs):
    model.train()
    for j in range(len(T_X_train)):
        y_pred = model(T_X_train[j], device).to(device)
        single_loss = loss_function(y_pred, T_y_train[j], torch.tensor([1, 3, n_features], dtype=torch.int), torch.tensor([1, 3, n_features], dtype=torch.int))
        loss.append(single_loss.item())

        single_loss.backward()
        optimizer.step()
        model.zero_grad()
    if i%5 == 1:
        print(f'epoch: {i+1:3} loss: {single_loss.item():10.10f}')

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

inp = dataPrep.reverseOneHot(X_train[:l], oneHot_to_word)
out = dataPrep.reverseOneHot(y_train[:l], ix_to_word)
predictions = dataPrep.oneHotClean(predictions, ix_to_word)

print("input:", inp)
print("predicted", predictions)
print("output", out)

#TODO tester de nuit avec corpus entier et ~10000 epochs
#TODO modifier loss function pour l'encodage index

