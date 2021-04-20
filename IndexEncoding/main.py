import torch
import dataPrep
import models
import numpy as np
import random

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

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

#vocab construct random order
vocab = dataPrep.vocabConstruct(data)
vocab = sorted(vocab, key=str.lower) #tri du vocabulaire par ordre alphabétique

#make each phrase as an entry of array
data = dataPrep.dataAsArray(data)
data = dataPrep.rmSpaces(data)

#word encoding
word_to_ix, ix_to_word = dataPrep.encodeWord(vocab, 'index')

#scaler creationa and fitting
scaler = dataPrep.fitScaler(data, word_to_ix)

#limit lenght of each phrase to 8 words
data = dataPrep.limitLength(data, 6)
data = data[:100]
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
X_train = dataPrep.convertPhrasetoIx(X_train, word_to_ix)
y_train = dataPrep.convertPhrasetoIx(y_train, word_to_ix)

X_test = dataPrep.convertPhrasetoIx(X_test, word_to_ix)
y_test = dataPrep.convertPhrasetoIx(y_test, word_to_ix)

print("rescale data")
#rescale data
data = scaler.transform(X_train)
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
    T_X_train.append(torch.FloatTensor(X_train[i]))
    T_y_train.append(torch.FloatTensor(y_train[i]))
for i in range(len(X_train)):
    T_X_train[i] = torch.reshape(T_X_train[i], (1, 3, 1)).to(device)
    T_y_train[i] = torch.reshape(T_y_train[i], (1, 3, 1)).to(device)

for i in range(len(X_test)):
    T_X_test.append(torch.FloatTensor(X_test[i]))
    T_y_test.append(torch.FloatTensor(y_test[i]))
for j in range(len(X_test)):
    T_X_test[j] = torch.reshape(T_X_test[j], (1, 3, 1)).to(device)
    T_y_test[j] = torch.reshape(T_y_test[j], (1, 3, 1)).to(device)


print("model declaration")
#model declaration
model = models.LSTM(input_size=1, hidden_size=128, nfeatures=1, num_layers=2).to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
batch_size = 4

loss = []

print("training model")
#model training
for i in range(epochs):
    model.train()
    for j in range(len(T_X_train)):
        y_pred = model(T_X_train[j], device).to(device)
        single_loss = loss_function(y_pred, T_y_train[j])
        loss.append(single_loss.item())

        single_loss.backward()
        optimizer.step()
        model.zero_grad()
    if i%5 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.4f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.4f}')

print("model predicting")
#model predictions
predictions = []
l = len(T_X_test)
for j in range(l):
    predictions.append(model(T_X_train[j], device).to(device))

print("reverse predicted tensors to CPU")
#moving back tensors to CPU to treat tensors as numpy array
converted = dataPrep.reverseTensor(predictions)

start = T_X_train[:l]
start = dataPrep.reverseTensor(start)
start = scaler.inverse_transform(start)
converted = scaler.inverse_transform(converted)
expected = scaler.inverse_transform(y_train)

print("input :",start)
print("output :", converted)
print("target :", expected[:l])

start = dataPrep.convertIxtoPhrase(start, ix_to_word)
converted = dataPrep.convertIxtoPhrase(converted, ix_to_word)
expected = dataPrep.convertIxtoPhrase(expected, ix_to_word)

print("input :",start)
print("output :", converted)
print("target :", expected[:l])

#TODO impact de la couche de sortie sur les performances (Linear, Tanh, Sigmoid, etc...)
#TODO mesurer impact loss function et optimizer (attention de ne pas faire d'associations peu fiables)