import torch
import dataPrep
import models


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
"""vocab = sorted(vocab, key=str.lower) #tri du vocabulaire par ordre alphabétique"""

#make each phrase as an entry of array
data = dataPrep.dataAsArray(data)
data = dataPrep.rmSpaces(data)

#word encoding
word_to_ix, ix_to_word = dataPrep.encodeWord(vocab, 'index')

scaler = dataPrep.fitScaler(data, word_to_ix)

#limit lenght of each phrase to 8 words
data = dataPrep.limitLength(data, 8)
data = data[:10]
print("vocabulary construction")

#split dataset into trainset and testset
ind = int(len(data)*0.7)
train = data[:ind]
test = data[ind:]

print("split data into sets")
#split sets into input and output for training and testing
X_train, y_train = dataPrep.splitX_y(train, 4)
X_test, y_test = dataPrep.splitX_y(test, 4)

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
    T_X_train[i] = torch.reshape(T_X_train[i], (1, 4, 1)).to(device)
    T_y_train[i] = torch.reshape(T_y_train[i], (1, 4, 1)).to(device)

for i in range(len(X_test)):
    T_X_test.append(torch.FloatTensor(X_test[i]))
    T_y_test.append(torch.FloatTensor(y_test[i]))

for j in range(len(X_test)):
    T_X_test[j] = torch.reshape(T_X_test[j], (1, 4, 1)).to(device)
    T_y_test[j] = torch.reshape(T_y_test[j], (1, 4, 1)).to(device)


print("model declaration")
#model declaration
model = models.LSTM(input_size=1, hidden_size=4, nfeatures=1, num_layers=2).to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
batch_size = 4

print("training model")
for i in range(epochs):
    for j in range(len(T_X_train)):
        model.train()
        optimizer.zero_grad()
        y_pred = model(T_X_train[j], device).to(device)

        single_loss = loss_function(y_pred, T_y_train[j])
        single_loss.backward()
        optimizer.step()
    if i%5 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.4f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.4f}')

print("model predicting")
predictions = []
l = len(T_X_test)
for j in range(l):
    predictions.append(model(T_X_train[j], device).to(device))

print("reverse predicted tensors to CPU")
#moving back tensors to CPU to treat tensors as numpy array
converted = []
for p in predictions:
    a = []
    p = p.cpu()
    p = p.detach().numpy()
    for t in p[0]:
        a.append(t[0])
    converted.append(a)

print(converted)
converted = scaler.inverse_transform(converted)
print(converted)
print(scaler.inverse_transform(y_test))

#TODO comment decoder output of LSTM
#TODO impact de la couche de sortie sur les performances (Linear, Tanh, Sigmoid, etc...)
#TODO mesurer impact loss function et optimizer (attention de ne pas faire d'associations peu fiables)