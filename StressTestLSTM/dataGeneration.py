#TODO tester les limites du LSTM en utilisant des prédictions simples (in: a out: a; in: ab out: ab)

import random
import torch
import torch.nn as nn
torch.manual_seed(0)
#________________________________________________________________________________________________________________________________________________________#
class LSTM(nn.Module):
    def __init__(self, hidden_size=256, nfeatures=1, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = nfeatures

        self.lstm1 = nn.LSTM(input_size=nfeatures, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)   #first lstm layer
        self.fc = nn.Linear(hidden_size, nfeatures) #linear layer to convert hidden processed data into 1 prediction

    def forward(self, x, device):
        h0_0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)     #hidden layer random init
        c0_0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)     #cells zero init
        x, (hn, cn) = self.lstm1(x, (h0_0, c0_0))
        x = self.fc(x)
        return x

#________________________________________________________________________________________________________________________________________________________#

#MAIN#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

phrase0 = 'how are you'.split()
phrase1 = 'who are you'.split()
phrase2 = 'are you ok'.split()
phrase3 = 'i am good'.split()

word_to_ix = {'how':0, 'are':1, 'you':2, 'who':3, 'ok':4, 'i':5, 'am':6, 'good':7}

X_train = []
y_train = []

length = 400

for i in range(length):
    if i%4==0:
        X_train.append(torch.FloatTensor([0, 1]))
        y_train.append(torch.FloatTensor([2, 2]))
    elif i%4==1:
        X_train.append(torch.FloatTensor([3, 1]))
        y_train.append(torch.FloatTensor([2, 2]))
    elif i%4==2:
        X_train.append(torch.FloatTensor([1, 2]))
        y_train.append(torch.FloatTensor([[4, 4]]))
    else:
        X_train.append(torch.FloatTensor([5, 6]))
        y_train.append(torch.FloatTensor([7, 7]))
for i in range(len(X_train)):
    X_train[i] = torch.reshape(X_train[i], (1, 2, 1)).to(device)
    y_train[i] = torch.reshape(y_train[i], (1, 2, 1)).to(device)

X_test = torch.FloatTensor([[0, 1], [3, 1], [1, 2], [5, 6], [1, 3]])
X_test = torch.reshape(X_test, (-1, 2, 1)).to(device)

print(X_train)
print("\n***********\n")
model = LSTM(hidden_size=32, nfeatures=1, num_layers=1).to(device) #2 couches 512 cells pour 26000 mots
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200
loss = []

for i in range(epochs):
    model.train()
    for j in range(len(X_train)):
        y_pred = model(X_train[j], device).to(device)
        single_loss = loss_function(y_pred, y_train[j])
        loss.append(single_loss.item())

        single_loss.backward()
        optimizer.step()
        model.zero_grad()
    if i % 5 == 1:
        print(f'epoch: {i + 1:3} loss: {single_loss.item():10.10f}')

print(f'epoch: {i + 1:3} loss: {single_loss.item():10.10f}')

print("X_test", X_test)
print(model(X_test, device).to(device))

#TODO implémentation word embedding pour LSTM (MSE possiblement utilisable)
#TODO implémenter nouvelle Loss
# https://towardsdatascience.com/building-a-next-word-predictor-in-tensorflow-e7e681d4f03f