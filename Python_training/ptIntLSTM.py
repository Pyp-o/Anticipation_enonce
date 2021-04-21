#------------------------------#
"""
Voici un script déclarant un LSTM faisant des prédictions d'entier
Il se sert ici de cuda (GPU) s'il est disponible, sinon du CPU
"""
#------------------------------#
import torch
import torch.nn as nn
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

def sliding_window(passengers, window):
    X = []
    y = []
    data = []
    i=0
    while(i+window+1<len(passengers)):
        data = passengers[i :i+window]
        X.append(data)
        y.append(passengers[i+window+1])
        i+=1
    return X, y

class Model(nn.Module):
    def __init__(self, seq_len=1, hidden_size=128, output_size=1, num_layers=1, batch_size=1):
        super().__init__()
        self.input_size=seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=False, bidirectional=False)   #first lstm layer
        self.fc = nn.Linear(self.hidden_size, self.output_size) #linear layer to convert hidden processed data into 1 prediction

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x)
        return x

######### DATA handling #########
#import data
flight_data = sns.load_dataset("flights")
# 144 mois de données = 12*12
passengers = flight_data['passengers'].values.astype(float)
scaler = MinMaxScaler(feature_range=(-1., 1.))
data_normalized = scaler.fit_transform(passengers.reshape(-1, 1))

X_train, y_train = sliding_window(data_normalized, 1)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_train = torch.reshape(X_train, (X_train.shape[0], -1, 1))
y_train = torch.reshape(y_train, (y_train.shape[0], 1, 1))

print(len(X_train))

######### MODEL #########
#model
model = Model(seq_len=1, hidden_size = 32, num_layers = 2, batch_size = 4, output_size=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = .01) #différence de performance est quasiment entierement liée à l'optimizer

epochs = 1000
Losses = np.zeros(epochs)


model.train()
for i in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    single_loss = loss_function(y_pred, y_train)
    Losses[i] = single_loss
    single_loss.backward()
    optimizer.step()
    if i%10 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

model.eval()
pred_data = []
real_pred_data = []
pred = model(X_train)
pred = pred.detach().numpy().reshape(-1)
pred_passengers = scaler.inverse_transform(pred.reshape(-1,1))
print(pred_passengers.reshape(-1))

# Loss
plt.figure()
plt.plot(np.log10(Losses))
plt.title('Learning curve')
plt.ylabel('loss: log10(MSE)')
plt.xlabel('epoch')
#plt.legend(['train', 'valid'], loc = 'upper left')
plt.show()
plt.close('all')

"""
# prediction
predYPlot = np.array(pred_passengers.reshape(-1))
predYPlot = np.pad(predYPlot, (inputSeqLength, 0), 'constant', constant_values=(np.NaN))
plt.figure()
plt.plot(predYPlot)
XPlot = np.arange(len(passengers) - inputSeqLength, dtype=float)
XPlot = np.pad(XPlot, (0, inputSeqLength), 'constant', constant_values=(np.NaN))
plt.scatter(XPlot, np.array(passengers), color='C2', marker='o')
plt.title('Prediction')
plt.ylabel('air passengegers')
plt.xlabel('months')
#plt.legend(['train', 'valid'], loc = 'upper left')
pathToFig = os.path.join('.', 'airPassengers_pred_LSTM.pdf')
plt.savefig(pathToFig, dpi = 96, papertype = 'a7')
plt.close('all')
"""