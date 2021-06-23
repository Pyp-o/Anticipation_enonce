"""
la différence de performances entre le modèle de Pierre C et le mien viens de l'optimizer
(Adamn contre SGD), lors du changement de l'un vers l'autre les résultats deviennent similaires
"""


import torch
import torch.nn as nn
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os

torch.manual_seed(0)

def sliding_window(passengers, window):
    X = []
    y = []
    i=0
    while(i+window+1<len(passengers)):
        data = passengers[i :i+window]
        X.append(data)
        y.append(passengers[i+window+1])
        i+=1
    return X, y

class LSTM(nn.Module):
    def __init__(self, hidden_size=256, nfeatures=1, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = nfeatures

        self.lstm1 = nn.LSTM(input_size=nfeatures, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)   #first lstm layer
        self.fc = nn.Linear(hidden_size, nfeatures) #linear layer to convert hidden processed data into 1 prediction

    def forward(self, x):
        #default: h0 and c0 full of zeros
        x, prev_state = self.lstm1(x)
        x = self.fc(x)
        return x

def plotTrain(Losses, output_file):
    # Loss
    plt.figure()
    plt.plot(np.log10(Losses))
    plt.title('Learning curve')
    plt.ylabel('loss: log10(MSE)')
    plt.xlabel('epoch')
    # plt.legend(['train', 'valid'], loc = 'upper left')
    #plt.show()
    out = os.path.join('.', 'train'+output_file)
    plt.savefig(out, dpi=96, papertype='a7')
    plt.close('all')
    return

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

def plotResults(pred_passengers, inputSeqLength, passengers, output_file):
    predYPlot = np.array(pred_passengers.reshape(-1))
    predYPlot = np.pad(predYPlot, (inputSeqLength, 0), 'constant', constant_values=(np.NaN))
    plt.figure()
    plt.plot(predYPlot)
    XPlot = np.arange(len(passengers) - inputSeqLength, dtype=float)
    XPlot = np.pad(XPlot, (0, inputSeqLength), 'constant', constant_values=(np.NaN))
    plt.plot(XPlot, np.array(passengers), color='C2')
    plt.title('Prediction')
    plt.ylabel('air passengers')
    plt.xlabel('months')
    plt.legend(['valid', 'train'], loc='upper left')
    #plt.show()

    out = os.path.join('.', 'values_' + output_file)
    plt.savefig(out, dpi=96, papertype='a7')
    plt.close('all')


def main(BATCH_SIZE, LEARNING_RATE, N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, EPOCHS, OUTPUT_FILE):


    T_X_train = []
    T_y_train = []
    T_X_test = []
    T_y_test = []
    # -------------- convert arrays as tensors
    T_X_train = torch.FloatTensor(X_train)
    T_y_train = torch.FloatTensor(Y_train)
    T_X_train = torch.reshape(T_X_train, (-1, BATCH_SIZE, 1, N_FEATURES))
    T_y_train = torch.reshape(T_y_train, (-1, BATCH_SIZE, 1, N_FEATURES))
    print("T_X_train.shape", T_X_train.shape)

    T_X_test = torch.FloatTensor(X_test)
    T_y_test = torch.FloatTensor(Y_test)
    T_X_test = torch.reshape(T_X_test, (-1, 1, N_FEATURES))
    T_y_test = torch.reshape(T_y_test, (-1, 1, N_FEATURES))
    print("T_X_test.shape", T_X_test.shape)

    print(len(X_train))

    ######### MODEL #########
    # model
    model = LSTM(hidden_size=HIDDEN_SIZE, nfeatures=N_FEATURES, num_layers=NUM_LAYERS)
    loss_function = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
    test_losses = []

    for i in range(EPOCHS):
        model.train()
        loss = 0
        predictions = model(T_X_test)
        single_loss = loss_function(predictions, T_y_test)
        test_losses.append(single_loss.item())
        model.zero_grad()

        for j in range(len(T_X_train)):
            y_pred = model(T_X_train[j])
            single_loss = loss_function(y_pred, T_y_train[j])
            loss += single_loss.item()
            single_loss.backward()
            optimizer.step()
            model.zero_grad()
        losses.append(loss / len(T_X_train))  # loss cumulée pour chaque epoch
        if i % 5 == 1:
            print(f'epoch:{i - 1:5}/{EPOCHS:3}\tloss: {single_loss.item():10.10f}')
    print(f'epoch: {i + 1:5}/{EPOCHS:5}\tloss: {single_loss.item():10.10f}')

    model.eval()
    pred_data = []
    real_pred_data = []
    pred = model(T_X_test)
    pred = pred.detach().numpy().reshape(-1)
    pred_passengers = scaler.inverse_transform(pred.reshape(-1, 1))

    #plotTrain(losses, test_losses, OUTPUT_FILE)
    plotTest(losses, test_losses, OUTPUT_FILE)

    plotResults(pred_passengers, 1, passengers, OUTPUT_FILE)

    return 0

######### DATA handling #########


BATCH_SIZE = [1, 2, 5, 10, 26, 65, 130]
LEARNING_RATE = [0.01, 0.001, 0.0001, 1e-5]
N_FEATURES = 1
HIDDEN_SIZE = [2, 4, 8, 16, 32, 64, 128, 256]
NUM_LAYERS = range(1,4)
EPOCHS = 1000

# import data
flight_data = sns.load_dataset("flights")
# 144 mois de données = 12*12
passengers = flight_data['passengers'].values.astype(float)
scaler = MinMaxScaler(feature_range=(-1., 1.))
data_normalized = scaler.fit_transform(passengers.reshape(-1, 1))
train = data_normalized[:132]
test = data_normalized
X_train, Y_train = sliding_window(train, 1)
X_test, Y_test = sliding_window(test, 1)

for bs in BATCH_SIZE:
    print("batch size = ", bs)
    output_file = 'bs_prelim_lr0-001_hs64_bs'+str(bs)+'_nl1_e1000.png'
    main(BATCH_SIZE=bs, LEARNING_RATE=.0001, N_FEATURES=N_FEATURES, HIDDEN_SIZE=64, NUM_LAYERS=2, EPOCHS=EPOCHS, OUTPUT_FILE=output_file)

for lr in LEARNING_RATE:
    print("learning rate = ", lr)
    output_file = 'lr_prelim_lr'+str(lr)+'_hs64_bs1_nl1_e1000.png'
    main(26, lr, N_FEATURES, 128, 2, EPOCHS, output_file)

for hs in HIDDEN_SIZE:
    print("hidden size = ", hs)
    output_file = 'hs_prelim_lr0-001_hs'+str(hs)+'_bs1_nl1_e1000.png'
    main(26, 0.0001, N_FEATURES, hs, 2, EPOCHS, output_file)

for nl in NUM_LAYERS:
    print("num layers = ", nl)
    output_file = 'nl_prelim_lr0-001_hs64_bs1_nl'+str(nl)+'_e1000.png'
    main(26, 0.0001, N_FEATURES, 128, nl, EPOCHS, output_file)