##### LINKS #####

# https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
# https://discuss.pytorch.org/t/please-help-lstm-input-output-dimensions/89353
# https://medium.com/@masterofchaos/lstms-made-easy-a-simple-practical-approach-to-time-series-prediction-using-pytorch-fastai-103dd4f27b82

##### IMPORTS #####

import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#data preparation, return [number of passengers for 12 months, number of passengers for the next month]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

#LSTM model creation
#input_size : 1 feature described here : number of passengers
#hidden_layer_size : 100 neurons
#output size : prediction on 1 month
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=256, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm1 = nn.LSTM(input_size, hidden_layer_size)
        
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm1(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

#load dataset
flight_data = sns.load_dataset("flights")

"""
#plot number of passenger per month
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.xlabel('Months')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(flight_data['passengers'])
plt.show()
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set all data "passengers" to float
all_data = flight_data['passengers'].values.astype(float)
#print(all_data)

#parsing all data into test and train datasets
test_data_size = 12
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

#get min and max value to normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

#convert training dataset into tensor
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

#sequence length is 12 (12 months in a year)
train_window = 12

#dataprep
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

#model creation
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training model
epochs = 150

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%5 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

#predictions
fut_pred = 12

test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()

#prediction for 12 months -> call 12 times prediction of 1 month
for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

#reverse normalize predicted values
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))

#index for predicted months
x = np.arange(132, 132+fut_pred, 1)

#plot predicted values and real values
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['passengers'])
plt.plot(x,actual_predictions)
plt.show()

#plot zoom on predicted values
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)

plt.plot(flight_data['passengers'][-train_window:])
plt.plot(x,actual_predictions)
plt.show()