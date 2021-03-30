import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, output_size=1, num_layers=1, seq_length=12, batch_size=1):
        super().__init__()
        input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)   #first lstm layer
        #self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size*2, num_layers=num_layers, batch_first=True, bidirectional=False)    #second lstm layer
        self.fc = nn.Linear(hidden_size, output_size) #linear layer to convert hidden processed data into 1 prediction

    def forward(self, x, device):
        h0_0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)     #hidden layer random init
        c0_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)     #cells zero init
        x, (hn, cn) = self.lstm1(x, (h0_0, c0_0))
        x = self.fc(x)

        return x