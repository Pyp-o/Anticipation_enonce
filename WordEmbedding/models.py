import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, hidden_size=256, nfeatures=1, num_layers=1, dropout=0.3):
        super().__init__()
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = nfeatures
        """

        self.lstm1 = nn.LSTM(input_size=nfeatures, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout)   #first lstm layer
        self.fc = nn.Linear(hidden_size, nfeatures) #linear layer to convert hidden processed data into 1 prediction

    def forward(self, x, state=None):
        #default: h0 and c0 full of zeros
        x, prev_state = self.lstm1(x, state)
        x = self.fc(x)
        return x, prev_state

"""
liste des fonctions de sortie à tester
logsoftmax
sigmoid
relu
"""