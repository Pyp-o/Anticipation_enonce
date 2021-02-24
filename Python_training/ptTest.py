import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

num_layers=1
input_size=4
batch = 1
hidden_size = 2

#hidden_size - the number of LSTM blocks per layer.
#input_size - the number of input features per time-step.
#num_layers - the number of hidden layers.
#number of LSTM blocks = hidden_size * num_layers
rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

#input_dim : (seq_len, batch, input_size).
#seq_len - the number of time steps in each input stream.
#batch - the size of each batch of input sequences.
input = Variable(torch.randn(1, batch, input_size)) # (seq_len, batch, input_size)
h0 = Variable(torch.randn(num_layers, batch, hidden_size)) # (num_layers, batch, hidden_size)
c0 = Variable(torch.randn(num_layers, batch, hidden_size))
output, hn = rnn(input, (h0, c0))
affine1 = nn.Linear(hidden_size, input_size)

print(output)
print(hn)