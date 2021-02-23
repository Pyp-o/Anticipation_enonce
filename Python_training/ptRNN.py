import torch
import torch.nn as nn

########## CLASS ##########
class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(LSTM, self).__init__()	#calling RNN constructor

		self.hidden_size = hidden_size	#number of features in the hidden state

		self.i2h = nn.LSTM(input_size, hidden_size)
		self.i2o = nn.LSTM(hidden_size, output_size)

	def forward(self, data, last_hidden):
		input = torch.cat((data, last_hidden),1)
		hidden = self.i2h(input)
		output = self.i2o(hidden)

		return hidden, output

########## MAIN ##########
rnn = RNN(10, 20, 2)

print(rnn)
#loss parameter
loss_fn = nn.MSELoss()

#data parameters
batch_size = 10
TIMESTEPS = 5

# Create some fake data
batch = torch.randn(batch_size, 10)
hidden = torch.zeros(batch_size, 20)
target = torch.zeros(batch_size, 2)

#training
loss = 0
for t in range(TIMESTEPS):
    # yes! you can reuse the same network several times,
    # sum up the losses, and call backward!
    hidden, output = rnn(batch, hidden)
    loss += loss_fn(output, target)
loss.backward()

hidden, pred = rnn(batch, hidden)
