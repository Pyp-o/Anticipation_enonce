import torch
import torch.nn as nn

# Target are to be un-padded
T = 50      # Input sequence length
N = 16      # Batch size
C = 20      # Number of classes (including blank)

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)

# Initialize random batch of targets (0 = blank, 1:C = classes)
target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)
ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()

print("input", input.shape)                     #log probs: (T, N, C)
print("input length", input_lengths.shape)      #input lengths: (N)
print("input length", input_lengths)
print()
print("target", target.shape)                   #target: (N, S)
print("target length", target_lengths.shape)    #target lengths: (N)
print("target", target)

"""
T : input length
N: batch size
C: nfeatures
S: max target length
"""