#------------------------------#
"""
PyTorch permet de créer des modèles de word embedding from scratch et de passer d'un vecteur à un mot en cherchant la distance euclidienne la plus faible entre deux vecteurs
Celle-ci se calcule par le dot-product des vecteurs, pour trouver la distance minimale, il faut un dot-product maximal
"""
#------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def print3DVectors(vectors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for vector in vectors:
        ax.quiver(0,0,0, vector[0], vector[1], vector[2])
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    plt.show()
    return 0

def reverseEmbedding(model, vector, ix_to_word):
    distance = torch.norm(model.embeddings.weight.data - vector, dim=1)
    nearest = torch.argmin(distance)
    nearest = int(nearest.numpy())
    return ix_to_word[nearest]

def wordEmbed(index,word_to_ix,ix_to_word, model):
    lookup_tensor = torch.tensor([word_to_ix[ix_to_word[index]]], dtype=torch.long)
    return model.embeddings(lookup_tensor)
########## MAIN ##########

CONTEXT_SIZE = 2
EMBEDDING_DIM = 3   #projection en 3D pour affichage, en temps normal, projection dans espace vectoriel plus étendu

# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now

# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2]) for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    print(f"epoch {epoch:1} -- loss : {total_loss:10.10f}")
    losses.append(total_loss)

out=[]
vec=[]
coo=[]
for context, target in trigrams:
    # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
    # into integer indices and wrap them in tensors)
    context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
    out.append(model.embeddings(context_idxs))


for o in out:
    o=o.detach().numpy()
    vec.append(o[0])

#centre des vecteurs en 0 pour obtenir coordonnées des points du vecteur
for i in range(len(vec)):
    coo.append( [0, 0, 0,vec[i][0], vec[i][1], vec[i][2]])

#print3DVectors(coo)

#word to vect
index = 4
tens = wordEmbed(index,word_to_ix,ix_to_word, model)

#vect to word
word = reverseEmbedding(model, tens, ix_to_word)
print(word)
print(tens)
print3DVectors(tens.detach().numpy())