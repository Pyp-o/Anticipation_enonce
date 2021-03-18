import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

#non stochastic model
torch.manual_seed(1)

######################################################################

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

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

index = random.randint(0, len(word_to_ix)-1)

print(word_to_ix[ix_to_word[index]]) #index
print(ix_to_word[index])            #word linked to the index


#word to vector to tensor
embeds = nn.Embedding(len(vocab), 3)  # n words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix[ix_to_word[index]]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)

#revert embedding
distance = torch.norm(embeds.weight.data - hello_embed, dim=1)
nearest = torch.argmin(distance)
nearest = int(nearest.numpy())
word = ix_to_word[nearest]
