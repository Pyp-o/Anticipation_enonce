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

############### FUNCTIONS ###############
#DATA
# tableau en entrée a chaque dialogue séparé par un retour à la ligne
# séparation de chaque dialogue et stockage dans un tableau
def parseDialogs(tab):
    data = []
    for t in tab:
        data.append(t.split("\n"))
    return data

# chaque dialogue est séparé par "__eou__"
# séparation de chaque prise de parole dans chaque dialogue
def parseUtterances(tab):
    data = []
    for t in tab:
        data.append(t[0].split("__eou__"))
    return data

# les fonctions ci dessus permettent de séparer les prises de paroles mais introduisent des cases videss
# suppression des cases vides situées en fin de dialogue
def deleteLastElement(tab):
    for t in tab:
        del t[-1]
    return tab

def vocabConstruct(tab):
    data = []
    for t in tab:
        data.append(t.split())
    return data

############### MAIN ###############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#importing and preparing data data
df = open(r'../../DataBase/dialog/dialogues_text.txt', encoding='utf-8')
data = parseDialogs(df)
print(vocabConstruct(df))
df.close()
data = parseUtterances(data)
deleteLastElement(data)



"""
#vocab construct
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}
"""