import timeit
import sys


setup1 = """
from gensim.models import KeyedVectors

print("importing GloVe...")
filename = '../embedding/glove.6B.100d.txt.word2vec'            #relative path from main
glove = KeyedVectors.load_word2vec_format(filename, binary=False)
print("GloVe imported")

word = "text"
vector = glove.get_vector(word)
"""

setup2 = """
import torch
import dataPrep
import torch.nn as nn
import numpy as np
import random
from gensim.models import KeyedVectors
import dataHandlingWE
import pickle
from os.path import exists

#-------------- No random --------------#
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)



#-------------- Parametres --------------#
FILENAME = "./Data/WEdata2.txt"
SUBSAMPLE = 100       #si 0 on prend tout le jeu de données
DATA_SUBSAMPLE = int(SUBSAMPLE/0.9) #number of phrases in the whole set
BATCH_SIZE = 250  #number oh phrases in every subsample (must respect SUBSAMPLE*BATCH_SIZE*(UTT_LEN/2)*N_FEATURES=tensor_size)
MIN_LEN = 4
MAX_LEN = 10
TEST_SIZE = 20

LEARNING_RATE = 0.001
N_FEATURES = 100    #100 pour GloVe
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 20000

TRAIN_SET = "train"  #"test"/"train"
PATH = "../../models/WE_trained_model_2_256_30000.pt"
NAME = "../../models/WE_trained_model_"+str(NUM_LAYERS)+"_"+str(HIDDEN_SIZE)+"_"+str(SUBSAMPLE)+".pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LSTM(nn.Module):
    def __init__(self, hidden_size=256, nfeatures=1, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = nfeatures

        self.lstm1 = nn.LSTM(input_size=nfeatures, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout)   #first lstm layer
        self.fc = nn.Linear(hidden_size, nfeatures) #linear layer to convert hidden processed data into 1 prediction

    def forward(self, x, prev_state=None):
        #default: h0 and c0 full of zeros
        x, prev_state = self.lstm1(x, prev_state)
        x = self.fc(x)
        return x, prev_state
        
if exists(FILENAME):
    print("importing data...")
    with open(FILENAME, "rb") as fp:
        data = pickle.load(fp)

else:
    print("preparing data...")
    data, glove = dataHandlingWE.prepareData('../../DataBase/dialog/dialogues_text.txt')
    print("data and GloVe imported !")

#-------------- limit lenght of each phrase to 8 words
data = dataPrep.limitLength2(data, min=MIN_LEN, max=MAX_LEN)  #limit length of phrases bewteen 4 and 10 by default
print(len(data[0]))
if DATA_SUBSAMPLE!=0:
    data = data[:DATA_SUBSAMPLE]
else :
    SUBSAMPLE = int(len(data)*0.7)

print("SUBSAMPLE", SUBSAMPLE)

#-------------- split dataset into trainset and testset
test = data[:1]
X_test, Y_test = dataPrep.splitX_y2(test)


print("converting arrays to tensors...")
T_X_test = []
#-------------- convert arrays as tensors
T_X_test = torch.FloatTensor(X_test)
T_X_test = torch.reshape(T_X_test, (-1, int(MAX_LEN/2), N_FEATURES)).to(device)
print("T_X_train.shape", T_X_test.shape)

print("model declaration")
#-------------- model declaration
model = LSTM(hidden_size=HIDDEN_SIZE, nfeatures=N_FEATURES, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
losses = []

h = torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE).to(device)
c = torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE).to(device)

model.load_state_dict(torch.load(PATH))
model.eval()

"""

setup3 ="""
from next_word_prediction import GPT2
gpt2 = GPT2()
sentence = 'Hello'"""


stmt1 = 'v=glove.get_vector(word)'
stmt2 = 'w=glove.similar_by_vector(vector, topn=1)'
stmt3 = 'predictions, (_,_) = model(T_X_test)'
stmt4 = """
for i in range(0,5):
    predict = gpt2.predict_next(sentence, 1)
    sentence += predict[0]"""


"""
NUMBER = [1, 10, 100, 1000]
timer1=[]
for num in NUMBER:
    print(num)
    timer1.append(timeit.timeit(stmt=stmt4, setup=setup3, number=num)/num)
print("temps encodage:",timer1)
"""

