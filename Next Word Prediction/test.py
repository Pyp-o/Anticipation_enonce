import torch
import dataPrep
import models
import numpy as np
import random
from gensim.models import KeyedVectors
import dataHandlingWE
import pickle
from os.path import exists
import sys

FILENAME = "WEdata.txt"






if exists(FILENAME):
    print("importing data...")
    with open(FILENAME, "rb") as fp:
        data = pickle.load(fp)
    print("data imported")

data = dataPrep.limitLength2(data, min=4, max=10)  # limit length of phrases bewteen 4 and 10 by default

X, Y = sliding_XY(data)