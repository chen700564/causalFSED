from allennlp.data.vocabulary import Vocabulary
from model.net import *
from allennlp.common import Params
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from reader import *


class Config:
    LR = 2e-5  # learning rate
    BATCH = 1


    PRE_FILE = 'bert-base-uncased'
    

    valquery = 'data/custom/harddevquery.json'

    trainreader2 = FewEventDetection_trainbaselinereader
    trainreader = FewEventDetection_traincausalreader
    devreader = FewEventDetection_devquerybaselinereader
    testreader = FewEventDetection_testbaselinereader

    
    Q = 2
    noiselength = 5
    nanoiselength = 5
    posnum = 10
    negativerate = 10


    
    sentence = "data/custom/sentence.json"
    trainfile = json.load(open('data/custom/train.json'))
    devfile = json.load(open('data/custom/dev.json'))
    testfile = json.load(open('data/custom/test.json'))


    instancenum = 2
    devinstancenum = 2
    testinstancenum = 4
    maxlength = 60

    model = Bert_causal
    model2 = Bert_proto

    labelnum = len(trainfile.keys())


    epochnum = instancenum * labelnum
    devepochnum = devinstancenum * len(devfile.keys())

    backdooruse = {
        'proto':'support',
        'relation':'support+query'
    }
    


config = Config()
