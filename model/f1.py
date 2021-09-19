import logging
from overrides import overrides
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
import copy

torch.set_printoptions(threshold=np.inf) 
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class F1(Metric):
    def __init__(self,k=1) -> None:
        self.tt = 0
        self.predictnums = 0
        self.goldnums = 0
        self.nalabels = 0
        self.k = k
        self.predictresult = []
        self.goldresult = []

    def decoder(self,logits,labels,triggerset):
        predict_label = torch.argmax(logits, dim=-1)
        predict_label = predict_label.masked_fill(labels==-1,-1)
        if logits.size(-1) == 3:
            predict_label = predict_label.masked_fill(predict_label==1,0)
            predict_label = predict_label.masked_fill(predict_label==2,1)
        predict_trigger = []
        goldnum = 0
        predictnum = 0
        tt = 0
        assert len(triggerset) == labels.size(0)
        for i in range(labels.size(0)):
            predict_trigger.append([])
            flag = 0
            goldnum += len(triggerset[i])
            for j in range(labels.size(1)):
                if predict_label[i][j] > 0:
                    if flag == 0:
                        newtrigger = [predict_label[i][j].item(),j,j]
                        flag = 1
                    else:
                        if newtrigger[0] == predict_label[i][j]:
                            newtrigger[-1] = j
                        else:
                            predictnum += 1
                            predict_trigger[i].append(newtrigger)
                            if newtrigger in triggerset[i]:
                                    tt += 1
                            newtrigger = [predict_label[i][j].item(),j,j]
                else:
                    if flag == 1:
                        predict_trigger[i].append(newtrigger)
                        if newtrigger in triggerset[i]:
                            tt += 1
                        predictnum += 1
                    flag = 0
            if flag == 1:
                predictnum += 1
                predict_trigger[i].append(newtrigger)
                if newtrigger in triggerset[i]:
                        tt += 1
        self.predictresult += predict_trigger
        self.goldresult += triggerset
        return predictnum,goldnum,tt,predict_trigger
                
                

    def __call__(self,logits, labels, triggerset = None):
        '''

        logits : (Batchsize, q, tokens, d) / (Batchsize, q*tokens, d)  ...  按照词语分类 / 按照句子分类
        labels : (Batchsize, q, tokens) / (Batchsize, q*tokens)


        '''
        # print(predictions)
        logits = logits.detach()
        labels = labels.detach()
        if triggerset is not None:
            if len(labels.size()) == 3:
                b,q,c = labels.size()
                logits = logits.reshape([b*q,c,-1])
                labels = labels.reshape([b*q,c])
                newtriggerset = []
                for i in triggerset:
                    newtriggerset += i
            else:
                q,c = labels.size()
                logits = logits.reshape([q,c,-1])
                newtriggerset = copy.copy(triggerset)
            predictnums,goldnums,tt,predict_trigger = self.decoder(logits,labels,newtriggerset)
        else:
            if len(logits.size()) == 2:
                predict_label = torch.argmax(logits, dim=-1)
                index1 = torch.where(labels > 0)[0]
                index2 = torch.where(predict_label > 0)[0]
                goldnums = len(index1)
                predictnums = len(index2)
                tt = len(torch.where(predict_label[index1] == labels[index1])[0])

        self.goldnums += goldnums
        self.predictnums += predictnums
        self.tt += tt

    def getmacro(self,goldtriggerset,predicttriggerset):
        assert len(goldtriggerset) == len(predicttriggerset)
        for i in range(len(goldtriggerset)):
            if len(goldtriggerset) > 0 :
                for j in goldtriggerset[i]:
                    goldlabel = j[0]
                    if goldlabel not in self.classset:
                        self.classset.append(goldlabel)
                    self.classgoldnums[goldlabel] += 1
            if len(predicttriggerset) > 0 :
                for j in predicttriggerset[i]:
                    predictlabel = j[0]
                    self.classpredictnums[predictlabel] += 1
                    if j in goldtriggerset[i]:
                        self.classtt[predictlabel] += 1

    def get_metric(self, reset: bool = False):
        result = {}
        if self.predictnums == 0:
            result["precision"] = 0
        else:
            result["precision"] = self.tt/self.predictnums
        if self.goldnums == 0:
            result["recall"] = 0
        else:
            result["recall"] = self.tt/self.goldnums
        if result["precision"] <= 0 and result["recall"] <= 0:
            result["f1"] = 0
        else:
            result["f1"] = 2*max(0, result["precision"]) * max(0, result["recall"]) / \
                (max(0, result["precision"])+max(0, result["recall"]))
        if reset:
            self.reset()
        return result

    @overrides
    def reset(self):
        self.goldnums = 0
        self.predictnums = 0
        self.tt = 0
        self.predictresult = []
        self.goldresult = []
