from typing import Dict, Optional

import torch
import torch.nn as nn
from model.f1 import *

from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from transformers import BertModel

class Bert_proto(Model):
    def __init__(self, vocab,
                 pretrainpath,
                 metric='proto',
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Bert_proto, self).__init__(vocab, regularizer)
        self.encoder = BertModel.from_pretrained(pretrainpath)
        self.f1 = F1()
        self.metric = metric
        if self.metric == 'relation':
            self.relation = nn.Sequential(
                nn.Linear(768*3,512),
                nn.ReLU(),
                nn.Linear(512,1)
            )
        self.loss = nn.NLLLoss()
        initializer(self)
        print(metric)

    def forward(self,
                support_tokenid,
                support_mask,
                support_index,
                query_tokenid,
                query_mask,
                query_index,
                support_label = None,
                query_label = None,
                query_triggerset=None,
                classname=None):
        '''
        support batchsize, k, token_num
        query batchsize, q
        index batchsize, q, candidate_num, 2
        support index 
        '''

        batchsize = support_tokenid.size(0)
        q = query_tokenid.size(1)
        self.K = support_tokenid.size(1)
        candidate_num_q = query_index.size(2)
        candidate_num_s = support_index.size(2)

        support_tokenid_reshape = support_tokenid.reshape([-1, support_tokenid.size(-1)]).long()
        query_tokenid_reshape = query_tokenid.reshape([-1, query_tokenid.size(-1)]).long()

        support_mask = support_mask.reshape([-1, support_mask.size(-1)]).float()
        query_mask = query_mask.reshape([-1, query_mask.size(-1)]).float()


        support_feature = self.encoder(input_ids=support_tokenid_reshape, attention_mask=support_mask)[0] #BK,L,D
        query_feature = self.encoder(input_ids=query_tokenid_reshape, attention_mask=query_mask)[0] #BQ,L,D

        support_feature = support_feature.reshape([batchsize,self.K,-1,support_feature.size(-1)])
        query_feature = query_feature.reshape([batchsize,q,-1,query_feature.size(-1)])

        support_token = support_feature.new_zeros(batchsize,self.K,candidate_num_s,support_feature.size(-1))
        query_event = query_feature.new_zeros(batchsize,q,candidate_num_q,query_feature.size(-1))

        support_event = support_feature.new_zeros(batchsize,2,self.K,support_feature.size(-1))
        query_index = query_index.long()
        support_index = support_index.long()

        for i in range(batchsize):
            for j in range(self.K):
                for k in range(candidate_num_s):
                    if support_label[i,j,k] >= 0:
                        support_token[i,j,k] = torch.mean(support_feature[i,j,support_index[i,j,k,0]:support_index[i,j,k,1],:],dim=0)
                index1 = torch.where(support_label[i,j] == 0)[0]
                if len(index1) > 0:
                    support_event[i,0,j] = torch.mean(support_token[i,j,index1],dim=0)
                index2 = torch.where(support_label[i,j] == 1)[0]
                if len(index2) > 0:
                    support_event[i,1,j] = torch.mean(support_token[i,j,index2],dim=0)
            for j in range(q):
                for k in range(candidate_num_q):
                    if query_index[i,j,k,0] > 0:
                        query_event[i,j,k] = torch.mean(query_feature[i,j,query_index[i,j,k,0]:query_index[i,j,k,1],:],dim=0)

        support_event = torch.mean(support_event,dim=2) #B,2,d
        
        support_event_reshape = support_event.unsqueeze(1).unsqueeze(1).repeat([1,q,candidate_num_q,1,1]) #B,1,1,2,d
        query_event_reshape = query_event.unsqueeze(3).repeat([1,1,1,2,1])


        if self.metric == 'proto':
            class_logits = -torch.sum((support_event_reshape-query_event_reshape)**2,dim=-1) #B,q,c,2
        elif self.metric == 'relation':
            info_agg = torch.cat([
                support_event_reshape,
                query_event_reshape,
                torch.abs(support_event_reshape-query_event_reshape),
            ],dim=-1)
            class_logits = self.relation(info_agg).squeeze(-1) #B,q,c,2
        class_logits = torch.softmax(class_logits,dim=-1)

        if not self.training:
            self.f1(class_logits,query_label,triggerset = query_triggerset)

        class_logits_reshape = class_logits.reshape([-1,class_logits.size(-1)])
        label = query_label.reshape([-1])
        index1 = torch.where(label!=-1)[0]
        class_logits_reshape = class_logits_reshape[index1]
        label = label[index1]
        loss = self.ce_loss(class_logits_reshape,label)
        if self.training:
            self.f1(class_logits_reshape,label)

        output_dict = {
            'loss': loss,
            'class_logits':class_logits,
        }
        return output_dict
    
    def ce_loss(self,logits, labels):
        logits_log = torch.log(logits + 1e-16)
        return self.loss(logits_log, labels)

    def get_metrics(self, reset: bool = False):
        result = {}
        metric = self.f1.get_metric(reset)
        result["microp"] = metric["precision"]
        result["micror"] = metric["recall"]
        result["microf1"] = metric["f1"]
        return result

class Bert_causal(Model):
    def __init__(self, vocab,
                 pretrainpath,
                 metric='proto',
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Bert_causal, self).__init__(vocab, regularizer)
        self.encoder = BertModel.from_pretrained(pretrainpath)
        self.f1 = F1()
        self.metric = metric
        if self.metric == 'relation':
            self.relation = nn.Sequential(
                nn.Linear(768*3,512),
                nn.ReLU(),
                nn.Linear(512,1),
            )
        self.loss = nn.NLLLoss()
        initializer(self)
        print(metric)
    
    def getnegfeature(self,tokenid,mask,index,label):
        batchsize = tokenid.size(0)
        snum = tokenid.size(1)
        cnum = index.size(2)
        tokenid_reshape = tokenid.reshape([-1, tokenid.size(-1)]).long()
        mask = mask.reshape([-1, mask.size(-1)]).float()
        feature = self.encoder(input_ids=tokenid_reshape, attention_mask=mask)[0] #BK,L,D
        feature = feature.reshape([batchsize,snum,-1,feature.size(-1)])
        token = feature.new_zeros(batchsize,snum,cnum,feature.size(-1))
        event = feature.new_zeros(batchsize,snum,feature.size(-1))
        index = index.long()
        for i in range(batchsize):
            for j in range(snum):
                for k in range(cnum):
                    if index[i,j,k,0] > 0:
                        token[i,j,k] = torch.mean(feature[i,j,index[i,j,k,0]:index[i,j,k,1],:],dim=0)
                index1 = torch.where(label[i,j] == 0)[0]
                if len(index1) > 0:
                    event[i,j] = torch.mean(token[i,j,index1],dim=0)
        return event
    
    def getfeature(self,tokenid,mask,index,triggerp=None):
        batchsize = tokenid.size(0)
        snum = tokenid.size(1)
        cnum = index.size(2)
        tokenid_reshape = tokenid.reshape([-1, tokenid.size(-1)]).long()
        mask = mask.reshape([-1, mask.size(-1)]).float()
        feature = self.encoder(input_ids=tokenid_reshape, attention_mask=mask)[0] #BK,L,D
        if triggerp is not None:
            feature = feature.reshape([batchsize,snum,cnum,-1,feature.size(-1)])
        else:
            feature = feature.reshape([batchsize,snum,-1,feature.size(-1)])
        event = feature.new_zeros(batchsize,snum,cnum,feature.size(-1))
        index = index.long()
        for i in range(batchsize):
            for j in range(snum):
                for k in range(cnum):
                    if index[i,j,k,0] > 0:
                        if triggerp is not None:
                            event[i,j,k] = torch.mean(feature[i,j,k,index[i,j,k,0]:index[i,j,k,1],:],dim=0)
                        else:
                            event[i,j,k] = torch.mean(feature[i,j,index[i,j,k,0]:index[i,j,k,1],:],dim=0)
    
        if triggerp is not None:
            meanevent = event * triggerp.unsqueeze(-1)
            meanevent = torch.sum(meanevent,dim=2)
            return meanevent
        return event
    
    def getsupportfeature(self,support_tokenid,support_mask,support_index,support_label):
        batchsize = support_tokenid.size(0)
        self.K = support_tokenid.size(1)
        candidate_num_s = support_index.size(2)

        support_tokenid_reshape = support_tokenid.reshape([-1, support_tokenid.size(-1)]).long()

        support_mask = support_mask.reshape([-1, support_mask.size(-1)]).float()


        support_feature = self.encoder(input_ids=support_tokenid_reshape, attention_mask=support_mask)[0] #BK,L,D

        support_feature = support_feature.reshape([batchsize,self.K,-1,support_feature.size(-1)])

        support_token = support_feature.new_zeros(batchsize,self.K,candidate_num_s,support_feature.size(-1))
        support_event = support_feature.new_zeros(batchsize,2,self.K,support_feature.size(-1))

        support_index = support_index.long()

        for i in range(batchsize):
            for j in range(self.K):
                for k in range(candidate_num_s):
                    if support_label[i,j,k] >= 0:
                        support_token[i,j,k] = torch.mean(support_feature[i,j,support_index[i,j,k,0]:support_index[i,j,k,1],:],dim=0)
                index1 = torch.where(support_label[i,j] == 0)[0]
                if len(index1) > 0:
                    support_event[i,0,j] = torch.mean(support_token[i,j,index1],dim=0)
                index2 = torch.where(support_label[i,j] == 1)[0]
                if len(index2) > 0:
                    support_event[i,1,j] = torch.mean(support_token[i,j,index2],dim=0)
        return support_event

    def forward(self,
                support_tokenid,
                support_mask,
                support_index,
                query_tokenid,
                query_mask,
                query_index,
                support_postokenid=None,
                support_posmask=None,
                support_posindex=None,
                query_eventtokenid=None,
                support_triggerp=None,
                query_triggerp=None,
                query_eventmask=None,
                query_eventindex=None,
                query_eventlabel = None,
                support_label = None,
                query_label = None,
                query_triggerset=None):
        '''
        support batchsize, k, token_num
        query batchsize, q
        index batchsize, q, candidate_num, 2
        support index 
        '''

        batchsize = support_tokenid.size(0)
        q = query_tokenid.size(1)
        self.K = support_tokenid.size(1)
        candidate_num_q = query_index.size(2)
        candidate_num_s = support_index.size(2)

        if self.training:
            support_posevent = self.getfeature(support_postokenid,support_posmask,support_posindex,support_triggerp) #B,K,D
            support_posevent = torch.mean(support_posevent,dim=1).unsqueeze(1) #B,1,D

            support_negevent = self.getnegfeature(support_tokenid,support_mask,support_index, support_label) #B,K,c,D
            support_negevent = torch.mean(support_negevent,dim=1).unsqueeze(1) #B,1,D

            support_event = torch.cat([support_negevent,support_posevent],dim=1)

            query_event1 = self.getfeature(query_eventtokenid,query_eventmask,query_eventindex,query_triggerp) #B,q,D

            query_event2 = self.getfeature(query_tokenid,query_mask,query_index) #B,q,c,D
            query_event2 = query_event2.reshape([batchsize,q*candidate_num_q,-1])

            query_event = torch.cat([query_event1,query_event2],dim=1)

            query_label = torch.cat([query_eventlabel,query_label.reshape([batchsize,-1])],dim=-1)
            
            num = query_event.size(1)
            assert num == query_label.size(1)

            support_event_reshape = support_event.unsqueeze(1).repeat([1,num,1,1])
            query_event_reshape = query_event.unsqueeze(2).repeat([1,1,2,1])

        else:

            support_event = self.getsupportfeature(support_tokenid,support_mask,support_index,support_label) #B,2,k,d
            support_event = torch.mean(support_event,dim=2) #B,2,d

            query_event = self.getfeature(query_tokenid,query_mask,query_index)

            support_event_reshape = support_event.unsqueeze(1).unsqueeze(1).repeat([1,q,candidate_num_q,1,1]) #B,1,1,2,d
            query_event_reshape = query_event.unsqueeze(3).repeat([1,1,1,2,1])


        if self.metric == 'proto':
            class_logits = -torch.sum((support_event_reshape-query_event_reshape)**2,dim=-1) #B,q,c,2
        elif self.metric == 'relation':
            info_agg = torch.cat([
                support_event_reshape,
                query_event_reshape,
                torch.abs(support_event_reshape-query_event_reshape),
            ],dim=-1)
            class_logits = self.relation(info_agg).squeeze(-1) #B,q,c,2
        class_logits = torch.softmax(class_logits,dim=-1)

        if not self.training:
            self.f1(class_logits,query_label,triggerset = query_triggerset)

        class_logits_reshape = class_logits.reshape([-1,class_logits.size(-1)])
        label = query_label.reshape([-1])
        index1 = torch.where(label!=-1)[0]
        class_logits_reshape = class_logits_reshape[index1]
        label = label[index1]
        loss = self.ce_loss(class_logits_reshape,label)
        if self.training:
            self.f1(class_logits_reshape,label)

        output_dict = {
            'loss': loss,
            'class_logits':class_logits,
        }
        return output_dict
    
    def ce_loss(self,logits, labels):
        logits_log = torch.log(logits + 1e-16)
        return self.loss(logits_log, labels)

    def get_metrics(self, reset: bool = False):
        result = {}
        metric = self.f1.get_metric(reset)
        result["microp"] = metric["precision"]
        result["micror"] = metric["recall"]
        result["microf1"] = metric["f1"]
        return result